# import necessary modules
import glob
import numpy as np
import os
import pandas as pd
import random
import torch

from functools import reduce
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.morphology import binary_dilation
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


class PolarCircle(Dataset):
    def __init__(self, config, mode):
        super().__init__()
        self.dir = config["dir"]

        # set the train/val split
        self.fold = config["fold"]
        self.splits = pd.read_pickle(os.path.join(os.path.dirname(os.path.realpath(__file__)), "splits_5_final.pkl"))

        # dataset config
        self.normalize = config["normalize"]
        self.pad_t = config["pad"][0]
        self.pad_z = config["pad"][1]
        self.pad_i = config["pad"][2]

        # get the image list
        self.images = []
        for opt in ("train", "val"):
            images = list(self.splits[0][opt])
            for image in images:
                split = image.split("_", 1)
                self.images.append(os.path.join(self.dir, split[0], "image",
                                                f"image{split[1]}", f"{split[1]}.mhd"))
        self.images = sorted(self.images)

        # get all polar pre-processed data
        self.cuts = {}
        for image in self.images:
            cuts = glob.glob(os.path.join(os.path.dirname(image.replace("image", "cut")), "polar_img*"))
            self.cuts[image] = cuts

        # set data
        base_image = os.path.join(self.dir, "{}", "image", "image{}", "{}.mhd")
        data = self.splits[self.fold][mode]
        data = [base_image.format(pat.split("_", 1)[0], pat.split("_", 1)[1], pat.split("_", 1)[1]) for pat in data]
        self.id_list = data

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image = self.id_list[index]
        image_cuts = self.cuts[image]

        img, cp, l, ncp, out = self.extract_cut(image_cuts)

        # separate healthy vessel data from plaque-rich vessel data
        loc_l = np.nonzero(l.sum(0))[0]
        loc_out = np.where(out.sum(0) > 0)
        loc_cp = cp.sum(0) > 0
        loc_ncp = ncp.sum(0) > 0
        loc_mix = np.where(binary_dilation(loc_cp & loc_ncp, iterations=self.pad_i + self.pad_z))

        loc_cp = np.where(loc_cp)
        loc_cp = np.setdiff1d(loc_cp, loc_mix)
        loc_ncp = np.where(loc_ncp)
        loc_ncp = np.setdiff1d(loc_ncp, loc_mix)

        start_and_end = reduce(np.union1d, (loc_l, loc_out, loc_mix))
        start_and_end = np.concatenate((start_and_end[: self.pad_i], start_and_end[-self.pad_i:]))

        loc_out = np.setdiff1d(loc_out, start_and_end)
        loc_cp = np.setdiff1d(loc_cp, start_and_end)
        loc_ncp = np.setdiff1d(loc_ncp, start_and_end)
        loc_mix = np.setdiff1d(loc_mix, start_and_end)

        loc_l = np.setdiff1d(loc_l, loc_out)
        loc_l = np.setdiff1d(loc_l, start_and_end)

        # take sub-cut
        rand = random.random()
        if rand < 0.8 and loc_ncp.size > 0:
            center = np.random.choice(loc_ncp)
        elif rand < 0.5 and loc_cp.size > 0:
            center = np.random.choice(loc_cp)
        elif loc_mix.size > 0:
            center = np.random.choice(loc_mix)
        elif loc_l.size > 0:
            center = np.random.choice(loc_l)
        else:
            o = np.nonzero(l.sum(axis=(0, 1)))[0]
            o = np.setdiff1d(o, np.concatenate((o[: 10], o[-10:])))
            center = np.random.choice(o)

        # hard-coded, make this variable at some point
        center += 24
        img = img[24 - self.pad_t: -24 + self.pad_t, :,
                  center - self.pad_z - self.pad_i: center + self.pad_z + self.pad_i + 1]

        center -= 24
        r_l = l[:, center - self.pad_i: center + self.pad_i + 1]
        r_out = out[:, center - self.pad_i: center + self.pad_i + 1]
        r_cp = cp[:, center - self.pad_i: center + self.pad_i + 1]
        r_ncp = ncp[:, center - self.pad_i: center + self.pad_i + 1]

        cls, img, r_cp, r_l, r_ncp, r_out = self.to_torch(img, r_cp, r_l, r_ncp, r_out)

        if random.random() < 0.5:
            img = torch.flip(img, [-1])
            r_l = torch.flip(r_l, [-1])
            r_out = torch.flip(r_out, [-1])
            r_cp = torch.flip(r_cp, [-1])
            r_ncp = torch.flip(r_ncp, [-1])
            cls = torch.flip(cls, [-1])
        if random.random() < 0.5:
            img = torch.flip(img, [0])
            r_l = torch.flip(r_l, [0])
            r_out = torch.flip(r_out, [0])
            r_cp = torch.flip(r_cp, [0])
            r_ncp = torch.flip(r_ncp, [0])
            cls = torch.flip(cls, [0])

        if self.normalize:
            img = self.norm(img)
        return img[None, ...], (r_l, r_out, r_cp, r_ncp, cls)

    @staticmethod
    def to_torch(img, r_cp, r_l, r_ncp, r_out):
        cls = np.zeros_like(r_out, dtype=np.int8)
        cls[r_cp != 0] += 1
        cls[r_ncp != 0] += 2

        cls = torch.from_numpy(cls).long()
        img = torch.from_numpy(img).float()
        r_l = torch.from_numpy(r_l).float()
        r_out = torch.from_numpy(r_out).float()
        r_cp = torch.from_numpy(r_cp).float()
        r_ncp = torch.from_numpy(r_ncp).float()
        return cls, img, r_cp, r_l, r_ncp, r_out

    @staticmethod
    def extract_cut(image_cuts):
        cut = random.choice(image_cuts)
        img = np.load(cut)

        r_l = np.load(cut.replace("img", "l"))
        r_cp = np.load(cut.replace("img", "cp"))
        r_ncp = np.load(cut.replace("img", "ncp"))
        r_out = r_cp + r_ncp
        return img, r_cp, r_l, r_ncp, r_out

    @staticmethod
    def norm(mpr):
        mpr = np.clip(mpr, a_min=-760, a_max=1240)
        return (mpr - 240) / 1000


class PolarAll(PolarCircle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        image = self.id_list[index]
        image_cuts = [cut for cut in self.cuts[image] if len(os.path.basename(cut)) <= 17]

        img, r_cp, r_l, r_ncp, r_out = self.extract_cut(image_cuts)

        loc_l = np.nonzero(r_l.sum(0))[0]
        img = img[24 - self.pad_t: -24 + self.pad_t, :, loc_l[0] + 24 - self.pad_z: loc_l[-1] + 24 + self.pad_z + 1]
        r_l = r_l[:, loc_l[0]: loc_l[-1] + 1]
        r_cp = r_cp[:, loc_l[0]: loc_l[-1] + 1]
        r_ncp = r_ncp[:, loc_l[0]: loc_l[-1] + 1]
        r_out = r_out[:, loc_l[0]: loc_l[-1] + 1]

        cls, img, r_cp, r_l, r_ncp, r_out = self.to_torch(img, r_cp, r_l, r_ncp, r_out)

        if self.normalize:
            img = self.norm(img)
        return img[None, ...], (r_l, r_out, r_cp, r_ncp, cls)


class CADRAData(Dataset):
    def __init__(self, config, mode, dset):
        self.config = config
        self.dir = os.path.join(config["dir"], dset)
        self.dir_signal = os.path.join(config["dir_signal"], dset, "plaque_major", "RoelOriginSegm",
                                       "CircleNN_VanillaClassifierRGMeanMax", "plaque{}", "{}", "signal.npy")
        self.mode = mode

        self.sheet_include = os.path.join(config["dir"], "annotations", "CADRADS_curated.xlsx")
        if dset == "IDR_CADRADS":
            self.sheet = os.path.join(config["dir"], "annotations", "CADRADS_IDR.xlsx")
        else:
            self.sheet = self.sheet_include

        self.images = []
        self.images_train = []
        self.images_test = []

        self.CADRADS = {}
        self.modifiers = {}
        self.signal = {}
        self.counts = {"0": 0., "1": 0., "2": 0., "3": 0., "4": 0., "5": 0.}
        remove, sheet_include = self.make_dset(config, dset)

        # create folds
        if self.mode != "test":
            self.folds = []
            for ind_train, ind_val in KFold(n_splits=5, random_state=1, shuffle=True).split(self.images_train):
                fold_train = []
                for ind in ind_train:
                    fold_train.append(self.images_train[ind])

                fold_val = []
                for ind in ind_val:
                    fold_val.append(self.images_train[ind])
                self.folds.append([fold_train, fold_val])

        if self.mode == "test":
            self.id_list = self.images_test
        elif self.mode == "all":
            self.id_list = self.images.copy()

        if dset == "Cardiac_UMCU":
            sheet_include = pd.read_excel(self.sheet_include, sheet_name="Cardiac_UMCU_train", index_col=0)
        for image in self.images:
            im = os.path.basename(image)[:-4]
            signal = self.dir_signal.format(im, im)

            if dset == "Cardiac_UMCU":
                id_ = os.path.basename(image)[:-4]
                value = sheet_include.loc[id_]
                remove = value["modifier_major"]

            if os.path.exists(signal):
                signal = np.load(signal)
                if dset == "Cardiac_UMCU" and isinstance(remove, str):
                    # CAD-RADS N
                    remove = remove.split(", ")
                    for r in remove:
                        if r == "RCA":
                            signal[0] = 0.
                        elif r == "LAD":
                            signal[1] = 0.
                        elif r == "RCA":
                            signal[2] == 0.
                    if signal.sum() == 0:
                        # failed to extract centerlines
                        print(image)
                        self.id_list.remove(image)
            else:
                if image in self.id_list:
                    self.id_list.remove(image)
                continue

            # combine plaque signals, remove VAE encoding
            signal[:, 1] = signal[:, 1] + signal[:, 2]
            signal = signal[:, :2]

            if config["cutoff"]:
                # remove signal after non-significant radius threshold
                for i in range(len(signal)):
                    if len(np.nonzero(signal[i, 0])[0]) == 0:
                        continue

                    irr = gaussian_filter1d(signal[i, 0], 5, mode="nearest") < (np.pi * 0.75 ** 2)
                    irr = np.flatnonzero(np.diff(np.r_[~irr[0], irr]))
                    signal[i, :, irr[-1]:] = 0

            if config["normalize"]:
                for i in range(len(signal)):
                    if len(np.nonzero(signal[i, 0])[0]) == 0:
                        continue

                    # percentual difference
                    signal[i, 0, 1:] = np.divide(np.diff(signal[i, 0]), signal[i, 0, 1:], out=np.zeros_like(signal[i, 0, 1:]), where=signal[i, 0, 1:] != 0)
                    signal[i, 0, 0] = 0

            self.signal[image] = signal

    def make_dset(self, dset):
        for opt in ("image", "test"):
            opt_ = "train" if opt == "image" else opt
            try:
                sheet = pd.read_excel(self.sheet, index_col=0)
                sheet_include = pd.read_excel(self.sheet_include, sheet_name=dset + "_" + opt_, index_col=0)
            except ValueError:
                continue

            images = os.path.join(self.dir, opt, "*", "*.mhd")
            images = sorted(glob.glob(images))
            for image in images:
                id_ = os.path.basename(image)[:-4]

                try:
                    remove = sheet_include.loc[id_]["remove"]
                except KeyError:
                    remove = sheet_include.loc[int(id_)]["remove"]

                if remove == "T" and self.config["cleaned"]:
                    continue

                if dset == "IDR_CADRADS":
                    value = sheet.loc[sheet["ID"] == id_]
                    score = value["CADRADS"].item()
                else:
                    value = sheet_include.loc[id_]
                    try:
                        score = value["score"].item()
                    except AttributeError:
                        score = value["score"]

                if score != score:
                    # handle NaN's
                    continue
                elif isinstance(score, str):
                    if any(char.isdigit() for char in score):
                        # we have either an in-between score or a modifier on top of CADRADS
                        self.CADRADS[image] = np.zeros(5, dtype=np.float32)
                        for s in score:
                            """
                            A: Severe stenosis in one or two vessels >=70 percent
                            B: Severe stenosis in three vessels >=70 percent, or LM >=50 percent
                            S: Stent
                            N: Nondiagnostic, additional evaluation needed
                            """
                            if s.isnumeric():
                                self.counts[s] += 1
                                self.CADRADS[image][:int(s)] += 1
                            elif s == "-":
                                self.modifiers[image] = 0
                            elif s == "A":
                                self.modifiers[image] = 1
                            elif s == "B":
                                self.modifiers[image] = 2
                            elif s == "S":
                                self.modifiers[image] = 3
                            elif s == "N":
                                self.modifiers[image] = 4
                        if np.max(self.CADRADS[image]) != 1:
                            self.CADRADS[image] /= 2
                elif isinstance(score, int):
                    # purely the CADRADS score
                    self.counts[str(score)] += 1
                    self.modifiers[image] = 0

                    self.CADRADS[image] = np.zeros(5, dtype=np.float32)
                    self.CADRADS[image][:score] = 1
                elif isinstance(score, float):
                    if score.is_integer():
                        score = int(score)

                        # purely the CADRADS score
                        self.counts[str(score)] += 1
                        self.modifiers[image] = 0

                        self.CADRADS[image] = np.zeros(5, dtype=np.float32)
                        self.CADRADS[image][:score] = 1
                    else:
                        lower = int(np.floor(score))
                        upper = int(np.ceil(score))

                        self.counts[str(lower)] += 1
                        self.counts[str(upper)] += 1
                        self.modifiers[image] = 0

                        self.CADRADS[image] = np.zeros(5, dtype=np.float32)
                        self.CADRADS[image][:lower] += 1
                        self.CADRADS[image][:upper] += 1
                        self.CADRADS[image] /= 2

                self.images.append(image)
                if opt == "image":
                    self.images_train.append(image)
                elif opt == "test":
                    self.images_test.append(image)
        return remove, sheet_include

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        image = self.id_list[index]

        coronaries = torch.from_numpy(self.signal[image])
        if coronaries.sum() == 0.:
            coronaries[:, 0] = 1

        score = torch.FloatTensor(self.CADRADS[image])
        modifier = self.modifiers[image]
        return coronaries, score, modifier
