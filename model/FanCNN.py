# import necessary modules
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import SimpleITK as sitk

from glob import glob
from os.path import basename, dirname, join, realpath
from scipy.ndimage.morphology import binary_dilation
from stl import mesh
from torch import Tensor
from torch.nn import *
from utils import build_tube_graph, fast_bilinear_interpolation, get_stretched_mpr, region_growing, separate


class FanCNN(Module):
    """
    CNN for processing polar-transformed coronary MPR data
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        dim = self.config["dim"]
        dim_max = self.config["dim_max"]
        self.dim = [dim * 2 ** k for k in range(self.config["n_layers"] + 1)]
        self.dim = [dim_max if i >= dim_max else i for i in self.dim]

        # properties
        self.dil_theta = self.config["dil_theta"]
        self.dil_z = self.config["dil_z"]
        self.n_channels = self.config["n_channels"]
        self.n_classes = self.config["n_classes"]
        self.n_groups = self.config["n_groups"]
        self.n_layers = self.config["n_layers"]

        # 1D (R-direction) convs
        self.convr = Sequential(
            Conv3d(self.n_channels, self.dim[0], kernel_size=(1, 7, 1), bias=False),
            BatchNorm3d(self.dim[0]),
            ReLU(inplace=True),
            Conv3d(self.dim[0], self.dim[0], kernel_size=(1, 7, 1), bias=False),
            BatchNorm3d(self.dim[0]),
            ReLU(inplace=True)
        )

        # 3D convs
        self.layers = ModuleList()
        self.bns = ModuleList()
        for i in range(self.n_layers):
            block = Sequential()
            block.append(Conv3d(self.dim[i], self.dim[i+1],
                                kernel_size=(3, 3, 3), dilation=(self.dil_theta[i], 1, self.dil_z[i]),
                                groups=self.n_groups, bias=False))
            self.bns.append(BatchNorm3d(self.dim[i+1]))

            if i == 0:
                block.append(BatchNorm3d(self.dim[i+1]))
                block.append(ReLU(inplace=True))
                block.append(Conv3d(self.dim[i+1], self.dim[i+1],
                                    kernel_size=(1, 3, 1), groups=self.n_groups, bias=False))
            self.layers.append(block)

        self.pre_final = Sequential(Conv3d(self.dim[-1], self.dim[-1],
                                           kernel_size=(1, 4, 1), groups=self.n_groups, bias=False),
                                    BatchNorm3d(self.dim[-1]))

        # output
        self.l = Conv3d(self.dim[-1], 1, kernel_size=(1, 1, 1))
        self.cp = Conv3d(self.dim[-1], 1, kernel_size=(1, 1, 1))
        self.ncp = Conv3d(self.dim[-1], 1, kernel_size=(1, 1, 1))
        self.cls = Conv3d(self.dim[-1], self.n_classes, kernel_size=(1, 1, 1))

    def encode(self, x):
        # R-direction convs
        x = self.convr(x)

        # 3D layers
        for i in range(self.n_layers):
            x = self.layers[i](x)
            x = self.bns[i](x)
            x = F.relu(x)

        return F.relu(self.pre_final(x))

    def forward(self, x):
        x = self.encode(x)

        r_l = F.leaky_relu(self.l(x))
        r_cp = F.leaky_relu(self.cp(x))
        r_ncp = F.leaky_relu(self.ncp(x))
        cls = F.leaky_relu(self.cls(x))
        return r_l, r_cp, r_ncp, cls


class TestFanCNN(FanCNN):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        self.to(self.device)

        self.pad = [7, 7]
        self.v_rad = np.linspace(0, self.config["n_rad"] * 2, self.config["n_rad"])
        self.v_theta = np.linspace(0.0, 2.0 * np.pi, (self.config["n_theta"] + 1))[:self.config["n_theta"]]

    def process(self, path_image, path_centerlines):
        image, spacing, offset = self.load_image(path_image)
        print(f"processing {basename(path_image)}...")

        for path_ctl in path_centerlines:
            try:
                ctls = [np.loadtxt(path_ctl)[:, :4]]
            except ValueError:
                # ASOCA data
                ctl = np.loadtxt(path_ctl, usecols=(1, 2, 3))
                ctl = np.concatenate((ctl, np.zeros((ctl.shape[0], 1))), axis=1)
                ctls = separate(ctl)

            for i, ctl in enumerate(ctls):
                mpr, rotmats, ctl = get_stretched_mpr(
                    centerline=ctl,
                    image=image,
                    mpr_width=self.config["ps"],
                    mpr_voxelsize=self.config["dr"],
                    offset=offset,
                    spacing=spacing,
                    point_spacing=self.config["dz"],
                    resample_line=True,
                )

                ctl_id = basename(path_ctl).replace(".txt", ".mhd").replace(".ctl.anno", ".mhd")
                ctl_name = join(dirname(path_ctl), ctl_id.replace("centerline", "mpr"))
                # if len(ctl) < 75:  # shorter than 3.75 cm
                #     print(f"skipping {basename(ctl_name)} with length {len(ctl) * self.config['dz']} mm")
                #     continue

                # create the image-level meshes
                path_out = join(dirname(dirname(path_ctl)), "automatic_segmentations", ctl_id.replace(".mhd", ""), "img")
                os.makedirs(path_out, exist_ok=True)
                meshes, _ = self.meshify(mpr, ctl, spacing, rotmats)

                for name, msh in meshes.items():
                    name = os.path.join(path_out, ctl_id.replace(".mhd", f"_{i}_{name}.stl"))
                    msh.save(name)

                # create the mpr-level meshes
                path_out = join(dirname(dirname(path_ctl)), "automatic_segmentations", ctl_id.replace(".mhd", ""), "mpr")
                os.makedirs(path_out, exist_ok=True)
                meshes, _ = self.meshify(mpr)

                for name, msh in meshes.items():
                    name = os.path.join(path_out, ctl_id.replace(".mhd", f"_{i}_{name}.stl"))
                    msh.save(name)
                self.save_image(join(path_out, ctl_id.replace(".mhd", f"_{i}.mhd")),
                                mpr, (self.config["dr"], self.config["dr"], self.config["dz"]), (0, 0, 0))

    def meshify(self, mpr, ctl=None, spacing=None, rotmats=None):
        """
        Partially legacy code as well, thank you Jelmer

        :param mpr: multi-planar reformatted image of coronary artery
        :param spacing_mpr: spacing of the mpr image
        :param ctl: the coronary artery centerline corresponding to the mpr image
        :param spacing: spacing of the original image
        :param rotmats: rotation matrices of the mpr image
        """
        mpr_polar = self.to_polar(mpr)
        z = mpr_polar.shape[2]

        # to torch
        mpr_polar = np.pad(mpr_polar, ((self.pad[0], self.pad[0]), (0, 0), (0, 0)), "wrap")
        mpr_polar = np.pad(mpr_polar, ((0, 0), (0, 0), (self.pad[1], self.pad[1])), "reflect")
        mpr_polar = torch.from_numpy(mpr_polar[None, None]).to(self.device)

        # forward
        r_l = np.zeros((len(self.config["weights"]), self.config["n_theta"], z), dtype=np.float32)
        r_cp = np.zeros((len(self.config["weights"]), self.config["n_theta"], z), dtype=np.float32)
        r_ncp = np.zeros((len(self.config["weights"]), self.config["n_theta"], z), dtype=np.float32)
        cls = torch.zeros(len(self.config["weights"]), 1, self.n_classes, self.config["n_theta"], 1, z).float().to(self.device)

        with torch.no_grad():
            for m, model in enumerate(self.config["weights"]):
                path = dirname(dirname(realpath(__file__)))
                ckpt = sorted(glob(join(path, "lightning_logs", model, "model", "*.pt")))[0]
                self.load(ckpt)

                outputs = self(mpr_polar)
                r_l[m] = torch.squeeze(outputs[0]).detach().cpu().data.numpy()
                r_cp[m] = torch.squeeze(outputs[1]).detach().cpu().data.numpy()
                r_ncp[m] = torch.squeeze(outputs[2]).detach().cpu().data.numpy()
                cls[m] = outputs[3]

        # regressors
        r_l = np.mean(r_l, axis=0) * self.config["ps"] / 63
        r_cp = np.mean(r_cp, axis=0) * self.config["ps"] / 63
        r_cp[r_cp < 3] = 0

        r_ncp_mx = np.max(r_ncp, axis=0) * self.config["ps"] / 63
        r_ncp_mn = np.mean(r_ncp, axis=0) * self.config["ps"] / 63
        r_ncp_mx[r_ncp_mx < 3] = 0
        r_ncp_mn[r_ncp_mn < 3] = 0

        # classifier
        cls = F.softmax(torch.mean(cls, dim=0), 1)
        cls = np.squeeze(cls.detach().cpu().data.numpy())
        cls_ = np.copy(cls)
        cls = np.argmax(cls, axis=0)

        # classifier attention
        cp = (cls == 1) | (cls == 3)
        ncp = (cls == 2) | (cls == 3)
        if self.config["aggregate"] == "mean":
            # r_ncp: mean aggregation
            ncp = region_growing(ncp, r_ncp_mn >= 3)

            r_ncp = r_ncp_mn
        elif self.config["aggregate"] == "max":
            # r_ncp: max aggregation
            ncp = region_growing(ncp, r_ncp_mx >= 3)

            r_ncp_mx[r_cp > 3] = r_ncp_mn[r_cp > 3]
            r_ncp = r_ncp_mx
        elif self.config["aggregate"] == "mean_max":
            # r_ncp: mean aggregation if mixed plaque, else max aggregation
            mp = region_growing(cp, (r_ncp_mn >= 3) | (r_cp >= 3)) * region_growing(ncp, r_ncp_mn >= 3)
            mp_mx = region_growing(mp, r_ncp_mx >= 3)
            ncp = region_growing(cls == 2, r_ncp_mx >= 3)
            ncp[mp_mx] = False

            r_ncp_mx[~ncp] = 0
            r_ncp_mx[mp] = r_ncp_mn[mp]
            r_ncp = r_ncp_mx
            ncp = r_ncp >= 3

        cp = binary_dilation(cp, iterations=1)
        r_cp = np.abs(r_l + r_ncp + r_cp * cp)
        r_ncp = np.abs(r_l + r_ncp * ncp)

        # initialize final pointclouds
        cloud_l = np.zeros((z * self.config["n_theta"], 3), dtype="float32")
        cloud_cp = np.zeros((z * self.config["n_theta"], 3), dtype="float32")
        cloud_ncp = np.zeros((z * self.config["n_theta"], 3), dtype="float32")
        cloud_cls = np.zeros((z * self.config["n_theta"], self.n_classes), dtype="float32")

        phis = np.arange(0, 2.0 * np.pi, 2.0 * np.pi / self.config["n_theta"])
        if ctl is not None:
            for c in range(3):
                ctl[:, c] -= spacing[c] * self.config["dz"]

        rads = {"l": r_l, "cp": r_cp, "ncp": r_ncp}
        for z_ in range(z):
            cls_slice = cls_[:, :, z_].T
            cloud_cls[z_ * self.config["n_theta"]: (z_ + 1) * self.config["n_theta"]] = cls_slice
            for name, r in rads.items():
                if ctl is not None:
                    coords = np.zeros((self.config["n_theta"], 4), dtype="float32")
                else:
                    coords = np.zeros((self.config["n_theta"], 3), dtype="float32")
                for x in range(self.config["n_theta"]):
                    coords[x, 0] = (r[x, z_] * np.cos(phis[x])) * self.config["dr"]
                    coords[x, 1] = (r[x, z_] * np.sin(phis[x])) * self.config["dr"]
                    if ctl is None:
                        coords[x, 2] = z_ * self.config["dz"]

                if ctl is not None:
                    coords = np.dot(rotmats[z_], coords.transpose()).transpose()
                    coords = coords[:, :3]
                    coords = coords + ctl[z_, :3]
                else:
                    # don't forget to compensate for dx and dy
                    coords[:, 0] += (mpr.shape[0] / 2) * self.config["dr"]
                    coords[:, 1] += (mpr.shape[1] / 2) * self.config["dr"]

                if name == "l":
                    cloud_l[z_ * self.config["n_theta"]: (z_ + 1) * self.config["n_theta"], :] = coords
                elif name == "cp":
                    cloud_cp[z_ * self.config["n_theta"]: (z_ + 1) * self.config["n_theta"], :] = coords
                elif name == "ncp":
                    cloud_ncp[z_ * self.config["n_theta"]: (z_ + 1) * self.config["n_theta"], :] = coords

        edge_index, faces = build_tube_graph(n_nodes=(self.config["n_theta"], z))
        clouds = {"l": cloud_l, "cp": cloud_cp, "ncp": cloud_ncp}

        meshes = {}
        for name, cloud in clouds.items():
            # add caps and triangulate to avoid candy wrapper nonsense
            new_faces = []
            for i in range(self.config["n_theta"]):
                j = cloud.shape[0] - self.config["n_theta"]
                new_faces.append(np.array([(i + 1) % self.config["n_theta"], i, cloud.shape[0]]))
                new_faces.append(np.array([(i + j), ((i + 1) % self.config["n_theta"]) + j, cloud.shape[0] + 1]))

            faces_all = np.concatenate((faces, np.array(new_faces)), axis=0)

            # wrap around
            if ctl is not None:
                cloud = np.concatenate((cloud, ctl[0, :3].reshape(1, 3)), axis=0)
                cloud = np.concatenate((cloud, ctl[-1, :3].reshape(1, 3)), axis=0)
            else:
                start = np.array([(mpr.shape[0] / 2) * self.config["dr"],
                                  (mpr.shape[1] / 2) * self.config["dr"],
                                  0])
                end = np.array([(mpr.shape[0] / 2) * self.config["dr"],
                                (mpr.shape[1] / 2) * self.config["dr"],
                                (mpr.shape[2] - 1) * self.config["dz"]])
                cloud = np.concatenate((cloud, start.reshape(1, 3)), axis=0)
                cloud = np.concatenate((cloud, end.reshape(1, 3)), axis=0)

            # create the mesh
            surface_mesh = mesh.Mesh(np.zeros(faces_all.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces_all):
                for j in range(3):
                    surface_mesh.vectors[i][j] = cloud[f[j], :]
            meshes[name] = surface_mesh

        rads["cls"] = cls_
        return meshes, rads

    def load(self, ckpt, verbose=False):
        state_dict = torch.load(ckpt, map_location="cuda:0")
        if self.config["load"] == "new":
            state_dict_new = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            # make state_dict compatible with old model
            state_dict_new = {}
            for k, v in state_dict.items():
                if "conv1" in k:
                    state_dict_new[k.replace("conv1", "convr.0")] = state_dict[k]
                elif "bn1" in k:
                    state_dict_new[k.replace("bn1", "convr.1")] = state_dict[k]
                elif "conv2" in k:
                    state_dict_new[k.replace("conv2", "convr.3")] = state_dict[k]
                elif "bn2" in k:
                    state_dict_new[k.replace("bn2", "convr.4")] = state_dict[k]
                elif "feasts" in k:
                    state_dict_new[k.replace("feasts", "layers")] = state_dict[k]
                elif "pre_conv" in k:
                    state_dict_new[k.replace("pre_conv", "pre_final.0")] = state_dict[k]
                elif "pre_bn" in k:
                    state_dict_new[k.replace("pre_bn", "pre_final.1")] = state_dict[k]
                elif "lum" in k:
                    state_dict_new[k.replace("lum", "l")] = state_dict[k]
                elif "non_clc" in k:
                    state_dict_new[k.replace("non_clc", "ncp")] = state_dict[k]
                elif "clc" in k:
                    state_dict_new[k.replace("clc", "cp")] = state_dict[k]
                else:
                    state_dict_new[k] = state_dict[k]
        log = self.load_state_dict(state_dict_new, strict=False)
        if verbose:
            print("missing keys:", log.missing_keys)
            print("unexpected keys:", log.unexpected_keys)

    def load_image(self, path, swap=True):
        image = sitk.ReadImage(path)
        spacing = image.GetSpacing()
        offset = image.GetOrigin()

        image = sitk.GetArrayFromImage(image)
        if swap:
            image = np.swapaxes(image, 0, 2)

        image = image.astype(np.float32)
        return self.norm(image), spacing, offset

    def save_image(self, path, image, spacing, offset, swap=True):
        image = self.denorm(image)
        if swap:
            image = np.swapaxes(image, 0, 2)

        image = sitk.GetImageFromArray(image.astype(np.int16))
        image.SetSpacing(spacing)
        image.SetOrigin(offset)

        writer = sitk.ImageFileWriter()
        writer.SetUseCompression(True)
        writer.SetFileName(path)
        writer.Execute(image)

    def norm(self, image):
        if image.min() >= -10:
            image -= 1024

        if self.config["load"] == "new":
            image = np.clip(image, a_min=-760, a_max=1240)
            image = (image - 240) / 1000
        return image

    def denorm(self, image):
        if self.config["load"] == "new":
            image = image * 1000 + 240
        return image

    def to_polar(self, mpr):
        # create polar-transformed mpr from ray-casts
        mpr_polar = np.zeros((self.config["n_theta"], self.config["n_rad"], mpr.shape[2]), dtype="float32")
        for z in range(mpr.shape[2]):
            for angle_i in range(self.v_theta.shape[0]):
                coords = np.zeros((self.config["n_rad"], 3), dtype="float32")
                coords[:, 0] = (
                        self.v_rad * np.cos(self.v_theta[angle_i])
                        + float(mpr.shape[0]) / 2.0
                )
                coords[:, 1] = (
                        self.v_rad * np.sin(self.v_theta[angle_i])
                        + float(mpr.shape[0]) / 2.0
                )
                mpr_polar[angle_i, :, z] = fast_bilinear_interpolation(
                    mpr[:, :, z], coords[:, 0], coords[:, 1]
                )[::-1]

        return mpr_polar


class WeightedL1Loss(L1Loss):
    def __init__(self, *args, **kwargs):
        super(WeightedL1Loss, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_0, input_1 = torch.masked_select(input, target == 0), torch.masked_select(input, target != 0)
        target_0, target_1 = torch.masked_select(target, target == 0), torch.masked_select(target, target != 0)

        loss_0, loss_1 = super().forward(input_0, target_0), super().forward(input_1, target_1)
        if torch.isnan(loss_1).any():
            return loss_0
        else:
            return loss_0 + loss_1


class FanLoss(Module):
    def __init__(self, cls="CrossEntropyLoss", inner="MSELoss", outer="WeightedL1Loss"):
        super(FanLoss, self).__init__()
        self.loss_cls = self.str_to_attr(cls)()
        self.loss_inner = self.str_to_attr(inner)()
        self.loss_outer = self.str_to_attr(outer)()

    def forward(self, inputs, targets):
        # classifier loss
        cls_inp, cls_tar = inputs[3][:, :, :, 0, :], targets[4]
        loss_cls = self.loss_cls(cls_inp, cls_tar)

        # inner loss
        l_inp, l_tar = torch.squeeze(inputs[0]), torch.squeeze(targets[0])
        loss_l = self.loss_inner(l_inp, l_tar)

        # outer loss
        cp_inp, cp_tar = torch.squeeze(inputs[1]), torch.squeeze(targets[2])
        ncp_inp, ncp_tar = torch.squeeze(inputs[2]), torch.squeeze(targets[3])
        out_tar = torch.squeeze(targets[1])
        loss_cp = self.loss_outer(cp_inp, cp_tar)
        loss_ncp = self.loss_outer(ncp_inp, ncp_tar)
        loss_out = self.loss_outer(cp_inp + ncp_inp, out_tar)
        return {"cls": loss_cls, "l": loss_l, "cp": loss_cp, "ncp": loss_ncp, "out": loss_out}

    @staticmethod
    def str_to_attr(classname):
        return getattr(sys.modules[__name__], classname)

