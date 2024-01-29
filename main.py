# import necessary modules
import argparse
import datetime
import json
import lightning.pytorch as pl
import os
import sys
import utils

from data import *
from glob import glob
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from model import *
from os.path import basename, dirname, exists, join
from torch.utils.data import DataLoader


def train(config):
    now = datetime.datetime.now()
    now = f"{str(now.day).zfill(2)}-{str(now.month).zfill(2)}-{now.year}_"\
          f"{str(now.hour).zfill(2)}-{str(now.minute).zfill(2)}-{str(now.second).zfill(2)}"
    config_d, config_g, config_m, config_o = config["DATA"], config["GENERAL"], config["MODEL"], config["OPTIMIZATION"]

    # save config
    config_file = join(os.getcwd(), f"lightning_logs/{config_g['name']}/{now}", "config.json")
    if not exists(dirname(config_file)):
        os.makedirs(dirname(config_file))
    with open(config_file, "w") as outfile:
        json.dump(config, outfile)

    # outline model
    model = LightningFan(config)

    # outline data
    data_train = PolarCircle(config_d, mode="train")
    data_val = PolarAll(config_d, mode="val")
    loader_train = DataLoader(data_train,
                              batch_size=config_o["batch_size"],
                              num_workers=config_o["n_workers"],
                              pin_memory=config_o["pin_memory"],
                              shuffle=True)
    loader_val = DataLoader(data_val,
                            batch_size=1,
                            num_workers=config_o["n_workers"],
                            pin_memory=config_o["pin_memory"],
                            shuffle=False)

    # outline trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"lightning_logs/{config_g['name']}/{now}/model",
        save_top_k=5,
        save_last=True,
        monitor=f"val/corr_ncp",
        mode="max",
    )
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=config_g["name"],
        version=now,
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        callbacks=[checkpoint_callback,
                   LearningRateMonitor(logging_interval='epoch'),
                   RichProgressBar()],
        check_val_every_n_epoch=config_o["eval_every"],
        deterministic=False,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config_o["clip_grad"],
        logger=logger,
        log_every_n_steps=5,
        max_epochs=config_o["n_iter"],
        num_sanity_val_steps=2,
    )

    ckpt = join(os.getcwd(), "lightning_logs", config_o["resume"], "model", "last.ckpt") if config_o["resume"] else None
    trainer.fit(model, loader_train, loader_val, ckpt_path=ckpt)


def test(config):
    model = TestFanCNN(config["MODEL"])
    paths_image = sorted(glob(config["DATA"]["dir_img"] + os.sep + "*.mhd"))

    for path_image in paths_image[:35] + paths_image[36:]:
        path_centerlines = sorted(glob(join(config["DATA"]["dir_ctl"], basename(path_image)[:-4] + "*.ctl.anno")))
        model.process(path_image, path_centerlines)


if __name__ == "__main__":
    # parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The configuration file for training/testing the FanCNN')

    args = parser.parse_args()
    config = json.load(open(args.config))
    config_g = config["GENERAL"]
    print(json.dumps(config, sort_keys=True, indent=4))

    # init
    utils.seed_everything(config_g["seed"])
    utils.set_gpu(config_g["gpu"])

    if config_g["mode"] == "train":
        train(config)
    elif config_g["mode"] == "test":
        test(config)
