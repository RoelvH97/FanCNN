# import necessary modules
import argparse
import json
import utils

from data import CADRAData
from model import CADResNet
from torch.utils.data import DataLoader


def test(config):
    model = CADResNet(config["MODEL"])
    data_test = CADRAData(config["DATA"], mode="test", dset="IDR_CADRADS")
    loader_test = DataLoader(data_test, batch_size=config["DATA"]["batch_size"], shuffle=False, num_workers=0)

    conf_matrix, y_pred, y_true = model.infer(loader_test)
    print(conf_matrix)


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

    if config_g["mode"] == "test":
        test(config)
