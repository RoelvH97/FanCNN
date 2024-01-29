# import necessary modules
import os


def set_gpu(index: int):
    print("Setting GPU to: {}".format(index))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
