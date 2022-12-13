import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

INPUT_SIZE = (3, 600, 338)


def main():
    model_name = 'u2net'  # u2netp
    model_dir = "saved_models/u2net/u2net_bce_itr_112000_train_0.529639_tar_0.067323.pth"

    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, INPUT_SIZE, requires_grad=True)
    dummy_input = dummy_input.type(torch.FloatTensor)

    # Export the model
    torch.onnx.export(net,         # model being run
                      # model input (or a tuple for multiple inputs)
                      dummy_input,
                      "output.onnx",       # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],   # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},    # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    main()
