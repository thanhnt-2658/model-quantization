# This file demonstrate how to use pytorch_quantization for Post-training quantization (PTQ),
# the original guide is: https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/docs/source/tutorials/quant_resnet50.rst
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
quant_modules.initialize() # this line should run before any model initialization
from torchvision import models


def collect_stats(model, data_loader, num_batches):
     """Feed data to the network and collect statistic"""

     # Enable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.disable_quant()
                 module.enable_calib()
             else:
                 module.disable()

     for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
         model(image.cuda())
         if i >= num_batches:
             break

     # Disable calibrators
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 module.enable_quant()
                 module.disable_calib()
             else:
                 module.enable()

 def compute_amax(model, **kwargs):
     # Load calib result
     for name, module in model.named_modules():
         if isinstance(module, quant_nn.TensorQuantizer):
             if module._calibrator is not None:
                 if isinstance(module._calibrator, calib.MaxCalibrator):
                     module.load_calib_amax()
                 else:
                     module.load_calib_amax(**kwargs)
             print(F"{name:40}: {module}")
     model.cuda()


# select method for clibration
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

model = None # init your model here
model.cuda()

# you will need data_loader for calibration and testing
data_loader = None
data_loader_test = None

# Collect statistics and perform calibration. It is a bit slow since we collect histograms on CPU
with torch.no_grad():
     collect_stats(model, data_loader, num_batches=2)
     compute_amax(model, method="percentile", percentile=99.99)

# evaluate model after PTQ
criterion = nn.CrossEntropyLoss()
with torch.no_grad():
    evaluate(model, criterion, data_loader_test, device="cuda", print_freq=20) # you will need to define your evaluate functiongit



