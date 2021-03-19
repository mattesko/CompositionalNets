import os
import torchvision.models as models
from CompositionalNets.Code.model import resnet_feature_extractor
from CompositionalNets.Code.config import data_path, model_save_dir, vc_num, backbone_type, dataset
from torch import load, device
from torch.cuda import is_available as cuda_is_available
from collections import OrderedDict
import re
from functools import reduce

from src.config import Directories
from src.models import UNet, set_weights_to_ones

# Setup work
device_ids = [0]

# dataset = 'chaos' # pascal3d+

#vgg, resnet50, resnext, resnet152
# TODO: Add the ability to use U-Net's bottleneck's output as the feature map
# The U-Net trained on the CHAOS dataset most definitely extracts more meaningful
# features than something pre-trained on ImageNet

# nn_type = 'vgg'
nn_type = backbone_type
unet_filename = 'unet_liver_2020-10-31_19:19:13.pth'

# vMF parameter
vMF_kappa = 30

# Number of vMF clusters, used as argument for cls_num in vMFMM init
# This needs to the same as the extractor's output channel size
# because the vMF clusters are used as weights for a Convolutional operation 
# applied on the extractor's output feature map
# Convolutional Function Dimension = [H * W * VC_NUM]
# Feature Map Dimensions = [VC_NUM * H * W]
# Convolutional Function * Feature Map Dimensions = vMF Activations
# vc_num = vc_num

categories = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa',
              'train', 'tvmonitor']
cat_test = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']

def calculate_receptive_fields(kernel_sizes, strides):
    
    r_fields = []
    prev_r_field = 1
    
    for i, kernel_size in enumerate(kernel_sizes, start=1):
        product_of_strides = reduce(lambda x,y: x*y, strides[:i])
        
        curr_r_field = prev_r_field + (kernel_size - 1) * product_of_strides
        r_fields.append(curr_r_field)
        prev_r_field = curr_r_field
        
    return r_fields

if nn_type =='vgg':
    layer = 'pool5'  # 'pool5','pool4'
    if layer == 'pool4':
        extractor=models.vgg16(pretrained=True).features[0:24]
    elif layer =='pool5':
        if vc_num == 512:
            extractor = models.vgg16(pretrained=True).features
        elif vc_num == 256:
            extractor = models.vgg16(pretrained=True).features[:12]
        elif vc_num == 128:
            extractor = models.vgg16(pretrained=True).features[:10]
            
elif nn_type[:6]=='resnet' or nn_type=='resnext' or nn_type=='alexnet':
    layer = 'last' # 'last','second'
    extractor=resnet_feature_extractor(nn_type,layer)
    
elif nn_type == 'unet':
    layer = 'pool5'
    path_to_unet = os.path.join(Directories.CHECKPOINTS, unet_filename)
    unet = UNet(pretrained=True)
    device = device('cuda:0' if cuda_is_available() else 'cpu')
    unet.load_state_dict(load(path_to_unet, map_location=device)['model_state_dict'])
    
    unet_ones = UNet(pretrained=False)
    unet_ones = unet_ones.get_features()
    
    if vc_num == 1024:
        extractor = unet.get_features()[:24]
    elif vc_num == 512:
        extractor = unet.get_features()[:19]
    elif vc_num == 256:
        extractor = unet.get_features()[:15]
    elif vc_num == 128:
        extractor = unet.get_features()[:10]
    else:
#         extractor = unet.get_features()[:9]
        unet_layer = 15
        extractor = unet.get_features()[:unet_layer] #256
        
        unet_ones = set_weights_to_ones(unet_ones)[:unet_layer]
#         extractor = unet.get_features()[:4] #64
#         extractor = unet.get_features()[:2] #64

elif nn_type == 'unet_lits':
    layer = 'pool5'
    unet_filepath = os.path.join(Directories.CHECKPOINTS, 'model_UNet.pth')
    state_dict = load(unet_filepath)['model_state_dict']
    new_dict = OrderedDict()
    for curr_key, value in state_dict.items():
        new_key = re.findall('conv.+', curr_key)[0]
        new_dict[new_key] = state_dict[curr_key]
    unet_filepath = os.path.join(Directories.CHECKPOINTS, 'model_UNet.pth')
    unet = UNet(in_channels=3)
    unet.load_state_dict(new_dict)
    
    if vc_num == 1024:
        extractor = unet.get_features()[:24]
    elif vc_num == 512:
        extractor = unet.get_features()[:19]
    elif vc_num == 256:
        extractor = unet.get_features()[:15]
    elif vc_num == 128:
        extractor = unet.get_features()[:10]
    else:
#         extractor = unet.get_features()[:9]
        extractor = unet.get_features()[:15]
#         extractor = unet.get_features()[:4]

if cuda_is_available():
    extractor.cuda(device_ids[0]).eval()

init_path = model_save_dir+'init_{}/'.format(nn_type)
if not os.path.exists(init_path):
    os.makedirs(init_path)

dict_dir = init_path+'dictionary_{}/'.format(nn_type)
if not os.path.exists(dict_dir):
    os.makedirs(dict_dir)

sim_dir = init_path+'similarity_{}_{}_{}/'.format(nn_type,layer,dataset)

Astride_set = [2, 4, 8, 16, 32]  # stride size
featDim_set = [64, 128, 256, 512, 512]  # feature dimension
Arf_set = [6, 16, 44, 100, 212]  # receptive field size for vgg
Apad_set = [2, 6, 18, 42, 90]  # padding size

# Why are these initialized to those values?
# Why an offset of 3?
if layer =='pool4' or layer =='second':
    Astride = Astride_set[3]
    Arf = Arf_set[3]
    Apad = Apad_set[3]
    offset = 3
    
elif layer =='pool5' or layer == 'last':
#     Astride = Astride_set[3]
#     Arf = 170
#     Apad = Apad_set[4]
#     offset = 3
    
    Astride = 32
    Arf = 300
    Apad = 20
    offset = 3

if nn_type == 'unet':
    num_layers = len(extractor)
    kernels = [3] * (num_layers // 2)
    strides = [1] * (num_layers // 2)
    padding = 1
    
    r_fields = calculate_receptive_fields(kernels, strides)
    
    num_max_pools = unet_layer // 5
    
    Astride = 1
    Arf = r_fields[-1] * (2 ** num_max_pools)
#     Apad = Arf*2
    Apad = 0
    offset = 0
    
