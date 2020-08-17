import os
import torchvision.models as models
from CompositionalNets.Code.model import resnet_feature_extractor
from CompositionalNets.Code.config import data_path, model_save_dir

# Setup work
device_ids = [0]

dataset = 'pascal3d+' # pascal3d+

#vgg, resnet50, resnext, resnet152
# TODO: Add the ability to use U-Net's bottleneck's output as the feature map
# The U-Net trained on the CHAOS dataset most definitely extracts more meaningful
# features than something pre-trained on ImageNet
nn_type = 'vgg' 

# vMF parameter
vMF_kappa = 30

# Number of vMF clusters, used as argument for cls_num in vMFMM init
# This needs to the same as the extractor's output channel size
# because the vMF clusters are used as weights for a Convolutional operation 
# applied on the extractor's output feature map
# Convolutional Function Dimension = [H * W * VC_NUM]
# Feature Map Dimensions = [VC_NUM * H * W]
# Convolutional Function * Feature Map Dimensions = vMF Activations
vc_num = 512

categories = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa',
              'train', 'tvmonitor']
cat_test = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']

if nn_type =='vgg':
    layer = 'pool5'  # 'pool5','pool4'
    if layer == 'pool4':
        extractor=models.vgg16(pretrained=True).features[0:24]
    elif layer =='pool5':
        extractor = models.vgg16(pretrained=True).features
elif nn_type[:6]=='resnet' or nn_type=='resnext' or nn_type=='alexnet':
    layer = 'last' # 'last','second'
    extractor=resnet_feature_extractor(nn_type,layer)

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
Arf_set = [6, 16, 44, 100, 212]  # receptive field size
Apad_set = [2, 6, 18, 42, 90]  # padding size

# Why are these initialized to those values?
# Why an offset of 3?
if layer =='pool4' or layer =='second':
    Astride = Astride_set[3]
    Arf = Arf_set[3]
    Apad = Apad_set[3]
    offset = 3
elif layer =='pool5' or layer == 'last':
    Astride = Astride_set[3]
    Arf = 170
    Apad = Apad_set[4]
    offset = 3
