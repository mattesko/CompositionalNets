from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from .vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
import os
from .config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, dict_dir, sim_dir, extractor
from CompositionalNets.Code.helpers import getImg, imgLoader, Imgset, myresize
from CompositionalNets.Code.config import init_path
from torch.utils.data import DataLoader
import numpy as np
import math
import torch
import pdb

paral_num = 3
nimg_per_cat = 5000
imgs_par_cat =np.zeros(len(categories))
occ_level='ZERO'
occ_type=''

def getImg(archive, mode, categories, dataset, data_path, 
           cat_test=None, occ_level='ZERO', occ_type=None, bool_load_occ_mask = False):

    if mode == 'train':
        train_imgs = []
        train_labels = []
        train_masks = []
        for category in categories:
            if dataset == 'pascal3d+':
                if occ_level == 'ZERO':
                    filelist = 'pascal3d+_occ/' + category + '_imagenet_train' + '.txt'
                    img_dir = 'pascal3d+_occ/TRAINING_DATA/' + category + '_imagenet'
            elif dataset == 'coco':
                if occ_level == 'ZERO':
                    img_dir = 'coco_occ/{}_zero'.format(category)
                    filelist = 'coco_occ/{}_{}_train.txt'.format(category, occ_level)

            with archive.open(filelist, 'r') as fh:
                contents = fh.readlines()
            img_list = [cc.strip().decode('ascii') for cc in contents]
            label = categories.index(category)
            for img_path in img_list:
                if dataset=='coco':
                    if occ_level == 'ZERO':
                        img = img_dir + '/' + img_path + '.jpg'
                    else:
                        img = img_dir + '/' + img_path + '.JPEG'
                else:
                    img = img_dir + '/' + img_path + '.JPEG'
                occ_img1 = []
                occ_img2 = []
                train_imgs.append(img)
                train_labels.append(label)
                train_masks.append([occ_img1,occ_img2])
        
        return train_imgs, train_labels, train_masks

    else:
        test_imgs = []
        test_labels = []
        occ_imgs = []
        for category in cat_test:
            if dataset == 'pascal3d+':
                filelist = data_path + 'pascal3d+_occ/' + category + '_imagenet_occ.txt'
                img_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level
                if bool_load_occ_mask:
                    if  occ_type=='':
                        occ_mask_dir = 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask_object'
                    else:
                        occ_mask_dir = 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask'
                    occ_mask_dir_obj = 'pascal3d+_occ/0_old_masks/'+category+'_imagenet_occludee_mask/'
            elif dataset == 'coco':
                if occ_level == 'ZERO':
                    img_dir = 'coco_occ/{}_zero'.format(category)
                    filelist = 'coco_occ/{}_{}_test.txt'.format(category, occ_level)
                else:
                    img_dir = 'coco_occ/{}_occ'.format(category)
                    filelist = 'coco_occ/{}_{}.txt'.format(category, occ_level)

#             if os.path.exists(filelist):
            with archive.open(filelist, 'r') as fh:
                contents = fh.readlines()
            img_list = [cc.strip().decode('ascii') for cc in contents]
            label = categories.index(category)
            for img_path in img_list:
                if dataset != 'coco':
                    if occ_level=='ZERO':
                        img = img_dir + occ_type + '/' + img_path[:-2] + '.JPEG'
                        occ_img1 = []
                        occ_img2 = []
                    else:
                        img = img_dir + occ_type + '/' + img_path + '.JPEG'
                        if bool_load_occ_mask:
                            occ_img1 = occ_mask_dir + '/' + img_path + '.JPEG'
                            occ_img2 = occ_mask_dir_obj + '/' + img_path + '.png'
                        else:
                            occ_img1 = []
                            occ_img2 = []

                else:
                    img = img_dir + occ_type + '/' + img_path + '.jpg'
                    occ_img1 = []
                    occ_img2 = []

                test_imgs.append(img)
                test_labels.append(label)
                occ_imgs.append([occ_img1,occ_img2])
#             else:
#                 print('FILELIST NOT FOUND: {}'.format(filelist))
        return test_imgs, test_labels, occ_imgs


def imgLoader(archive, img_path,mask_path,bool_resize_images=True,bool_square_images=False):
    
    archive_img_path = archive.open(img_path)
    input_image = Image.open(archive_img_path)
    if bool_resize_images:
        if bool_square_images:
            input_image.resize((224,224),Image.ANTIALIAS)
        else:
            sz=input_image.size
            min_size = np.min(sz)
            if min_size!=224:
                input_image = input_image.resize((np.asarray(sz) * (224 / min_size)).astype(int),Image.ANTIALIAS)
    preprocess =  transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = preprocess(input_image)

    if mask_path[0]:
        f = archive.open(mask_path[0])
        mask1 = np.array(Image.open(f))
        f.close()
        mask1 = myresize(mask1, 224, 'short')
        try:
            mask2 = cv2.imread(mask_path[1])[:, :, 0]
            mask2 = mask2[:mask1.shape[0], :mask1.shape[1]]
        except:
            mask = mask1
        try:
            mask = ((mask1 == 255) * (mask2 == 255)).astype(np.float)
        except:
            mask = mask1
    else:
        mask = np.ones((img.shape[0], img.shape[1])) * 255.0

    mask = torch.from_numpy(mask)
    return img,mask


class Imgset():
    def __init__(self, archive, imgs, masks, labels, loader,bool_square_images=False):
        self.archive = archive
        self.images = imgs
        self.masks 	= masks
        self.labels = labels
        self.loader = loader
        self.bool_square_images = bool_square_images

    def __getitem__(self, index):
        fn = self.images[index]
        label = self.labels[index]
        mask = self.masks[index]
        img,mask = self.loader(self.archive,fn,mask,bool_resize_images=True,bool_square_images=self.bool_square_images)
        return img, mask, label

    def __len__(self):
        return len(self.images)

def save_checkpoint(state, filename, is_best):
    if is_best:
        print("=> Saving new checkpoint")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")

print('max_images {}'.format(nimg_per_cat))

import zipfile
from PIL import Image
from torchvision import transforms
from src.config import directories
data_path = os.path.join(directories['CompositionalNets'], 'data')
archive = zipfile.ZipFile(os.path.join(data_path, 'CompNet_data.zip'))

if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)

#############################
# BEWARE THIS IS RESET TO LOAD OLD VCS AND MODEL
#############################
def compute_similarity_matrix(data_loader, category, save_name,
                              sim_dir_name=f'similarity_vgg_pool5_{dataset}',
                              u_out_name=f'dictionary_{layer}_{vc_num}_u.pickle',
                              N_sub = 50, num_layer_features=100, 
                             num_centers_threshold=100):
    
    # binary encode feature matrices with vmf kernels
    # computes binary distance (computes similarity between two images)
    # compute similarity between all images, such that spectral clustering could be
    # used for mixture model learning
    
    with open(os.path.join(dict_dir, u_out_name), 'rb') as fh:
        centers = pickle.load(fh)
    num_centers = len(centers)
##HERE
    bool_pytorch = True
#     imgs, labels, masks = getImg('train', [category], dataset, data_path, cat_test, occ_level, occ_type, bool_load_occ_mask=False)
    N=len(data_loader.dataset)
    ##HERE
#     imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False,bool_cutout=False,bool_pytorch=bool_pytorch)
    savepath = os.path.join(init_path, sim_dir_name, save_name)
    if not os.path.exists(os.path.join(init_path, sim_dir_name)):
        os.makedirs(os.path.join(init_path, sim_dir_name))
    i = 0
#     if not os.path.exists(savepath):
    r_set = [None for nn in range(N)]
    for ii,data in enumerate(data_loader):
        if len(data) == 3:
            input, mask, label = data
        elif len(data) == 2:
            input, label = data

        if i<N:
            with torch.no_grad():
                layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
            iheight,iwidth = layer_feature.shape[1:3]
            lff = layer_feature.reshape(layer_feature.shape[0],-1).T
            lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10
            if num_centers > num_centers_threshold:
                with open(os.path.join(dict_dir, f'r_set{ii}.pkl'), 'wb') as fh:
                    cos_dist = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
                    pickle.dump(cos_dist, fh)
            else:
                r_set[ii] = cdist(lff_norm, centers, 'cosine').reshape(iheight,iwidth,-1)
            i+=1

    print('Determine best threshold for binarization - {} ...'.format(category))
#     pdb.set_trace()
    
    nthresh=20
    magic_thhs=range(nthresh)
    coverage = np.zeros(nthresh)
    act_per_pix = np.zeros(nthresh)
    layer_feature_b = [None for nn in range(num_layer_features)]
    magic_thhs = np.asarray([x*1/nthresh for x in range(nthresh)])
    for idx,magic_thh in enumerate(magic_thhs):
        for nn in range(num_layer_features):
            
            if num_centers > num_centers_threshold:
                with open(os.path.join(dict_dir, f'r_set{nn}.pkl'), 'rb') as fh:
                    r = pickle.load(fh)
                    layer_feature_b[nn] = (r<magic_thh).astype(int).T
            else:
                layer_feature_b[nn] = (r_set[nn]<magic_thh).astype(int).T
            coverage[idx] 	+= np.mean(np.sum(layer_feature_b[nn],axis=0)>0)
            act_per_pix[idx] += np.mean(np.sum(layer_feature_b[nn],axis=0))
            layer_feature_b[nn] = None
    
#     pdb.set_trace()
    
    coverage=coverage/num_layer_features
    act_per_pix=act_per_pix/num_layer_features
    best_loc=(act_per_pix>2)*(act_per_pix<15)
    if np.sum(best_loc):
        best_thresh = np.min(magic_thhs[best_loc])
    else:
        best_thresh = 0.45
    layer_feature_b = [None for nn in range(N)]
    
#     pdb.set_trace()
    
    if num_centers > num_centers_threshold:
        for nn in range(N):
            with open(os.path.join(dict_dir, f'r_set{nn}.pkl'), 'rb') as fh1:
                r = pickle.load(fh1)
                feature = (r<best_thresh).astype(int).T
                with open(os.path.join(dict_dir, f'feature_b_{nn}.pkl'), 'wb') as fh2:
                    pickle.dump(feature, fh2)
    else:
        for nn in range(N):
            layer_feature_b[nn] = (r_set[nn]<best_thresh).astype(int).T

    print('Start compute sim matrix ... magicThresh {}'.format(best_thresh))
    _s = time.time()

    mat_dis1 = np.ones((N,N))
    mat_dis2 = np.ones((N,N))
#     N_sub = 2000
#     N_sub = 50
    sub_cnt = int(math.ceil(N/N_sub))
    for ss1 in range(sub_cnt):
        start1 = ss1*N_sub
        end1 = min((ss1+1)*N_sub, N)
        
        if num_centers > num_centers_threshold:
            for i in range(start1, end1+1):
                with open(os.path.join(dict_dir, f'feature_b_{i}.pkl'), 'rb') as fh:
                    layer_feature_b[i] = pickle.load(fh)
                    
        layer_feature_b_ss1 = layer_feature_b[start1:end1]
        for ss2 in range(ss1,sub_cnt):
            print('iter {1}/{0} {2}/{0}'.format(sub_cnt, ss1+1, ss2+1))
            _ss = time.time()
            start2 = ss2*N_sub
            end2 = min((ss2+1)*N_sub, N)
            if ss1==ss2:
                inputs = [(layer_feature_b_ss1, nn) for nn in range(end2-start2)]
                para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs))

            else:
                layer_feature_b_ss2 = layer_feature_b[start2:end2]
                inputs = [(layer_feature_b_ss2, lfb) for lfb in layer_feature_b_ss1]
                para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral_full)(i) for i in inputs))

            mat_dis1[start1:end1, start2:end2] = para_rst[:,0]
            mat_dis2[start1:end1, start2:end2] = para_rst[:,1]

            _ee = time.time()
            print('comptSimMat iter time: {}'.format((_ee-_ss)/60))

    _e = time.time()
    print('comptSimMat total time: {}'.format((_e-_s)/60))

    with open(savepath, 'wb') as fh:
        print('saving at: '+savepath)
        pickle.dump([mat_dis1, mat_dis2], fh)
    return mat_dis1, mat_dis2

if __name__ == '__main__':
    for category in categories:
        
        imgs, labels, masks = getImg(archive, 'train', [category], dataset, data_path, cat_test, occ_level, occ_type, bool_load_occ_mask=False)
        imgs=imgs[:nimg_per_cat]
        imgset = Imgset(archive, imgs, masks, labels, imgLoader, bool_square_images=False)
        data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)
        
        compute_similarity_matrix(data_loader, category, 
                                  f'simmat_mthrh045_{category}_K{vc_num}.pickle')
    