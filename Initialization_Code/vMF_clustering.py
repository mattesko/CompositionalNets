from CompositionalNets.Code.vMFMM import *
from .config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer,init_path, nn_type, dict_dir, offset, extractor, unet_ones
from CompositionalNets.Code.helpers import getImg, imgLoader, Imgset, myresize
import torch
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available
import cv2
import glob
import pickle
import os
from tqdm.autonotebook import tqdm
import pdb

import os
import psutil

# Number of images to train on per category
# Clustering ignored after this threshold is met
img_per_cat = 1000

# Number of feature vectors to sample from each image's feature map
# samp_size_per_img = 20
# samp_size_per_img = 60
# samp_size_per_img = 80
samp_size_per_img = 2000


def learn_vmf_clusters(data_loader, categories=categories, img_per_cat=1000,
                       max_it=150, tol=5e-5,
                       u_out_name=f'dictionary_{layer}_{vc_num}.pickle',
                       p_out_name=f'dictionary_{layer}_{vc_num}_p.pickle',
                       verbose=False, kappas=[], use_mask=False):
    """Learn VMF clusters on feature maps extracted from the given images"""
    
    imgs_par_cat = np.zeros(len(categories))
    loc_set = []
    feat_set = []
    pbar = tqdm(enumerate(data_loader), disable=not verbose, 
                desc="Sampling DNN features from dataset", leave=False)
    
#     pdb.set_trace()
    for ii, data in pbar:
        if len(data) == 3:
#             use_mask = True
            images, mask, label = data
            mask = mask.squeeze().numpy()
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
        elif len(data) == 2:
#             use_mask = False
            images, label = data
            mask = images

        if imgs_par_cat[label]<img_per_cat:
            
            with torch.no_grad():
                
                if cuda_is_available:
                    images = images.cuda(device_ids[0])
                
                feature_map = extractor(images)[0].detach().cpu().numpy()
                
                if use_mask:
                    mask_feature_map = unet_ones(mask)
                    mask_feature_map = mask_feature_map.squeeze().detach().cpu().numpy()
                    
                    mask_min, mask_max = mask_feature_map.min(), mask_feature_map.max()
                    mask_feature_map = (mask_feature_map - mask_min) / (mask_max - mask_min + 1e-12)
                
            height, width = feature_map.shape[1:3]
            
            
#             if use_mask:
                
#                 mask = cv2.resize(mask, (height, width))
#                 feature_map = feature_map * mask
#                 mask = mask.reshape(-1)
            
            # Crop image by some offset, why? How to choose offset?
            # Feed segmentation map (same spatial dimension as the feature map)
            feature_map = feature_map[:,offset:height - offset, offset:width - offset]
            
            # Flatten image at each channel
            gtmp = feature_map.reshape(feature_map.shape[0], -1)
            if use_mask:
                
                mask_feature_map = mask_feature_map.reshape(feature_map.shape[0], -1)
            
            threshold = 0.75
            if gtmp.shape[1] >= samp_size_per_img:
                if use_mask:
#                     pdb.set_trace()
#                     rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
                    rand_idx = np.argwhere(mask_feature_map[0] > threshold).reshape(-1)
                    np.random.shuffle(rand_idx)
                    rand_idx = rand_idx[:samp_size_per_img]
                
                else:
                    rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
            else:
                
                new_num_features = samp_size_per_img - gtmp.shape[1]
                print(f'Number of features desired is greater than what\'s available. Clipping number of features to {new_num_features}')
                
                if use_mask:
                    rand_idx = np.random.permutation(gtmp.shape[1])[:new_num_features]
                    
                    rand_idx = np.argwhere(mask_feature_map[0] > threshold).reshape(-1)
                    np.random.shuffle(rand_idx)
                    rand_idx = rand_idx[:samp_size_per_img]
                    
                else:
                    rand_idx = np.random.permutation(gtmp.shape[1])[:new_num_features]
                #rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
            
            # select indices such that they are value of 1 within the segmentation map
            
            tmp_feats = gtmp[:, rand_idx].T
            
            cnt = 0
#             pdb.set_trace()
            for rr in rand_idx:
                ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
                hi = (ihi+offset)*(images.shape[2]/height)-Apad
                wi = (iwi + offset)*(images.shape[3]/width)-Apad
                if hi < ihi:
                    hi = ihi
                if wi < iwi:
                    wi = wi
                #hi = Astride * (ihi + offset) - Apad
                #wi = Astride * (iwi + offset) - Apad

                #assert (hi >= 0)
                #assert (wi >= 0)
                #assert (hi <= img.shape[0] - Arf)
                #assert (wi <= img.shape[1] - Arf)
                loc_set.append([label, ii, hi,wi,hi+Arf,wi+Arf])
                feat_set.append(tmp_feats[cnt,:])
                cnt+=1

            imgs_par_cat[label]+=1
    
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / 1024 ** 2
    print(f'{memory_used} MB')
    feat_set = np.asarray(feat_set)
    loc_set = np.asarray(loc_set).T

#     print(f'Number of features: {feat_set.shape}')

    # could loop over different kappa values which will affect the similarity matrix
    # compute average of similarity matrix (pick kappa with lowest similarity). similarity should plateau
    # after a certain kappa value
    model = vMFMM(vc_num, 'k++')
    best_kappa = 0
    best_similarity = 1
    best_model = None
    
    if (type(kappas) == list or type(kappas) == range) and kappas:
        pbar = tqdm(kappas, disable=not verbose, 
                desc="Selecting best Kappa", leave=False)
        for kappa in kappas:
            model.fit(feat_set, kappa, max_it, tol, verbose=False)
            
            mat = np.dot(model.mu, model.mu.T)
            average_similarity = np.average(mat)
            
            if average_similarity < best_similarity:
                best_similarity = average_similarity
                best_model = model
                best_kappa = kappa
        print(f'Best Kappa: {best_kappa}. Lowest Similarity: {best_similarity}')
    elif (type(kappas) == int):
        model.fit(feat_set, kappas, max_it, tol, verbose=verbose)
        best_model = model
    else:
        model.fit(feat_set, vMF_kappa, max_it, tol, verbose=verbose)
        best_model = model
    
    filepath = os.path.join(dict_dir, u_out_name)
    with open(filepath, 'wb') as fh:
        # could compute similarity matrix between the kernels. want kernels to be as different as possible.
        # so that they focus on different image patters. essentially dot product between kernels
        # good average similarity: .3 or .4. This depends on the kappa.
        # dot product between each vector, and get a matrix. compute dot prod between each kernel with every other one
        # 1s on diagonal, symmetric matrix
        pickle.dump(best_model.mu, fh)
    
    filepath = os.path.join(dict_dir, p_out_name)
    with open(filepath, 'wb') as fh:
        pickle.dump(best_model.p, fh)
    
    return best_model, loc_set


def save_cluster_images(model, loc_set, in_images, num_images=50,
                        max_num_clusters=-1,
                        out_dir_name=f'cluster_images_{layer}_{vc_num}',
                        verbose=False):
    """Save images from the vMF model cluster"""
    
    SORTED_IDX = []
    SORTED_LOC = []
    for vc_i in range(vc_num):
        sort_idx = np.argsort(-model.p[:, vc_i])[0:num_images]
        SORTED_IDX.append(sort_idx)
        tmp=[]
        for idx in range(num_images):
            iloc = loc_set[:, sort_idx[idx]]
            tmp.append(iloc)
        SORTED_LOC.append(tmp)

    print(f'Saving top {num_images} images for each cluster')
    
    out_dir = os.path.join(dict_dir, out_dir_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    pbar = tqdm(range(vc_num), disable=not verbose)
    for vc_i in pbar:
        patch_set = np.zeros(((Arf**2)*3, num_images)).astype('uint8')
        sort_idx = SORTED_IDX[vc_i]#np.argsort(-p[:,vc_i])[0:num_images]
        opath = os.path.join(out_dir, str(vc_i))
        if not os.path.exists(opath):
            os.makedirs(opath)
        locs=[]
        for idx in range(num_images):
            iloc = loc_set[:,sort_idx[idx]]
            category = iloc[0]
            loc = iloc[1:6].astype(int)
            if not loc[0] in locs:
                locs.append(loc[0])
                img = in_images[int(loc[0])]
                #img = myresize(img, 224, 'short')
                patch = img[loc[1]:loc[3], loc[2]:loc[4], :]
                #patch_set[:,idx] = patch.flatten()
                if patch.size:
                    filepath = os.path.join(opath, f'{idx:03}.JPEG')
                    patch = cv2.resize(patch, (50, 50), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(filepath, patch)
        if vc_i == max_num_clusters: break
    
    # print summary for each vc
    #if layer=='pool4' or layer =='last': # somehow the patches seem too big for p5
    summary_dir = os.path.join(out_dir, 'canvas')
    if not os.path.exists(summary_dir): os.makedirs(summary_dir)
    for c in range(vc_num):
        iidir = os.path.join(out_dir, str(c))
        files = glob.glob(os.path.join(iidir, '*.JPEG'))
        width = 100
        height = 100
        canvas = np.zeros((0,4*width,3))
        cnt = 0
        for jj in range(4):
            row = np.zeros((height,0,3))
            ii=0
            tries=0
            next=False
            for ii in range(4):
                if (jj*4+ii)< len(files):
                    img_file = files[jj*4+ii]
                    if os.path.exists(img_file):
                        img = cv2.imread(img_file)
                    img = cv2.resize(img, (width,height), interpolation=cv2.INTER_NEAREST)
                else:
                    img = np.ones((height, width, 3))
                row = np.concatenate((row, img), axis=1)
            canvas = np.concatenate((canvas,row),axis=0)
        filepath = os.path.join(summary_dir, f'{str(c)}.JPEG')
        cv2.imwrite(filepath, canvas)
        if c == max_num_clusters: break
    

if __name__ == "__main__":
    imgs_par_cat = np.zeros(len(categories))
    bool_load_existing_cluster = False
    
    occ_level = 'ZERO'
    occ_type = ''
    imgs, labels, masks = getImg('train', categories, dataset, data_path, cat_test, occ_level, occ_type, bool_load_occ_mask=False)
    imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False)
    data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)

    loc_set = []
    feat_set = []
    nfeats = 0
    
    for ii,data in enumerate(data_loader):
        input, mask, label = data
        if np.mod(ii,500)==0:
            print('{} / {}'.format(ii,len(imgs)))

        fname = imgs[ii]
        category = labels[ii]

        if imgs_par_cat[label]<img_per_cat:
            with torch.no_grad():
                tmp = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
            height, width = tmp.shape[1:3]
            img = cv2.imread(imgs[ii])

            # Crop image by some offset
            tmp = tmp[:,offset:height - offset, offset:width - offset]

            # Flatten image at each channel
            gtmp = tmp.reshape(tmp.shape[0], -1)
            if gtmp.shape[1] >= samp_size_per_img:
                rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
            else:
                rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
                #rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
            tmp_feats = gtmp[:, rand_idx].T

            cnt = 0
            for rr in rand_idx:
                ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset))
                hi = (ihi+offset)*(input.shape[2]/height)-Apad
                wi = (iwi + offset)*(input.shape[3]/width)-Apad
                #hi = Astride * (ihi + offset) - Apad
                #wi = Astride * (iwi + offset) - Apad

                #assert (hi >= 0)
                #assert (wi >= 0)
                #assert (hi <= img.shape[0] - Arf)
                #assert (wi <= img.shape[1] - Arf)
                loc_set.append([category, ii, hi,wi,hi+Arf,wi+Arf])
                feat_set.append(tmp_feats[cnt,:])
                cnt+=1

            imgs_par_cat[label]+=1


    feat_set = np.asarray(feat_set)
    loc_set = np.asarray(loc_set).T

    print(feat_set.shape)
    model = vMFMM(vc_num, 'k++')
    model.fit(feat_set, vMF_kappa, max_it=150)
    with open(dict_dir+'dictionary_{}_{}_u.pickle'.format(layer,vc_num), 'wb') as fh:
        pickle.dump(model.mu, fh)


    num = 50
    SORTED_IDX = []
    SORTED_LOC = []
    for vc_i in range(vc_num):
        sort_idx = np.argsort(-model.p[:, vc_i])[0:num]
        SORTED_IDX.append(sort_idx)
        tmp=[]
        for idx in range(num):
            iloc = loc_set[:, sort_idx[idx]]
            tmp.append(iloc)
        SORTED_LOC.append(tmp)

    with open(dict_dir + 'dictionary_{}_{}_p.pickle'.format(layer,vc_num), 'wb') as fh:
        pickle.dump(model.p, fh)
    p = model.p

    print('save top {0} images for each cluster'.format(num))
    example = [None for vc_i in range(vc_num)]
    out_dir = dict_dir + '/cluster_images_{}_{}/'.format(layer,vc_num)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('')

    for vc_i in range(vc_num):
        patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
        sort_idx = SORTED_IDX[vc_i]#np.argsort(-p[:,vc_i])[0:num]
        opath = out_dir + str(vc_i) + '/'
        if not os.path.exists(opath):
            os.makedirs(opath)
        locs=[]
        for idx in range(num):
            iloc = loc_set[:,sort_idx[idx]]
            category = iloc[0]
            loc = iloc[1:6].astype(int)
            if not loc[0] in locs:
                locs.append(loc[0])
                img = cv2.imread(imgs[int(loc[0])])
                img = myresize(img, 224, 'short')
                patch = img[loc[1]:loc[3], loc[2]:loc[4], :]
                #patch_set[:,idx] = patch.flatten()
                if patch.size:
                    cv2.imwrite(opath+str(idx)+'.JPEG',patch)
        #example[vc_i] = np.copy(patch_set)
        if vc_i%10 == 0:
            print(vc_i)

    # print summary for each vc
    #if layer=='pool4' or layer =='last': # somehow the patches seem too big for p5
    for c in range(vc_num):
        iidir = out_dir + str(c) +'/'
        files = glob.glob(iidir+'*.JPEG')
        width = 100
        height = 100
        canvas = np.zeros((0,4*width,3))
        cnt = 0
        for jj in range(4):
            row = np.zeros((height,0,3))
            ii=0
            tries=0
            next=False
            for ii in range(4):
                if (jj*4+ii)< len(files):
                    img_file = files[jj*4+ii]
                    if os.path.exists(img_file):
                        img = cv2.imread(img_file)
                    img = cv2.resize(img, (width,height))
                else:
                    img = np.zeros((height, width, 3))
                row = np.concatenate((row, img), axis=1)
            canvas = np.concatenate((canvas,row),axis=0)

        cv2.imwrite(out_dir+str(c)+'.JPEG',canvas)
