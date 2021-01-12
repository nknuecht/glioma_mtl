# Nicholas Nuechterlein

import numpy as np
import nibabel as nib
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def get_input_params(dataformat, mtl=True):
    '''
    This function returns a parameters used in our dataloader
    Arguments
    ---------
    dataformat: one of 'raw3D', 'crop3Dslice', 'raw3D'
               - 'modality3D' <- single MRI sequence cropped to tumor boundary (1 channel volume)
               - 'crop3Dslice' <- 4 MRI sequences cropped to tumor boundary (4 channel volume)
               - 'raw3D' <- 4 whole-brain MRI sequences (4 channel volume)
    mtl: whether or not to use unlabeled MRI sequences; i.e., a choice between simple CNNs or MTL network

    Outputs
    ---------
    dataformat:      Revise dataformat string if MTL is used
    channels:        Input channels for MRI input
    resize_shape:    Hard coded input sizes for indiviual MRI sequences: either (64, 64, 64) or (144, 144, 144)
    '''
    try:
        if dataformat == 'modality3D':
            channels = 1
            resize_shape = (64, 64, 64)
            if mtl:
                dataformat = 'modality3D_mtl'
        elif dataformat == 'crop3Dslice':
            channels = 4
            resize_shape = (64, 64, 64)
            if mtl:
                dataformat = 'cropped3D_mtl'
        elif dataformat == 'raw3D':
            channels = 4
            resize_shape = (144, 144, 144)
            if mtl:
                dataformat = 'raw3D_mtl'

        return dataformat, channels, resize_shape
    except:
        print('Incorrect dataformat')
        return _, _, _

def get_data_splits(metadata_df, task='idh', mtl = False):

    '''
    This function returns pandas dataframes containing training and validation indices and sample metadata
    Arguments
    ---------
    metadata_df:   For each sample this dataframe indicates
                        1) whether it is in the labeled training, unlabeled training set, or valiation set
                        2) its idh status, and 1p19q status
    task:          Either 'idh' or '1p19q'
    mtl:           If True, MRI data without IDH mutation or 1p/19q co-deletion labels will be included in the traing set

    Outputs
    ---------
    train_df:      Dataframe of training samples
    val_df:        Dataframe of validation samples
    classes:       Names of numerical labels
    '''

    # validation set
    if task == 'idh':
        classes = ['wildtype', 'mutant']
        val_df = metadata_df.loc[(metadata_df['phase'] == 'val')
                                 & (metadata_df[task].isin([0,1]))] # check whether IDH status known
    elif task == '1p19q':
        classes = ['non-codel', 'oligo']
        val_df = metadata_df.loc[metadata_df['phase'] == 'val']
    else:
        print('ERROR: task must be "idh" or "1p19q"')

    # training set
    if mtl:
        train_df = metadata_df.loc[metadata_df['phase'].isin(['train', 'unlabeled train'])]
    else:
        train_df = metadata_df.loc[(metadata_df['phase'] == 'train')
                                   & (metadata_df[task].isin([0,1]))] # only labeled data (0/1)

    return train_df, val_df, classes

def class_accuracies(pred_tensor, label_tensor, dice_dict=None, probs_list = [], classes=['wildtype', 'oligo', 'mutant'],
                        verbose=True, phase='train'):

    # put labels and predictions on the cpu.
    try:
        pred_arr = np.asarray([int(x.cpu().numpy()) for x in pred_tensor])
        label_arr = np.asarray([int(x.cpu().numpy()) for x in label_tensor])
    except:
        pred_arr = np.asarray([int(x) for x in pred_tensor])
        label_arr = np.asarray([int(x) for x in label_tensor])


    # get accuracy on each class
    acc_dict = {}
    for i in range(len(classes)):
        labels = label_arr[label_arr == i]
        preds = pred_arr[label_arr == i]
        acc = np.sum(labels == preds)/labels.shape[0]
        acc_dict[classes[i]] = acc

    # get AUC for 2 two class prediction
    if len(classes) == 2:
        try:
            auc_score = roc_auc_score(label_arr, np.asarray(probs_list))
            f1 = f1_score(label_arr, pred_arr)
            precision = precision_score(label_arr, pred_arr, zero_division=0)
            recall = recall_score(label_arr, pred_arr)

        except:
            print(' - - - - ERROR in auc calculation - - - - ')
            auc_score = roc_auc_score(label_arr, np.asarray(probs_list))
            auc_score = -1
    else:
        auc_score = None
    average_acc = np.mean(list(acc_dict.values()))

    if dice_dict is not None:
        dice_wt = np.mean(dice_dict['dice_wt'])
        dice_core = np.mean(dice_dict['dice_core'])
        dice_enh = np.mean(dice_dict['dice_enh'])


    else:
        dice_wt, dice_core, dice_enh = -1, -1, -1



    return average_acc, auc_score, dice_wt, dice_core, dice_enh, f1, precision, recall



def dice_score(im1, im2, smooth=1):
    '''
    Compute the dice score between two input images or volumes. Note that we use a smoothing factor of 1.
    :param im1: Image 1
    :param im2: Image 2
    Outputs
    ---------
    Dice score
    '''
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() + smooth) / (im1.sum() + im2.sum() + smooth)

def get_whole_tumor_mask(data):
    # this function returns the Whole tumor mask (i.e. all labels are merged)
    return data > 0

def get_tumor_core_mask(data):
    # This function merges label 1 and 3 to get the Core Mask
    return np.logical_or(data == 1, data == 3)

def get_enhancing_tumor_mask(data):
    # This function returns the mask for enhancing tumor i.e. label 3
    # Note that in original files, there is no mask element with value 4. We have renamed label 4 to label 3 in
    # our transformations.
    return data == 3

def get_dice_scores(im1, im2):
    # This function computes the dice score for Whole tumor, core mask, and enhancing tumor.
    if torch.is_tensor(im1):
        if im1.is_cuda:
            im1 = im1.detach().cpu().numpy()
        else:
            im1 = im1.numpy()
    if torch.is_tensor(im2):
        if im2.is_cuda:
            im2 = im2.detach().cpu().numpy()
        else:
            im2 = im2.numpy()

    im1_wt = get_whole_tumor_mask(im1)
    im2_wt = get_whole_tumor_mask(im2)

    im1_core = get_tumor_core_mask(im1)
    im2_core = get_tumor_core_mask(im2)

    im1_enh = get_enhancing_tumor_mask(im1)
    im2_enh = get_enhancing_tumor_mask(im2)

    d_wt = dice_score(im1_wt, im2_wt)
    d_core = dice_score(im1_core, im2_core)
    d_enh = dice_score(im1_enh, im2_enh)

    return d_wt, d_core, d_enh


def seg_eval(model, flair_img, t1_img, t1ce_img, t2_img, device='cpu'):
    xmin, xmax, ymin, ymax, zmin, zmax = cropVolumes(flair_img, t1_img, t1ce_img, t2_img)

    flair_cropped = min_max(flair_img[xmin:xmax, ymin:ymax, zmin:zmax])
    t1_cropped   = min_max(t1_img[xmin:xmax,  ymin:ymax, zmin:zmax])
    t1ce_cropped  = min_max(t1ce_img[xmin:xmax,  ymin:ymax, zmin:zmax])
    t2_cropped    = min_max(t2_img[xmin:xmax,    ymin:ymax, zmin:zmax])

    # convert to Tensor
    resize = (1, flair_cropped.shape[0], flair_cropped.shape[1], flair_cropped.shape[2])
    flair_cropped = flair_cropped.reshape(resize)
    t1_cropped = t1_cropped.reshape(resize)
    t1ce_cropped = t1ce_cropped.reshape(resize)
    t2_cropped = t2_cropped.reshape(resize)

    flair_tensor = torch.from_numpy(flair_cropped)
    t1_tensor = torch.from_numpy(t1_cropped)
    t1ce_tensor = torch.from_numpy(t1ce_cropped)
    t2_tensor = torch.from_numpy(t2_cropped)
    del flair_cropped, t1_cropped, t1ce_cropped, t2_cropped

    # concat the tensors and then feed them to the model
    tensor_concat = torch.cat([flair_tensor, t1_tensor, t1ce_tensor, t2_tensor], 0)  # inputB #
    tensor_concat = torch.unsqueeze(tensor_concat, 0)
    tensor_concat = tensor_concat.to(device)

    test_resolutions = [None, (144, 144, 144)]
    output = None
    for res in test_resolutions:
        if output is not None:
            # test at the scaled resolution and combine them`
            output = output + model(tensor_concat.double(), inp_res=res)
        else:
            # test at the original resolution
            output = model(tensor_concat.double(), inp_res=res)
    output = output / len(test_resolutions)
    del tensor_concat

    # convert the output to segmentation mask and move to CPU
    # seg_image = output[0].max(0)[1].data.byte().cpu().numpy() # changed this (should double check)
    seg_image = output.max(1)[1].data.byte().cpu().numpy()
    return seg_image, output

def get_bb_3D_torch(img, pad=0):
    xs = torch.nonzero(torch.sum(torch.sum(img, axis=1), axis=1))
    ys = torch.nonzero(torch.sum(torch.sum(img, axis=0), axis=1))
    zs = torch.nonzero(torch.sum(torch.sum(img, axis=0), axis=0))
    if xs.shape[0] != 0:
        xmin, xmax = torch.min(xs), torch.max(xs)
    else:
        print('zero seg x dim')
        xmin, xmax = 0, img.shape[0]-1

    if ys.shape[0] != 0:
        ymin, ymax = torch.min(ys), torch.max(ys)
    else:
        print('zero seg y dim')
        ymin, ymax = 0, img.shape[1]-1

    if zs.shape[0] != 0:
        zmin, zmax = torch.min(zs), torch.max(zs)
    else:
        print('zero seg z dim')
        zmin, zmax = 0, img.shape[2]-1

    # ymin, ymax = torch.min(ys), torch.max(ys)
    # zmin, zmax = torch.min(zs), torch.max(zs)
    bbox = (xmin-pad, ymin-pad, zmin-pad, xmax+pad, ymax+pad, zmax+pad)
    return bbox

def cropVolume(img, data=False):
    '''
    Helper function that removes the black area around the brain in the MRI images
    Arguments
    ---------
    img: 3D volume
    data: nibabel (nib) allows you to access 3D volume data using the get_data(). If you have already used it before
    calling this function, then it is false

    Outputs
    ---------
    returns the crop positions acrss 3 axes (channel, width and height)
    '''
    if not data:
       img = img.get_data()
    sum_array = []


    for ch in range(img.shape[2]):
        values, indexes = np.where(img[:, :, ch] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    ch_s = np.nonzero(sum_array)[0][0]
    ch_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[0]):
        values, indexes = np.where(img[width, :, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    wi_s = np.nonzero(sum_array)[0][0]
    wi_e = np.nonzero(sum_array)[0][-1]
    sum_array = []
    for width in range(img.shape[1]):
        values, indexes = np.where(img[:, width, :] > 0)
        sum_val = sum(values)
        sum_array.append(sum_val)
    hi_s = np.nonzero(sum_array)[0][0]
    hi_e = np.nonzero(sum_array)[0][-1]

    return ch_s, ch_e, wi_s, wi_e, hi_s, hi_e


def cropVolumes(img1, img2, img3, img4):
    '''
    This function crops the 4 volumes that BRATS dataset provides
    :param img1: Volume 1
    :param img2: Volume 2
    :param img3: Volume 3
    :param img4: Volume 4
    :return: maximum dimensions across three dimensions
    '''
    zmin1, zmax1, xmin1, xmax1, ymin1, ymax1 = cropVolume(img1, True)
    zmin2, zmax2, xmin2, xmax2, ymin2, ymax2 = cropVolume(img2, True)
    zmin3, zmax3, xmin3, xmax3, ymin3, ymax3 = cropVolume(img3, True)
    zmin4, zmax4, xmin4, xmax4, ymin4, ymax4 = cropVolume(img4, True)

    # find the maximum dimensions

    xmin = min(xmin1, xmin2, xmin3, xmin4)
    xmax = max(xmax1, xmax2, xmax3, xmax4)
    ymin = min(ymin1, ymin2, ymin3, ymin4)
    ymax = max(ymax1, ymax2, ymax3, ymax4)
    zmin = min(zmin1, zmin2, zmin3, zmin4)
    zmax = max(zmax1, zmax2, zmax3, zmax4)

    return xmin, xmax, ymin, ymax, zmin, zmax

def get_bb(seg_img):
    '''
    This function returns a tumor 2D bounding box from a 2D MRI slice using a segmentation mask
    '''
    xmin, xmax = np.min(np.nonzero(np.sum(seg_img, axis=1))),  np.max(np.nonzero(np.sum(seg_img, axis=1)))
    ymin, ymax = np.min(np.nonzero(np.sum(seg_img, axis=0))),  np.max(np.nonzero(np.sum(seg_img, axis=0)))
    bbox = (xmin, ymin, xmax, ymax)
    return bbox

def get_bb_3D(img, pad=0):
    '''
    This function returns a tumor 3D bounding box using a segmentation mask
    '''
    xs = np.nonzero(np.sum(np.sum(img, axis=1), axis=1))
    ys = np.nonzero(np.sum(np.sum(img, axis=0), axis=1))
    zs = np.nonzero(np.sum(np.sum(img, axis=0), axis=0))
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    zmin, zmax = np.min(zs), np.max(zs)
    bbox = (xmin-pad, ymin-pad, zmin-pad, xmax+pad, ymax+pad, zmax+pad)
    return bbox


def get_maxslice_bb(seg_img, plane='axial', percentile=25):
    '''
    This function returns the largest 2D slice of the tumor on the MRI image
    '''

    if plane == 'axial':
        depth_sums = np.sum(np.sum(seg_img, axis=0), axis = 0)
        rand_slice = np.random.randint(seg_img.shape[2])
    elif plane == 'coronal':
        depth_sums = np.sum(np.sum(seg_img, axis=0), axis = 1)
        rand_slice = np.random.randint(seg_img.shape[1])
    elif plane == 'sagittal':
        depth_sums = np.sum(np.sum(seg_img, axis=1), axis = 1)
        rand_slice = np.random.randint(seg_img.shape[0])
    nonzero_depth_sums = depth_sums[depth_sums > 0]
    percentile_val = np.percentile(nonzero_depth_sums, percentile)

    while depth_sums[rand_slice] <= percentile_val:
        rand_slice = np.random.randint(seg_img.shape[2])
    if plane == 'axial':
        seg_slice = seg_img[:, :, rand_slice]
    elif plane == 'coronal':
        seg_slice = seg_img[:, rand_slice, :]
    elif plane == 'sagittal':
        seg_slice = seg_img[rand_slice, :, :]


    bbox = get_bb(seg_slice)
    return rand_slice, bbox




def largest_slice(seg_path, return_max_slice=True, axial_only=True, percentile=25, plane='axial'):
    '''
    This function returns the largest 2D slice of the brain on the MRI image
    '''
    seg_img = nib.load(seg_path).get_data()
    if return_max_slice and axial_only:
        max_seg_pixels = 0
        largest_slice = -1
        for frame in range(seg_img.shape[2]):

            img_slice = seg_img[:,:,frame]
            num_pixels = np.sum(img_slice[img_slice>0])
            if num_pixels > max_seg_pixels:
                max_seg_pixels = num_pixels
                largest_slice = frame
        return largest_slice, None, plane


    elif not return_max_slice and axial_only:
        # plane = 'axial' # 'plane' is in input
        rand_axial_slice, bbox = get_maxslice_bb(seg_img=seg_img, plane='axial', percentile=25)
        return rand_axial_slice, bbox, plane

    elif return_max_slice and not axial_only:
        print('TODO: implement for coronal and sagittal planes')
    elif not return_max_slice and not axial_only:
        plane_list = ['axial', 'coronal', 'sagittal']
        plane = plane_list[np.random.randint(3)]

        rand_axial_slice, bbox = get_maxslice_bb(seg_img=seg_img, plane=plane, percentile=25)
        return rand_axial_slice, bbox, plane


def min_max(img):
    '''
    Min-max normalization
    '''
    return (img - img.min()) / (img.max() - img.min())
