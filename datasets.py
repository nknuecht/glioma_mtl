import os
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from ast import literal_eval
import copy


from utils import largest_slice, min_max
from format_data import cropped3D_mtl, modality3D_mtl, raw3D_mtl
import skimage.transform as skTrans

from utils import min_max
import nibabel as nib


import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

from models.ESPNet import ESPNet, SegOut

class GeneralDataset(Dataset):

    def __init__(self,
                metadata_df,
                root_dir,
                genomic_csv_file = None,
                transform=None,
                seg_transform=None, ###
                label='cluster',
                classes=['wildtype', 'oligo', 'mutant'],
                dataformat=None, # indicates what shape (or content) should be returned (2D or 3D, etc.)
                returndims=None, # what size/shape 3D volumes should be returned as.
                return_max_slice=True, ## lose this.
                percentile=25, # when pulling slices out, what percentile of tumor area should be needed to make a slice a valid choice
                validate_on_volume=False,
                visualize=False,
                modality=None,
                brats2tcia_df=None,
                include_genomic_data=True,
                pad=2,
                pretrained=None,
                device='cpu'):
        """
        Args:
            csv_file (string): Path to the csv file bounding box and ?
            root_dir (string): Directory for MR images
            transform (callable, optional):
            dataformat: format (3D, 2D, cropped, etc.) to be returned.
        """
        self.device=device
        self.brats2tcia_df = brats2tcia_df
        if self.brats2tcia_df is not None:
            self.tcia2brats_df  = self.brats2tcia_df.dropna().reset_index().set_index('tciaID').rename(columns={'index':'bratsID'})
        else: self.tcia2brats_df = None

        # if os.path.isfile(str(csv_file)): self.metadata_df = pd.read_csv(csv_file, index_col=0)
        # else:
        self.metadata_df = metadata_df

        self.label = label
        self.root_dir = root_dir
        self.transform = transform
        self.seg_transform = seg_transform
        self.dataformat = dataformat

        self.returndims=returndims
        self.classes = classes
        self.return_max_slice = return_max_slice
        self.percentile = percentile
        self.validate_on_volume = validate_on_volume
        self.visualize = visualize
        self.modality = modality
        self.pad = pad
        self.include_genomic_data = include_genomic_data
        self.pretrained = pretrained
        # self.brats2tcia_df = brats2tcia_df

        self.seg_net = ESPNet(classes=4, channels=4)
        if self.pretrained is not None:
            assert os.path.isfile(self.pretrained), 'Pretrained file does not exist. {}'.format(pretrained)
            weight_dict = torch.load(self.pretrained, map_location=self.device)
            self.seg_net.load_state_dict(weight_dict)

        self.genomic_csv_file = genomic_csv_file
        if self.genomic_csv_file is not None:
            self.genomic_df = pd.read_csv(self.genomic_csv_file, index_col=0)
            self.genomic_df = pd.concat([self.tcia2brats_df, self.genomic_df], axis=1, join='inner').set_index('bratsID')


    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tciaID = self.metadata_df.iloc[idx]['tciaID']
        phase = self.metadata_df.iloc[idx]['phase']
        BraTS18ID = self.metadata_df.iloc[idx].name

        # get SCNA data if avalible
        if self.genomic_csv_file is not None and BraTS18ID in self.genomic_df.index:
            genomic_data = self.genomic_df.loc[BraTS18ID].values
            genomic_data = genomic_data.astype(np.float)
        else:
            genomic_data = None



        # make dictonary of paths to MRI volumnes (modalities) and segmenation masks
        mr_path_dict = {}
        sequence_type = ['seg', 'seg_wt', 't1', 't1ce', 'flair', 't2']
        for seq in sequence_type:
            mr_path_dict[seq] = os.path.join(self.root_dir,BraTS18ID,BraTS18ID + '_'+seq+'.nii.gz')



        # seg_image_wt = None


        if self.dataformat == 'cropped3D_mtl':
            image, seg_image, seg_image_wt = cropped3D_mtl(mr_path_dict=mr_path_dict, pad=self.pad)
        elif self.dataformat == 'modality3D_mtl':
            image, seg_image, seg_image_wt = modality3D_mtl(mr_path_dict, pad=self.pad, modality=self.modality)
        elif self.dataformat == 'raw3D_mtl':
            image, seg_image, seg_image_wt= raw3D_mtl(mr_path_dict=mr_path_dict, pad=self.pad)

        if seg_image is not None:
            seg_image[seg_image == 4] = 3

        # get sample metadata
        label = self.metadata_df.iloc[idx][self.label]
        survival = self.metadata_df.iloc[idx]['OS']
        event = self.metadata_df.iloc[idx]['OS_EVENT']
        tciaID = self.metadata_df.iloc[idx]['tciaID']

        # create null variable for samples who are missing data or labels
        image_null_value = np.zeros(image.shape).astype(np.float)
        shape = image.shape
        if len(image.shape) == 3:
            shape = (1, shape[0], shape[1], shape[2])
        seg_null_value = np.ones((shape[1], shape[2], shape[3])).astype(np.float)*255
        genomic_null_value = np.ones(self.genomic_df.shape[1]).astype(np.float) * 0
        null_int = 255

        # create null and non-null samples
        sample = {'image': image, 'seg_image':seg_image, 'label': label, 'OS':survival,
                  'event':event, 'image_paths':mr_path_dict, 'tciaID':tciaID, 'bratsID':BraTS18ID, 'genomic_data':genomic_data}

        null_sample = {'image': image_null_value, 'seg_image':seg_null_value, 'label': null_int, 'OS':0,
                  'event':0, 'image_paths':{}, 'tciaID':'', 'bratsID':BraTS18ID, 'genomic_data':genomic_null_value}

        if seg_image_wt is None:
            sample['seg_probs'] = image_null_value
        else:
            sample['seg_probs'] = seg_image_wt

        if not self.include_genomic_data:
            sample['genomic_data'] = null_sample['genomic_data']

        # check existance of ground truth segmentation label
        gt_seg = self.metadata_df.iloc[idx]['gt_seg']


        if self.transform:
            sample['image'] = self.transform(sample['image'])
        if self.seg_transform:
            sample['seg_image'] = self.seg_transform(sample['seg_image'])
        if self.seg_transform:
            sample['seg_probs'] = self.seg_transform(sample['seg_probs'])

        if tciaID != 0 and (label == 0 or label == 1): # scan from the tcia.
            ## if the sample is from the TCIA that means that you have survival (for sure)
            if gt_seg == 1:
                if phase in ['train', 'val']: # means we have subtype labels
                    return (sample['image'], sample['seg_image'], sample['genomic_data'], sample['seg_probs']), sample['label'], (sample['OS'], sample['event']), sample['bratsID']
                else:
                    # these are the tcia images that we don't have data on.
                    # there are 7 of these: 'Brats18_TCIA02_606_1', 'Brats18_TCIA03_138_1', 'Brats18_TCIA03_338_1', 'Brats18_TCIA03_474_1', 'Brats18_TCIA04_328_1', 'Brats18_TCIA08_278_1', 'Brats18_TCIA13_634_1'
                    return (sample['image'], sample['seg_image'], null_sample['genomic_data'], sample['seg_probs']), null_sample['label'],(sample['OS'], sample['event']), sample['bratsID']

            else: # we have a segmenation predicted from our network, not ground truth.
                if phase in ['train', 'val']: # means we have cluster labels
                    return (sample['image'], sample['seg_image'], sample['genomic_data'], sample['seg_probs']), sample['label'],(sample['OS'], sample['event']), sample['bratsID']
                else:
                    # these are the tcia images that we don't have data on.
                    # there is 1 of these: 'Brats18_TCIA11_612_1'
                    return (sample['image'], sample['seg_image'], null_sample['genomic_data'], sample['seg_probs']), null_sample['label'],(sample['OS'], sample['event']), sample['bratsID']

        else: # is a brats scan not from the tcia
            if gt_seg == 1:
                return (sample['image'], sample['seg_image'], null_sample['genomic_data'], sample['seg_probs']), null_sample['label'], (null_sample['OS'], null_sample['event']), sample['bratsID']

            else: # we have a segmenation predicted from our network, not ground truth.
                return (sample['image'], sample['seg_image'], null_sample['genomic_data'], sample['seg_probs']), null_sample['label'], (null_sample['OS'], null_sample['event']), sample['bratsID']
