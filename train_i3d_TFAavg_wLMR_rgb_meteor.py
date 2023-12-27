import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import videotransforms
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# from pytorch_i3d import InceptionI3d
# from pytorch_i3d import InceptionI3dwLMR
from pytorch_i3d import InceptionI3dTavgwLMR
from pytorch_i3d import InceptionI3dNoTavgwLMR
# from charades_dataset import Charades as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
args = parser.parse_args("")
args.mode = 'flow'
args.save_model = 'i3d_model_name'
args.root = '.'

#### Not following OOP approach because i am not comfortable with it. and moreover, non-oop will be helpful for reusability of this code for building X and RI models, also the soft labels are not taken care of in mmaction2 library, so debugging non-OOP code would be more tractable.
import os
import os.path as osp
import re
import warnings
from operator import itemgetter

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import random
from mmaction.apis import init_recognizer
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.cnn import normal_init
from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_recognizer
import copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, average_precision_score
from parallel import DataParallelModel, DataParallelCriterion
torch.random.manual_seed(1)


# config_file = '/scratch/cp_wks/codes/work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_meteor_alllabels_7expanded/tsn_config-activitynet-rgb_meteor_latest.py'
# config_file = '/scratch/cp_wks/codes/work_dirs/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_meteor_alllabels_7expanded_wClassBalancedSampling_RandomCropCorrected_wLMR_wfs20_FullRT/tsn_config-activitynet-rgb_meteor_latest_wCorrectCropAug.py'
# config_file = 'tsn_config-activitynet-rgb_meteor_latest.py'
# checkpoint_file = '/scratch/cp_wks/codes/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52.pth'
device = torch.device('cuda:0')

### pre-computed
# cls_counts = torch.tensor([60, 197, 211, 4811])
cls_counts = torch.tensor([62, 206, 297, 5107, 1526])
# print("Provided class counts:", cls_counts)
num_classes = len(cls_counts)

model_dir = 'work_dirs/saved_models/'
os.makedirs(model_dir, exist_ok=True)

# gpudeviceids = [0,1,2,3]
# batch_size = len(gpudeviceids)*len(cls_counts) #1*len(cls_counts) #2*len(cls_counts)  ### videos_per_gpu must be a multiple of num_classes
batch_size = 1*len(cls_counts)


### class balanced batch sampler
def prepLabelsDS(all_labels, num_samples=None, batch_size = 15):   ### batch size should be a multiple of N -> no. of classes
    S = dict()
    c_max = 0
    for idx in range(len(all_labels)):
        label = all_labels[idx]
        if label not in S:
            S[label] = list()
        S[label].append(idx)
        if(len(S[label]) > c_max):
            c_max = len(S[label])
    N = len(S)
    if(num_samples is not None):
        c_max = num_samples//N
    Sbkp = copy.deepcopy(S)
    return Sbkp, c_max    

def getClassBalancedBatches(Sbkp, c_max, batch_size = 15):   
    ### do this for every epoch
    N = len(Sbkp)
    num_batches = N*(c_max//batch_size)
    S = copy.deepcopy(Sbkp)
    ### oversample with shuffling
    for k in S:
        if(len(S[k]) < c_max):
            temp = []
            for ai in range(0, c_max//len(S[k])):
                random.shuffle(S[k])
                temp = temp + S[k]
            temp = temp + random.sample(S[k], c_max % len(S[k]))
            S[k] = temp
        else:
            S[k] = random.sample(S[k], c_max)
    ### preparing batches
    B = []
    for i in range(num_batches):
        Bi = []
        for k in S:
            start_idx = int(i*batch_size//N)
            end_idx = int((i+1)*batch_size//N)
            Bi += S[k][start_idx:end_idx]
        random.shuffle(Bi)
        B.append(Bi)
    random.shuffle(B)
    return B


# cfg = model.net.cfg

### rawframe data fetch inits
# modality = 'Flow'
modality = 'RGB'
# modality = cfg.data.test.get('modality', 'RGB')
if modality == 'Flow':
    filename_tmpl = 'flow_{}_{:05d}.jpg'
    start_index = 0
else:
    filename_tmpl = 'img_{:05}.jpg'
    start_index = 1
# filename_tmpl = cfg.data.test.get('filename_tmpl', 'img_{:05}.jpg')
# start_index = cfg.data.test.get('start_index', 1)
# count the number of frames that match the format of `filename_tmpl`
# RGB pattern example: img_{:05}.jpg -> ^img_\d+.jpg$
# Flow patteren example: {}_{:05d}.jpg -> ^x_\d+.jpg$
pattern = f'^{filename_tmpl}$'
if modality == 'Flow':
    pattern = pattern.replace('{}', 'x')
pattern = pattern.replace(pattern[pattern.find('{'):pattern.find('}') + 1], '\\d+')

### prepare train data pre-processing pipeline
train_pipeline = [{'type': 'SampleFrames', 'clip_len': 64, 'frame_interval': 1, 'num_clips': 1, 'test_mode': False},
                 {'type': 'RawFrameDecode'},
                 # {'type': 'RandomRescale', 'scale_range': (256, 320)},
                 # {'type': 'RandomCrop', 'size': 224},
                 {'type': 'Resize', 'scale': (-1, 256)},
                 {'type': 'RandomResizedCropWidthBounds',
                  'area_range': ((680*1720)/(1080*1920), 1.0), 'aspect_ratio_range': (1720/1080, 1920/680), 'min_width_ratio': 0.844},
                 {'type': 'Resize', 'scale': (224, 224), 'keep_ratio': False},
                 {'type': 'Flip', 'flip_ratio': 0.5},
                 {'type': 'Normalize', 'mean': [128, 128], 'std': [128, 128], 'to_bgr': False},
                 {'type': 'FormatShape', 'input_format': 'NCHW_Flow'},
                 {'type': 'Collect', 'keys': ['imgs', 'label'], 'meta_keys': ['frame_inds', 'original_shape', 'img_shape', 'keep_ratio', 'scale_factor', 'flip']},
                 #, 'crop_bbox'
                 {'type': 'ToTensor', 'keys': ['imgs', 'label']}]
# train_pipeline = copy.deepcopy(cfg.data.train.pipeline)
if modality == 'RGB':
    for i in range(len(train_pipeline)):
        if 'FormatShape' in train_pipeline[i]['type']:
            train_pipeline[i] = {'type': 'FormatShape', 'input_format': 'NCHW'}
        if 'Normalize' in train_pipeline[i]['type']:
            train_pipeline[i] = {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False}    
train_pipeline = Compose(train_pipeline)

### prepare val data pre-processing pipeline
val_pipeline = [{'type': 'SampleFrames', 'clip_len': 64, 'frame_interval': 1, 'num_clips': 1, 'test_mode': True},
                 {'type': 'RawFrameDecode'},
                 {'type': 'Resize', 'scale': (-1, 256)},
                 # {'type': 'CenterCrop', 'crop_size': (224, 224)},
                 {'type': 'CenterCrop', 'crop_size': (420, 208)},
                 {'type': 'Resize', 'scale': (224, 224), 'keep_ratio': False},
                 {'type': 'Normalize', 'mean': [128, 128], 'std': [128, 128], 'to_bgr': False},
                 {'type': 'FormatShape', 'input_format': 'NCHW_Flow'},
                 {'type': 'Collect', 'keys': ['imgs', 'label'], 'meta_keys': []},
                 {'type': 'ToTensor', 'keys': ['imgs']}]
# val_pipeline = copy.deepcopy(cfg.data.val.pipeline)
if modality == 'RGB':
    for i in range(len(val_pipeline)):
        if 'FormatShape' in val_pipeline[i]['type']:
            val_pipeline[i] = {'type': 'FormatShape', 'input_format': 'NCHW'}
        if 'Normalize' in val_pipeline[i]['type']:
            val_pipeline[i] = {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False}
val_pipeline = Compose(val_pipeline)



fps = 30

if modality == 'Flow':
    ann_file_train = 'dasets/meteor/original_data/rawframe_format_fullres/train_alllabels_flow_7expanded/annotations_oneclass.txt'
    ann_file_val = 'datasets/meteor/original_data/rawframe_format_fullres/val_alllabels_flow_7expanded/annotations_oneclass.txt'
else:
    ann_file_train = 'datasets/meteor/original_data/rawframe_format_fullres/train_alllabels_7expanded/annotations_oneclass.txt'
    ann_file_val = 'datasets/meteor/original_data/rawframe_format_fullres/val_alllabels_7expanded/annotations_oneclass.txt'
    
video_infos = {'video':[], 'label':[], 'duration':[], 'total_frames':[]}
for line in open(ann_file_train):
    video_infos['video'].append(line.strip().split(' ')[0])
    video_infos['total_frames'].append(int(line.strip().split(' ')[1]))
    video_infos['duration'].append(video_infos['total_frames'][-1] / fps)
    
    # label = np.zeros((num_classes,video_infos['total_frames'][-1]), np.float32)
    # label[int(line.strip().split(' ')[2]), :] = 1
    # video_infos['label'].append(label)
    video_infos['label'].append(int(line.strip().split(' ')[2]))

val_video_infos = {'video':[], 'label':[], 'duration':[], 'total_frames':[]}
for line in open(ann_file_val):
    val_video_infos['video'].append(line.strip().split(' ')[0])
    val_video_infos['total_frames'].append(int(line.strip().split(' ')[1]))
    val_video_infos['duration'].append(val_video_infos['total_frames'][-1] / fps)
    
    # label = np.zeros((num_classes,val_video_infos['total_frames'][-1]), np.float32)
    # label[int(line.strip().split(' ')[2]), :] = 1
    # val_video_infos['label'].append(label)
    val_video_infos['label'].append(int(line.strip().split(' ')[2]))
        
print("Train data length:", len(video_infos['video']))
print("Val data length:", len(val_video_infos['video']))


### Sbkp contains class-specific dataset-indices list; c_max_samples is the no. of samples to retrieve from each class (by over/under-sampling)
Sbkp, c_max_samples = prepLabelsDS(video_infos['label'], num_samples=len(video_infos['label']), batch_size = batch_size)
print("max samples per class after class-balancing:", c_max_samples)

class_counts = {}
for skey in Sbkp:
    class_counts[skey] = len(Sbkp[skey])
print("Class counts:", class_counts)

# exp_df = {'epoch':[], 'Yield': [], 'Cutting': [], 'RuleBreak': [], 'OverTaking': [], 'Deviate': [], 'AvgAcc':[], 'top1_acc':[]}  #, 'epoch_acc':[], 'loss':[]
exp_df = {'epoch':[], 'OverTaking': [], 'Deviate': [], 'RuleBreak': [], 'Cutting': [], 'Yield': [], 'mAP': [], 'AvgAcc':[], 'top1_acc':[]}  #, 'epoch_acc':[], 'loss':[]

########## revisit params (also scaling lr for multi gpus.
init_lr=0.001 #*len(gpudeviceids)   #0.0125  #0.1 ## 0.025 
#init_lr=0.01
# init_lr=0.1
max_steps=64e3 
mode='flow'    #'rgb'

# save_model='i3d_model_name'
# save_model='i3d_model_name_IB'
# save_model='i3d_model_name_CB_0.0025lr'
# save_model='i3d_model_name_CB_0.25lr' ### is actually 64tl --> clip-length
#save_model='i3d_model_name_CB_0.025lr'
# save_model='i3d_model_name_CB_0.1lr'
# save_model='i3d_model_name_CB_0.1lr_latelrsched'
# save_model='i3d_model_name_CB_0.0125lr_latelrsched'
# save_model='i3d_model_name_CB_0.0125lr_latelrsched_IB'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IB'
# save_model='i3d_model_name_CB_0.001lr_latelrsched_IB'
# save_model='i3d_model_name_CB_0.001lr_latelrsched_CB'
# save_model='i3d_model_name_0.0125lr_latelrsched_IB_0.001lr_LMR'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IB_0.001lr_LMR'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IB_0.01lr_LMR'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.01lr_LMR'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.001lr_LMR'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.01lr_LMR_avgtemporalpool'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.001lr_LMR_avgtemporalpool'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.01lr_LMR_Tavgagg'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.001lr_LMR_Tavgagg'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.001lr_LMR0.35l_Tavgagg'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.001lr_LMR0.25l0.25d_Tavgagg'
# save_model='i3d_model_name_CB_0.1lr_latelrsched_IBe18_0.001lr_LMR0.25l1.0d20B_Tavgagg'
# save_model='i3d_RGB_model_0.1lr_latelrsched_IBe18_0.001lr_LMR0.42l1.0d20B_Tavgagg'
# save_model='i3d_RGB_model_0.1lr_latelrsched_IB_mformerRandCrop018_LMR0.42l1.0d20B_Tavgagg'
# save_model='i3d_RGB_model_0.1lr_latelrsched_IBe18_0.1lr_LMR0.42l1.0d20B_Tavgagg'
# save_model='i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB017_0.001lr_LMR0.42l1.0d20B_Tavgagg'
# save_model='i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB028_0.001lr_LMR0.42l1.0d20B_Tavgagg'
#save_model='i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB028_0.01lr_LMR0.42l1.0d20B_Tavgagg'
# save_model='i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB028_0.1lr_LMR0.42l1.0d20B_Tavgagg'
# save_model='i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB028_0.001lr_LMR0.42l1.0d20B_Tavgagg_dropout0'
# save_model='i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB009_0.001lr_LMR0.42l0.25d20B_Tavgagg_dropout0'
# save_model='i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB009_0.001lr_LMR0.42l0.5d20B_Tavgagg_dropout0'
save_model='i3d_RGB_model_0.1lr_latelrsched_IB009_0.001lr_NestedLMR0.42l1.0d20B_Tavgagg_dropout0'
save_model='i3d_RGB_model_0.1lr_latelrsched_IB009_0.001lr_NestedLMR0.42l0.5d20B_Tavgagg_dropout0'

### recheck
save_model='i3d_RGB_model_0.1lr_latelrsched_IB009_0.001lr_LMR0.42l1.0d20B_Tavgagg_dropout0'

# save_model='i3d_RGB_model_0.1lr_latelrsched_IB_HDDpretrained012_0.001lr_LMR0.42l1.0d20B_Tavgagg_dropout0'
save_model='i3d_RGB_model_0.1lr_latelrsched_IB_HDDpretrained012_0.001lr_LMR0.42l0.5d20B_Tavgagg_dropout0'

save_model='i3d_model_name_CB_0.1lr_latelrsched_IB009_0.001lr_LMR0.42l0.5d20B_Tavgagg_dropout0_softCE'

print('\n',save_model,'\n')

sampling_type = 'CB'
# sampling_type = 'IB'

# setup the LMR model
cls_counts = torch.tensor([62, 206, 297, 5107, 1526])
# lrec = 0.25
# lrec = 0.35
lrec = 0.42

# decay = 1.0
# decay = 0.25
decay = 0.5
wthresh = 20.0
eplmr = 0.1
if modality == 'Flow':
    # i3d = InceptionI3dwLMR(num_classes=num_classes, in_channels=2, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr)
    i3d = InceptionI3dTavgwLMR(num_classes=num_classes, in_channels=2, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr)
    # i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    ### epoch should be ep-1
    # model_epoch_get = 19
    # i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_model_name_CB_0.0125lr_latelrsched_IB'+str(model_epoch_get-1).zfill(3)+'.pt', map_location=device))
    # i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_model_name_CB_0.1lr_latelrsched_IB011.pt', map_location=device))
    i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_model_name_CB_0.1lr_latelrsched_IB018.pt', map_location=device))
else:
#     # i3d = InceptionI3dwLMR(num_classes=num_classes, in_channels=3, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr)
#     # i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
#     # i3d = InceptionI3dTavgwLMR(num_classes=num_classes, in_channels=3, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr)
#     # i3d = InceptionI3dTavgwLMR(num_classes=num_classes, in_channels=3, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr, dropout_keep_prob=0.0)
    
#     # i3d = InceptionI3dTavgwLMR(num_classes=num_classes, in_channels=3, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr, dropout_keep_prob=0.0, nested_lmr=True)
    
    i3d = InceptionI3dTavgwLMR(num_classes=num_classes, in_channels=3, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr, dropout_keep_prob=0.0, nested_lmr=False)
    i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_RGB_model_0.1lr_latelrsched_IB009.pt', map_location=device))
    
#     # i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB017.pt', map_location=device))
#     # i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_RGB_model_0.1lr_latelrsched_cRT_IB_wholeRT_0.1lr_IB028.pt', map_location=device))
#     # i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_RGB_model_0.1lr_latelrsched_IB_mformerRandCrop018.pt', map_location=device))
    
    #i3d = InceptionI3dTavgwLMR(num_classes=num_classes, in_channels=3, class_counts=cls_counts, l=lrec, d=decay, omega=wthresh, epsilon=eplmr, dropout_keep_prob=0.0, nested_lmr=False)
    #i3d.load_state_dict(torch.load('work_dirs/saved_models/i3d_RGB_model_0.1lr_latelrsched_IB_HDDpretrained012.pt', map_location=device))
    print("trainlmr.............................")
# i3d.replace_logits(num_classes)
# i3d.replace_logits_wfclayer()
i3d.to(device)
##i3d = nn.DataParallel(i3d , device_ids=gpudeviceids)
# i3d = DataParallelModel(i3d)

optimizer = optim.SGD(i3d.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0)#0.0000001)
# lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30])
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [20])

### only helpful for sequence data, which may result in occupying large space to store output data and its gradients in a single gpu. then parallelization would be required. DDP is better (but a custom class-specific distributed sampler would be required).
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = DataParallelCriterion(loss_fn)
loss_fn = SoftTargetCrossEntropy()

# ### for best val accuracy
# best_loss = 0
num_steps_per_update = 1 ### accum gradient (for large batch size that does not fit into memory this should be higher)
num_epochs = 40
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            i3d.train()
            all_video_dirs = copy.deepcopy(video_infos['video'])
            all_video_total_frames = copy.deepcopy(video_infos['total_frames'])
            all_labels = copy.deepcopy(video_infos['label'])
            if(sampling_type=='CB'):
                epoch_batches = getClassBalancedBatches(Sbkp, c_max_samples, batch_size = batch_size)
            else:
                datset_indices = list(range(0, len(all_labels)))
                random.shuffle(datset_indices)
                epoch_batches = [list(range(batch_idx, min(batch_idx+batch_size, len(all_labels)))) for batch_idx in range(0, len(all_labels), batch_size)]
        else:
            i3d.eval()
            all_video_dirs = copy.deepcopy(val_video_infos['video'])
            all_video_total_frames = copy.deepcopy(val_video_infos['total_frames'])
            all_labels = copy.deepcopy(val_video_infos['label'])
            epoch_batches = [list(range(batch_idx, min(batch_idx+batch_size, len(all_labels)))) for batch_idx in range(0, len(all_labels), batch_size)]
        if(len(epoch_batches[-1])!=batch_size):
            epoch_batches = epoch_batches[:-1]
        ### temp test
        # epoch_batches = epoch_batches[:3]
        num_batches_ = len(epoch_batches)
        print("\ntotal_batches",num_batches_)
        
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        # pred_labels = []
        all_pred_labels = []
        all_pred_scores = []
        all_gt_labels = []
        num_iter = 0
        for iteration, batch_idxs in tqdm(enumerate(epoch_batches)):
            batched_data = None
            batched_data_imgs = torch.tensor([])
            batched_data_label = torch.tensor([])
            ### extracting data for each element in batch
            for idx in batch_idxs:
                if(sampling_type=='IB' and phase=='train'):
                    idx = datset_indices[idx]
                ### prepare data
                # video = all_video_dirs[idx]
                # total_frames = len(list(filter(lambda x: re.match(pattern, x) is not None, os.listdir(video))))
                ### not sure if total frames should be multiplied by 2, check output, also check if the label values are coming out alright???????????????
                data = dict(frame_dir=all_video_dirs[idx],
                            total_frames=all_video_total_frames[idx],
                            label=all_labels[idx],
                            start_index=start_index,
                            filename_tmpl=filename_tmpl,
                            modality=modality)
                if phase == 'train':
                    data = train_pipeline(data)
                    # ### useful for X model --> do stacking to collect it for all batch elements
                    # img_metas = copy.deepcopy(data['img_metas'])
                    data.pop('img_metas')
                else:
                    data = val_pipeline(data)
                
                ### reformatting for i3d model intake
                if modality == 'Flow':
                    data['imgs'] = torch.cat([data['imgs'][0, 0:data['imgs'].shape[1]:2, ...].unsqueeze(0), data['imgs'][0, 1:data['imgs'].shape[1]:2, ...].unsqueeze(0)])
                else:
                    data['imgs'] = data['imgs'].transpose(1, 0)
                
                data = collate([data], samples_per_gpu=1)
                
                batched_data_imgs = torch.cat([batched_data_imgs, data['imgs']])
                if(batched_data_label.shape[0]==0):
                    batched_data_label = copy.deepcopy(data['label'])
                else:
                    batched_data_label = torch.cat([batched_data_label, data['label']])
                
            batched_data_imgs = batched_data_imgs.to(device)
            batched_data_label = batched_data_label.squeeze().to(device)
            
            num_iter += 1
            temporal_length = batched_data_imgs.size(2)
            per_frame_logits, modified_batched_data_label = i3d(batched_data_imgs, batched_data_label)
            
            all_pred_labels.extend( torch.argmax(per_frame_logits, dim=1).cpu().detach().tolist() )
            all_pred_scores.extend( per_frame_logits.cpu().detach().tolist() )
            all_gt_labels.extend( batched_data_label.tolist() )
            
            # compute classification loss (with max-pooling along time B x C )
            #### Uncomment below line for normal cross entropy loss
            #cls_loss = F.binary_cross_entropy_with_logits(per_frame_logits, modified_batched_data_label)
            #### For soft cross entropy loss
            cls_loss = loss_fn(per_frame_logits, modified_batched_data_label)
            tot_cls_loss += cls_loss.data

            loss = cls_loss/num_steps_per_update
            tot_loss += loss.data
            if phase == 'train':
                loss.backward()
                
            if num_iter == num_steps_per_update and phase == 'train':
                num_iter = 0
                optimizer.step()
                optimizer.zero_grad()
                
                if iteration % 100 == 0:
                    print( '{} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_cls_loss/(100), tot_loss*num_steps_per_update/100))
                    tot_loss = tot_cls_loss = 0.

            ### free memory from gpu
            del loss, cls_loss, per_frame_logits, batched_data, batched_data_imgs, batched_data_label
    
        if phase == 'val':
            print( 'Epoch_{} {} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(epoch+1, phase, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) )
        else:
            torch.save(i3d.state_dict(), model_dir+save_model+str(epoch).zfill(3)+'.pt')
        
        ### train val acc.
        performance_out = classification_report(all_gt_labels, all_pred_labels, target_names= ['Yielding', 'Cutting', 'RB-WL', 'Overtake', 'Deviate'], zero_division=1)
        yield_acc, cutting_acc, rbwl_acc, overtake_acc, deviate_acc, avg_acc, wavg_acc = \
        float(performance_out.split('\n')[2].strip().split('     ')[2].strip()), \
        float(performance_out.split('\n')[3].strip().split('     ')[2].strip()), \
        float(performance_out.split('\n')[4].strip().split('     ')[2].strip()), \
        float(performance_out.split('\n')[5].strip().split('     ')[2].strip()), \
        float(performance_out.split('\n')[6].strip().split('     ')[2].strip()), \
        float(performance_out.split('\n')[9].strip().split('     ')[2].strip()), \
        float(performance_out.split('\n')[10].strip().split('     ')[2].strip())
        print("\nC/A, Avg C/A, Overall A")
        print([yield_acc, cutting_acc, rbwl_acc, overtake_acc, deviate_acc], avg_acc, wavg_acc)
        
        if phase == 'val':
            ### Compute AP
            all_gt_labels = np.array(all_gt_labels)
            all_pred_scores = np.array(all_pred_scores)
            target_names= ['Yield', 'Cutting', 'RuleBreak', 'OverTaking', 'Deviate']
            target_aps = []
            for cls in range(len(target_names)):
                class_ap = np.round(average_precision_score((all_gt_labels==cls).astype('int64'), all_pred_scores[:, cls]), 3)
                target_aps.append(class_ap) 
                exp_df[target_names[cls]].append(class_ap)
            # average_precision_score(y_true_oh, y_scores,average=None)

            print("\nclass APs, mAP")
            print(target_aps, np.round(np.mean(target_aps), 3))

            exp_df['epoch'].append(epoch+1)
            exp_df['top1_acc'].append(wavg_acc)
            exp_df['AvgAcc'].append(avg_acc)
            exp_df['mAP'].append(np.round(np.mean(target_aps), 3))
            # exp_df['Yield'].append(yield_acc)
            # exp_df['Cutting'].append(cutting_acc)
            # exp_df['RuleBreak'].append(rbwl_acc)
            # exp_df['OverTaking'].append(overtake_acc)
            # exp_df['Deviate'].append(deviate_acc)

            exp_df_ = pd.DataFrame(exp_df)
            exp_df_.to_csv(model_dir+save_model+'_all_epoch_models_accuracies.csv',index=False)

    lr_sched.step()


# print(exp_df)
exp_df_ = pd.DataFrame(exp_df)
exp_df_.to_csv(model_dir+save_model+'_all_epoch_models_accuracies.csv',index=False)
exp_df_
