import torch
from dataset import get_dataset, HyperX
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils_HSI import sample_gt
import tqdm
import clip
from model import build
from torch.optim import Adam, AdamW, SGD
import torch.nn.functional as F


device = 'cuda'

parser = argparse.ArgumentParser(description='PyTorch LDGnet')
parser.add_argument('--data_path', type=str, default='/media/data/ty/clip/Houston/',
                    help='the path to load the data')
parser.add_argument('--source_name', type=str, default='Houston13',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the test dir')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=5,
                    help='multiple of of data augmentation')
parser.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")


# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', default=True, action='store_true',
                    help="See data exploration visualization")



args = parser.parse_args()


img_src, gt_src, LABEL_VALUES_src, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                        args.data_path)
img_tar, gt_tar, LABEL_VALUES_tar, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                        args.data_path)
sample_num_src = len(np.nonzero(gt_src)[0])
sample_num_tar = len(np.nonzero(gt_tar)[0])
training_sample_tar_ratio = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar

num_classes=gt_src.max()
N_BANDS = img_src.shape[-1]
hyperparams = vars(args)
hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                    'device': device, 'center_pixel': False, 'supervision': 'full'})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

r = int(hyperparams['patch_size']/2)+1
img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))     

train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
img_src_con, train_gt_src_con = img_src, train_gt_src

for i in range(args.re_ratio-1): 
    img_src_con = np.concatenate((img_src_con,img_src))
    train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))

hyperparams_train = hyperparams.copy()
hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})
train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)






@torch.no_grad()
def test(model):
    model.eval()
    result = {'grass healthy':[],
              'grass stressed':[],
              'trees':[],
              'water':[],
              'residential buildings':[],
              'non-residential buildings':[],
              'road':[]}
    label_list = ['grass healthy', 'grass stressed', 'trees', 'water', 'residential buildings', 'non-residential buildings', 'road']
    for data, label in tqdm.tqdm(testloader):
        data = data.to('cuda')
        label = label.to('cuda')
        logits_per_image, logits_per_text = model(data, text)
        value, idx = torch.max(logits_per_image, dim=-1)
        label = label - 1
        if idx == label:
            result[label_list[int(label)]].append(1)
        else:
            result[label_list[int(label)]].append(0)
    for i in result:
        result[i] = sum(result[i])/len(result[i])
    return result






#you need to adjust
lr = 1e-3
batch_size = 32
weight_decay=0.05
epoch = 12
#




trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=2)
device = 'cuda'

model = build('vitb32', device=device)


optim = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

text = torch.cat([clip.tokenize(f'A hyperspectral image of {LABEL_VALUES_src[k]}') for k in [0,1,2,3,4,5,6]]).to(device)
for i in range(epoch):
    model.train()
    for data, label in tqdm.tqdm(trainloader):
        data, label = data.to(device), label.to(device)
        label = label - 1
        label = label
        

        logits_per_image, logits_per_text = model(data, text)
        # for name, value in model.named_parameters():
        #     if 'conv' in name:
        #         print(name)
        #         print(value)
        #loss = F.cross_entropy(logits_per_image, label.long(), reduction='none')
        loss = F.cross_entropy(logits_per_image, label.long())
        
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss.item())
    print(test(model))
    




