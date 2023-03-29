
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode
from typing import List
import os
import torch as ch
import torchvision
import numpy as np

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from urllib.request import urlretrieve

ORIG_DATAPATH = './dataset'
FFCV_DATAPATH = './dataset_ffcv'

if not os.path.exists(ORIG_DATAPATH):
    os.makedirs(ORIG_DATAPATH)

if not os.path.exists(FFCV_DATAPATH):
    os.makedirs(FFCV_DATAPATH)


# batch_size = 64

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/content/tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform
        self.id_dict = id

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        image = np.asarray(image, dtype=np.uint8)
        image = image.T
        label = self.id_dict[img_path.split('/')[4]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label

class TestTinyImageNetDataset(Dataset):
    def __init__(self, id, transform=None):
        self.filenames = glob.glob("/content/tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform
        self.id_dict = id
        self.cls_dic = {}
        for i, line in enumerate(open('/content/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0],a[1]
            self.cls_dic[img] = self.id_dict[cls_id]
 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        
        image = np.asarray(image, dtype=np.uint8)
        image = image.T
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label


def download_tinyImg200(path,
                     url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                     tarname='tiny-imagenet-200.zip'):
    if not os.path.exists(path):
        os.mkdir(path)
    urlretrieve(url, os.path.join(path,tarname))
    print (os.path.join(path,tarname))
    import zipfile
    zip_ref = zipfile.ZipFile(os.path.join(path,tarname), 'r')
    zip_ref.extractall()
    zip_ref.close()

download_tinyImg200(ORIG_DATAPATH)

id_dict = {}
for i, line in enumerate(open('/content/tiny-imagenet-200/wnids.txt', 'r')):
  id_dict[line.replace('\n', '')] = i

trainset = TrainTinyImageNetDataset(id=id_dict)
testset = TestTinyImageNetDataset(id=id_dict)


datasets = {
    'train': trainset,
    'val': testset
}

for (name, ds) in datasets.items():
    writer = DatasetWriter(f'./{FFCV_DATAPATH}/tinyimagenet_{name}.beton', {
        'image': RGBImageField(),
        'label': IntField()
    })
    writer.from_indexed_dataset(ds)

print("Datasets Created!")

