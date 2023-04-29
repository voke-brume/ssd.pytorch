from util import SPARKDataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import glob
import torchvision.utils 
from torch.utils.data.dataloader import default_collate
from torchvision.models import *
import random
import torch
from util import process_labels,PyTorchSparkDataset
from data import *


resize = transforms.Resize(size=(300,
            300))
trainTransforms = transforms.Compose([transforms.ToPILImage(),resize,
            transforms.ToTensor()])
class_map=[{'proba_2': 0,'cheops': 1,'debris':2,'double_star':3,'earth_observation_sat_1':4,'lisa_pathfinder':5,'proba_3_csc':6,'proba_3_ocs':7,'smart_1':8,'soho':9,'xmm_newton':10}]
dataset=PyTorchSparkDataset(class_map,'train',"E:\\CPE 620 final\\data\\",transform=trainTransforms)

# def create_mask(bb, x):
#     """Creates a mask for the bounding box of same shape as image"""
#     rows,cols,*_ = x.shape
#     Y = np.zeros((rows, cols))
#     bb = bb.astype(np.int)
#     Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
#     return Y
# def mask_to_bb(Y):
#     """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
#     cols, rows = np.nonzero(Y)
#     if len(cols)==0: 
#         return np.zeros(4, dtype=np.float32)
#     top_row = np.min(rows)
#     left_col = np.min(cols)
#     bottom_row = np.max(rows)
#     right_col = np.max(cols)
#     return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

# def create_bb_array(x):
#     """Generates bounding box array from a train_df row"""
#     return np.array([x[5],x[4],x[7],x[6]])


print(len(dataset))
dl = DataLoader(dataset, batch_size=4, shuffle=True)
print(len(dl))
batch_iterator = iter(dl)
for i in range(20):
    images, targets1 = next(batch_iterator)
    print(targets1)
    targets=targets1[1][:]
    # print(targets)


