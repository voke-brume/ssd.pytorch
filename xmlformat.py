import xml.etree.cElementTree as ET
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
from xml.dom import minidom

from util import process_labels,PyTorchSparkDataset
from data import *

def _pretty_print(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        _pretty_print(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('\t' * depth)
        else:
            parent[index - 1].tail = '\n' + ('\t' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('\t' * (depth - 1))
resize = transforms.Resize(size=(300,
            300))
trainTransforms = transforms.Compose([transforms.ToPILImage(),resize,
            transforms.ToTensor()])
class_map=[{'proba_2': 0,'cheops': 1,'debris':2,'double_star':3,'earth_observation_sat_1':4,'lisa_pathfinder':5,'proba_3_csc':6,'proba_3_ocs':7,'smart_1':8,'soho':9,'xmm_newton':10}]
dataset=PyTorchSparkDataset(class_map,'train',"E:\\CPE 620 final\\data\\",transform=trainTransforms)
for i in range(66000):
    torch_image,bbox,labels=dataset[i]



    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = 'E:\\CPE 620 final\\data\\train_1' # set correct folder name
    ET.SubElement(root, 'filename').text = str(os.path.splitext(os.path.basename(torch_image))[0][:])

    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(300)
    ET.SubElement(size, 'height').text = str(300)
    ET.SubElement(size, 'depth').text = str(3)

    ET.SubElement(root, 'segmented').text = '0'

    obj = ET.SubElement(root, 'object')
    ET.SubElement(obj, 'name').text = str(labels)
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'occluded').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'

    bx = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bx, 'xmin').text = str(bbox[1])
    ET.SubElement(bx, 'ymin').text = str(bbox[0])
    ET.SubElement(bx, 'xmax').text = str(bbox[3])
    ET.SubElement(bx, 'ymax').text = str(bbox[2])


    tree = ET.ElementTree(root)
    
    txt=os.path.splitext(os.path.basename(torch_image))[0][:]    
    txt=txt+'.xml'
    print(txt)
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent = "   ")
    with open(os.path.join("E:\\CPE 620 final\\data\\VOCdevkit\\SPARK\\Annotations\\",str(txt)), "w") as f:
        f.write(xmlstr)
    # tree.write("E:\\CPE 620 final\\data\\VOCdevkit\\SPARK\\Annotations\\"+str(txt))