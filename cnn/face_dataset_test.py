import os
import torchvision.transforms as transforms
import torch
import numpy 
import random
from PIL import Image
from sklearn.model_selection import train_test_split

def idWithZeros(id1):
  idStr = str(id1)
  while (len(idStr) < 4):
    idStr = "0" + idStr
  return idStr

class FaceDataset(torch.utils.data.Dataset):
  def __init__(self, in_path, img_size=(112,112)):
    super(FaceDataset, self).__init__()

    self.in_path = in_path #
    self.img_size = img_size # 
    self.transform = transforms.Compose([
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])
    print(self.in_path)
    print("adding paths")
    
    filelist = []
        
    with open(self.in_path + "/pairs.txt") as fp:
      cnt = 1
      same_pairs = True
      split = 0
      first = True
      for line in fp:
        if(first):
          first = False
          continue
        elif (same_pairs):
          values = line.strip().split('\t')
          name_1 = values[0]
          id_1 = values[1]
          name_2 = name_1
          id_2 = values[2]
          expected = 1
        else:
          values = line.strip().split('\t')
          name_1 = values[0]
          id_1 = values[1]
          name_2 = values[2]
          id_2 = values[3]
          expected = 0
        filename_1 = self.in_path + "/" + name_1 + "/" + name_1 + "_" + idWithZeros(id_1) + ".jpg"
        filename_2 = self.in_path + "/" + name_2 + "/" + name_2 + "_" + idWithZeros(id_2) + ".jpg"
        filelist.append([filename_1, filename_2, same_pairs])
        if cnt%300 == 0:
          same_pairs = not same_pairs
        cnt = cnt+1

    self.data = filelist

  def __len__(self):
    print(len(self.data))
    return len(self.data)

  def __getitem__(self, idx):
    data = self.data
    
    anchor_img = Image.open(data[idx][0]).convert('RGB')
    other_img = Image.open(data[idx][1]).convert('RGB')

    w,h,cw,ch = other_img.size[0], other_img.size[1], min(other_img.size), min(other_img.size)
    box = (w-cw)//2, (h-ch)//2, (w+cw)//2, (h+ch)//2  # left, upper, right, lower
    other_img = other_img.crop(box)
    other_img = other_img.resize(self.img_size)

    w,h,cw,ch = anchor_img.size[0], anchor_img.size[1], min(anchor_img.size), min(anchor_img.size)
    box = (w-cw)//2, (h-ch)//2, (w+cw)//2, (h+ch)//2  # left, upper, right, lower
    anchor_img = anchor_img.crop(box)
    anchor_img = anchor_img.resize(self.img_size)

    anchor_img_t = transforms.ToTensor()(numpy.array(anchor_img))
    other_img_t = transforms.ToTensor()(numpy.array(other_img))
    return self.transform(anchor_img_t),  self.transform(other_img_t), data[idx][2], data[idx][0], data[idx][1]


dataset_dir = "../lfw_funneled"

class DataLoaderFaceTest():
    def __init__(self, workers):
        super(DataLoaderFaceTest, self).__init__()
        self.testloader = torch.utils.data.DataLoader(FaceDataset(dataset_dir), batch_size=1, num_workers=workers, pin_memory=True)

    def get_testloader(self):
        return self.testloader
