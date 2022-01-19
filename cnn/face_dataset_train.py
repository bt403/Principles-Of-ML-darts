import os
import torchvision.transforms as transforms
import torch
import numpy 
import random
from PIL import Image
from sklearn.model_selection import train_test_split

class FaceDataset(torch.utils.data.Dataset):
  def __init__(self, in_path, limit=10000, mode='train', img_size=(112,112)):
    super(FaceDataset, self).__init__()

    self.mode = mode #train or test
    self.in_path = in_path #
    self.img_size = img_size # 
    self.transform = transforms.Compose([
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])
    print(self.in_path)
    self.labels = []
    self.imgs_path = []
    self.imgs_path_val = []
    self.labels_val = []
    c = 0
    self.limit = limit
    print("adding paths")
    with open('data_set.txt', 'w') as f:
      for (dirpath, dirnames, filenames) in os.walk(dataset_dir_ms1m):
        if c >= limit:
          print("break")
          break;
        else:
          for file in filenames:
            _, ext = os.path.splitext(file)
            if ext == ".jpeg" and c < limit:
              self.labels.append(os.path.basename(os.path.normpath(dirpath)))
              self.imgs_path.append(os.path.join(dirpath, file))
              c = c+1
              item = os.path.join(dirpath, file) + "," + os.path.basename(os.path.normpath(dirpath))
              f.write("%s\n" % item)
            else:
              c = c+1
              break;
    print("finished adding paths")

    self.imgs_path ,self.imgs_path_val = train_test_split(self.imgs_path,test_size=0.1, random_state=42)
    self.labels ,self.labels_val = train_test_split(self.labels,test_size=0.1, random_state=42)
    print("finished splits")
    self.data =  numpy.array(list(zip(self.imgs_path, self.labels)))        
    self.data_val =  numpy.array(list(zip(self.imgs_path_val, self.labels_val)))

  def __len__(self):
    if (self.mode == "train"):
      print("length train")
      print(len(self.imgs_path))
      return len(self.imgs_path)
    else:
      print("length val")
      print(len(self.imgs_path_val))
      return len(self.imgs_path_val)

  def __getitem__(self, idx):
    if self.mode == "train":
      data = self.data
    else:
      data = self.data_val

    index = idx
    positive_list = numpy.array([])
    while (positive_list.size == 0):
      anchor_label = data[index][1]
      mask = numpy.ones(data.shape[0], dtype=bool)
      mask[index] = False
      positive_list = data[mask]
      positive_list = positive_list[positive_list[:,1]==anchor_label] 
      if (positive_list.size == 0):
        index = random.randrange(len(data))

    positive_item = random.choice(positive_list)
    positive_img = Image.open(positive_item[0]).convert('RGB')

    negative_list = data[mask]
    negative_list = negative_list[negative_list[:,1]!=anchor_label]
    negative_item = random.choice(negative_list)
    
    negative_img = Image.open(negative_item[0]).convert('RGB')
    anchor_img = Image.open(data[index][0]).convert('RGB')

    w,h,cw,ch = positive_img.size[0], positive_img.size[1], min(positive_img.size), min(positive_img.size)
    box = (w-cw)//2, (h-ch)//2, (w+cw)//2, (h+ch)//2  # left, upper, right, lower
    positive_img = positive_img.crop(box)
    positive_img = positive_img.resize(self.img_size)

    w,h,cw,ch = negative_img.size[0], negative_img.size[1], min(negative_img.size), min(negative_img.size)
    box = (w-cw)//2, (h-ch)//2, (w+cw)//2, (h+ch)//2  # left, upper, right, lower
    negative_img = negative_img.crop(box)
    negative_img = negative_img.resize(self.img_size)

    w,h,cw,ch = anchor_img.size[0], anchor_img.size[1], min(anchor_img.size), min(anchor_img.size)
    box = (w-cw)//2, (h-ch)//2, (w+cw)//2, (h+ch)//2  # left, upper, right, lower
    anchor_img = anchor_img.crop(box)
    anchor_img = anchor_img.resize(self.img_size)

    anchor_img_t = transforms.ToTensor()(numpy.array(anchor_img))
    positive_img_t = transforms.ToTensor()(numpy.array(positive_img))
    negative_img_t = transforms.ToTensor()(numpy.array(negative_img))
    return self.transform(anchor_img_t),  self.transform(positive_img_t),  self.transform(negative_img_t), anchor_label, negative_item[1]


dataset_dir = "./lfw_funneled"
dataset_dir_td = "./TD_RGB"
dataset_dir_ms1m = "../ms1m-data"
#dataset_dir_ms1m = "../TD_RGB"

class DataLoaderFace():
    def __init__(self, batch_size, workers, limit=10000):
        super(DataLoaderFace, self).__init__()
        self.trainloader = torch.utils.data.DataLoader(FaceDataset(dataset_dir_ms1m, limit=limit), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        #self.searchloader = torch.utils.data.DataLoader(FaceDataset(dataset_dir_ms1m, mode="val", limit=limit), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        self.searchloader = ""
        self.valloader = torch.utils.data.DataLoader(FaceDataset(dataset_dir_ms1m, mode="val", limit=limit), batch_size=4, shuffle=True, num_workers=workers, pin_memory=True)

    def get_trainloader(self):
        return self.trainloader

    def get_searchloader(self):
        return self.searchloader
    
    def get_valloader(self):
        return self.valloader