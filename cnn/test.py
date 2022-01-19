import os
import sys
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from face_dataset_train import DataLoaderFace
from face_dataset_test import DataLoaderFaceTest

import math

from torch.autograd import Variable
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--testset', type=str, default='ms1m', help='which architecture to use')
parser.add_argument('--sample_limit', type=int, default=10000, help='subsampling limit for search training')
parser.add_argument('--threshold', type=int, default=0.5, help='threshold value for testing')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

output_dimension = 128
threshold = args.threshold

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  testset = args.testset
  
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, output_dimension, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  if (testset == "lfw"):
    dataLoaderFace = DataLoaderFaceTest(workers=4)
    valid_queue = dataLoaderFace.get_testloader()
  else:
    dataLoaderFace = DataLoaderFace(args.batch_size, workers=4, limit=args.sample_limit)
    valid_queue = dataLoaderFace.get_valloader()

  model.drop_path_prob = args.drop_path_prob
  accuracy = infer(valid_queue, model, testset)
  logging.info('Accuracy %f', accuracy)


def infer(valid_queue, model, testset):
  model.eval()
  accuracy = utils.AvgrageMeter()
  results = []
  for step, data in enumerate(valid_queue):
    if (testset == "lfw"):
      anchor_img, other_img, truth, anchor_label, other_label = data[0], data[1], data[2], data[3], data[4]
      anchor_img = Variable(anchor_img, requires_grad=False).cuda()
      other_img = Variable(other_img, requires_grad=False).cuda()
    else:
      anchor_img, positive_img, negative_img, anchor_label, negative_label = data[0], data[1], data[2], data[3], data[4]
      anchor_img = Variable(anchor_img, requires_grad=False).cuda()
      positive_img = Variable(positive_img, requires_grad=False).cuda()
      negative_img = Variable(negative_img, requires_grad=False).cuda()

    n = anchor_img.shape[0]
    
    dist_p = np.array([])
    dist_n = np.array([])
    if (testset == "lfw"):
      anchor_out, _ = model(anchor_img)
      other_out, _ = model(other_img)
      if(truth == 1):
        labels_p = torch.from_numpy(np.ones((1, other_img.shape[0]), dtype=None)).cuda(non_blocking=True)
        dist_p = (other_out - anchor_out).pow(2).sum(1)
      else:
        labels_n = torch.from_numpy(np.zeros((1, other_img.shape[0]), dtype=None)).cuda(non_blocking=True)
        dist_n = (other_out - anchor_out).pow(2).sum(1)
    else:
      labels_p = torch.from_numpy(np.ones((1, positive_img.shape[0]), dtype=None)).cuda(non_blocking=True)
      labels_n = torch.from_numpy(np.zeros((1, negative_img.shape[0]), dtype=None)).cuda(non_blocking=True)
        
      anchor_out, _ = model(anchor_img)
      positive_out, _ = model(positive_img)
      negative_out, _ = model(negative_img)

      dist_p = (positive_out - anchor_out).pow(2).sum(1)
      dist_n = (negative_out - anchor_out).pow(2).sum(1)
    
    prec = utils.accuracy_face(dist_p, dist_n, threshold)
    accuracy.update(prec, n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %f', step, accuracy.avg)

  return accuracy.avg


if __name__ == '__main__':
  main() 

