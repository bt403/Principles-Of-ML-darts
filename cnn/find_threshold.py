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
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

output_dimension = 128
testset = args.testset

def obtainMetrics(results, threshold):
    results_matrix = np.array(results)
    true_labels = np.array(results_matrix[:, 1]).astype(int)
    distance = np.array(results_matrix[:, 0]).astype(float)
    pred_labels = np.where(distance <= threshold, 1, 0)
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    FP_Rate = FP*100.0/(FP + TN)
    FN_Rate = FN*100.0/(FN + TP)

    if (math.isnan(FP_Rate)):
        FP_Rate = 0.0
    if (math.isnan(FN_Rate)):
        FN_Rate = 0.0
    Accuracy = (TP + TN)*100.0/(TP + TN + FP + FN)
    return (FP_Rate, FN_Rate, Accuracy)

def obtainThreshold(results):
  threshold = 0.2
  FP_Rate, FN_Rate, Accuracy = obtainMetrics(results, threshold)
  old = (FP_Rate, FN_Rate, Accuracy, threshold)

  #Increase threshold by 0.1
  while FP_Rate < FN_Rate and threshold:
    old = (FP_Rate, FN_Rate, Accuracy, threshold)
    threshold = threshold + 0.1
    FP_Rate, FN_Rate, Accuracy = obtainMetrics(results, threshold)
  FP_Rate, FN_Rate, Accuracy, threshold = old

  #Increase threshold by 0.05
  while FP_Rate < FN_Rate and threshold:
    old = (FP_Rate, FN_Rate, Accuracy, threshold)
    threshold = threshold + 0.05
    FP_Rate, FN_Rate, Accuracy = obtainMetrics(results, threshold)
  FP_Rate, FN_Rate, Accuracy, threshold = old

  #Increase threshold by 0.01
  while FP_Rate < FN_Rate and threshold:
    old = (FP_Rate, FN_Rate, Accuracy, threshold)
    threshold = threshold + 0.01
    FP_Rate, FN_Rate, Accuracy = obtainMetrics(results, threshold)
  FP_Rate, FN_Rate, Accuracy, threshold = old

  #Increase threshold by 0.001
  while FP_Rate < FN_Rate and threshold:
    old = (FP_Rate, FN_Rate, Accuracy, threshold)
    threshold = threshold + 0.001
    FP_Rate, FN_Rate, Accuracy = obtainMetrics(results, threshold)

  if abs(old[0]-old[1]) < abs(FP_Rate - FN_Rate):
    FP_Rate, FN_Rate, Accuracy, threshold = old
  return (FP_Rate, FN_Rate, Accuracy, threshold)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

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
  results = infer(valid_queue, model, testset)
  FP_Rate, FN_Rate, Accuracy, threshold = obtainThreshold(results)
  print(FP_Rate, FN_Rate, Accuracy, threshold)
  logging.info('Accuracies %f %f %f %f', FP_Rate, FN_Rate, Accuracy, threshold)


def infer(valid_queue, model, testset):
  model.eval()
  results = []
  print("running")
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
    c = 0
    for i in dist_p:
        row = [dist_p[c].item(), 1]
        results.append(row)
        c = c+1
    c = 0
    for j in dist_n:
        row = row = [dist_n[c].item(), 0]
        results.append(row)
        c = c+1

    if step % args.report_freq == 0:
      logging.info('valid %03d', step)

  return results





if __name__ == '__main__':
  main() 

