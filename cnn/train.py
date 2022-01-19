import os
import sys
import time
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


from torch.autograd import Variable
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--sample_limit', type=int, default=10000, help='subsampling limit for search training')

args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

output_dimension = 128
threshold = 0.5

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = torch.mean(1/2*(label) * torch.pow(dist, 2) +
                                      1/2*(1-label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

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
  if(args.model_path != "saved_models"):
    utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = torch.jit.script(ContrastiveLoss())
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  #train_transform, valid_transform = utils._data_transforms_cifar10(args)
  print("creating dataloader")
  dataLoaderFace = DataLoaderFace(args.batch_size, workers=4, limit=args.sample_limit)
  train_queue = dataLoaderFace.get_trainloader()
  print("finished creating dataloader 1")
  valid_queue = dataLoaderFace.get_valloader()
  print("finished creating dataloader 2")

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, str(epoch) + '_weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  accuracy = utils.AvgrageMeter()
  model.train()

  for step, data in enumerate(train_queue):
    anchor_img, positive_img, negative_img, anchor_label, negative_label = data[0], data[1], data[2], data[3], data[4]

    anchor_img = Variable(anchor_img, requires_grad=False).cuda()
    positive_img = Variable(positive_img, requires_grad=False).cuda()
    negative_img = Variable(negative_img, requires_grad=False).cuda()
    labels_p = torch.from_numpy(np.ones((1, positive_img.shape[0]), dtype=None)).cuda(non_blocking=True)
    labels_n = torch.from_numpy(np.zeros((1, negative_img.shape[0]), dtype=None)).cuda(non_blocking=True)
 
    optimizer.zero_grad()
    
    anchor_out, anchor_out_aux = model(anchor_img)
    positive_out, positive_out_aux = model(positive_img)
    negative_out, negative_out_aux = model(negative_img)

    dist_p = (positive_out - anchor_out).pow(2).sum(1)
    dist_n = (negative_out - anchor_out).pow(2).sum(1)
    loss_p = criterion(dist_p, labels_p)
    loss_n = criterion(dist_n, labels_n)
    loss = loss_n + loss_p
    
    if args.auxiliary:
      dist_p_aux = (positive_out_aux - anchor_out_aux).pow(2).sum(1)
      dist_n_aux = (negative_out_aux - anchor_out_aux).pow(2).sum(1)
      loss_p_aux = criterion(dist_p_aux, labels_p)
      loss_n_aux = criterion(dist_n_aux, labels_n)
      loss_aux = loss_n_aux + loss_p_aux
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    n = positive_img.shape[0]
    prec = utils.accuracy_face(dist_p, dist_n, threshold)
    objs.update(loss.data.item(), n)
    accuracy.update(prec, n)
    
    #top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      print("--- Distances ---")
      print(dist_p)
      print(dist_n)
      logging.info('train %03d %f %f', step, objs.avg, accuracy.avg)

  return accuracy.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  accuracy = utils.AvgrageMeter()
  model.eval()

  for step, data in enumerate(valid_queue):
    anchor_img, positive_img, negative_img, anchor_label, negative_label = data[0], data[1], data[2], data[3], data[4]
    anchor_img = Variable(anchor_img, requires_grad=False).cuda()
    positive_img = Variable(positive_img, requires_grad=False).cuda()
    negative_img = Variable(negative_img, requires_grad=False).cuda()

    n = positive_img.shape[0]

    labels_p = torch.from_numpy(np.ones((1, positive_img.shape[0]), dtype=None)).cuda(non_blocking=True)
    labels_n = torch.from_numpy(np.zeros((1, negative_img.shape[0]), dtype=None)).cuda(non_blocking=True)

    anchor_out, _ = model(anchor_img)
    positive_out, _ = model(positive_img)
    negative_out, _ = model(negative_img)

    dist_p = (positive_out - anchor_out).pow(2).sum(1)
    dist_n = (negative_out - anchor_out).pow(2).sum(1)
    loss_p = criterion(dist_p, labels_p)
    loss_n = criterion(dist_n, labels_n)
    loss = loss_n + loss_p

    prec = utils.accuracy_face(dist_p, dist_n, threshold)
    objs.update(loss.data.item(), n)
    accuracy.update(prec, n)
    
    if step % args.report_freq == 0:
      logging.info('valid %03d %f %f', step, objs.avg, accuracy.avg)

  return accuracy.avg, objs.avg


if __name__ == '__main__':
  main() 

