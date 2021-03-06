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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from face_dataset_search import DataLoaderFace
from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--sample_limit', type=int, default=10000, help='subsampling limit for search training')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  criterion = torch.jit.script(ContrastiveLoss())
  criterion = criterion.cuda()
  model = Network(args.init_channels, output_dimension, args.layers, criterion)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  if (args.model_path != "saved_models"):
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_alphas(checkpoint['alphas_normal'], checkpoint['alphas_reduce'])
  
  model = model.cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  #optimizer = optim.Adam(face_model.parameters(), lr=0.0001)
  dataLoaderFace = DataLoaderFace(args.batch_size, workers=4, limit=args.sample_limit)
  train_queue = dataLoaderFace.get_trainloader()
  search_queue = dataLoaderFace.get_searchloader()
  valid_queue = dataLoaderFace.get_valloader()

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, search_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    utils.save_checkpoint_search(model, os.path.join(args.save, 'checkpoint_weights' + str(epoch+1)+ '.tar'), epoch, optimizer)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  accuracy = utils.AvgrageMeter()

  for step, data in enumerate(train_queue):
    model.train()

    anchor_img, positive_img, negative_img, anchor_label, negative_label = data[0], data[1], data[2], data[3], data[4]
    anchor_img_search , positive_img_search , negative_img_search , anchor_label_search , negative_label_search = next(iter(valid_queue))

    anchor_img = Variable(anchor_img, requires_grad=False).cuda()
    positive_img = Variable(positive_img, requires_grad=False).cuda()
    negative_img = Variable(negative_img, requires_grad=False).cuda()

    n = positive_img.shape[0]

    anchor_img_search = Variable(anchor_img_search, requires_grad=False).cuda()
    positive_img_search = Variable(positive_img_search, requires_grad=False).cuda()
    negative_img_search = Variable(negative_img_search, requires_grad=False).cuda()

    labels_p = torch.from_numpy(np.ones((1, positive_img.shape[0]), dtype=None)).cuda(non_blocking=True)
    labels_n = torch.from_numpy(np.zeros((1, negative_img.shape[0]), dtype=None)).cuda(non_blocking=True)
    labels_p_search = torch.from_numpy(np.zeros((1, positive_img_search.shape[0]), dtype=None)).cuda(non_blocking=True)
    labels_n_search = torch.from_numpy(np.zeros((1, negative_img_search.shape[0]), dtype=None)).cuda(non_blocking=True)

    architect.step(anchor_img, positive_img, negative_img, labels_p, labels_n,
      anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()

    anchor_out = model(anchor_img)
    positive_out = model(positive_img)
    negative_out = model(negative_img)

    dist_p = (positive_out - anchor_out).pow(2).sum(1)
    dist_n = (negative_out - anchor_out).pow(2).sum(1)
    loss_p = criterion(dist_p, labels_p)
    loss_n = criterion(dist_n, labels_n)
    loss = loss_n + loss_p

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec = utils.accuracy_face(dist_p, dist_n, threshold)
    objs.update(loss.data.item(), n)
    accuracy.update(prec, n)

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

    anchor_out = model(anchor_img)
    positive_out = model(positive_img)
    negative_out = model(negative_img)

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

