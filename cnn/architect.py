import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, anchor_img, positive_img, negative_img, labels_p, labels_n, eta, network_optimizer):
    loss = self.model._loss(anchor_img, positive_img, negative_img, labels_p, labels_n)
    theta = _concat(self.model.parameters()).data
    try:
      #for v in self.model.parameters():
      #  print(network_optimizer.state[v])
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except Exception as e:
      print("except")
      print(repr(e))
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, anchor_img, positive_img, negative_img, labels_p, labels_n,
      anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search,  eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(anchor_img, positive_img, negative_img, labels_p, labels_n, anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search, eta, network_optimizer)
    else:
        self._backward_step(anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search)
    self.optimizer.step()

  def _backward_step(self, anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search):
    loss = self.model._loss(anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search)
    loss.backward()

  def _backward_step_unrolled(self, anchor_img, positive_img, negative_img, labels_p, labels_n,
      anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(anchor_img, positive_img, negative_img, labels_p, labels_n, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(anchor_img_search, positive_img_search, negative_img_search, labels_p_search, labels_n_search)
    unrolled_loss.backward()

    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, anchor_img, positive_img, negative_img, labels_p, labels_n)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input_p, target_p, input_n, target_n, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      print(type(p.data))
      print(type(p.data))
      print("type r")
      print(type(R))
      print("type v")
      print(type(v))
      #p.data.add_(R, v)
      p.data.add_(R)
      p.data.add_(v)
    loss_p = self.model._loss(input_p, target_p)
    loss_n = self.model._loss(input_n, target_n)
    loss = loss_p + loss_n
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss_p = self.model._loss(input_p, target_p)
    loss_n = self.model._loss(input_p, target_p)
    loss = loss_p + loss_n
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

