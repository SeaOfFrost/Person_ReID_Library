# encoding: utf-8


import torch.nn.functional as F

from .triplet_loss import TripletLoss
from .imptriplet_loss import ImpTripletLoss

def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    imptriplet = ImpTripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    
    if sampler == 'softmax':
        def loss_func(score, feat, labels):
            return F.cross_entropy(score, labels)
    elif sampler == 'triplet':
        def loss_func(score, feat, labels):
            return triplet(feat, labels)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, labels):
            return F.cross_entropy(score, labels) + triplet(feat, labels)[0]
    elif sampler == 'softmax_imptriplet':
        def loss_func(score, feat, labels):
            return F.cross_entropy(score, labels) + imptriplet(feat, labels)[0]
    elif sampler == 'MGN':
        def loss_func(outputs, labels):
            T_Loss = [triplet(output, labels)[0] for output in outputs[1:4]]
            T_Loss = sum(T_Loss) / len(T_Loss)
            C_Loss = [F.cross_entropy(output, labels) for output in outputs[4:]]
            C_Loss = sum(C_Loss) / len(C_Loss)
            return T_Loss + 2 * C_Loss
    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_imptriplet, ''but got {}'.format(sampler))
        
    return loss_func
