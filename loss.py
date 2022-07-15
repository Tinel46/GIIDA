import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1)
    qlogp = (q * logp).sum(dim=1)
    return qlogq - qlogp

def consistency_loss(logits_w, logits_s, T=1.0, p_cutoff=0.9):

    logits_w = logits_w.detach() ##weak aug gradient detach
    logits_w = logits_w / T
    logits_s = logits_s / T
    pseudo_label = torch.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)
    mask_binary = max_probs.ge(p_cutoff)
    mask = mask_binary.float()
    masked_loss = kl_div_with_logit(logits_w, logits_s) * mask
    return masked_loss.mean(), mask.mean()

def ent(logits, temperature):

    B, C = logits.shape
    epsilon = 1e-5
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
    return torch.sum(torch.sum(-pred * torch.log(pred + epsilon) / weight_sum * entropy_weight, dim=-1))

def entropy(predictions, reduction='none'):

    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

def kld(logits, num_classes, temperature):

    B, C = logits.shape
    epsilon = 1e-5
    pred = F.softmax(logits / temperature, dim=1)
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)

    #### mini entropy rank
    # indices = torch.argsort(entropy_weight, dim=0, descending=True).squeeze()
    # ratio = 0.8
    # indice_high = indices[:int(len(indices) * ratio)]
    # indice_low = indices[int(len(indices) * ratio):int(len(indices))]
    # H1 = torch.sum(torch.sum(-pred[indice_low] * torch.log(pred[indice_low] + epsilon) / weight_sum * entropy_weight[indice_low],dim=-1))
    # H2 = torch.sum(torch.sum(-torch.log(pred[indice_high] + epsilon) / (num_classes * weight_sum) * entropy_weight[indice_high], dim=-1))
    # H = H1+H2
    return torch.sum(torch.sum(-torch.log(pred + epsilon)/(num_classes*weight_sum)* entropy_weight,dim=-1))

def l2(logits, temperature):

    B, C = logits.shape
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
    return torch.sum(torch.sum(pred**2/ weight_sum*entropy_weight, dim=-1))

def reg(logits, temperature):
    B, C = logits.shape
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    weight_sum = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
    return torch.sum((1 - torch.sum(pred ** 2 / weight_sum * entropy_weight, dim=-1)))

def weightAnchor(logits, temperature):
    B, C = logits.shape
    pred = F.softmax(logits / temperature, dim=1)  ##
    entropy_weight = entropy(pred).detach()
    entropy_weight = 1 + torch.exp(-entropy_weight)
    entropy_weight = (B * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
    return entropy_weight


def get_trans_matrix_prob(p_s_n, y_s, num_class):
    # p_s_n is probability form
    # onehot label
    y_s_mask = torch.zeros(y_s.size(0), num_class).to(device).scatter_(1, y_s.unsqueeze(1), 1)

    T_all = y_s_mask.unsqueeze(dim=-1).matmul(p_s_n.unsqueeze(dim=-2)).sum(0)
    count = torch.sum(T_all, dim=-1)
    count = torch.where(count == 0., torch.tensor(1.).to(device), count)
    count = (1.0 / count).unsqueeze(dim=-1)

    T = torch.mul(T_all, count)

    return T





