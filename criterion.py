import os
import sys
import torch




def mixup_criterion_su(criterion, pred, y_a, y_b, lam, weight_a=None, weight_b=None):
	if weight_a is not None and weight_b is not None:
		return lam * torch.mean(criterion(pred, y_a) * weight_a) + (1 - lam) * torch.mean(criterion(pred, y_b) * weight_b)
	else:
		return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_criterion_unsu(criterion, pred, y_a, y_b, lam, weight_a=None, weight_b=None):
	if weight_a is not None and weight_b is not None:
		loss = lam * torch.mean(criterion(pred, y_a) * weight_a) + (1 - lam) * torch.mean(criterion(pred, y_b) * weight_b)
	else:
		loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
	return loss
