import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import mixup_data


## code for CNN13 from https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/architectures.py
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls, BasicBlock, ResNet




# class BasicBlockWrapper(BasicBlock):

# 	def forward(self, x):
# 		identity = x

# 		out = self.conv1(x)
# 		out = self.bn1(out)
# 		out = self.relu(out)

# 		out = self.conv2(out)
# 		out = self.bn2(out)

# 		if self.downsample is not None:
# 			identity = self.downsample(x)

# 		out += identity
# 		out = self.relu(out)

# 		return out



class ResNetWrapper(ResNet):

	def __init__(self, block, layers, num_classes=100, zero_init_residual=False, dropout_rate=0.5):
		super(ResNetWrapper, self).__init__(block, layers, num_classes=128, zero_init_residual=zero_init_residual)
		
		# self.dropout = nn.Dropout(dropout_rate)
		self.activation = nn.LeakyReLU(0.1)
		self.fc2 = weight_norm(nn.Linear(128, num_classes))


	def forward(self, x, target=None, mixup_hidden=False,  mixup_alpha=0.1, layers_mix=None, ext_feature=False):

		if mixup_hidden == True:
			layer_mix = random.randint(0,layers_mix)

			out = x
			if layer_mix == 0:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.conv1(out)
			out = self.bn1(out)
			out = self.relu(out)
			out = self.maxpool(out)
			out = self.layer1(out)

			if layer_mix == 1:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.layer2(out)

			if layer_mix == 2:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.layer3(out)
			if layer_mix == 3:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.layer4(out)
			if layer_mix == 4:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.avgpool(out)
			out = out.view(out.size(0), -1)
			out = self.fc(out)

			if ext_feature:
				return out, y_a, y_b, lam
			else:
				# out = self.dropout(out)
				out = self.activation(out)
				out = self.fc2(out)
				return out, y_a, y_b, lam

		else:

			out = x

			out = self.conv1(out)
			out = self.bn1(out)
			out = self.relu(out)
			out = self.maxpool(out)
			out = self.layer1(out)
			out = self.layer2(out)
			out = self.layer3(out)
			out = self.layer4(out)
			out = self.avgpool(out)
			out = out.view(out.size(0), -1)
			out = self.fc(out)
			
			if ext_feature:
				return out
			else:
				# out = self.dropout(out)
				out = self.activation(out)
				out = self.fc2(out)
				return out
				

class ResNetWrapper2(ResNet):

	def __init__(self, block, layers, num_classes=100, zero_init_residual=False, dropout_rate=0.5):
		super(ResNetWrapper2, self).__init__(block, layers, num_classes=num_classes, zero_init_residual=zero_init_residual)


	def forward(self, x, target=None, mixup_hidden=False,  mixup_alpha=0.1, layers_mix=None, ext_feature=False):

		if mixup_hidden == True:
			layer_mix = random.randint(0,layers_mix)

			out = x
			if layer_mix == 0:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.conv1(out)
			out = self.bn1(out)
			out = self.relu(out)
			out = self.maxpool(out)
			out = self.layer1(out)

			if layer_mix == 1:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.layer2(out)

			if layer_mix == 2:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.layer3(out)
			if layer_mix == 3:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.layer4(out)
			if layer_mix == 4:
				out, y_a, y_b, lam = mixup_data(out, target, mixup_alpha)

			out = self.avgpool(out)
			out = out.view(out.size(0), -1)
	
			if ext_feature:
				return out, y_a, y_b, lam
			else:
				out = self.fc(out)
				return out, y_a, y_b, lam

		else:

			out = x

			out = self.conv1(out)
			out = self.bn1(out)
			out = self.relu(out)
			out = self.maxpool(out)
			out = self.layer1(out)
			out = self.layer2(out)
			out = self.layer3(out)
			out = self.layer4(out)
			out = self.avgpool(out)
			out = out.view(out.size(0), -1)
			
			if ext_feature:
				return out
			else:
				out = self.fc(out)
				return out
				

def resnet18(pretrained=True, num_classes=100, dropout_rate=0.0):
	"""Constructs a ResNet-18 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""

	# two fc layers
	# feature from first fc layer
	# and classification by second fc layer
	model = ResNetWrapper(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_rate=dropout_rate)

	# one fc layer
	# feature from last conv layer
	# and classification by fc layer
	# model = ResNetWrapper2(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_rate=dropout_rate)


	if pretrained:
		state_dict = model_zoo.load_url(model_urls['resnet18'])
		del state_dict['fc.weight']
		del state_dict['fc.bias']
		model.load_state_dict(state_dict, strict=False)
	return model

