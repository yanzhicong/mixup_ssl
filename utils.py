import os
import sys
import errno
import pickle as pkl
import numpy as np

import torch
from torch.autograd import Variable
from scipy import linalg


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from itertools import repeat, cycle



def to_var(x,requires_grad=True):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x,requires_grad=requires_grad)

def denorm(x):
	out = (x+1)/2
	return out.clamp(0,1)


def make_dir_if_not_exists(path):
	"""Make directory if doesn't already exists"""
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise 


def to_cuda(var):
	if torch.cuda.is_available():
		return var.cuda()
	else:
		return var


def denorm(x):
	out = (x+1)/2
	return out.clamp(0,1)



def to_one_hot(inp):
	y_onehot = torch.FloatTensor(inp.size(0), 10)
	y_onehot.zero_()

	y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
	
	return Variable(y_onehot.cuda(),requires_grad=False)


def mixup_data_su(x, y, alpha=1.0):
	'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1.
	batch_size = x.size()[0]
	index = np.random.permutation(batch_size)
	mixed_x = lam * x + (1 - lam) * x[index,:]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_data(x, y, alpha):
	'''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1.
	batch_size = x.size()[0]
	index = torch.randperm(batch_size).cuda()
	mixed_x = lam * x + (1 - lam) * x[index,:]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam



def print_and_write(content, filep):
	print(content)
	filep.write(content+'\n')



def eval_pesudo_label_acc(plabel, weight, true_label, filep):
	"""Computes the precision and weighted precision of pesudo-label"""

	acc = float(np.sum((plabel == true_label).astype(np.int32))) / float(len(plabel))
	weighted_acc = float(np.sum((plabel == true_label).astype(np.float32) * weight)) / float(np.sum(weight))

	print_and_write("Pesudo Label Acc: %0.5f"%(acc * 100.0), filep)
	print_and_write("Pesudo Label Weighted Acc: %0.5f"%(weighted_acc * 100.0), filep)

	weight = np.sort(weight)[::-1]
	wi = [1, 1000, 2000, 4000, 8000, 16000, 32000]
	ind_str_list = [str(ind) for ind in wi]
	weight_str_list = ["%0.5f"%weight[ind] for ind in wi]

	print_and_write("Pesudo Label Weight(" + "/".join(ind_str_list) + "): " + "/".join(weight_str_list), filep)

	return acc, weighted_acc





def update_ema_variables(model, ema_model, alpha, global_step):
	# Use the true average until the exponential average is more correct
	alpha = min(1 - 1 / (global_step + 1), alpha)
	for ema_param, param in zip(ema_model.parameters(), model.parameters()):
		ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
	


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	#labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8).type(torch.cuda.FloatTensor)
	minibatch_size = len(target)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / minibatch_size))
	return res




def load_data_subset(data_aug, batch_size, workers, dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
	## copied from GibbsNet_pytorch/load.py
	import numpy as np
	from functools import reduce
	from operator import __or__
	from torch.utils.data.sampler import SubsetRandomSampler
		
	if dataset == 'cifar10':
		mean = [x / 255 for x in [125.3, 123.0, 113.9]]
		std = [x / 255 for x in [63.0, 62.1, 66.7]]
	elif dataset == 'cifar100':
		mean = [x / 255 for x in [129.3, 124.1, 112.4]]
		std = [x / 255 for x in [68.2, 65.4, 70.4]]
	elif dataset == 'svhn':
		mean = [x / 255 for x in [127.5, 127.5, 127.5]]
		std = [x / 255 for x in [127.5, 127.5, 127.5]]
	elif dataset == 'mnist':
		pass 
	else:
		assert False, "Unknow dataset : {}".format(dataset)
	
	if data_aug==1:
		print ('data aug')
		if dataset == 'svhn':
			train_transform = transforms.Compose(
											[ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
											transforms.Normalize(mean, std)])
			test_transform = transforms.Compose(
											[transforms.ToTensor(), transforms.Normalize(mean, std)])
		elif dataset == 'mnist':
			hw_size = 24
			train_transform = transforms.Compose([
								transforms.RandomCrop(hw_size),                
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
						])
			test_transform = transforms.Compose([
								transforms.CenterCrop(hw_size),                       
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
						])
		else:    
			train_transform = transforms.Compose(
												[transforms.RandomHorizontalFlip(),
												transforms.RandomCrop(32, padding=2),
												transforms.ToTensor(),
												transforms.Normalize(mean, std)])
			test_transform = transforms.Compose(
												[transforms.ToTensor(), transforms.Normalize(mean, std)])
	else:
		print ('no data aug')
		if dataset == 'mnist':
			hw_size = 28
			train_transform = transforms.Compose([
								transforms.ToTensor(),       
								transforms.Normalize((0.1307,), (0.3081,))
						])
			test_transform = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
						])
				
		else:   
			train_transform = transforms.Compose(
												[transforms.ToTensor(),
												transforms.Normalize(mean, std)])
			test_transform = transforms.Compose(
												[transforms.ToTensor(), transforms.Normalize(mean, std)])
	if dataset == 'cifar10':
		train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
		test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
		num_classes = 10
	elif dataset == 'cifar100':
		train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
		test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
		num_classes = 100
	elif dataset == 'svhn':
		train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
		test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
		num_classes = 10
	elif dataset == 'mnist':
		train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
		test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
		num_classes = 10
	#print ('svhn', train_data.labels.shape)
	elif dataset == 'stl10':
		train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
		test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
		num_classes = 10
	elif dataset == 'imagenet':
		assert False, 'Do not finish imagenet code'
	else:
		assert False, 'Do not support dataset : {}'.format(dataset)

		
	n_labels = num_classes
	
	def get_sampler(labels, n=None, n_valid= None):
		# Only choose digits in n_labels
		# n = number of labels per class for training
		# n_val = number of lables per class for validation
		#print type(labels)
		#print (n_valid)

		(indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
		# Ensure uniform distribution of labels
		np.random.shuffle(indices)
		
		indices_valid = np.hstack([
					list(filter(lambda idx: labels[idx] == i, indices))[:n_valid]
					for i in range(n_labels)])
		indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
		indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:] for i in range(n_labels)])
		#print (indices_train.shape)
		#print (indices_valid.shape)
		#print (indices_unlabelled.shape)
		indices_train = torch.from_numpy(indices_train)
		indices_valid = torch.from_numpy(indices_valid)
		indices_unlabelled = torch.from_numpy(indices_unlabelled)
		sampler_train = SubsetRandomSampler(indices_train)
		sampler_valid = SubsetRandomSampler(indices_valid)
		sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
		return sampler_train, sampler_valid, sampler_unlabelled
	
	#print type(train_data.train_labels)
	
	# Dataloaders for MNIST
	if dataset == 'svhn':
		train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class)
	elif dataset == 'mnist':
		train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class)
	else: 
		# train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels, labels_per_class, valid_labels_per_class)
		train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.targets, labels_per_class, valid_labels_per_class)
	
	
	labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
												sampler=train_sampler,
												num_workers=workers, pin_memory=True)

	validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
												sampler=valid_sampler,
												num_workers=workers, pin_memory=True)

	unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
												sampler=unlabelled_sampler,
												num_workers=workers, pin_memory=True)

	test = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
												shuffle=False, 
												num_workers=workers, pin_memory=True)

	return labelled, validation, unlabelled, test, num_classes





class SSLDataset(object):
	
	def __init__(self, data_aug, batch_size, workers, dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
		## copied from GibbsNet_pytorch/load.py
		import numpy as np
		from functools import reduce
		from operator import __or__
		from torch.utils.data.sampler import SubsetRandomSampler
			
		if dataset == 'cifar10':
			mean = [x / 255 for x in [125.3, 123.0, 113.9]]
			std = [x / 255 for x in [63.0, 62.1, 66.7]]
		elif dataset == 'cifar100':
			mean = [x / 255 for x in [129.3, 124.1, 112.4]]
			std = [x / 255 for x in [68.2, 65.4, 70.4]]
		elif dataset == 'svhn':
			mean = [x / 255 for x in [127.5, 127.5, 127.5]]
			std = [x / 255 for x in [127.5, 127.5, 127.5]]
		elif dataset == 'mnist':
			pass 
		else:
			assert False, "Unknow dataset : {}".format(dataset)
		
		if data_aug==1:
			print ('data aug')
			if dataset == 'svhn':
				train_transform = transforms.Compose(
												[ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
												transforms.Normalize(mean, std)])
				test_transform = transforms.Compose(
												[transforms.ToTensor(), transforms.Normalize(mean, std)])
			elif dataset == 'mnist':
				hw_size = 24
				train_transform = transforms.Compose([
									transforms.RandomCrop(hw_size),                
									transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,))
							])
				test_transform = transforms.Compose([
									transforms.CenterCrop(hw_size),                       
									transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,))
							])
			else:    
				train_transform = transforms.Compose(
													[transforms.RandomHorizontalFlip(),
													transforms.RandomCrop(32, padding=2),
													transforms.ToTensor(),
													transforms.Normalize(mean, std)])
				test_transform = transforms.Compose(
													[transforms.ToTensor(), transforms.Normalize(mean, std)])
		else:
			print ('no data aug')
			if dataset == 'mnist':
				hw_size = 28
				train_transform = transforms.Compose([
									transforms.ToTensor(),       
									transforms.Normalize((0.1307,), (0.3081,))
							])
				test_transform = transforms.Compose([
									transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,))
							])
					
			else:   
				train_transform = transforms.Compose(
													[transforms.ToTensor(),
													transforms.Normalize(mean, std)])
				test_transform = transforms.Compose(
													[transforms.ToTensor(), transforms.Normalize(mean, std)])

		if dataset == 'cifar10':
			train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
			test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
			num_classes = 10
		elif dataset == 'cifar100':
			train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
			test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
			num_classes = 100
		elif dataset == 'svhn':
			train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
			test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
			num_classes = 10
		elif dataset == 'mnist':
			train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
			test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
			num_classes = 10
		#print ('svhn', train_data.labels.shape)
		elif dataset == 'stl10':
			train_data = datasets.STL10(data_target_dir, split='train', transform=train_transform, download=True)
			test_data = datasets.STL10(data_target_dir, split='test', transform=test_transform, download=True)
			num_classes = 10
		elif dataset == 'imagenet':
			assert False, 'Do not finish imagenet code'
		else:
			assert False, 'Do not support dataset : {}'.format(dataset)

		n_labels = num_classes

		def get_indices(labels, n=None, n_valid=None):
			# Only choose digits in n_labels
			# n = number of labels per class for training
			# n_val = number of lables per class for validation
			# print(type(labels))
			# print(type(labels[0]))
			# print((n_valid))

			(indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
			# Ensure uniform distribution of labels
			np.random.shuffle(indices)
			
			indices_valid = np.hstack([
						list(filter(lambda idx: labels[idx] == i, indices))[:n_valid]
							for i in range(n_labels)])
			indices_train = np.hstack([
						list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n]
							for i in range(n_labels)])
			indices_unlabelled = np.hstack([
						list(filter(lambda idx: labels[idx] == i, indices))[n_valid:]
							for i in range(n_labels)])

			indices_train = indices_train
			indices_valid = indices_valid
			indices_unlabelled = indices_unlabelled

			return indices_train, indices_valid, indices_unlabelled

		# Dataloaders for MNIST
		if dataset == 'svhn':
			indices_train, indices_valid, indices_unlabelled = get_indices(train_data.labels, labels_per_class, valid_labels_per_class)
		elif dataset == 'mnist':
			indices_train, indices_valid, indices_unlabelled = get_indices(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class)
		else: 
			# indices_train, indices_valid, indices_unlabelled = get_indices(train_data.train_labels, labels_per_class, valid_labels_per_class)
			indices_train, indices_valid, indices_unlabelled = get_indices(train_data.targets, labels_per_class, valid_labels_per_class)


		self.indices_train = indices_train
		self.indices_valid = indices_valid
		self.indices_unlabelled = indices_unlabelled


		# print(train_data.targets[0:100])
		# print(len(self.indices_train))
		# print(len(self.indices_valid))
		# print(len(self.indices_unlabelled))

		train_sampler = SubsetRandomSampler(torch.from_numpy(indices_train))
		valid_sampler = SubsetRandomSampler(torch.from_numpy(indices_valid))
		unlabelled_sampler = SubsetRandomSampler(torch.from_numpy(indices_unlabelled))
		
		self.batch_size = batch_size
		self.train_data = train_data
		self.test_data = test_data

		labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
													sampler=train_sampler,
													num_workers=workers, pin_memory=True)

		validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
													sampler=valid_sampler,
													num_workers=workers, pin_memory=True)

		unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
													sampler=unlabelled_sampler,
													num_workers=workers, pin_memory=True)

		train = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
													shuffle=False,
													num_workers=workers, pin_memory=True)

		test = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
													shuffle=False, 
													num_workers=workers, pin_memory=True)

		# return labelled, validation, unlabelled, test, num_classes

		self.labelled = labelled
		self.validation = validation
		self.unlabelled = unlabelled
		self.train = train
		self.test = test
		self.num_classes = num_classes


	@property
	def total_size(self):
		return len(self.train_data.targets)

	@property
	def label(self):
		targets = self.label_all
		targets = targets[self.indices_train]
		return targets.astype(np.int32), self.indices_train.astype(np.int32), self.total_size, self.num_classes

	@property
	def label_all(self):
		targets = np.array(self.train_data.targets)
		if len(targets.shape) > 1:
			targets = targets[:, 0]
		return targets.astype(np.int32)

	def iter_data_val(self):
		return self.validation

	def iter_data_train(self):
		return self.train

	def iter_data_test(self):
		return self.test

	@property
	def len_sl(self):
		return len(self.labelled)

	def iter_sl(self):
		return self.labelled

	@property
	def len_ssl(self):
		return len(self.unlabelled)

	def iter_ssl(self):
		return zip(cycle(self.labelled), self.unlabelled)

	def set_pesudo_label(self, pesudo_label):

		targets = np.array(self.train_data.targets)
		print(targets.shape)
		if len(targets.shape) > 1:
			# clear old pesudo label
			targets = targets[:, 0]

		targets = targets.reshape([targets.shape[0], 1])
		assert pesudo_label.shape[0] == targets.shape[0]
		self.train_data.targets = np.concatenate([targets, pesudo_label], axis=1)



if __name__ == '__main__':
	# labelled, validation, unlabelled, test, num_classes = load_data_subset(data_aug=1, 
	#                 batch_size=32, workers=1, dataset='cifar10', 
	#                 data_target_dir="./data/cifar10", labels_per_class=100, valid_labels_per_class = 500)
	# for (inputs, targets), (u, t) in zip(cycle(labelled), unlabelled):
	#     print(inputs.shape, targets.shape, u.shape, t.shape)
	#     print(inputs.type(), targets.type(), u.type(), t.type())

	dataset = SSLDataset(data_aug=1, 
					batch_size=32,workers=1,dataset='cifar100', 
					data_target_dir="./data/cifar100", labels_per_class=100, valid_labels_per_class=500)

	pkl.dump(dataset.label_all, open('cifar100_label_all.pkl', 'wb'))

	# print(len(dataset.iter_data_train()))

	# dataset.set_pesudo_label(np.zeros([60000, 20]))
	
	# for (inputs, targets) in dataset.iter_data_train():
	#     print(inputs.shape, targets.shape)

	# dataset.set_pesudo_label(np.zeros([60000, 23]))

	# for (inputs, targets) in dataset.iter_data_train():
	#     print(inputs.shape, targets.shape)


	# for (inputs, targets), (u, t) in dataset.iter_ssl():
	#     print(inputs.shape, targets.shape, u.shape, t.shape)
	#     print(inputs.type(), targets.type(), u.type(), t.type())

