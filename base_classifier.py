import os
import sys
import numpy as np
import pickle as pkl
import torch
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *



# class BaseClassifier(object):
# 	def __init__(self, args, dataset):

# 		# self.ema_model = getNetwork(args, dataset.num_classes, ema=True)

# 		self.build_model(args, dataset)

# 		if args.optimizer == "sgd":
# 			self.optimizer = torch.optim.SGD(self.model.parameters(), args.lr,
# 										momentum=args.momentum,
# 										weight_decay=args.weight_decay,
# 										nesterov=args.nesterov)
# 		elif args.optimizer == "adam":
# 			self.optimizer = torch.optim.Adam(self.model.parameters(), args.lr, weight_decay=args.weight_decay)

# 		self.exp_name = experiment_name(args)
# 		self.exp_dir = os.path.join(args.root_dir, self.exp_name)
# 		print (self.exp_dir)
# 		if not os.path.exists(self.exp_dir):
# 			os.makedirs(self.exp_dir)

# 		self.result_path = os.path.join(self.exp_dir , 'out.txt')
# 		if os.path.exists(self.result_path):
# 			print("Result path is already exists!")
# 			if input() != "yes":
# 				exit(-1)

# 		self.filep = open(self.result_path, 'w')
		
# 		out_str = str(args)
# 		self.filep.write(out_str + '\n')
# 		self.global_step = 0

# 	def print_and_write(self, str):
# 		print(str)
# 		self.filep.write(str+'\n')

# 	def eval_pesudo_label_acc(self, plabel, weight, true_label):

# 		self.print_and_write("Pesudo Label Acc: %0.5f"%(
# 				float(np.sum((plabel == true_label).astype(np.int32))) 
# 						/ float(len(plabel)) * 100.0))
# 		self.print_and_write("Pesudo Label Weighted Acc: %0.5f"%(
# 				float(np.sum((plabel == true_label).astype(np.float32) * weight)) 
# 						/ float(np.sum(weight)) * 100.0))

# 		weight = np.sort(weight)[::-1]
# 		wi = [1, 1000, 2000, 4000, 8000, 16000, 32000]
# 		ind_str_list = [str(ind) for ind in wi]
# 		weight_str_list = ["%0.5f"%weight[ind] for ind in wi]

# 		self.print_and_write("Pesudo Label Weight(" + "/".join(ind_str_list) + "): " + "/".join(weight_str_list))

# 	def validate(self, dataset, global_step, epoch, ema = False, testing = False):
# 		class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
# 		meters = AverageMeterSet()

# 		# switch to evaluate mode
# 		if ema:
# 			self.ema_model.eval()
# 		else:
# 			self.model.eval()


# 		end = time.time()

# 		if testing:
# 			iterator = dataset.iter_data_test()
# 		else:
# 			iterator = dataset.iter_data_val()

# 		for i, (input, target) in enumerate(iterator):
# 			meters.update('data_time', time.time() - end)
		
# 			if args.dataset == 'cifar10':
# 				input = apply_zca(input, zca_mean, zca_components)
				
# 			with torch.no_grad():        
# 				input_var = torch.autograd.Variable(input.cuda())

# 			with torch.no_grad():
# 				if len(target.size()) > 1:
# 					target = target[:, 0]
# 				target_var = torch.autograd.Variable(target.long().cuda())

# 			minibatch_size = len(target_var)
# 			labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)    
# 			assert labeled_minibatch_size > 0
# 			meters.update('labeled_minibatch_size', labeled_minibatch_size)

# 			# compute output
# 			if ema:
# 				output1 = self.ema_model(input_var)
# 			else:
# 				output1 = self.model(input_var)

# 			softmax1 = F.softmax(output1, dim=1)
# 			class_loss = class_criterion(output1, target_var) / minibatch_size

# 			# measure accuracy and record loss
# 			prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
# 			meters.update('class_loss', class_loss.item(), minibatch_size)
# 			meters.update('top1', prec1[0], minibatch_size)
# 			meters.update('error1', 100.0 - prec1[0], minibatch_size)
# 			meters.update('top5', prec5[0], minibatch_size)
# 			meters.update('error5', 100.0 - prec5[0], minibatch_size)

# 			# measure elapsed time
# 			meters.update('batch_time', time.time() - end)
# 			end = time.time()
			
# 		self.print_and_write(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'
# 			.format(top1=meters['top1'], top5=meters['top5']))

# 		if testing == False:
# 			if ema:
# 				val_ema_class_loss_list.append(meters['class_loss'].avg)
# 				val_ema_error_list.append(meters['error1'].avg)
# 			else:
# 				val_class_loss_list.append(meters['class_loss'].avg)
# 				val_error_list.append(meters['error1'].avg)

# 		return meters['top1'].avg



