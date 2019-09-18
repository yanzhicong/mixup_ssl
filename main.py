import os
import sys
import re
import argparse
import shutil
import time
import pickle as pkl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


from mean_teacher import architectures, data, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

from utils import *
from label_propagation_utils import *
from criterion import *

from networks.wide_resnet import *
from networks.lenet import *

from plotter import *

parser = argparse.ArgumentParser(description='Deep Semi-supervised Learning with Mixup Regularization and Label Propagation')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
						choices=['cifar10', 'cifar100', 'svhn'],
						help='dataset: cifar10, cifar100 or svhn' )
parser.add_argument('--num_labeled', default=100, type=int, metavar='L',
					help='number of labeled samples per class')

parser.add_argument('--num_valid_samples', default=500, type=int, metavar='V',
					help='number of validation samples per class')

parser.add_argument('--arch', default='cnn13', type=str, help='either of cnn13, WRN28_2 , cifar_shakeshake26')
parser.add_argument('--dropout', default=0.0, type=float,
					metavar='DO', help='dropout rate')

parser.add_argument('--sl', action='store_true', default=False,
					help='only supervised learning: no use of unlabeled data')
parser.add_argument('--pseudo_label', choices=['single','mean_teacher'], default="mean_teacher",
						help='pseudo label generated from either a single model or mean teacher model')

parser.add_argument('--epochs', default=400, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch_size', default=100, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='max learning rate')
parser.add_argument('--initial_lr', default=0.0, type=float,
					metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr_rampup', default=0, type=int, metavar='EPOCHS',
					help='length of learning rate rampup in the beginning')
parser.add_argument('--lr_rampdown_epochs', default=450, type=int, metavar='EPOCHS',
					help='length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate \
					reaches to zero')

parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')


parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
						help='optimizer we are going to use. can be either adam of sgd')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--nesterov', action='store_true',
					help='use nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
					help='ema variable decay rate (default: 0.999)')

parser.add_argument('--loss_lambda', default=10.0, type=float,
					help='consistency coeff for mixup usup loss')


parser.add_argument('--consistency_rampup_starts', default=0, type=int, metavar='EPOCHS',
					help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=100, type=int, metavar='EPOCHS',
					help='lepoch at which consistency loss ramp-up ends')

parser.add_argument('--mixup_sup_alpha', default=0.2, type=float,
					help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
parser.add_argument('--mixup_usup_alpha', default=0.2, type=float,
					help='for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')


parser.add_argument('--lp_alpha', default=0.99, type=float,
					help='for label propagation, the alpha paramter that controls the propagation range')
parser.add_argument('--lp_n', default=10, type=int,
					help='for nearest neighbor graph construction in label propagation, ')


parser.add_argument('--mixup_hidden', action='store_true', default=False,
					help='apply mixup in hidden layers')
parser.add_argument('--num_mix_layer`', default=3, type=int,
					help='number of layers on which mixup is applied including input layer')
parser.add_argument('--checkpoint_epochs', default=50, type=int,
					metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
parser.add_argument('--evaluation_epochs', default=1, type=int,
					metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')

parser.add_argument('--print_freq', '-p', default=100, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
					help='evaluate model on evaluation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--root_dir', type = str, default = 'experiments',
						help='folder where results are to be stored')
parser.add_argument('--data_dir', type = str, default = 'data/cifar10/',
						help='folder where data is stored')
parser.add_argument('--n_cpus', default=0, type=int,
					help='number of cpus for data loading')

parser.add_argument('--job_id', type=str, default='lp_test_mm')


args = parser.parse_args()
print(args)
use_cuda = torch.cuda.is_available()


best_prec1 = 0
global_step = 0

##get number of updates etc#####

if args.dataset in ['cifar10', 'cifar100']:
	len_data = args.num_labeled
	num_updates = int((50000/args.batch_size))*args.epochs 
elif args.dataset == 'svhn':
	len_data = args.num_labeled
	num_updates = int((73250/args.batch_size)+1)*args.epochs 
print ('number of updates', num_updates)


dataset = SSLDataset(1, args.batch_size, args.n_cpus, args.dataset, args.data_dir,
		labels_per_class = args.num_labeled,
		valid_labels_per_class = args.num_valid_samples)


if args.dataset == 'cifar10':
	zca_components = np.load(args.data_dir +'zca_components.npy')
	zca_mean = np.load(args.data_dir +'zca_mean.npy')


### get net####
def getNetwork(args, num_classes, ema= False):
	
	if args.arch in ['cnn13','WRN28_2']:
		net = eval(args.arch)(num_classes, args.dropout)
	elif args.arch in ['cifar_shakeshake26']:
		model_factory = architectures.__dict__[args.arch]
		model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
		net = model_factory(**model_params)
	else:
		print('Error : Network should be either [cnn13/ WRN28_2 / cifar_shakeshake26')
		sys.exit(0)
	
	if ema:
		for param in net.parameters():
			param.detach_()

	return net


def experiment_name(args):
	if args.sl:
		exp_name = 'SL_'
	else:
		exp_name = 'SSL_'
	exp_name += str(args.dataset)
	exp_name += '_l' + str(args.num_labeled)
	exp_name += '_u' + str(args.num_valid_samples)
	
	exp_name += '_arch'+ str(args.arch)
	
	exp_name += '_lr'+str(args.lr)
	exp_name += '_init_lr'+ str(args.initial_lr)
	exp_name += '_rup'+ str(args.lr_rampup)
	exp_name += '_rdn'+ str(args.lr_rampdown_epochs)
	
	exp_name += '_emad'+ str(args.ema_decay)
	exp_name += '_mcons'+ str(args.loss_lambda)
	exp_name += '_ramp'+ str(args.consistency_rampup_starts)
	exp_name += '_'+ str(args.consistency_rampup_ends)

	exp_name += '_lpa'+str(args.lp_alpha)
	exp_name += '_lpn'+str(args.lp_n)

	exp_name += '_l2'+str(args.weight_decay)
	exp_name += '_eph'+str(args.epochs)
	exp_name += '_bs'+str(args.batch_size)
	
	if args.mixup_sup_alpha:
		exp_name += '_msa'+str(args.mixup_sup_alpha)
	if args.mixup_usup_alpha:
		exp_name += '_m_usup_a'+str(args.mixup_usup_alpha)
	if args.mixup_hidden:
		exp_name += 'm_hidden'
		exp_name += str(args.num_mix_layer)

	# exp_name += '_pl_'+str(args.pseudo_label)

	if args.job_id is not None:
		exp_name += '_id_'+str(args.job_id)

	exp_name = os.path.basename(str(__file__)).split('.')[0] + '/' + exp_name

	# exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
	print('experiement name: ' + exp_name)
	return exp_name






def main():
	global global_step
	global best_prec1
	global best_test_ema_prec1
	
	print('| Building net type [' + args.arch + ']...')
	model = getNetwork(args, dataset.num_classes)
	ema_model = getNetwork(args, dataset.num_classes, ema=True)

	if use_cuda:
		model.cuda()
		ema_model.cuda()
		cudnn.benchmark = True

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay,
								nesterov=args.nesterov)

	exp_name = experiment_name(args)

	exp_dir = os.path.join(args.root_dir, exp_name)
	print (exp_dir)
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	
	result_path = os.path.join(exp_dir , 'out.txt')
	filep = open(result_path, 'w')
	out_str = str(args)
	filep.write(out_str + '\n')

	indices_pkl_filepath = os.path.join(exp_dir, 'indices.pkl')
	pkl.dump([dataset.indices_train, dataset.indices_valid, dataset.indices_unlabelled], open(indices_pkl_filepath, 'wb'))
	
	plotter = Plotter()

	if args.evaluate:
		print("Evaluating the primary model:")
		validate(dataset, model, global_step, args.start_epoch, filep, plotter)
		print("Evaluating the EMA model:")
		validate(dataset, ema_model, global_step, args.start_epoch, filep, plotter)
		return

	for epoch in range(args.start_epoch, args.epochs):
		start_time = time.time()

		if epoch < 20:
			lp = True
		elif epoch < 50:
			lp = epoch % 3 == 0
		elif epoch < 90:
			lp = epoch % 6 == 0
		else:
			lp = epoch % 10 == 0

		train_ssl(dataset, model, ema_model, optimizer, epoch, filep, exp_dir, plotter, ssl=not args.sl, lp=lp)

		print_and_write("--- training epoch in %s seconds ---\n" % (time.time() - start_time), filep)

		if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
			start_time = time.time()

			print_and_write("Evaluating the primary model on validation set:", filep)
			prec1 = validate(dataset, model, global_step, epoch + 1, filep, plotter)

			print_and_write("Evaluating the EMA model on validation set:", filep)
			ema_prec1 = validate(dataset, ema_model, global_step, epoch + 1, filep, plotter, ema=True)

			print_and_write("--- validation in %s seconds ---\n" % (time.time() - start_time), filep)

			# if args.pseudo_label == 'single':
			# 	is_best = prec1 > best_prec1
			# 	best_prec1 = max(prec1, best_prec1)
			# else:
			is_best = ema_prec1 > best_prec1
			best_prec1 = max(ema_prec1, best_prec1)

			if is_best:
				start_time = time.time()
				print_and_write("Evaluating the primary model on test set:", filep)
				best_test_prec1 = validate(dataset, model, global_step, epoch + 1, filep, plotter, testing=True)
				print_and_write("Evaluating the EMA model on test set:", filep)
				best_test_ema_prec1 = validate(dataset, ema_model, global_step, epoch + 1, filep, plotter, ema=True, testing=True)
				print_and_write("--- testing in %s seconds ---\n" % (time.time() - start_time), filep)

		else:
			is_best = False
		
		print_and_write("Test error on the model with best validation error %s\n" % (best_test_prec1), filep)
		print_and_write("Test error on the model with best validation error %s\n" % (best_test_ema_prec1), filep)
		
		if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
			save_checkpoint({
				'epoch': epoch + 1,
				'global_step': global_step,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'ema_state_dict': ema_model.state_dict(),
				'best_prec1': best_prec1,
				'optimizer' : optimizer.state_dict(),
			}, is_best, exp_dir, epoch + 1) 

		filep.flush()

		plotter.to_csv(os.path.join(exp_dir, 'plot_output'))
		plotter.to_html_report(os.path.join(exp_dir, 'plot_output', 'report.html'))



def train_ssl(dataset, model, ema_model, optimizer, epoch, filep, exp_dir, plotter, ssl=True, lp=True):
	global global_step
	
	class_criterion = nn.CrossEntropyLoss().cuda()
	class_criterion2 = nn.CrossEntropyLoss(reduction="none").cuda()

	meters = AverageMeterSet()

	if epoch == 0 or epoch < args.consistency_rampup_starts:
		# in the start, set the pesudo label and weight all to zero
		dataset.set_pesudo_label(np.zeros([dataset.total_size, 4]))

	elif (ssl and lp) or epoch == args.consistency_rampup_starts:
		# extract features from model and ema_model
		model.eval()
		ema_model.eval()

		print("start cal features")
		feature_list = []
		ema_feature_list = []
		for input, _ in dataset.iter_data_train():
			if args.dataset == 'cifar10':
				input = apply_zca(input, zca_mean, zca_components)
			input_var = Variable(to_cuda(input))
			feature = model(input_var, ext_feature=True).detach().cpu().numpy()
			ema_feature = ema_model(input_var, ext_feature=True).detach().cpu().numpy()
			feature_list.append(feature)
			ema_feature_list.append(ema_feature)

		feature_list = np.concatenate(feature_list, axis=0)
		ema_feature_list = np.concatenate(ema_feature_list, axis=0)

		if epoch == 1 or epoch % 20 == 0:
			pkl.dump((feature_list, ema_feature_list), open(os.path.join(exp_dir, 'feature_list_%d.pkl'%epoch), 'wb'))

		# perform label propagation, and eval pesudo label accuracy
		Y, W = graph_laplace(feature_list, dataset, n=args.lp_n, alpha=args.lp_alpha)
		Y2, W2 = graph_laplace(ema_feature_list, dataset, n=args.lp_n, alpha=args.lp_alpha)

		acc1, wacc1 = eval_pesudo_label_acc(Y.astype(np.int32), W, dataset.label_all, filep)
		acc2, wacc2 = eval_pesudo_label_acc(Y2.astype(np.int32), W2, dataset.label_all, filep)

		plotter.scalar('plabel_acc', epoch, acc1)
		plotter.scalar('plabel_weighted_acc', epoch, wacc1)
		plotter.scalar('plabel_acc_ema', epoch, acc2)
		plotter.scalar('plabel_weighted_acc_ema', epoch, wacc2)

		# set pesudo label
		dataset.set_pesudo_label(np.concatenate([
				Y[:, np.newaxis], 
				W[:, np.newaxis],
				Y2[:, np.newaxis], 
				W2[:, np.newaxis]
				], axis=1))

	# switch to train mode
	model.train()
	ema_model.train()

	end = time.time()
	i = -1

	for (input, target), (uinput, utarget) in dataset.iter_ssl():
		# measure data loading time
		i = i+1
		meters.update('data_time', time.time() - end)
		
		if input.shape[0]!= uinput.shape[0]:
			bt_size = np.minimum(input.shape[0], uinput.shape[0])
			input = input[0:bt_size]
			target = target[0:bt_size]
			uinput = uinput[0:bt_size]
			utarget = utarget[0:bt_size]

		target = target[:, 0].long()


		if args.pseudo_label == 'single':
			# use label propagation by features generated from the model
			pesudo_label = utarget[:, 1:3]
		else:
			# use label propagation by features generated from the ema_model
			pesudo_label = utarget[:, 3:5]


		if args.dataset == 'cifar10':
			input = apply_zca(input, zca_mean, zca_components)
			uinput = apply_zca(uinput, zca_mean, zca_components) 

		lr = adjust_learning_rate(optimizer, epoch, i, dataset.len_ssl)
		meters.update('lr', optimizer.param_groups[0]['lr'])
		
		# calculate supervised loss
		if args.mixup_sup_alpha >= 1e-5:
			
			if use_cuda:
				input , target = input.cuda(), target.cuda()
			input_var, target_var = Variable(input), Variable(target)
			
			if args.mixup_hidden:
				# if use Manifold mixup
				output_mixed_l, target_a_var, target_b_var, lam = model(input_var, target_var, mixup_hidden=True,  mixup_alpha=args.mixup_sup_alpha, layers_mix=args.num_mix_layer)
			else:
				mixed_input, target_a, target_b, lam = mixup_data_su(input, target, args.mixup_sup_alpha)
				mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
				output_mixed_l = model(mixed_input_var)

			su_loss = mixup_criterion_su(class_criterion, output_mixed_l, target_a_var, target_b_var, lam)

		else:
			input_var = torch.autograd.Variable(input.cuda())
			target_var = torch.autograd.Variable(target.cuda())
			output = model(input_var)

			su_loss = class_criterion(output, target_var)

		meters.update('su_loss', su_loss.item())

		
	
		uinput_var = torch.autograd.Variable(uinput.cuda())


		### get ema loss. We use the actual samples(not the mixed up samples ) for calculating EMA loss

		minibatch_size = len(target_var)

		if args.mixup_sup_alpha:
			class_logit = model(input_var)
		else:
			class_logit = output

		ema_class_logit = ema_model(input_var)
		ema_class_loss = class_criterion(ema_class_logit, target_var)# / minibatch_size
		

		# calculate unsupervised loss
		if args.loss_lambda and ssl:
			if args.mixup_hidden:
				output_mixed_u, plabel_a, plabel_b, lam = model(uinput_var, pesudo_label, mixup_hidden=True,  
										mixup_alpha = args.mixup_sup_alpha, layers_mix=args.num_mix_layer)
			else:
				mixedup_x, plabel_a, plabel_b, lam = mixup_data(uinput_var, pesudo_label, args.mixup_usup_alpha)
				output_mixed_u = model(mixedup_x)

			target_a_var = Variable(to_cuda(plabel_a[:, 0].long()))
			weight_a_var = Variable(to_cuda(plabel_a[:, 1].float()))
			target_b_var = Variable(to_cuda(plabel_b[:, 0].long()))
			weight_b_var = Variable(to_cuda(plabel_b[:, 1].float()))

			unsu_loss = mixup_criterion_unsu(class_criterion2, output_mixed_u, target_a_var, target_b_var, lam, weight_a_var, weight_b_var)
			meters.update('unsu_loss', unsu_loss.item())

			if epoch < args.consistency_rampup_starts:
				unsupervised_loss_weight = 0.0
			else:
				unsupervised_loss_weight = get_current_consistency_weight(args.loss_lambda, epoch, i, dataset.len_ssl)

			meters.update('unsu_loss_weight', unsupervised_loss_weight)
			unsu_loss = unsupervised_loss_weight * unsu_loss

		else:
			unsu_loss = 0
			meters.update('unsu_loss', 0)

		if ssl:
			loss = su_loss + unsu_loss
		else:
			loss = su_loss



		meters.update('loss', loss.item())


		prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
		meters.update('top1', prec1, minibatch_size)
		meters.update('error1', 100. - prec1, minibatch_size)
		meters.update('top5', prec5, minibatch_size)
		meters.update('error5', 100. - prec5, minibatch_size)

		ema_prec1, ema_prec5 = accuracy(ema_class_logit.data, target_var.data, topk=(1, 5))
		meters.update('ema_top1', ema_prec1, minibatch_size)
		meters.update('ema_error1', 100. - ema_prec1, minibatch_size)
		meters.update('ema_top5', ema_prec5, minibatch_size)
		meters.update('ema_error5', 100. - ema_prec5, minibatch_size)

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		global_step += 1
		update_ema_variables(model, ema_model, args.ema_decay, global_step)

		# measure elapsed time
		meters.update('batch_time', time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print_and_write(
				'Epoch: [{0}][{1}/{2}]\t'
				'Time {meters[batch_time]:.3f}\t'
				'Data {meters[data_time]:.3f}\t'
				'Class {meters[su_loss]:.4f}\t'
				# 'Class2 {meters[class_loss2]:.4f}\t'
				'Mixup Cons {meters[unsu_loss]:.4f}\t'
				'Prec@1 {meters[top1]:.3f}\t'
				'Prec@5 {meters[top5]:.3f}'.format(
					epoch, i, dataset.len_ssl, meters=meters), filep)


	plotter.dist2('train_su_loss', epoch, meters['su_loss'].record)
	plotter.dist2('train_unsu_loss', epoch, meters['unsu_loss'].record)
	plotter.dist2('train_loss', epoch, meters['loss'].record)

	plotter.scalar('train_error_top1', epoch, meters['error1'].avg)
	plotter.scalar('train_error_top5', epoch, meters['error5'].avg)
	plotter.scalar('train_ema_error_top1', epoch, meters['ema_error1'].avg)
	plotter.scalar('train_ema_error_top5', epoch, meters['ema_error5'].avg)

	plotter.dist2('learning_rate', epoch, meters['lr'].record)
	plotter.dist2('unsu_loss_weight', epoch, meters['unsu_loss_weight'].record)



def validate(dataset, model, global_step, epoch, filep, plotter, ema=False, testing = False):
	class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
	meters = AverageMeterSet()

	# switch to evaluate mode
	model.eval()

	if testing:
		iterator = dataset.iter_data_test()
	else:
		iterator = dataset.iter_data_val()

	end = time.time()
	for i, (input, target) in enumerate(iterator):
		meters.update('data_time', time.time() - end)
	
		if args.dataset == 'cifar10':
			input = apply_zca(input, zca_mean, zca_components)
			
		with torch.no_grad():        
			input_var = torch.autograd.Variable(input.cuda())
		with torch.no_grad():
			if len(target.size()) > 1:
				target = target[:, 0]
			# target_var = torch.autograd.Variable(target.cuda(async=True))
			target_var = torch.autograd.Variable(target.long().cuda())

		minibatch_size = len(target_var)
		labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
		assert labeled_minibatch_size > 0
		meters.update('labeled_minibatch_size', labeled_minibatch_size)

		# compute output
		output1 = model(input_var)
		softmax1 = F.softmax(output1, dim=1)
		class_loss = class_criterion(output1, target_var) / minibatch_size

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
		meters.update('class_loss', class_loss.item(), minibatch_size)
		meters.update('top1', prec1, minibatch_size)
		meters.update('error1', 100.0 - prec1, minibatch_size)
		meters.update('top5', prec5, minibatch_size)
		meters.update('error5', 100.0 - prec5, minibatch_size)

		# measure elapsed time
		meters.update('batch_time', time.time() - end)
		end = time.time()
		
	print_and_write(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
		  .format(top1=meters['top1'], top5=meters['top5']), filep)
	
	if testing == False:
		# pass
		if ema:
			plotter.scalar('test_ema_class_loss', epoch, meters['class_loss'].avg)
			plotter.scalar('test_ema_error_top1', epoch, meters['error1'].avg)
		else:
			plotter.scalar('test_class_loss', epoch, meters['class_loss'].avg)
			plotter.scalar('test_error_top1', epoch, meters['error1'].avg)

	
	return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
	filename = 'checkpoint.{}.ckpt'.format(epoch)
	checkpoint_path = os.path.join(dirpath, filename)
	best_path = os.path.join(dirpath, 'best.ckpt')
	torch.save(state, checkpoint_path)
	print("--- checkpoint saved to %s ---" % checkpoint_path)
	if is_best:
		shutil.copyfile(checkpoint_path, best_path)
		print
		("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
	lr = args.lr
	epoch = epoch + step_in_epoch / total_steps_in_epoch

	# LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
	lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

	# Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
	if args.lr_rampdown_epochs:
		assert args.lr_rampdown_epochs >= args.epochs
		lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		
	return lr


def adjust_learning_rate_step(optimizer, epoch, gammas, schedule):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr
	assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
	for (gamma, step) in zip(gammas, schedule):
		if (epoch >= step):
			lr = lr * gamma
		else:
			break
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr


def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch, total_steps_in_epoch):
	# Consistency ramp-up from https://arxiv.org/abs/1610.02242
	epoch = epoch - args.consistency_rampup_starts
	epoch = epoch + step_in_epoch / total_steps_in_epoch
	return final_consistency_weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )

if __name__ == '__main__':
	main()

	
