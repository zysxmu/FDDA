"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
import random
import pickle
import tqdm
__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G, train_loader, test_loader,
				 settings, logger, opt_type="SGD", optimizer_state=None, use_FDDA=False, batch_index=None,
				 model_name='resnet18',
				 D_BNSLoss_weight=0.1, C_BNSLoss_weight=0.01, FDDA_iter=1, BNLoss_weight=0.1):
		"""
		init trainer
		"""
		
		self.settings = settings
		
		self.model = utils.data_parallel(
			model, self.settings.nGPU, self.settings.GPU)
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		self.generator = utils.data_parallel(
			generator, self.settings.nGPU, self.settings.GPU)

		self.train_loader = train_loader
		self.test_loader = test_loader
		self.criterion = nn.CrossEntropyLoss().cuda()
		self.kdloss_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
		self.bce_logits = nn.BCEWithLogitsLoss().cuda()
		self.MSE_loss = nn.MSELoss().cuda()
		self.L1Loss = nn.L1Loss().cuda()
		self.lr_master_S = lr_master_S
		self.lr_master_G = lr_master_G
		self.opt_type = opt_type
		self.use_FDDA = use_FDDA
		self.D_BNSLoss_weight = D_BNSLoss_weight
		self.C_BNSLoss_weight = C_BNSLoss_weight
		self.batch_index = batch_index
		self.FDDA_iter = FDDA_iter
		self.model_name = model_name

		self.BNLoss_weight = BNLoss_weight

		self.logger = logger
		self.mean_list = []
		self.var_list = []
		self.teacher_running_mean = []
		self.teacher_running_var = []
		self.save_BN_mean = []
		self.save_BN_var = []


		self.fix_G = False
		self.use_range_limit = False
		self.cosine_epoch = 100

		self.logger.info('--------------')
		self.logger.info('BNLoss_weight is:' + str(self.BNLoss_weight))
		self.logger.info('--------------')

		if self.use_FDDA:
			self.logger.info('--------------')
			self.logger.info('Use use_FDDA!')
			self.logger.info('D_BNSLoss_weight is:' + str(self.D_BNSLoss_weight))
			self.logger.info('C_BNSLoss_weight is:' + str(self.C_BNSLoss_weight))
			self.logger.info('FDDA_iter is:' + str(self.FDDA_iter))

			self.true_mean = {}
			self.true_var = {}
			if self.settings.dataset in ["imagenet"]:
				# assert False, "unsupport data set: " + self.settings.dataset
				head = './save_ImageNet'
				if self.batch_index is None:
					batch_index = random.randint(0, 0)
				bias = 1
				if self.model_name == 'resnet18':
					BN_layer_num = 20
				elif self.model_name == 'mobilenet_w1':
					BN_layer_num = 27
				elif self.model_name == 'mobilenetv2_w1':
					BN_layer_num = 52
				elif self.model_name == 'regnetx_600m':
					BN_layer_num = 53
				else:
					assert False, "unsupport model: " + self.model_name
			else:
				assert False, "unsupport data set: " + self.settings.dataset

			self.start_layer = int((BN_layer_num + 1) / 2) - 2

			mean_pickle_path = '/' + self.model_name + "_mean_Categorical_bs_1.pickle"
			var_pickle_path = '/' + self.model_name + "_var_Categorical_bs_1.pickle"
			teacher_output_pickle_path = '/' + self.model_name + "_teacher_output_Categorical_1.pickle"

			#################
			self.teacher_output_Categorical = []
			self.teacher_output_Categorical_correct = set()
			with open(head + teacher_output_pickle_path, "rb") as fp:
				mydict = pickle.load(fp)
			for k in mydict:
				self.teacher_output_Categorical.append(mydict[k])
				if np.argmax(mydict[k].data.cpu().numpy(), axis=1) == k:
					self.teacher_output_Categorical_correct.add(k)
			self.teacher_output_Categorical = torch.cat(self.teacher_output_Categorical, dim=0)
			self.logger.info('--------------')
			self.logger.info(
				'len self.teacher_output_Categorical_correct: ' + str(len(self.teacher_output_Categorical_correct)))
			self.logger.info(
				'teacher_output_Categorical shape: ' + str(self.teacher_output_Categorical.shape))
			self.logger.info('--------------')
			#################

			self.logger.info("Use: " + head + mean_pickle_path)
			self.logger.info("Use: " + head + var_pickle_path)
			if self.batch_index is None:
				self.logger.info('re-random batch_index!')
			else:
				self.logger.info('batch_index have been set alreay!')
			self.logger.info('batch_index is:' + str(batch_index))
			self.logger.info('--------------')

			with open(head + mean_pickle_path, "rb") as fp:  # Pickling
				mydict = pickle.load(fp)
			for l in range(self.settings.nClasses):
				self.true_mean[l] = []
				for layer_index in range(BN_layer_num):
					BN_nums = mydict[l][batch_index + l * bias][layer_index]
					BN_nums = BN_nums.cuda()
					self.true_mean[l].append(BN_nums)

			with open(head + var_pickle_path, "rb") as fp:  # Pickling
				mydict = pickle.load(fp)
			for l in range(self.settings.nClasses):
				self.true_var[l] = []
				for layer_index in range(BN_layer_num):
					BN_nums = mydict[l][batch_index + l * bias][layer_index]
					BN_nums = BN_nums.cuda()
					self.true_var[l].append(BN_nums)

		if opt_type == "SGD":
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.settings.lr_S,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
			)
		elif opt_type == "RMSProp":
			self.optimizer_S = torch.optim.RMSprop(
				params=self.model.parameters(),
				lr=self.settings.lr,
				eps=1.0,
				weight_decay=self.settings.weightDecay,
				momentum=self.settings.momentum,
				alpha=self.settings.momentum
			)
		elif opt_type == "Adam":
			self.optimizer_S = torch.optim.Adam(
				params=self.model.parameters(),
				lr=self.settings.lr,
				eps=1e-5,
				weight_decay=self.settings.weightDecay
			)

		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)
		self.scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_S,
																	  T_max=self.cosine_epoch*200, eta_min=0.)

		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
											betas=(self.settings.b1, self.settings.b2))

	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_G = self.lr_master_G.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr_G
		return
	
	def loss_fn_kd(self, output, labels, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha

		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""

		criterion_d = nn.CrossEntropyLoss().cuda()
		kdloss = nn.KLDivLoss(reduction='batchmean').cuda()
		# kdloss = nn.KLDivLoss().cuda()
		alpha = self.settings.alpha
		T = self.settings.temperature
		a = F.log_softmax(output / T, dim=1)
		b = F.softmax(teacher_outputs / T, dim=1)
		c = (alpha * T * T)
		d = criterion_d(output, labels)
		KD_loss = kdloss(a, b) * c + d
		return KD_loss
	
	def forward(self, images, teacher_outputs, labels=None):
		"""
		forward propagation
		"""
		# forward and backward and optimize


		output = self.model(images)
		if labels is not None:
			loss = self.loss_fn_kd(output, labels, teacher_outputs)
			return output, loss
		else:
			return output, None
	
	def backward_G(self, loss_G):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		loss_G.backward()
		self.optimizer_G.step()

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss_S.backward()
		self.optimizer_S.step()

	def backward(self, loss):
		"""
		backward propagation
		"""
		self.optimizer_G.zero_grad()
		self.optimizer_S.zero_grad()
		loss.backward()
		self.optimizer_G.step()
		self.optimizer_S.step()

	def hook_fn_forward(self, module, input, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		# use biased var in train
		var = input.var([0, 2, 3], unbiased=False)

		self.mean_list.append(mean)
		self.var_list.append(var)
		self.teacher_running_mean.append(module.running_mean)
		self.teacher_running_var.append(module.running_var)

	def hook_fn_forward_saveBN(self,module, input, output):
		self.save_BN_mean.append(module.running_mean.cpu())
		self.save_BN_var.append(module.running_var.cpu())

	def cal_true_BNLoss(self):

		D_BNS_loss = torch.zeros(1).cuda()
		C_BNS_loss = torch.zeros(1).cuda()
		loss_one_hot_BNScenters = torch.zeros(1).cuda()


		import random

		l = random.randint(0, self.settings.nClasses - 1)
		#################
		if self.epoch > 4:
			while l not in self.teacher_output_Categorical_correct:
				l = random.randint(0, self.settings.nClasses-1)
		#################

		self.mean_list.clear()
		self.var_list.clear()

		z = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()
		labels = Variable(torch.randint(l, l + 1, (self.settings.batchSize,))).cuda()

		z = z.contiguous()
		labels = labels.contiguous()
		images = self.generator(z, labels)
		output_teacher_batch = self.model_teacher(images)

		if self.epoch <= 4:
			if l not in self.teacher_output_Categorical_correct:
				for num in range(len(self.mean_list)):
					D_BNS_loss += self.MSE_loss(self.mean_list[num], torch.randn(self.var_list[num].shape).cuda()) \
									 + self.MSE_loss(self.var_list[num], torch.randn(self.var_list[num].shape).cuda())
				D_BNS_loss = 2.0 * D_BNS_loss / len(self.mean_list)
			else:
				for num in range(self.start_layer, len(self.mean_list)):
					D_BNS_loss += self.MSE_loss(self.mean_list[num], torch.normal(mean=self.true_mean[l][num], std=0.5).cuda()) \
									 + self.MSE_loss(self.var_list[num], torch.normal(mean=self.true_var[l][num], std=1.0).cuda())

					C_BNS_loss += self.MSE_loss(self.mean_list[num], self.true_mean[l][num].cuda()) \
									 + self.MSE_loss(self.var_list[num], self.true_var[l][num].cuda())
				D_BNS_loss = D_BNS_loss / (len(self.mean_list) - self.start_layer)
				C_BNS_loss = C_BNS_loss / (len(self.mean_list) - self.start_layer)
		else:
			if l not in self.teacher_output_Categorical_correct:
				for num in range(self.start_layer, len(self.mean_list)):
					D_BNS_loss += self.MSE_loss(self.mean_list[num], torch.normal(mean=self.true_mean[l][num], std=0.5).cuda()) \
									 + self.MSE_loss(self.var_list[num], torch.normal(mean=self.true_var[l][num], std=1.0).cuda())

					C_BNS_loss += self.MSE_loss(self.mean_list[num], self.true_mean[l][num].cuda()) \
								  + self.MSE_loss(self.var_list[num], self.true_var[l][num].cuda())

				D_BNS_loss = D_BNS_loss / (len(self.mean_list) - self.start_layer)
				C_BNS_loss = C_BNS_loss / (len(self.mean_list) - self.start_layer)
			else:
				for num in range(self.start_layer, len(self.mean_list)):
					D_BNS_loss += self.MSE_loss(self.mean_list[num], torch.normal(mean=self.true_mean[l][num], std=0.5).cuda()) \
									 + self.MSE_loss(self.var_list[num], torch.normal(mean=self.true_var[l][num], std=1.0).cuda())

					C_BNS_loss += self.MSE_loss(self.mean_list[num], self.true_mean[l][num].cuda()) \
								  + self.MSE_loss(self.var_list[num], self.true_var[l][num].cuda())

				D_BNS_loss = D_BNS_loss / (len(self.mean_list) - self.start_layer)
				C_BNS_loss = C_BNS_loss / (len(self.mean_list) - self.start_layer)

		loss_one_hot_BNScenters += self.criterion(output_teacher_batch, labels)
		return D_BNS_loss, loss_one_hot_BNScenters, C_BNS_loss

	def train(self, epoch, true_data_loader=None):
		"""
		training
		"""

		self.epoch = epoch

		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		fp_acc = utils.AverageMeter()

		iters = 200
		self.update_lr(epoch)

		self.model.eval()
		self.model_teacher.eval()
		self.generator.train()
		
		start_time = time.time()
		end_time = start_time

		if epoch == 0:
			for m in self.model_teacher.modules():
				if isinstance(m, nn.BatchNorm2d):
					m.register_forward_hook(self.hook_fn_forward)

		if true_data_loader is not None:
			iterator = iter(true_data_loader)

		for i in range(iters):

			start_time = time.time()
			data_time = start_time - end_time

			if epoch >= self.settings.warmup_epochs:
				try:
					images, _, labels = next(iterator)
				except:
					self.logger.info('re-iterator of true_data_loader')
					iterator = iter(true_data_loader)
					images, _, labels = next(iterator)
				images, labels = images.cuda(), labels.cuda()

			z = Variable(torch.randn(self.settings.batchSize, self.settings.latent_dim)).cuda()
			G_labels = Variable(torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))).cuda()
			z = z.contiguous()
			G_labels = G_labels.contiguous()
			G_images = self.generator(z, G_labels)
		
			self.mean_list.clear()
			self.var_list.clear()
			G_output_teacher_batch = self.model_teacher(G_images)

			loss_one_hot = self.criterion(G_output_teacher_batch, G_labels)
			BNS_loss = torch.zeros(1).cuda()
			for num in range(len(self.mean_list)):
				BNS_loss += self.MSE_loss(self.mean_list[num], self.teacher_running_mean[num]) + self.MSE_loss(
					self.var_list[num], self.teacher_running_var[num])

			BNS_loss = BNS_loss / len(self.mean_list)
			BNS_loss = self.BNLoss_weight * BNS_loss

			if self.use_FDDA and i % self.FDDA_iter == 0:
				D_BNS_loss, loss_one_hot_BNScenters, C_BNS_loss = self.cal_true_BNLoss()
				D_BNS_loss = self.D_BNSLoss_weight * D_BNS_loss
				C_BNS_loss = self.C_BNSLoss_weight * C_BNS_loss

				loss_one_hot_BNScenters = 0.5 * loss_one_hot_BNScenters
				loss_one_hot = loss_one_hot * 0.5
				loss_G = loss_one_hot + BNS_loss + D_BNS_loss + loss_one_hot_BNScenters + C_BNS_loss
			else:
				loss_G = loss_one_hot + BNS_loss

			self.backward_G(loss_G)

			if epoch >= self.settings.warmup_epochs:
				self.mean_list.clear()
				self.var_list.clear()
				output_teacher_batch = self.model_teacher(images)

				output, loss_S = self.forward(torch.cat((images, G_images.detach())).detach(),
											  torch.cat((output_teacher_batch.detach(),
														 G_output_teacher_batch.detach())).detach(),
											  torch.cat((labels, G_labels.detach())).detach())
				self.backward_S(loss_S)
				self.scheduler_S.step()
			else:
				output, loss_S = self.forward(G_images.detach(), G_output_teacher_batch.detach(), G_labels.detach())

			end_time = time.time()
			
			gt = G_labels.data.cpu().numpy()
			d_acc = np.mean(np.argmax(G_output_teacher_batch.data.cpu().numpy(), axis=1) == gt)
			fp_acc.update(d_acc)

		if self.use_FDDA and i % self.FDDA_iter == 0:
			self.logger.info(
				"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f]"
				" [D_BNS_loss:%f] [loss_one_hot_BNScenters:%f] [C_BNS_loss:%f] [S loss: %f] "
				% (epoch + 1, self.settings.nEpochs, i+1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(),
				   BNS_loss.item(), D_BNS_loss.item(), loss_one_hot_BNScenters.item(), C_BNS_loss.item(), loss_S.item())
			)
		else:
			self.logger.info(
				"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [One-hot loss: %f] [BNS_loss:%f] [S loss: %f] "
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, 100 * fp_acc.avg, loss_G.item(), loss_one_hot.item(),
				   BNS_loss.item(), loss_S.item())
			)

		return 0, 0, 0

	def test(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()
		
		self.model.eval()
		self.model_teacher.eval()
		
		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):
				start_time = time.time()
				
				labels = labels.cuda()
				images = images.cuda()
				output = self.model(images)

				loss = torch.ones(1)
				self.mean_list.clear()
				self.var_list.clear()

				single_error, single_loss, single5_error = utils.compute_singlecrop(
					outputs=output, loss=loss,
					labels=labels, top5_flag=True, mean_flag=True)

				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))
				
				end_time = time.time()
		
		self.logger.info(
			"[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
		)
		return top1_error.avg, top1_loss.avg, top5_error.avg

	def test_teacher(self, epoch):
		"""
		testing
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		top5_error = utils.AverageMeter()

		self.model_teacher.eval()

		iters = len(self.test_loader)
		start_time = time.time()
		end_time = start_time

		with torch.no_grad():
			for i, (images, labels) in enumerate(self.test_loader):

				if i % 100 == 0:
					print(i)
				start_time = time.time()
				data_time = start_time - end_time

				labels = labels.cuda()
				if self.settings.tenCrop:
					image_size = images.size()
					images = images.view(
						image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
					images_tuple = images.split(image_size[0])
					output = None
					for img in images_tuple:
						if self.settings.nGPU == 1:
							img = img.cuda()
						img_var = Variable(img, volatile=True)
						temp_output, _ = self.forward(img_var)
						if output is None:
							output = temp_output.data
						else:
							output = torch.cat((output, temp_output.data))
					single_error, single_loss, single5_error = utils.compute_tencrop(
						outputs=output, labels=labels)
				else:
					if self.settings.nGPU == 1:
						images = images.cuda()

					output = self.model_teacher(images)

					loss = torch.ones(1)
					self.mean_list.clear()
					self.var_list.clear()

					single_error, single_loss, single5_error = utils.compute_singlecrop(
						outputs=output, loss=loss,
						labels=labels, top5_flag=True, mean_flag=True)
				#
				top1_error.update(single_error, images.size(0))
				top1_loss.update(single_loss, images.size(0))
				top5_error.update(single5_error, images.size(0))

				end_time = time.time()
				iter_time = end_time - start_time

		self.logger.info(
				"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
				% (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00 - top1_error.avg))
		)

		return top1_error.avg, top1_loss.avg, top5_error.avg
