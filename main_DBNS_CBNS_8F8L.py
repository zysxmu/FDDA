import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import hubconf
# option file should be modified according to your expriment
from options import Option

from dataloader import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from trainer_DBNS_CBNS import Trainer
from PIL import Image

import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dsets
import os
import shutil

class Generator_imagenet(nn.Module):
	def __init__(self, options=None, conf_path=None):
		self.settings = options or Option(conf_path)

		super(Generator_imagenet, self).__init__()

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img

class imagenet_dataset(Dataset):
	def __init__(self, path_label_Categorical, batch_index):

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		self.test_transform = transforms.Compose([
			transforms.Resize(256),
			# transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize
		])

		self.train_transform = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize, ])


		self.path_label = []
		for l in path_label_Categorical:
			self.path_label.append(path_label_Categorical[l][batch_index+l])

	def __getitem__(self, index):
		# print(self.path_label[index])
		path = self.path_label[index][0][0]
		label = self.path_label[index][1].item()

		with open(path, 'rb') as f:
			img = Image.open(f)
			img = img.convert('RGB')

		# img = self.test_transform(img)
		img = self.train_transform(img)
		return img, path, label

	def __len__(self):
		return len(self.path_label)


class ExperimentDesign:
	def __init__(self, generator=None, model_name=None, options=None, conf_path=None):
		self.settings = options or Option(conf_path)
		self.generator = generator
		self.train_loader = None
		self.test_loader = None
		self.model_name = model_name
		self.model = None
		self.model_teacher = None
		
		self.optimizer_state = None
		self.trainer = None

		self.unfreeze_Flag = True

		self.batch_index = None # for use true BNLoss

		self.true_data_loader = None
		
		os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
		
		self.settings.set_save_path()
		shutil.copyfile(conf_path, os.path.join(self.settings.save_path, conf_path))
		shutil.copyfile('./main_DBNS_CBNS.py', os.path.join(self.settings.save_path, 'main_DBNS_CBNS.py'))
		shutil.copyfile('./trainer_DBNS_CBNS.py', os.path.join(self.settings.save_path, 'trainer_DBNS_CBNS.py'))
		self.logger = self.set_logger()
		self.settings.paramscheck(self.logger)

		self.prepare()
	
	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.INFO)

		return logger

	def prepare(self):
		self._set_gpu()
		self._set_dataloader()
		self._true_data_loader()
		self._set_model()
		self._replace()
		self.logger.info(self.model)
		self._set_trainer()

	def _true_data_loader(self):
		import pickle
		import random
		if self.settings.dataset in ["imagenet"]:
			# assert False, "unsupport data set: " + self.settings.dataset
			head = './save_ImageNet'
			self.batch_index = random.randint(0, 0)
		else:
			assert False, "unsupport data set: " + self.settings.dataset

		path_label_pickle_path = '/' + self.model_name + "_path_label_Categorical_bs_1.pickle"

		self.logger.info('--------------')
		self.logger.info('Use true_data_loader!')
		self.logger.info("Use: " + head + path_label_pickle_path)
		self.logger.info('batch_index is:' + str(self.batch_index))
		self.logger.info('--------------')

		self.paths = {}
		with open(head + path_label_pickle_path, "rb") as fp:  # Pickling
			mydict = pickle.load(fp)

		if self.settings.dataset in ["imagenet"]:
			dataset = imagenet_dataset(mydict, self.batch_index)
		true_data_loader = torch.utils.data.DataLoader(dataset,
													   batch_size=min(self.settings.batchSize, len(dataset)),
													   shuffle=True,
													   num_workers=0,
													   pin_memory=True,
													   drop_last=True)

		self.logger.info('len(true_data_loader) is: ' + str(len(true_data_loader)))
		self.logger.info('len(dataset) is: ' + str(len(dataset)))
		self.true_data_loader = true_data_loader

	def _set_gpu(self):
		self.logger.info('settings.manualSeed is:' + str(self.settings.manualSeed))
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		data_loader = DataLoader(dataset=self.settings.dataset,
		                         batch_size=self.settings.batchSize,
		                         data_path=self.settings.dataPath,
		                         n_threads=self.settings.nThreads,
		                         ten_crop=self.settings.tenCrop,
		                         logger=self.logger)
		
		self.train_loader, self.test_loader = data_loader.getloader()

	def _set_model(self):
		if self.settings.dataset in ["imagenet"]:

			if self.model_name == 'resnet18':
				self.model_teacher = ptcv_get_model('resnet18', pretrained=True)
				self.model = ptcv_get_model('resnet18', pretrained=True)
			elif self.model_name == 'mobilenet_w1':
				self.model_teacher = ptcv_get_model('mobilenet_w1', pretrained=True)
				self.model = ptcv_get_model('mobilenet_w1', pretrained=True)
			elif self.model_name == 'mobilenetv2_w1':
				self.model_teacher = eval('hubconf.{}(pretrained=True)'.format('mobilenetv2'))
				self.model = eval('hubconf.{}(pretrained=True)'.format('mobilenetv2'))
			elif self.model_name == 'regnetx_600m':
				self.model_teacher = ptcv_get_model('regnetx_600m', pretrained=True)
				self.model = ptcv_get_model('regnetx_600m', pretrained=True)
			else:
				assert False, "unsupport model: " + self.model_name
			self.model_teacher.eval()
		else:
			assert False, "unsupport data set: " + self.settings.dataset

	def _set_trainer(self):
		lr_master_G = utils.LRPolicy(self.settings.lr_G,
									 self.settings.nEpochs,
									 self.settings.lrPolicy_G)
		params_dict_G = {
			'step': self.settings.step_G,
			'decay_rate': self.settings.decayRate_G
		}

		lr_master_G.set_params(params_dict=params_dict_G)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			generator=self.generator,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			lr_master_S=None,
			lr_master_G=lr_master_G,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			use_FDDA=self.settings.use_FDDA,
			batch_index=self.batch_index,
			model_name=self.model_name,
			D_BNSLoss_weight=self.settings.D_BNSLoss_weight,
			C_BNSLoss_weight=self.settings.C_BNSLoss_weight,
			FDDA_iter=self.settings.FDDA_iter,
			BNLoss_weight=self.settings.BNLoss_weight
		)

	def quantize_model_resnet18(self, model, bit=None, module_name='model'):
		"""
	    Recursively quantize a pretrained single-precision model to int8 quantized model
	    model: pretrained single-precision model
	    """
		weight_bit = self.settings.qw
		act_bit = self.settings.qa

		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			if bit is not None:
				quant_mod = Quant_Conv2d(weight_bit=bit)
			else:
				quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			# quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod = Quant_Linear(weight_bit=8)
			quant_mod.set_param(model)
			return quant_mod

		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			# import IPython
			# IPython.embed()
			if module_name == 'model.features.stage4.unit2.activ':
				return nn.Sequential(*[model, QuantAct(activation_bit=8)])
			if bit is not None:
				return nn.Sequential(*[model, QuantAct(activation_bit=bit)])
			else:
				return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])

		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				if n == 'init_block':
					mods.append(self.quantize_model_resnet18(m, 8, module_name + '.' + n))
				else:
					mods.append(self.quantize_model_resnet18(m, bit, module_name + '.' + n))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)

			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model_resnet18(mod, bit, module_name + '.' + attr))
			return q_model

	def quantize_model_regnetx600m(self, model, bit=None, module_name='model'):
		"""
	    Recursively quantize a pretrained single-precision model to int8 quantized model
	    model: pretrained single-precision model
	    """
		weight_bit = self.settings.qw
		act_bit = self.settings.qa

		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			if module_name == 'model.features.init_block.conv':
				quant_mod = Quant_Conv2d(weight_bit=8)
			else:
				quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			# quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod = Quant_Linear(weight_bit=8)
			quant_mod.set_param(model)
			return quant_mod

		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			# import IPython
			# IPython.embed()
			if module_name == 'model.features.stage4.unit7.activ' or module_name == 'model.features.init_block.activ':
				return nn.Sequential(*[model, QuantAct(activation_bit=8)])
			if bit is not None:
				return nn.Sequential(*[model, QuantAct(activation_bit=bit)])
			else:
				return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])

		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model_regnetx600m(m, bit, module_name + '.' + n))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)

			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model_regnetx600m(mod, bit, module_name + '.' + attr))
			return q_model

	def quantize_model_mobilenetv2_w1(self, model, bit=None, module_name='model'):
		"""
	    Recursively quantize a pretrained single-precision model to int8 quantized model
	    model: pretrained single-precision model
	    """
		weight_bit = self.settings.qw
		act_bit = self.settings.qa

		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			if module_name == 'model.features.0.0':
				quant_mod = Quant_Conv2d(weight_bit=8)
			else:
				quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			# quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod = Quant_Linear(weight_bit=8)
			quant_mod.set_param(model)
			return quant_mod

		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			# import IPython
			# IPython.embed()
			if module_name == 'model.features.18.2' or module_name == 'model.features.0.2':
				return nn.Sequential(*[model, QuantAct(activation_bit=8)])
			else:
				return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])

		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model_mobilenetv2_w1(m, bit, module_name + '.' + n))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)

			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model_mobilenetv2_w1(mod, bit, module_name + '.' + attr))
			return q_model

	def quantize_model_mobilenetv1_w1(self, model, bit=None, module_name='model'):
		"""
	    Recursively quantize a pretrained single-precision model to int8 quantized model
	    model: pretrained single-precision model
	    """
		weight_bit = self.settings.qw
		act_bit = self.settings.qa

		# quantize convolutional and linear layers
		if type(model) == nn.Conv2d:
			if module_name == 'model.features.init_block.conv':
				quant_mod = Quant_Conv2d(weight_bit=8)
			else:
				quant_mod = Quant_Conv2d(weight_bit=weight_bit)
			quant_mod.set_param(model)
			return quant_mod
		elif type(model) == nn.Linear:
			# quant_mod = Quant_Linear(weight_bit=weight_bit)
			quant_mod = Quant_Linear(weight_bit=8)
			quant_mod.set_param(model)
			return quant_mod

		# quantize all the activation
		elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
			# import IPython
			# IPython.embed()
			if module_name == 'model.features.stage5.unit2.pw_conv.activ' or module_name == 'model.features.init_block.activ':
				return nn.Sequential(*[model, QuantAct(activation_bit=8)])
			else:
				return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])

		# recursively use the quantized module to replace the single-precision module
		elif type(model) == nn.Sequential:
			mods = []
			for n, m in model.named_children():
				mods.append(self.quantize_model_mobilenetv1_w1(m, bit, module_name + '.' + n))
			return nn.Sequential(*mods)
		else:
			q_model = copy.deepcopy(model)

			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					setattr(q_model, attr, self.quantize_model_mobilenetv1_w1(mod, bit, module_name + '.' + attr))
			return q_model

	def _replace(self):

		if self.model_name == 'resnet18':
			self.model = self.quantize_model_resnet18(self.model)
		elif self.model_name == 'mobilenet_w1':
			self.model = self.quantize_model_mobilenetv1_w1(self.model)
		elif self.model_name == 'mobilenetv2_w1':
			self.model = self.quantize_model_mobilenetv2_w1(self.model)
		elif self.model_name == 'regnetx_600m':
			self.model = self.quantize_model_regnetx600m(self.model)
		else:
			assert False, "unsupport model: " + self.model_name

	def freeze_model(self,model):
		"""
		freeze the activation range
		"""
		if type(model) == QuantAct or type(model) == QuantAct_MSE or type(model) == QuantAct_percentile:
			model.fix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.freeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.freeze_model(mod)
			return model
	
	def unfreeze_model(self,model):
		"""
		unfreeze the activation range
		"""
		if type(model) == QuantAct or type(model) == QuantAct_MSE or type(model) == QuantAct_percentile:
			model.unfix()
		elif type(model) == nn.Sequential:
			for n, m in model.named_children():
				self.unfreeze_model(m)
		else:
			for attr in dir(model):
				mod = getattr(model, attr)
				if isinstance(mod, nn.Module) and 'norm' not in attr:
					self.unfreeze_model(mod)
			return model

	def run(self):
		best_top1 = 100
		best_top5 = 100
		start_time = time.time()

		test_error, test_loss, test5_error = self.trainer.test_teacher(0)

		try:
			self.start_epoch = 0

			for epoch in range(self.start_epoch, self.settings.nEpochs):
				self.epoch = epoch

				self.freeze_model(self.model)

				if epoch < 4:
					self.logger.info("\n self.unfreeze_model(self.model)\n")
					self.unfreeze_model(self.model)

				_, _, _ = self.trainer.train(epoch=epoch, true_data_loader=self.true_data_loader)

				self.freeze_model(self.model)
				if self.settings.dataset in ["imagenet"]:
					if epoch > self.settings.warmup_epochs - 2:
						test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
					else:
						test_error = 100
						test5_error = 100
				else:
					assert False, "invalid data set"

				if best_top1 >= test_error:
					best_top1 = test_error
					best_top5 = test5_error
					self.logger.info(
						'Save generator! The path is' + os.path.join(self.settings.save_path, "generator.pth"))
					torch.save(self.generator.state_dict(), os.path.join(self.settings.save_path, "generator.pth"))
					self.logger.info(
						'Save model! The path is' + os.path.join(self.settings.save_path, "model.pth"))
					torch.save(self.model.state_dict(), os.path.join(self.settings.save_path, "model.pth"))
				
				self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
				self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1,
				                                                                                       100 - best_top5))

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return best_top1, best_top5


def main():
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument('--model_name', type=str)
	parser.add_argument('--id', type=int, metavar='experiment_id',
	                    help='Experiment ID')
	args = parser.parse_args()
	
	option = Option(args.conf_path)
	option.manualSeed = args.id + 3
	option.experimentID = option.experimentID + "{:0>2d}_repeat".format(args.id)

	if option.dataset in ["imagenet"]:
		generator = Generator_imagenet(option)
	else:
		assert False, "invalid data set"
	experiment = ExperimentDesign(generator, model_name=args.model_name, options=option, conf_path=args.conf_path)
	experiment.run()


if __name__ == '__main__':
	main()
