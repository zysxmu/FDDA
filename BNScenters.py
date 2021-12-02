import argparse
import os
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from pytorchcv.model_provider import get_model as ptcv_get_model
import torchvision

import torch.nn as nn
import utils as utils
import copy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import os
import torch
import torchvision.datasets as dsets
import pickle
import hubconf


__all__ = ["Trainer"]


class Trainer(object):
	"""
	trainer for training network, use SGD
	"""

	def __init__(self, model_teacher, train_loader):
		"""
		init trainer
		"""
		self.model_teacher = utils.data_parallel(model_teacher, 1)

		self.train_loader = train_loader
		self.mean_list = {}
		self.var_list = {}
		self.batch_index = 0
		self.register()
	def hook_fn_forward(self, module, input, output):
		input = input[0]
		mean = input.mean([0, 2, 3])
		var = input.var([0, 2, 3], unbiased=False)
		if self.batch_index not in self.mean_list:
			self.mean_list[self.batch_index] = []
			self.var_list[self.batch_index] = []
		self.mean_list[self.batch_index].append(mean.data.cpu())
		self.var_list[self.batch_index].append(var.data.cpu())

	def register(self):
		for m in self.model_teacher.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.register_forward_hook(self.hook_fn_forward)

	def only_find_BN(self, loader, l):

		path_label = {}

		self.mean_list.clear()
		self.var_list.clear()
		self.model_teacher.eval()
		with torch.no_grad():
			for i, (images, path, label) in enumerate(loader):
				images = images.cuda()
				output = self.model_teacher(images)
				path_label[self.batch_index] = (path, label)
				self.batch_index += 1
				break
		return self.mean_list, self.var_list, path_label, output

class imagenet_dataset(Dataset):
	def __init__(self, split_points, total_dataset, l):

		self.l = l
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
						normalize,])

		self.train_data = (total_dataset.imgs[split_points[l]:split_points[l+1]])

	def __getitem__(self, index):
		path = self.train_data[index][0]
		label = self.train_data[index][1]
		assert label == self.l
		with open(path, 'rb') as f:
			img = Image.open(f)
			img = img.convert('RGB')

		img = self.test_transform(img)
		return img, path, label

	def __len__(self):
		return len(self.train_data)

class ExperimentDesign:
	def __init__(self, model_name='resnet18'):
		self.train_loader = None
		self.model_teacher = None
		# for imagenet
		self.split_points = None
		self.total_dataset = None
		self.trainer = None
		self.model_name = model_name
		
		os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
		self.prepare()

	def prepare(self):
		self._set_gpu()
		self._set_model()
		self._set_trainer()
	
	def _set_gpu(self):
		torch.manual_seed(0)
		torch.cuda.manual_seed(0)
		cudnn.benchmark = True

	def _set_dataloader(self, l, dataPath=None, trueBN_batch_size=1):
		# create data loader

		if self.total_dataset is None:
			print('search for split points!')
			import torchvision.datasets as dsets
			traindir = os.path.join(dataPath, "train")
			self.total_dataset = dsets.ImageFolder(traindir)
			self.split_points = [0]

			for i, label in enumerate(self.total_dataset.targets):
				if i == 0:
					continue
				if label != self.total_dataset.targets[i-1]:
					self.split_points.append(i)
				if i == len(self.total_dataset.targets)-1:
					self.split_points.append(i+1)
			print('search end!')

		dataset = imagenet_dataset(self.split_points, self.total_dataset, l)
		trainloader = torch.utils.data.DataLoader(dataset,
												  batch_size=trueBN_batch_size,
												  shuffle=True,
												  num_workers=0,
												  pin_memory=True)

		self.train_loader = trainloader
		return

	def _set_model(self):

		print('load ' + self.model_name)
		if self.model_name == 'resnet18':
			self.model_teacher = ptcv_get_model('resnet18', pretrained=True)
		elif self.model_name == 'mobilenet_w1':
			self.model_teacher = ptcv_get_model('mobilenet_w1', pretrained=True)
		elif self.model_name == 'mobilenetv2_w1':
			self.model_teacher = eval('hubconf.{}(pretrained=True)'.format('mobilenetv2'))
		elif self.model_name == 'regnetx_600m':
			self.model_teacher = ptcv_get_model('regnetx_600m', pretrained=True)
		else:
			assert False, "unsupport model: " + self.model_name
		self.model_teacher.eval()
		print(self.model_teacher)

	def _set_trainer(self):
		# set trainer
		self.trainer = Trainer(
			model_teacher=self.model_teacher,
			train_loader=self.train_loader)

	def only_find_BN(self, dataPath=None, trueBN_batch_size=1):
		mean_Categorical, var_Categorical, path_label_Categorical, teacher_output_Categorical = {}, {}, {}, {}
		for l in range(1000):
			self._set_dataloader(l, dataPath, trueBN_batch_size)
			mean_l, var_l, path_label, output_l = self.trainer.only_find_BN(self.train_loader, l)
			mean_Categorical[l], var_Categorical[l] = copy.deepcopy(mean_l), copy.deepcopy(var_l)
			path_label_Categorical[l] = copy.deepcopy(path_label)
			teacher_output_Categorical[l] = copy.deepcopy(output_l.cpu())
			print('label:', l, 'len', len(self.train_loader), len(mean_Categorical),
				  len(var_Categorical), len(path_label_Categorical))

		head = './save_ImageNet'
		with open(head + "/"+self.model_name+"_mean_Categorical_bs_1.pickle", "wb") as fp:
			pickle.dump(mean_Categorical, fp, protocol=pickle.HIGHEST_PROTOCOL)
		with open(head + "/"+self.model_name+"_var_Categorical_bs_1.pickle", "wb") as fp:
			pickle.dump(var_Categorical, fp, protocol=pickle.HIGHEST_PROTOCOL)
		with open(head + "/"+self.model_name+"_path_label_Categorical_bs_1.pickle", "wb") as fp:
			pickle.dump(path_label_Categorical, fp, protocol=pickle.HIGHEST_PROTOCOL)
		with open(head + "/"+self.model_name+"_teacher_output_Categorical_1.pickle", "wb") as fp:
			pickle.dump(teacher_output_Categorical, fp, protocol=pickle.HIGHEST_PROTOCOL)
		return None


def main():
	parser = argparse.ArgumentParser(description='Baseline')
	parser.add_argument('--dataPath', type=str)
	parser.add_argument('--model_name', type=str)
	args = parser.parse_args()

	experiment = ExperimentDesign(args.model_name)
	experiment.only_find_BN(dataPath=args.dataPath, trueBN_batch_size=1)


if __name__ == '__main__':
	main()
