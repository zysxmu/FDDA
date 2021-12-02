import argparse
import datetime
import os
import time
import traceback
import sys
import copy
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import logging
import hubconf

# option file should be modified according to your expriment
from options import Option

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dsets
import os
import shutil


class DataLoader(object):
    """
    data loader for CV data sets
    """

    def __init__(self, dataset, batch_size, n_threads=4,
                 ten_crop=False, data_path='/home/dataset/', logger=None):
        """
        create data loader for specific data set
        :params n_treads: number of threads to load data, default: 4
        :params ten_crop: use ten crop for testing, default: False
        :params data_path: path to data set, default: /home/dataset/
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        self.data_path = data_path
        self.logger = logger
        self.dataset_root = data_path

        if self.dataset in ["imagenet"]:
            self.train_loader, self.test_loader = self.imagenet(
                dataset=self.dataset)
        else:
            assert False, "invalid data set"

    def getloader(self):
        """
        get train_loader and test_loader
        """
        return self.train_loader, self.test_loader

    def imagenet(self, dataset="imagenet"):


        testdir = os.path.join(self.data_path, "val")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        test_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(testdir, test_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_threads,
            pin_memory=False)
        return None, test_loader


def test(model, test_loader):
    """
    testing
    """
    top1_error = utils.AverageMeter()
    top1_loss = utils.AverageMeter()
    top5_error = utils.AverageMeter()

    model.eval()
    iters = len(test_loader)
    print('total iters', iters)
    start_time = time.time()
    end_time = start_time

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i % 100 == 0:
                print(i)
            start_time = time.time()

            labels = labels.cuda()
            images = images.cuda()
            output = model(images)

            loss = torch.ones(1)
            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, loss=loss,
                labels=labels, top5_flag=True, mean_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))
            end_time = time.time()

            if i % 500 == 0:
                print(i)

    return top1_error.avg, top1_loss.avg, top5_error.avg

class ExperimentDesign:
    def __init__(self, model_name=None, model_path=None, options=None, conf_path=None):
        self.settings = options or Option(conf_path)

        self.model_name = model_name
        self.model_path = model_path

        self.test_loader = None
        self.model = None
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        self.prepare()

    def prepare(self):
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._replace()

    def _set_model(self):

        if self.settings.dataset in ["imagenet"]:
            if self.model_name == 'resnet18':
                self.model = ptcv_get_model('resnet18', pretrained=False)
            elif self.model_name == 'mobilenet_w1':
                self.model = ptcv_get_model('mobilenet_w1', pretrained=False)
            elif self.model_name == 'mobilenetv2_w1':
                self.model = eval('hubconf.{}(pretrained=False)'.format('mobilenetv2'))
            elif self.model_name == 'regnetx_600m':
                self.model = ptcv_get_model('regnetx_600m', pretrained=False)
            else:
                assert False, "unsupport model: " + self.model_name
            self.model.eval()
        else:
            assert False, "unsupport data set: " + self.settings.dataset

    def _set_gpu(self):
        torch.manual_seed(self.settings.manualSeed)
        torch.cuda.manual_seed(self.settings.manualSeed)
        assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        cudnn.benchmark = True

    def _set_dataloader(self):
        # create data loader
        data_loader = DataLoader(dataset=self.settings.dataset,
                                 batch_size=32,
                                 data_path=self.settings.dataPath,
                                 n_threads=self.settings.nThreads,
                                 ten_crop=self.settings.tenCrop,
                                 logger=None)

        self.train_loader, self.test_loader = data_loader.getloader()

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
        print(self.model)

    def freeze_model(self, model):
        """
        freeze the activation range
        """
        if type(model) == QuantAct:
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

    def run(self):
        best_top1 = 100
        best_top5 = 100
        start_time = time.time()

        pretrained_dict = torch.load(self.model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('cur_x' not in k)}

        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('load!')
        self.model = self.model.cuda()
        try:

            self.freeze_model(self.model)
            if self.settings.dataset in ["imagenet"]:
                test_error, test_loss, test5_error = test(model=self.model, test_loader=self.test_loader)

            else:
                assert False, "invalid data set"
            print("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(test_error, test5_error))
            print("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - test_error,
                                                                                                   100 - test5_error))
        except BaseException as e:
            print("Training is terminating due to exception: {}".format(str(e)))
            traceback.print_exc()

        end_time = time.time()
        time_interval = end_time - start_time
        t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
        print(t_string)

        return best_top1, best_top5


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--conf_path', type=str, metavar='conf_path',
                        help='input the path of config file')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    option = Option(args.conf_path)
    experiment = ExperimentDesign(model_name=args.model_name, model_path=args.model_path, options=option, conf_path=args.conf_path)
    experiment.run()


if __name__ == '__main__':
    main()
