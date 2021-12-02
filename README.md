# Fine-grained Data Distribution Alignment for Post-Training Quantization [paper](https://arxiv.org/abs/2109.04186)

## Requirements

- Python >= 3.7.10
- Pytorch >= 1.2.0, Pytorch >= 1.7.0 (For regnetx600m specially)
- Torchvision >= 0.4.0

## Reproduce the Experiment Results 

1. The pre-trained model will be downloaded automatically. If the download process fails, please use the URL in the console to download manually.

2. Randomly select one image per class to generate corresponding BNS centers, selected images will be formulating the calibration dataset, run:
    
    `cd FDDA`

    `mkdir save_ImageNet`

    `CUDA_VISIBLE_DEVICES=0 python BNScenters.py --dataPath PathToImageNetDataset --model_name resnet18/mobilenet_w1/mobilenetv2_w1/regnetx_600m`  
   
   Noted that a model will generate corresponding BNS centers that can't be used by other model.

4. Use FDDA to train a quantized model. Modify the `qw, qa` in imagenet_config.hocon to set desired bit-width. Modify the `dataPath` in imagenet_config.hocon to the path of ImageNet Dataset. For all layers are quantized to same bit-width, run:

    `CUDA_VISIBLE_DEVICES=0 python main_cosine_CBNS.py --model_name resnet18/mobilenet_w1/mobilenetv2_w1/regnetx_600m --conf_path imagenet_config.hocon --id=0`

   For F8L8, run:
   
   `CUDA_VISIBLE_DEVICES=0 python main_cosine_CBNS_8F8L.py --model_name resnet18/mobilenet_w1/mobilenetv2_w1/regnetx_600m --conf_path imagenet_config.hocon --id=0`

## Evaluate Our Models

We also provide **original** (before the code clean up, thus, the logs will be a little messy.) training logs and trained models for test. 
They can be downloaded from [here](https://drive.google.com/drive/folders/1LNhxoYKG2fz3D3-7A7WiMpdjAh8f-HZH?usp=sharing) 

Due to different remote servers have different hardware and software constraints, we use different versions of PyTorch to finish our experiments.
Noted that the PyTorch version in test should be the same as the PyTorch version in train to fully recover the accuracy.
Please use the PyTorch version in train in the first line of `train_test.log`.


To test our models, download it and then modify the `qw, qa` in imagenet_config.hocon to set desired bit-width. For all layers are quantized to same bit-width, run:

   `CUDA_VISIBLE_DEVICES=0 python test.py --conf_path imagenet_config.hocon --model_name resnet18/mobilenet_w1/mobilenetv2_w1/regnetx_600m --model_path PathToModel`

   For F8L8, run:
   
   `CUDA_VISIBLE_DEVICES=0 python test_8F8L.py --conf_path imagenet_config.hocon --model_name resnet18/mobilenet_w1/mobilenetv2_w1/regnetx_600m --model_path PathToModel`

Following results can be obtained:

| Model     | Bit-width| Dataset  | Top-1 Acc.  |
| --------- | -------- | -------- | ----------- | 
| resnet18  | W4A4-F4L4 | ImageNet | 68.744%    | 
| resnet18  | W4A4-F8L8 | ImageNet | 69.758%    | 
| resnet18  | W5A5-F5L5 | ImageNet | 70.558%    | 
| resnet18  | W5A5-F8L8 | ImageNet | 70.864%    | 
| --------- | -------- | -------- | ----------- | 
| mobilenetv1  | W4A4-F4L4 | ImageNet | 63.748%    | 
| mobilenetv1  | W4A4-F8L8 | ImageNet | 65.760%    | 
| mobilenetv1  | W5A5-F5L5 | ImageNet | 70.258%    | 
| mobilenetv1  | W5A5-F8L8 | ImageNet | 71.764%    | 
| --------- | -------- | -------- | ----------- | 
| mobilenetv2  | W4A4-F4L4 | ImageNet | 68.382%    | 
| mobilenetv2  | W4A4-F8L8 | ImageNet | 69.322%    | 
| mobilenetv2  | W5A5-F5L5 | ImageNet | 71.63%    | 
| mobilenetv2  | W5A5-F8L8 | ImageNet | 71.99%    | 
| --------- | -------- | -------- | ----------- | 
| regnetx600m  | W4A4-F4L4 | ImageNet | 68.960%    | 
| regnetx600m  | W4A4-F8L8 | ImageNet | 70.326%    | 
| regnetx600m  | W5A5-F5L5 | ImageNet | 73.620%    | 
| regnetx600m  | W5A5-F8L8 | ImageNet | 73.996%    | 

### Contact

For any question, be free to contact: viper.zhong@gmail.com. The github issue is also welcome.