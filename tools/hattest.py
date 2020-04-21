import time

import torch
from torchsummary import summary

from thop import profile
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import cv2
# config_file = '../configs/ttfnet/ttfnet_re_1x_hat.py' # 网络配置文件
config_file = '../configs/ttfnet/ttfnet_r18_1x.py'
# checkpoint_file = '../checkpoints/work_dir/epoch_50.pth' # 刚刚下载的模型文件地址
checkpoint_file = '../checkpoints/ttfnet18_1x-fe6884.pth'
# build the model from a config file and a checkpoint file

model = init_detector(config_file, checkpoint_file, device='cuda:0')



input = torch.randn(1, 3, 512, 512)
# flops = profile(model, inputs=(input, ))

print((model))
#test a single image and show the results
img = '../data/hat/VOC2007/JPEGImages/000003.jpg' # 测试图像地址
# or img = mmcv.imread(img), which will only load it once

result = inference_detector(model, img)

show_result(img, result, model.CLASSES)

# test a list of images and write the results to image files
#imgs = ['../images/16004479832_a748d55f21_k.jpg', #'../images/17790319373_bd19b24cfc_k.jpg']
#for i, result in enumerate(inference_detector(model, imgs)):
#    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))

#test a video and show the results
# path = '../hattest.mp4'
# path3 = 0
# path2 = "rtsp://admin:wq123456@192.168.1.62:554/h264/ch1/main/av_stream"
# video = mmcv.VideoReader(path)
# for frame in video:
#
#     start = time.time()
#     result = inference_detector(model, frame)
#     end = time.time()
#     # Time elapsed
#     seconds = end - start
#     #print("Time taken : {0} seconds".format(seconds))
#     # Calculate frames per second
#     fps = 1 / seconds
#     print("Estimated frames per second : {0}".format(fps))
#     show_result(frame, result, model.CLASSES, wait_time=1)
