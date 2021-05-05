#!/usr/bin/env python3
###############################mike

#from tensorflow.keras.models import Sequential, model_from_json
import rospy
import math
from get_rs_image import Get_image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys
sys.path.insert(1, "/home/chien/.local/lib/python3.6/site-packages/")
from sort_data import Sort_data
##################################
from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
import tensorflow as tf
from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='/home/chien/graduate_program/src/yolact/src/weights/yolact_base_597_80000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=100, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.7, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')                   
    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)
    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

n = 10
state = False
old_cenX = 0
old_cenY = 0
frame_count = 0
count = 0
old_count = 0
first_state = True
centerX = []
centerY = []

arrayPx = [[ [] for i in range(0)] for j in range(20)]
arrayPy = [[ [] for i in range(0)] for j in range(20)]

point = [[ [] for i in range(0)] for j in range(20)]
old_data = [[ [] for i in range(0)] for j in range(20)]

predict_pos = [[ [] for i in range(0)] for j in range(20)]
for i in range(20):
    old_data[i] = [i]
    for k in range(6):
        old_data[i].append(0)
    for j in range(2):
        predict_pos[i].append([0])  
    predict_pos[i].append([])    
       
first_frame = False

state_pre = False

realX = []
realY = []
preX = []
preY = []
flag = False
misx = []
misy=[] 

old_obj_info = [[ [] for i in range(0)] for j in range(20)]
for i in range(20):
    old_obj_info[i] = [i] 
    for k in range(3):
        old_obj_info[i].append(0)
    old_obj_info[i].append([0,0,0,0,0,0])
    for j in range(2):   
        old_obj_info[i].append([]) 



#------------------------------------------------------------------------
mask_flag = False

origin_data = [[ [] for i in range(0)] for j in range(20)]
compare_data = [[ [] for i in range(0)] for j in range(20)]
old_obj_count = 0
total_count = 0
for i in range(20):
    #---------------------ID---------------------
    origin_data[i] = [i]
    compare_data[i] = [i]
    #--------------------------------------------
    for k in range(3):
        origin_data[i].append(0)
        compare_data[i].append(0)

    origin_data[i].append([0,0,0,0,0,0])
    compare_data[i].append([0,0,0,0,0,0])  

    for j in range(2):   
        origin_data[i].append([])
        compare_data[i].append([])
#---------------------------------------------------------------


def trans_degree(x,y,degree):
    
    degree=float(degree)*math.pi/180
    l=50
    #####
    f_x=float(x)-l*math.sin(degree)
    f_y=float(y)+l*math.cos(degree)
    #####
    tmp_x=x-round(f_x)
    tmp_y=y-round(f_y)
    return (round(tmp_x),round(tmp_y))

def arctan_recovery(cos_x,sin_x):
    global target_ang
    predict_degree = 0.5*np.arctan2(sin_x,cos_x)
    predict_degree=predict_degree/math.pi*180
    target_ang = predict_degree
    return predict_degree

def predict1_next(Img,Px, Py):
    global model, state_pre
    x_input = np.array([[  [Px[0],Py[0]], [Px[1],Py[1]],  [Px[2],Py[2]]  ]])
    x_input = x_input.reshape((1, 3, 2))
    x_input=x_input/100.

    photo = np.array(Img) 
    photo = np.expand_dims(photo, axis=0)
    photo=photo.reshape((-1,480,640,1))
    photo=photo/255
    predict_degree, yhat = model.predict([photo, x_input], verbose=0)
    degree = arctan_recovery(predict_degree[0][0],predict_degree[0][1])

    yhat*=100
    state_pre = True
    return degree, yhat


def predict_next(Px, Py):
    global model, state_pre
    x_input = np.array([[  [Px[0],Py[0]], [Px[1],Py[1]],  [Px[2],Py[2]]  ]])
    x_input = x_input.reshape((1, 3, 2))
    x_input=x_input/100.
    yhat = model.predict(x_input, verbose=0)
    yhat*=100
    state_pre = True
    return yhat
    
def plot_data(real_posX,real_posY, pre_posX, pre_posY):
    '''
    x = (np.sum(misx)/len(misx))
    y = (np.sum(misy)/len(misy))
    plt.xlabel('X (Pixel)')
    plt.ylabel('Y (Pixel)')
    plt.text(180,218, 'x average error: %f pixel'% x)
    plt.text(180,216, 'y average error: %f pixel'% y)
    plt.ylim(185,220)
    plt.title("Speed : 3.54 cm/s \n (Record every 10 fps )")
    '''
    
    #plt.plot(real_posX[0], real_posY[0], 'ko')
    plt.plot(real_posX, real_posY, 'ro--' , label='real_pos')
    plt.plot(pre_posX, pre_posY, 'bo--', label='pre_pos')
    plt.legend()
    plt.savefig('low_speed.png')

def Compare_Data(old_data, new_data, old_count, new_count):
    global origin_data
    num = 0
    list_info = []
    data_set = [[ [] for i in range(0)] for j in range(20)]
    for i in range(20):
        data_set[i] = [i] 
        for k in range(3):
            data_set[i].append(0)
        data_set[i].append([0,0,0,0,0,0])
        for j in range(2):   
            data_set[i].append([]) 

    for i in range(new_count):
        distance = []
        for j in range(old_count):
            distance.append(np.sqrt( pow((new_data[i][4][0] - old_data[j][4][0]),2) +  pow((new_data[i][4][1] - old_data[j][4][1]),2) ))

        if distance != []:
            min_id = distance.index(min(distance))
            
            if distance[min_id] < 30:

                if new_data[i][1] == old_data[min_id][1]:   
                    #currect object
                    data_set[min_id] = old_data[min_id]
                    '''
                    data_set[min_id] = new_data[i]
                    data_set[min_id][0] = min_id
                    '''
                   
                else:
                    #new object
                    data_set[20-1-num] = new_data[i]
                    data_set[20-1-num][0] = 20-1-num
                    num+=1
                    
            else: 
                #other the same object, origin object is disapear
                data_set[20-1-num] = new_data[i]
                data_set[20-1-num][0] = 20-1-num
                num+=1 
                
    if num != 0: 
        for i in range(20):
            if data_set[i][1] == 0:
                list_info.append(i)
        for i in range(num):
            #change obj info
            data_set[list_info[i]], data_set[20-1-i] = data_set[20-1-i], data_set[list_info[i]] 
            data_set[list_info[i]][0], data_set[20-1-i][0] = data_set[20-1-i][0], data_set[list_info[i]][0] 
    
    if new_count < old_count:
        total_obj = old_count  
    total_obj = new_count 

    new_data = data_set
    return new_data, total_obj

def data_save( mask_picture, classes, names, scorces, boxes):
    global origin_data, compare_data ,first_frame  , old_obj_count
    new_obj_count = 0
    total_count = 0
    compare_data = [[ [] for i in range(0)] for j in range(20)]
    for i in range(20):
        compare_data[i] = [i]
        for k in range(3):
            compare_data[i].append(0)
        compare_data[i].append([0,0,0,0,0,0])  
        for j in range(2):   
            compare_data[i].append([])

    if first_frame == False:
        for i in range(len(classes)):
            origin_data[i][1] = names[i]
            origin_data[i][2] = scorces[i]
            #origin_data[i][3] = mask_picture
            x1,y1,x2,y2 = boxes[i, :]
            origin_data[i][4] = [(x1+x2)/2, (y1+y2)/2, x1, x2, y1, y2]
            old_obj_count+=1
        new_data = origin_data
        first_frame = True
    else:
        for i in range(len(classes)):
            compare_data[i][1] = names[i]
            compare_data[i][2] = scorces[i]
            #compare_data[i][3] = mask_picture
            x1,y1,x2,y2 = boxes[i, :]
            compare_data[i][4] = [(x1+x2)/2, (y1+y2)/2, x1, x2, y1, y2]
            new_obj_count+=1
        
        #new_data, total_count = Compare_Data(origin_data, compare_data, old_obj_count, new_obj_count)       
        num = 0
        list_info = []
        data_set = [[ [] for i in range(0)] for j in range(20)]
        for i in range(20):
            data_set[i] = [i] 
            for k in range(3):
                data_set[i].append(0)
            data_set[i].append([0,0,0,0,0,0])
            for j in range(2):   
                data_set[i].append([]) 

        for i in range(new_obj_count):
            distance = []
            for j in range(old_obj_count):
                distance.append(np.sqrt( pow((compare_data[i][4][0] - origin_data[j][4][0]),2) +  pow((compare_data[i][4][1] - origin_data[j][4][1]),2) ))

            if distance != []:
                min_id = distance.index(min(distance))
                
                if distance[min_id] < 20:

                    if compare_data[i][1] == origin_data[min_id][1]:   
                        #currect object
                        data_set[min_id] = compare_data[i]
                        data_set[min_id][0] = min_id
                        data_set[min_id][5] = origin_data[min_id][5] 
                        data_set[min_id][6] = origin_data[min_id][6]         
                    else:
                        #new object
                        data_set[20-1-num] = compare_data[i]
                        data_set[20-1-num][0] = 20-1-num
                        num+=1
                        
                else: 
                    #other the same object, origin object is disapear
                    data_set[20-1-num] = compare_data[i]
                    data_set[20-1-num][0] = 20-1-num
                    num+=1 
                    
        if num != 0: 
            for i in range(3):
                if data_set[i][1] == 0:
                    list_info.append(i)
            for i in range(num):
                #change obj info
                data_set[list_info[i]], data_set[20-1-i] = data_set[20-1-i], data_set[list_info[i]] 
                data_set[list_info[i]][0], data_set[20-1-i][0] = data_set[20-1-i][0], data_set[list_info[i]][0] 
        
        if new_obj_count < old_obj_count:
            total_obj = old_obj_count  
        total_obj = new_obj_count 
        compare_data = data_set

        origin_data = compare_data
        old_obj_count = new_obj_count
        total_count = total_obj
    return compare_data, total_count


'''
def data_save(img_info, classes, scores, boxes):
    global old_data, new_data, first_frame, count, old_count
    
    new_data = [[ [] for i in range(0)] for j in range(20)]
    test = [[ [] for i in range(0)] for j in range(20)]
    for i in range(20):
        new_data[i] = [i]
        test[i] = [i]
        for k in range(3):
            new_data[i].append(0)
            test[i].append(0)
        test[i].append((0,0))
        new_data[i].append((0,0)) 
    num = 0
    num1 = 0
    count = 0
    if first_frame == False:
        for i in range(len(classes)):
            old_data[i][1] = cfg.dataset.class_names[classes[i]]
            old_data[i][2] = 1
            old_data[i][3] = scores[i]
            x1,y1,x2,y2 = boxes[i, :]
            old_data[i][4] = [(x1+x2)/2, (y1+y2)/2, x1, x2, y1, y2]
            #old_data[i][5] = img_info #0306
            old_count+=1
            
        first_frame = True 
        new_data = old_data   
    else:

        for i in range(len(classes)):
            
            test[i][1] = cfg.dataset.class_names[classes[i]]
            test[i][2] = 1
            test[i][3] = scores[i]
            x1,y1,x2,y2 = boxes[i, :]
            test[i][4] = [(x1+x2)/2, (y1+y2)/2, x1, x2, y1, y2]
            #test[i][5] = img_info #0306
            count +=1
             
        #rint('1   : ',test)
        for i in range(count):
            testlist = []
            for j in range(old_count):
                testlist.append(np.sqrt( pow((test[i][4][0] - old_data[j][4][0]),2) +  pow((test[i][4][1] - old_data[j][4][1]),2) )) 
            
            if testlist != []:
                minid = testlist.index(min(testlist))

                if testlist[minid] < 20:
                    if test[i][1] == old_data[minid][1]:   
                        new_data[minid] = test[i]
                        new_data[minid][0] = minid
                    else:
                        
                        new_data[count+num-1] = test[i]
                        new_data[count+num-1][0] = count+num-1
                        num+=1
                else: 

                    new_data[count+num-1] = test[i]
                    new_data[count+num-1][0] = count+num-1
                    num1+=1 
                    num+=1
        if num != 0:
            for i in range(count+num):
                if new_data[i][2] == 0:
                    if i < (count+num1 -2):
                        new_data[i], new_data[count+num1 -2] = new_data[count+num1-2], new_data[i]
                        new_data[i][0], new_data[count+num1 -2][0] = new_data[count+num1-2][0], new_data[i][0]
                        num1 -=1
    
        old_data = new_data 
        if count < old_count:
            count = old_count  
        old_count = count         
        #print(old_data)
        #print('-------------')    
    return new_data , old_count
'''
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    global first_frame, old_obj_info
    name = []
    mask_img = []
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save
        
    
    with timer.env('Copy'):
        
        #idx = t[1].argsort(0, descending=True)[:args.top_k]
        idx1 = t[1].argsort()
        idx = idx1.argsort()
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
            mask_picture = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
        for i in range(len(classes)):
            name.append(cfg.dataset.class_names[classes[i]])
            mask_img.append(mask_picture[i:i+1, :, :, None])

        #obj_info, obj_num = data_save(mask_img, classes, scores, boxes)
        
        obj_info, obj_num = test111.data_save(mask_img, classes, name, scores, boxes, first_frame, old_obj_info)
        first_frame = True
        
        
        #print(classes)
        #print('---------')
        #np.save('masks.npy', masks.cpu().numpy())
        #print(obj_info[0][4][0], obj_info[0][4][1])
    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (obj_info[j][0] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        
        masks = masks[:num_dets_to_consider, :, :, None]
        #img_gpu = img_gpu * (masks.sum(dim=0) > 0.5).float()  #only show mask
        #img_gpu = img_gpu * masks[0]
        

        #mike0225
        mask_img = img_gpu * (masks.sum(dim=0) > 0.5).float() #0209
        global mask_numpy 
        mask_numpy = (mask_img * 255).byte().cpu().numpy() #0209
        mask_numpy = cv2.cvtColor(mask_numpy, cv2.COLOR_BGR2GRAY)
        
        

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])

        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha
        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

        
    if args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        global  frame_count, state_pre, flag, predict_pos, centerX, centerY, preX, preY, degree ,mask_color, mask_flag
        frame_count+=1
        
        for j in range(obj_num):
            global img_num, temp_x, temp_y, yhat
            '''
            mask_image = mask_picture[j:j+1, :, :, None]
            mask_image = img_gpu * (mask_image.sum(dim=0) > 0.5).float() #0209
            mask_numpy1 = (mask_image * 255).byte().cpu().numpy() #0209
            mask_color = cv2.cvtColor(mask_numpy1, cv2.COLOR_BGR2GRAY)
            mask_flag = True
            '''
            if obj_info[j][2] != 0:
                
                
                #0502-------------------------------------------------------------------
                mask_image = img_gpu *(obj_info[j][3].sum(dim=0) > 0.5).float()
                mask_numpy1 = (mask_image * 255).byte().cpu().numpy() 
                mask_color= cv2.cvtColor(mask_numpy1, cv2.COLOR_BGR2GRAY)
                mask_flag = True
                #-------------------------------------------------------------------------
                
                if frame_count%10 == 3:
                    
                    #-----------------------------
                    obj_info[j][5].append(mask_color)
        
                    #cv2.imwrite('/home/chien/123/test_{}.jpg'.format(j),mask_numpy1)
                    
                    if len(obj_info[j][5]) > 2:
                        '''
                        for k in range(len(obj_info[j][5])):
                            cv2.imwrite('/home/chien/123/test_{}.jpg'.format(k),obj_info[j][5][k])
                        '''    
                        imagedata1 = np.array(obj_info[j][5])    
                        
                        imagedata1=imagedata1.reshape((-1,3,480,640,1))
                        imagedata1=imagedata1/255.
                        yhat = model.predict(imagedata1, verbose=0) 
                        #print(yhat)
                        
                        obj_info[j][5].pop(0) #0->1
                
                if len(obj_info[j][5]) == 2:
                    
                    for i in range(5):
                        x1 = yhat[1][0][i][1]*320+320
                        y1 = yhat[1][0][i][2]*240+240
                        degree1 = arctan_recovery(yhat[1][0][i][3],yhat[1][0][i][4])
                        temp_x1,temp_y1=trans_degree(x1,y1,degree1)
                        cv2.circle(img_numpy, (int(x1),int(y1)),5,(0, 0, 255),5)
                        cv2.line(img_numpy,(int(x1+temp_x1),int(y1+temp_y1)),(int(x1-temp_x1),int(y1-temp_y1)),(0,0,255),5)
                        
                    #for k in range(len(obj_info[j][5])):
                    #    cv2.imwrite('/home/chien/123/test_{}.jpg'.format(k),obj_info[j][5][k])
                       
                    #-----------------------------
                    
                    centerX.append(obj_info[j][4][0])
                    centerY.append(obj_info[j][4][1])

                    predict_pos[j][0].append(obj_info[j][4][0])
                    predict_pos[j][1].append(obj_info[j][4][1])
                    
                    if predict_pos[j][0][0] == 0:
                        predict_pos[j][0].pop(0)
                    if predict_pos[j][1][0] == 0:
                        predict_pos[j][1].pop(0) 

                    if len(predict_pos[j][0]) > 2:
                        #predict_pos[j][2] = predict_next( predict_pos[j][0], predict_pos[j][1]) 

                        '''
                        degree, predict_pos[j][2] = predict1_next( mask_numpy1, predict_pos[j][0], predict_pos[j][1]) # test0227
                        temp_x,temp_y=trans_degree(predict_pos[j][2][0,4,0],predict_pos[j][2][0,4,1],degree)
                        '''
                        
                        predict_pos[j][0].pop(0) #0->1
                        predict_pos[j][1].pop(0)
                '''
                if state_pre == True:
                    
                    if predict_pos[j][2] != []:
                        
                        for i in range(5):
                            if (predict_pos[j][2][0,i,0]) > 640 or (predict_pos[j][2][0,i,1]) > 480:
                                pass
                            else:    
                                pass
                                #cv2.circle(img_numpy,(predict_pos[j][2][0,i,0],predict_pos[j][2][0,i,1]),5,(0,0,213),-1)      
                        cv2.line(img_numpy,(int(obj_info[j][4][0]+temp_x),int(obj_info[j][4][1]+temp_y)),(int(obj_info[j][4][0]-temp_x),int(obj_info[j][4][1]-temp_y)),(0,0,255),3)
                        
                        if flag ==False:
                            for i in range(5):
                                preX.append(predict_pos[j][2][0,i,0])
                                preY.append(predict_pos[j][2][0,i,1])
                                #preY.append(num)
                        else:
                            preX.append(predict_pos[j][2][0,4,0])
                            preY.append(predict_pos[j][2][0,4,1])
                            #preY.append(num)

                        flag = True
                '''        
                color = get_color(obj_info[j][0])
                score = obj_info[j][3]
                
                if args.display_bboxes:
                    cv2.rectangle(img_numpy, (obj_info[j][4][2], obj_info[j][4][4]), (obj_info[j][4][3], obj_info[j][4][5]), color, 1)
                
                if args.display_text:
                    
                    _class = obj_info[j][1]
                    
                    #text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class
                    text_str = '%s: %s' % (obj_info[j][0],_class) if args.display_scores else _class

                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.6
                    font_thickness = 1

                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                    text_pt = (obj_info[j][4][2], obj_info[j][4][4] - 3)
                    text_color = [255, 255, 255]
                    
                    cv2.rectangle(img_numpy, (obj_info[j][4][2], obj_info[j][4][4]), (obj_info[j][4][2] + text_w, obj_info[j][4][4] - text_h - 4), color, -1)
                    cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
            '''
            else:
                for i in range(2):
                    predict_pos[j][i] = [0]
                predict_pos[j][2] = []
            '''
        old_obj_info = obj_info
        #print(old_obj_info)
        #print('#####################################')
    return img_numpy


class Detections:

    def __init__(self):
        self.bbox_data = []
        self.mask_data 
    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),
            (self.mask_data, args.mask_det_file)
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)
          

from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def evalvideo(vid, net:Yolact, out_path:str=None):
    cudnn.benchmark = True
    
    '''
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)
    '''
    #target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    
    num_frames = float('inf')

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()

    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    def cleanup_and_exit():
        print()
        pool.terminate()
        #vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            #frame = vid.read()[1]
            frame = sub_img.cv_image
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            out = net(imgs)
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    # All this timing code to make sure that 
    def play_video():
        try:
           
            nonlocal frame_buffer, running, num_frames, frames_displayed, vid_done
            while running:
                frame_time_start = time.time()
                global mask_numpy, mask_color
                if not frame_buffer.empty():
                    next_time = time.time()
                      
                    if out_path is None:
                        #if mask_flag==True:
                        #    cv2.imshow('test', mask_color)
                        #cv2.imshow('123',mask_numpy)
                        
                        cv2.imshow("yolact", frame_buffer.get())

                        '''
                        if cv2.waitKey(33) & 0xFF == ord('s'):
                            savedata(mask_numpy)
                        '''
                    else:
                
                        out.write(frame_buffer.get())

                if out_path is None and cv2.waitKey(1) == 27:
                    '''
                    global centerX, centerY
                    with open("test.txt","w") as f:
                        for i in range(len(centerX)):   
                            a = centerX[i]
                            f.write(str(a)+ ' ')  
                        f.write('\n' + 'aaaaaa' + '\n')
                        for k in range(len(centerY)):
                            b = centerY[k]
                            f.write(str(b)+ ' ')
                        f.write('\n')    
                    '''  

                    '''
                    global centerX, centerY , preX, preY
                    plot_data(centerX, centerY, preX, preY)
                    '''
  
                    # Press Escape to close
                    running = False
                if not (frames_displayed < num_frames):
                    running = False

        except:
            # See issue #197 for why this is necessary
            import traceback
            traceback.print_exc()


    extract_frame = lambda x, i: (x[0][i] if x[1][i]['detection'] is None else x[0][i].to(x[1][i]['detection']['box'].device), [x[1][i]])
    
    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]
    print()
    if out_path is None: print('Press Escape to close.')
    
    try:
        
        while running:
           
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)
            start_time = time.time()
            # Start loading the next frames from the disk
            if not vid_done:
                
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
                
            else:
                next_frames = None
            
            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args =  [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
                
                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())
                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)
                
                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence)-1})
                frame_times = (time.time() - start_time)
                fps = 1 / frame_times
                #print('fps', fps)
    except KeyboardInterrupt:
        print('\nStopping...')
    
    cleanup_and_exit()

def savedata(img):
    global take_picture_counter
    Object_Name = 'test'
    Train_Data_Dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    name = str(Train_Data_Dir + str(Object_Name + '_' + str(take_picture_counter+1) + ".jpg"))
    cv2.imwrite(name,img)
    print("[Save] ", name)
    take_picture_counter += 1


if __name__ == '__main__':
    
    
    global model, take_picture_counter
    take_picture_counter = 0
    parse_args()
    rospy.init_node('eval')
    sub_img = Get_image()

    test111 = Sort_data()

    torch.cuda.empty_cache()

    
    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    with torch.no_grad():
        #torch.cuda.set_per_process_memory_fraction(0.5,0)
        #torch.cuda.empty_cache()
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        dataset = None        
        
        print('Loading model...', end='')
    
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')
        
        #test weight
        '''
        tf.compat.v1.disable_eager_execution()
        tf_config=tf.compat.v1.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5 #最大使用率
        sess=tf.compat.v1.Session(config=tf_config)
        '''

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

        model = tf.keras.models.load_model('/home/chien/CNN/weight/model_0504.h5') 
        photo_array = []
        test_img1 = '/home/chien/123/test_0.jpg'#137
        photo1 = cv2.imread(test_img1,0)
        test_img2 = '/home/chien/123/test_1.jpg'
        photo2 = cv2.imread(test_img2,0)
        test_img3 = '/home/chien/123/test_2.jpg'#1307
        photo3 = cv2.imread(test_img3,0)

        photo_array.append(photo1)
        photo_array.append(photo2)
        photo_array.append(photo3)

        imagedata1 = np.array(photo_array)    
        imagedata1=imagedata1.reshape((-1,3,480,640,1))
        imagedata1=imagedata1/255.

        yhat = model.predict(imagedata1, verbose=0)  
        
        


           
        if args.cuda:
            net = net.cuda()
        #vid = cv2.VideoCapture(0)    
        net.detect.use_fast_nms = args.fast_nms
        net.detect.use_cross_class_nms = args.cross_class_nms
        cfg.mask_proto_debug = args.mask_proto_debug   

        evalvideo(sub_img.cv_image, net)
        
        #evalvideo(vid, net, args.video)
    

