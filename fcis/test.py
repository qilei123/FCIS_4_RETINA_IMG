# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Guodong Zhang, Haozhi Qi
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
import cv2
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    # general
    # configuration file is required
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--test_epoch',help='test epoch',default = 8,type=int)
    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))
config.test_epoch = args.test_epoch

import mxnet as mx
from function.test_fcis import test_fcis
from utils.create_logger import create_logger


def main():
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print ctx
    #print args
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    print 'final_output_path:'+final_output_path
    test_fcis(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path,
              ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), config.TEST.test_epoch,
              args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path)

if __name__ == '__main__':
    main()
