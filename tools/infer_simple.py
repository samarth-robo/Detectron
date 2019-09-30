#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args(myargv=None):
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--im_or_folder', help='image or folder of images', default=None
    )
    if myargv is None:
        myargv = sys.argv[1:]
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args(args=myargv)


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=args.thresh,
            kp_thresh=args.kp_thresh,
            ext=args.output_ext,
            out_when_no_box=args.out_when_no_box
        )


class HumanKPDetector:
  def __init__(self, myargv=None, detectron_dir=os.path.expanduser(os.path.join('~', 'libraries', 'detectron'))):
    workspace.GlobalInit(['caffe2'])
    setup_logging(__name__)
    self.logger = logging.getLogger(__name__)
    args = parse_args(myargv)
    
    # config
    # model_config_filename=os.path.join(detectron_dir, 'configs/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml')
    model_config_filename=os.path.join(detectron_dir, 'configs/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml')
    args.cfg = model_config_filename
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1

    # weights
    # model_weights = 'https://dl.fbaipublicfiles.com/detectron/37732318/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_55_09.Lx8H5JVu/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
    model_weights = 'https://dl.fbaipublicfiles.com/detectron/37697714/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml.08_44_03.qrQ0ph6M/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl'
    cfg.DOWNLOAD_CACHE = os.path.join(detectron_dir, 'detectron-download-cache')
    args.weights = cache_url(model_weights, cfg.DOWNLOAD_CACHE)

    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    self.model = infer_engine.initialize_model_from_cfg(args.weights)
    self.dummy_coco_dataset = dummy_datasets.get_coco_dataset()

  def detect(self, im):
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            self.model, im, None, timers=timers
        )
    self.logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    for k, v in timers.items():
        self.logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
    im_show = vis_utils.vis_one_image_opencv(im, cls_boxes, cls_segms, cls_keyps,
        dataset=self.dummy_coco_dataset)
    boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
        cls_boxes, cls_segms, cls_keyps)
    ids = [i for i,c in enumerate(classes) if c == 1]
    valid = len(ids) > 0
    kps = -np.ones((3, 17))
    if valid:
        scores = boxes[ids, -1]
        kps = keypoints[ids[np.argmax(scores)]][:3]
    return valid, kps, im_show


if __name__ == '__main__':
    # workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    # setup_logging(__name__)
    # args = parse_args()
    # main(args)
    im = cv2.imread('../demo/15673749081_767a7fa63a_k.jpg')
    hkp = HumanKPDetector()
    _, _, im_show = hkp.detect(im)
    # import cv2
    # cv2.imshow('detectron output', im_show)
    # cv2.waitKey(0)
