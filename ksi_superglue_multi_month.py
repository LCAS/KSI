#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import cv2
import cv2.mat_wrapper
import matplotlib.cm as cm
import torch
from ultralytics import YOLO
import numpy as np

from models.descriptor_encoder import *
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, make_matching_plot_fast_rgb, frame2tensor)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    yolo = YOLO("./models/weights/yolo.pt").to('cpu')
    
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors', 'indexes']

    vs_ref = VideoStreamer('assets/matches/blt2/march', opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs_ref.next_frame()
    #frame, ret = vs.next_frame()

    #frame = cv2.imread("assets/long/march/rgb/march_color_image_0.png")
    #frame = cv2.resize(frame,(640,480),cv2.INTER_AREA)

    rgb0 = frame
    assert ret, 'Error when reading the first frame (try different --input?)'

    yoloimg = frame #cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
    yoloimg = cv2.resize(yoloimg, (640, 640))
    '''
    background = yolo.predict(yoloimg,conf=0.2, classes=[0,1], verbose=False)
    if background[0].masks is not None:
        masks = np.array(background[0].masks.data.cpu())
        sem_background = np.any(masks, axis=0).astype(np.uint8)
        sem_background = cv2.resize(sem_background, (128, 128))
        sem_background[sem_background == 1] = 255 
    '''
    result = yolo.predict(yoloimg,conf=0.2, classes=[0, 4], verbose=False)
    if result[0].masks is not None:
        masks = np.array(result[0].masks.data.cpu())
        combined_mask0 = np.any(masks, axis=0).astype(np.uint8)
        masked_img = yoloimg * combined_mask0[:,:,np.newaxis]
        frame = cv2.resize(masked_img, (640, 480))
        resized_masks0 = np.empty((masks.shape[0], 480, 640), dtype=masks.dtype)
        for i in range(masks.shape[0]):
            resized_masks0[i] = cv2.resize(masks[i], (640, 480), interpolation=cv2.INTER_NEAREST)
            resized_masks0[i][resized_masks0[i] == 1] = 255
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #semantic_encoder = Semantic_DescEncoder()
    frame_tensor = frame2tensor(cv2.cvtColor(rgb0,cv2.COLOR_RGB2GRAY), device)

    last_data = matching.superpoint({'image': frame_tensor}, resized_masks0)
    #last_data = semantic_encoder.encode(frame_tensor,sem_background)
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = vs_ref.i

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()
    total_kp = 0
    total_confidence = 0
    total_zeros = 0

    while True:
        frame, ret = vs.next_frame()
        rgb1 = frame
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = vs_ref.i - 1, vs.i - 1
        eval_path = Path(opt.output_dir) / '{}_{}_evaluation.npz'.format(stem0, stem1)
        
        yoloimg = frame #cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        yoloimg = cv2.resize(yoloimg, (640, 640))

        '''
        background = yolo.predict(yoloimg,conf=0.2, classes=[0,1], verbose=False)
        if background[0].masks is not None:
            masks = np.array(background[0].masks.data.cpu())
            sem_background = np.any(masks, axis=0).astype(np.uint8)
            sem_background = cv2.resize(sem_background, (128, 128))
            sem_background[sem_background == 1] = 255
        '''
        result = yolo.predict(yoloimg,conf=0.2, classes=[0, 4], verbose=False)
        if result[0].masks is not None:
            masks = np.array(result[0].masks.data.cpu())
            combined_mask1 = np.any(masks, axis=0).astype(np.uint8)
            #combined_mask[combined_mask == 1] = 255
            masked_img = yoloimg * combined_mask1[:,:,np.newaxis]
            frame = cv2.resize(masked_img, (640, 480))
            resized_masks1 = np.empty((masks.shape[0], 480, 640), dtype=masks.dtype)
            for i in range(masks.shape[0]):
                resized_masks1[i] = cv2.resize(masks[i], (640, 480), interpolation=cv2.INTER_NEAREST)
                resized_masks1[i][resized_masks1[i] == 1] = 255
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)         
        else: 
            continue

        frame_tensor = frame2tensor(cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY), device)

        pred = matching({**last_data, 'image1': frame_tensor},resized_masks0,resized_masks1)
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        # Add valid parameter to semantic keypoints only
        #'''
        sem = last_data['indexes0'][0].cpu().numpy()
        valid1 = sem > -1
        valid2 = matches > -1
        valid = valid1 & valid2
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])

        '''
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        #'''
        if len(mkpts1) != 0:
            total_kp += len(mkpts1)
            total_confidence += np.mean(confidence[valid])
        else:
            total_zeros += 1

        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        #out = make_matching_plot_fast(
            #last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            #path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
        out = make_matching_plot_fast_rgb(
            rgb0, rgb1, combined_mask0, combined_mask1, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
        
        # Write the evaluation results to disk.
        out_eval = {'kpts0': kpts0,
                    'kpts1': kpts1,
                    'mkpts0': mkpts0,
                    'mkpts1': mkpts1,
                    'matches': matches,
                    'confidence': confidence}
        #np.savez(str(eval_path), **out_eval)

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        # Update last_data and last_frame for the next iteration
        #"""
        ##########################################
        frame, ret = vs_ref.next_frame()
        rgb1 = frame
        if not ret:
            print('Finished demo_superglue.py')
            break
        
        yoloimg = frame #cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        yoloimg = cv2.resize(yoloimg, (640, 640))

        result = yolo.predict(yoloimg,conf=0.2, classes=[0, 4], verbose=False)
        if result[0].masks is not None:
            masks = np.array(result[0].masks.data.cpu())
            combined_mask1 = np.any(masks, axis=0).astype(np.uint8)
            #combined_mask[combined_mask == 1] = 255
            masked_img = yoloimg * combined_mask1[:,:,np.newaxis]
            frame = cv2.resize(masked_img, (640, 480))
            resized_masks1 = np.empty((masks.shape[0], 480, 640), dtype=masks.dtype)
            for i in range(masks.shape[0]):
                resized_masks1[i] = cv2.resize(masks[i], (640, 480), interpolation=cv2.INTER_NEAREST)
                resized_masks1[i][resized_masks1[i] == 1] = 255
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)         
        else: 
            continue

        frame_tensor = frame2tensor(cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY), device)

        pred = matching({**last_data, 'image1': frame_tensor},resized_masks0,resized_masks1)
        ##############################################
        last_data = {k+'0': pred[k+'1'] for k in keys}
        last_data['image0'] = frame_tensor
        last_frame = frame
        last_image_id = (vs_ref.i - 1)
        rgb0 = rgb1
        combined_mask0 = combined_mask1
        #"""

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    print("Average Keypoints:", total_kp/88)
    print("Average Confidence:", total_confidence/88)
    print("Total Zeros:", total_zeros)
    cv2.destroyAllWindows()
    vs.cleanup()
