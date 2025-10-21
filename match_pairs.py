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
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
from ultralytics import YOLO
from models.perceptree import PercepTree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, estimate_pose_3d, estimate_scale, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, plot_3d_vectors, read_image, read_rgb_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, frame2tensor,project_images, compute_sem_match_stat, load_image_SFD2)
from models.LGutils import load_image_LG

torch.set_grad_enabled(False)

MONTH = 'woodland'
DESC = 'U-256U-256N-FN-SIFT'
#DESC = 'baseline-SIFT-SG'
TYPE = 'long'

if __name__ == '__main__':
    """ # Enable Stack Trace for warnings
    import warnings
    import traceback
    import logging
    _formatwarning = warnings.formatwarning

    def formatwarning_tb(*args, **kwargs):
        s = _formatwarning(*args, **kwargs)
        tb = traceback.format_stack()
        s += ''.join(tb[:-1])
        return s

    warnings.formatwarning = formatwarning_tb
    logging.captureWarnings(True)
    """
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--month', type=str, default=f'{MONTH}',
        help='Month which the data is from')
    parser.add_argument(
        '--desc', type=str, default=f'{DESC}',
        help='Descriptor identifier')
    parser.add_argument(
        '--input_pairs', type=str, default=f'assets/{TYPE}/{MONTH}_test_pairs_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default=f'assets/{TYPE}/{MONTH}/rgb/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default=f'dump_match_pairs/{DESC}/{MONTH}',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')
    parser.add_argument(
        '--extractor', type=str, default=f'superpoint',
        help='Keypoint extractor: superpoint,sfd2,sift,orb')
    parser.add_argument(
        '--matcher', type=str, default=f'superglue',
        help='Descriptor matcher: superglue,lightglue, nnm')
    parser.add_argument(
        '--ksi', action='store_true',
        help='Enable Keypoint Semantic Integration')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

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
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)
    
    if opt.month:
        opt.input_pairs = f'assets/{TYPE}/{opt.month}_test_pairs_gt.txt'
        opt.input_dir = f'assets/{TYPE}/{opt.month}/rgb/'
        opt.output_dir = f'dump_match_pairs/{opt.desc}/{opt.month}'
    
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

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

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))
    

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'extractor': opt.extractor,
        'matcher': opt.matcher,
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
            'ksi': opt.ksi
        },
        'sfd2': {
            'output': 'sfd2_features',
            'model': {
                'name': 'ressegnetv2',
                'use_stability': True,
                'max_keypoints': opt.max_keypoints,
                'conf_th': 0.001,
                'multiscale': False,
                'scales': [1.0],
                'model_fn': os.path.join(os.getcwd(), "models/weights/sfd2.pth"),
            },
            'preprocessing': {
                'grayscale': False,
                'resize_max': 640,
            },
            'matcher': {
                'output': 'NNM',
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                    'distance_threshold': None,
                    'match_threshold': 0.0, #Dummy value
                },
            },
            'ksi': opt.ksi,
            'keypoint_threshold': 0.001# Same as conf_th
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        },
        'NNM': {
                'output': 'NNM',
                'model': {
                    'name': 'nearest_neighbor',
                    'do_mutual_check': True,
                    'distance_threshold': None,
                    'match_threshold': 0.0, #Dummy value
                },
        }
    }
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    # Pick your panoptic segmentation model
    yolo = YOLO("./models/weights/yolo.pt").to(device)
    #perceptree = PercepTree(weights_path="./models/weights/ResNext-101_fold_01.pth", score_thresh=0.7)

    timer = AverageTimer(newline=True)
    epis = []

    conf_matrix_sum = None
    num_pairs = 0
    rp = [0.0, 0.0, 0.0]
    sem2sem = []
    bg2bg = []
    ins2ins = []

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False

                #'''
                # New Code: Calculate and save confusion matrix
                # True labels: whether the keypoints in image0 and corresponding match in image1 are foreground or background
                indexes0 = results['indexes0']
                indexes1 = results['indexes1']
                valid_matches = matches != -1  # Matches that are valid (not -1)
                
                matched_indexes0 = indexes0[valid_matches]  # Keypoints in image0 that have valid matches
                matched_indexes1 = matches[valid_matches]   # Corresponding keypoint indices in image1 from the matches
                matched_indexes1_labels = indexes1[matched_indexes1]  # Get labels of the matched keypoints in image1

                # True if both matched keypoints in image0 and image1 are in semantic (foreground) regions
                true_labels0 = matched_indexes0 >= 0
                true_labels1 = matched_indexes1_labels >= 0

                # Predicted labels: Consider matched keypoints as valid if they are in the semantic regions in both images
                predicted_labels = true_labels0 & true_labels1

                # Calculate confusion matrix
                conf_matrix = confusion_matrix(true_labels0, predicted_labels, labels=[0, 1])

                # Accumulate confusion matrices
                if conf_matrix_sum is None:
                    conf_matrix_sum = conf_matrix
                else:
                    conf_matrix_sum += conf_matrix
                num_pairs += 1

                # Create the output directory for confusion matrices if it does not exist
                conf_output_dir = Path(output_dir) / "conf_matrices"
                conf_output_dir.mkdir(exist_ok=True, parents=True)

                # Function to save confusion matrix as an image
                def save_confusion_matrix(conf_matrix, title, image_path, cmap='Blues'):
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(conf_matrix, annot=True, cmap=cmap, fmt='g')
                    plt.title(title)
                    plt.xlabel('Predicted Label')
                    plt.ylabel('True Label')
                    plt.tight_layout()
                    plt.savefig(image_path)
                    plt.close()

                # Save confusion matrices as images for each pair
                save_confusion_matrix(conf_matrix, "Confusion Matrix for Matches in Image 0",
                                    conf_output_dir / f"{stem0}_{stem1}_conf_matrix_image0.png")
                #'''

            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                do_eval = False

            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, opt.resize, rot1, opt.resize_float)
        
        # Load image pair for lightglue SIFT
        if opt.matcher == 'lightglue':
            rgb0 = load_image_LG(input_dir / name0, opt.resize).cuda()
            rgb1 = load_image_LG(input_dir / name1,opt.resize).cuda()
        elif opt.matcher == 'nnm':
            rgb0 = load_image_SFD2(input_dir / name0, opt.resize).cuda()
            rgb1 = load_image_SFD2(input_dir / name1,opt.resize).cuda()
        else:
            rgb0 = None
            rgb1 = None

        
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            #'''
            # Added for YOLO START
            resized_masks0 = None
            resized_masks1 = None

            yoloimg = cv2.imread(str(input_dir / name0))
            yoloimg = cv2.resize(yoloimg, (640, 640))
            result = yolo.predict(yoloimg,conf=0.2, classes=[0,4], verbose=False)
            #result = perceptree.predict(yoloimg, conf=0.7, classes=None, verbose=False)
            # 0-Building 1-Pipe 2-Pole 3-Robot 4-Trunk 5-Vehicle
            if result[0].masks is not None:
                masks = result[0].masks.data.cpu().numpy()
                #combined_mask = np.any(masks, axis=0).astype(np.uint8)
                #masked_img = yoloimg * combined_mask[:,:,np.newaxis]
                resized_masks0 = np.empty((masks.shape[0], 480, 640), dtype=masks.dtype)
                for j in range(masks.shape[0]):
                    resized_masks0[j] = cv2.resize(masks[j], (640, 480), interpolation=cv2.INTER_NEAREST)
                    resized_masks0[j][resized_masks0[j] == 1] = 255
                    resized_masks0[j][resized_masks0[j] != 255] = 0 

            yoloimg = cv2.imread(str(input_dir / name1))
            yoloimg = cv2.resize(yoloimg, (640, 640))
            result = yolo.predict(yoloimg,conf=0.2, classes=[0,4], verbose=False) 
            #result = perceptree.predict(yoloimg, conf=0.7, classes=None, verbose=False)
            # 0-Building 1-Pipe 2-Pole 3-Robot 4-Trunk 5-Vehicle
            if result[0].masks is not None:
                masks = result[0].masks.data.cpu().numpy()
                #combined_mask = np.any(masks, axis=0).astype(np.uint8)
                #masked_img = yoloimg * combined_mask[:,:,np.newaxis]
                resized_masks1 = np.empty((masks.shape[0], 480, 640), dtype=masks.dtype)
                for j in range(masks.shape[0]):
                    resized_masks1[j] = cv2.resize(masks[j], (640, 480), interpolation=cv2.INTER_NEAREST)
                    resized_masks1[j][resized_masks1[j] == 1] = 255
                    resized_masks1[j][resized_masks1[j] != 255] = 0
                    
            timer.update('YOLO')
            # Added for YOLO END
            #Note: Utils line 428 was added to set z=0.0
            #'''
            pred = matching({'image0': inp0, 'image1': inp1, 'gs0': image0, 'gs1': image1, 'rgb0': rgb0, 'rgb1': rgb1}, resized_masks0, resized_masks1)
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            indexes0 = pred['indexes0']
            indexes1 = pred['indexes1']
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf,
                            'indexes0':indexes0, 'indexes1':indexes1}
            np.savez(str(matches_path), **out_matches)
            #'''
            # New Code: Calculate and save confusion matrix
            # True labels: whether the keypoints in image0 and corresponding match in image1 are foreground or background
            valid_matches = matches != -1  # Matches that are valid (not -1)
            matched_indexes0 = indexes0[valid_matches]  # Keypoints in image0 that have valid matches
            matched_indexes1 = matches[valid_matches]   # Corresponding keypoint indices in image1 from the matches
            matched_indexes1_labels = indexes1[matched_indexes1]  # Get labels of the matched keypoints in image1
            # True if both matched keypoints in image0 and image1 are in semantic (foreground) regions
            true_labels0 = matched_indexes0 >= 0
            true_labels1 = matched_indexes1_labels >= 0

            # Predicted labels: Consider matched keypoints as valid if they are in the semantic regions in both images
            predicted_labels = true_labels0 & true_labels1

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(true_labels0, predicted_labels, labels=[0, 1])

            # Accumulate confusion matrices
            if conf_matrix_sum is None:
                conf_matrix_sum = conf_matrix
            else:
                conf_matrix_sum += conf_matrix
            num_pairs += 1

            # Create the output directory for confusion matrices if it does not exist
            conf_output_dir = Path(output_dir) / "conf_matrices"
            conf_output_dir.mkdir(exist_ok=True, parents=True)

            # Function to save confusion matrix as an image
            def save_confusion_matrix(conf_matrix, title, image_path, cmap='Blues'):
                plt.figure(figsize=(5, 4))
                sns.heatmap(conf_matrix, annot=True, cmap=cmap, fmt='g')
                plt.title(title)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.savefig(image_path)
                plt.close()

            # Save confusion matrices as images for each pair
            save_confusion_matrix(conf_matrix, "Confusion Matrix for Matches in Image 0",
                                conf_output_dir / f"{stem0}_{stem1}_conf_matrix_image0.png")
            #'''
        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        if do_eval:
            # Estimate the pose and compute the pose error.
            assert len(pair) == 38, 'Pair does not have ground truth info'
            K0_original = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1_original = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale the intrinsics to resized image.
            K0 = scale_intrinsics(K0_original, scales0)
            K1 = scale_intrinsics(K1_original, scales1)

            # Update the intrinsics + extrinsics if EXIF rotation was found.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0

            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            epis.append(np.mean(epi_errs))
            correct = epi_errs < 5e-4 # 2e-3 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            thresh = 1.  # In pixels relative to resized image size.

            # Get depth based pose estimation
            depth0 = cv2.imread(str(input_dir).replace("rgb", "depth")+"/"+name0.replace("color", "depth"),cv2.IMREAD_GRAYSCALE)
            depth1 = cv2.imread(str(input_dir).replace("rgb", "depth")+"/"+name1.replace("color", "depth"),cv2.IMREAD_GRAYSCALE)
            #ret = estimate_pose_3d(mkpts0, mkpts1, depth0, depth1, K0_original, K1_original, scales0, scales1)
            #plot_pointcloud_with_rgb(image0,depth0,K0)
            
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None:
                err_t, err_R = np.inf, np.inf
                R = np.eye(3)
                t = np.zeros(3)
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,    
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)

            _, iou, iou_indexes = project_images(resized_masks0, T_0to1[:3, :3], T_0to1[:3, 3], K0, resized_masks1, False)    
            stat = compute_sem_match_stat(kpts0, kpts1,indexes0, indexes1, iou_indexes, matches)     
            sem2sem.append(stat['semantics_to_semantics_pct'])
            bg2bg.append(stat['background_to_background_pct'])
            ins2ins.append(stat['correct_mask_pct'])          

            # Convert ground truth and recovered rotation/translation to transformation matrices
            gt_pose = T_0to1
            _, t_scaled = estimate_scale(mkpts0, mkpts1, depth0, depth1, scales0, scales1, K0_original, K1_original, R, t, abs(np.linalg.norm(gt_pose[:3,3])))

            if t_scaled is None:
                ret = None
            if ret is not None:
                recovered_pose = np.eye(4)
                recovered_pose[:3, :3] = R
                recovered_pose[:3, 3] = t_scaled
                rp += recovered_pose[:3, 3]
                #plot_3d_vectors(t_scaled, gt_pose[:3,3])
            else:
                recovered_pose = None

            # Save ground truth and recovered poses to an npz file
            output_pose_data = {
                'ground_truth_pose': gt_pose
            }
            if recovered_pose is not None:
                output_pose_data['recovered_pose'] = recovered_pose

            np.savez(str(output_dir / '{}_{}_poses.npz'.format(stem0, stem1)), **output_pose_data)

            timer.update('eval')

        if do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.extractor.config['keypoint_threshold']
            m_thresh = matching.matcher.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        if do_viz_eval:
            # Visualize the evaluation results for the image pair.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info (only works with --fast_viz).
            k_thresh = matching.extractor.config['keypoint_threshold']
            m_thresh = matching.matcher.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]
            
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)

            timer.update('viz_eval')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
    #'''
    # After processing all pairs, compute and save the average confusion matrix
    if num_pairs > 0:
        avg_conf_matrix = conf_matrix_sum / num_pairs

        # Save the average confusion matrices with a different colormap ('viridis')
        save_confusion_matrix(avg_conf_matrix, "Average Confusion Matrix for Matches in Image 0",
                            conf_output_dir / "average_conf_matrix_image0.png", cmap='viridis')
    # Create a DataFrame from the lists
    data = {
        'Semantics_to_Semantics_Percentage': sem2sem,
        'Background_to_Background_Percentage': bg2bg,
        'Correct_Mask_Percentage': ins2ins
    }
    df = pd.DataFrame(data)
    # Save the DataFrame as a CSV file
    csv_filename = f'{os.path.dirname(opt.output_dir)}/{opt.month}_{opt.desc}_semantic_match_statistics.csv'
    df.to_csv(csv_filename, index=False)

    #'''
    if opt.eval:
        # Collate the results into a final table and print to terminal.
        pose_errors = []
        precisions = []
        matching_scores = []
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.*yy for yy in aucs]
        prec = 100.*np.mean(precisions)
        ms = 100.*np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))
