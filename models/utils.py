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
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import re
matplotlib.use('TkAgg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            # Custom sort key to handle natural ordering
            def natural_key(path):
                parts = re.split(r'(\d+)', str(path))  # Split into text and number
                return [int(part) if part.isdigit() else part for part in parts]
            self.listing.sort(key=natural_key)
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim
    
    def load_rgb_image(self, impath):
        """ Read image as RGB and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            rgbim: numpy array sized H x W X 3.
        """
        rgbim = cv2.imread(impath)
        if rgbim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = rgbim.shape[1], rgbim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        rgbim = cv2.resize(
            rgbim, (w_new, h_new), interpolation=self.interp)
        return rgbim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_rgb_image(image_file) # Updated Load funciton
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def read_rgb_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path))
    
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new, 3))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    
    inp = frame2tensor(image, device)
    return image, inp, scales

def read_image(path, device, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales

def load_image_SFD2(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = image.astype(np.float32)
    image = image[:, :, ::-1]  # BGR to RGB
    image = image.astype(np.float32)
    if resize is not None:
        image = cv2.resize(image, (resize[0], resize[1]), interpolation=cv2.INTER_CUBIC)
    return numpy_image_to_torch(image)

# --- GEOMETRY ---

def project_images(input_images, rotation_matrices, translation_vectors, camera_matrix, expected_images=None, viz=False):
    """
    Projects a set of input binarized grayscale images based on given rotation, translation matrices, and camera intrinsic matrix.
    For each input image, computes IoU with a stack of expected images and visualizes the one with the best IoU.
    If the best IoU is below the threshold (0.1), returns -1 as the best index.
    When a match is found, the matched expected image is removed from the expected_images internally.

    Args:
        input_images (np.ndarray): A stack of input binarized grayscale images as a 3D NumPy array of shape (M, H, W).
                                   Each image should have values 0 or 255.
        rotation_matrices (np.ndarray): Either a single 3x3 rotation matrix or a stack of rotation matrices of shape (M, 3, 3).
        translation_vectors (np.ndarray): Either a single 3x1 translation vector or a stack of translation vectors of shape (M, 3, 1).
        camera_matrix (np.ndarray): A 3x3 intrinsic camera matrix.
        expected_images (np.ndarray, optional): A stack of expected binarized grayscale images to compare with the projected images.
                                                Should be a 3D NumPy array of shape (N, H, W), where N is the number of images.
        viz (bool): If True, visualize the original, projected, and best matching expected image with IoU overlay.

    Returns:
        list: A list of projected binarized grayscale images.
        list: A list of best IoU values for each input image.
        list: A list of indexes of the expected images with the best IoU (or -1 if below threshold).
    """
    if input_images is None:
        print("Warning: input_images is None, skipping processing.")
        return [], [], []  # Or some other default value indicating no operation was performed
    
    # IoU threshold
    IOU_THRESHOLD = 0.1

    # Ensure the input_images is a 3D array
    if len(input_images.shape) != 3:
        raise ValueError("Input images must be a stack of images with shape (M, H, W).")

    num_inputs = input_images.shape[0]
    height, width = input_images.shape[1], input_images.shape[2]

    # Handle single rotation matrix and translation vector
    if rotation_matrices.shape == (3, 3):
        # Replicate the rotation matrix for all input images
        rotation_matrices = np.tile(rotation_matrices, (num_inputs, 1, 1))
    elif rotation_matrices.shape != (num_inputs, 3, 3):
        raise ValueError(f"Rotation matrices must have shape ({num_inputs}, 3, 3) or (3, 3).")

    if translation_vectors.shape in [(3,), (3, 1)]:
        translation_vectors = translation_vectors.reshape(1, 3, 1)
        # Replicate the translation vector for all input images
        translation_vectors = np.tile(translation_vectors, (num_inputs, 1, 1))
    elif translation_vectors.shape != (num_inputs, 3, 1):
        raise ValueError(f"Translation vectors must have shape ({num_inputs}, 3, 1) or (3, 1).")

    # Initialize lists to store results
    projected_images = []
    best_ious = []
    best_indexes = []

    # Keep track of the original indices of expected images
    if expected_images is not None:
        expected_indices = list(range(expected_images.shape[0]))
        # Convert expected_images to a list for internal management
        expected_images_list = [expected_images[i] for i in range(expected_images.shape[0])]
    else:
        expected_indices = []
        expected_images_list = []

    for i in range(num_inputs):
        input_image = input_images[i]
        rotation_matrix = rotation_matrices[i]
        translation_vector = translation_vectors[i]

        # Ensure the image is grayscale and binarized
        if len(input_image.shape) != 2:
            raise ValueError(f"Input image at index {i} must be a grayscale image.")

        unique_vals = np.unique(input_image)
        if not (np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]) or np.array_equal(unique_vals, [0, 255])):
            raise ValueError(f"Input image at index {i} must be binarized with values 0 and 255.")

        # Generate a grid of (x, y, 1) homogeneous coordinates
        y_coords, x_coords = np.indices((height, width))
        ones = np.ones_like(x_coords)
        homogeneous_coords = np.stack((x_coords, y_coords, ones), axis=-1).reshape(-1, 3).T  # shape: (3, N)

        # Convert pixel coordinates to normalized camera coordinates
        inv_K = np.linalg.inv(camera_matrix)
        normalized_coords = inv_K @ homogeneous_coords  # shape: (3, N)

        # Assume the plane is at Z = 0
        R_2x2 = rotation_matrix[:2, :2]
        # Ensure t_2x1 has shape (2, 1)
        t_2x1 = translation_vector[:2].reshape(2, 1)

        # Perform the coordinate transformation
        transformed_coords = R_2x2 @ normalized_coords[:2, :] + t_2x1

        # Project back to pixel coordinates
        projected_coords = camera_matrix[:2, :2] @ transformed_coords + camera_matrix[:2, 2:3]

        # Reshape to image shape
        map_x = projected_coords[0, :].reshape(height, width).astype(np.float32)
        map_y = projected_coords[1, :].reshape(height, width).astype(np.float32)

        # Remap the image
        projected_image = cv2.remap(input_image, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        projected_images.append(projected_image)

        best_iou = 0
        best_index = -1
        best_expected_image = None

        if expected_images_list:
            num_expected = len(expected_images_list)

            # Ensure all expected images have the correct size
            for idx in range(num_expected):
                exp_img = expected_images_list[idx]
                if exp_img.shape != (height, width):
                    raise ValueError(f"Expected image at index {expected_indices[idx]} must have the same size as the input image.")

                # Validate that exp_img is binarized
                unique_vals = np.unique(exp_img)
                if not (np.array_equal(unique_vals, [0]) or np.array_equal(unique_vals, [255]) or np.array_equal(unique_vals, [0, 255])):
                    raise ValueError(f"Expected image at index {expected_indices[idx]} must be binarized with values 0 and 255.")

                # Compute Intersection and Union
                intersection = cv2.bitwise_and(projected_image, exp_img)
                union = cv2.bitwise_or(projected_image, exp_img)

                # Calculate IoU
                intersection_count = np.count_nonzero(intersection)
                union_count = np.count_nonzero(union)

                # Handle the case where both masks are empty (union_count == 0)
                if union_count == 0:
                    if np.count_nonzero(projected_image) == 0 and np.count_nonzero(exp_img) == 0:
                        iou = 1.0  # Both masks are empty; define IoU as 1
                    else:
                        iou = 0.0  # One mask is empty; define IoU as 0
                else:
                    iou = intersection_count / union_count

                # Update best IoU and expected image
                if iou > best_iou:
                    best_iou = iou
                    best_index = idx  # Index within expected_images_list
                    best_expected_image = exp_img

            # Apply IoU threshold
            if best_iou < IOU_THRESHOLD:
                best_index = -1  # No expected image meets the threshold
                best_expected_image = None  # No visualization for below-threshold match
            else:
                # Remove the matched expected image and its index
                matched_expected_index = expected_indices.pop(best_index)
                expected_images_list.pop(best_index)
                best_index = matched_expected_index  # Use the original index
        else:
            best_index = -1
            best_iou = 0
            best_expected_image = None

        best_ious.append(best_iou)
        best_indexes.append(best_index)

        if viz:
            if best_expected_image is not None:
                # Visualization code for matched image
                overlay = np.zeros((height, width, 3), dtype=np.uint8)

                # Assign colors: projected_image in red, best_expected_image in green, intersection in yellow
                overlay[(projected_image == 255) & (best_expected_image == 0)] = [0, 0, 255]        # Red
                overlay[(projected_image == 0) & (best_expected_image == 255)] = [0, 255, 0]        # Green
                overlay[(projected_image == 255) & (best_expected_image == 255)] = [0, 255, 255]    # Yellow

                # Add IoU text to the overlay image
                cv2.putText(overlay, f'Best IoU: {best_iou:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, f'Best Index: {best_index}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Convert images to color for visualization
                input_color = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                expected_color = cv2.cvtColor(best_expected_image, cv2.COLOR_GRAY2BGR)
                projected_color = cv2.cvtColor(projected_image, cv2.COLOR_GRAY2BGR)

                # Add labels to each image
                cv2.putText(input_color, 'Input Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(expected_color, 'Expected Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(projected_color, 'Projected Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(overlay, 'Overlay', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Stack images side by side for visualization
                top_row = np.hstack((input_color, expected_color))
                bottom_row = np.hstack((projected_color, overlay))
                combined_image = np.vstack((top_row, bottom_row))

                # Resize combined image if it's too large
                screen_res = 1280, 720
                scale_width = screen_res[0] / combined_image.shape[1]
                scale_height = screen_res[1] / combined_image.shape[0]
                scale = min(scale_width, scale_height, 1)
                window_width = int(combined_image.shape[1] * scale)
                window_height = int(combined_image.shape[0] * scale)
                combined_image_resized = cv2.resize(combined_image, (window_width, window_height))

                cv2.imshow(f"Visualization for Input Image {i}", combined_image_resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                # Visualization code when no match is found
                input_color = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                projected_color = cv2.cvtColor(projected_image, cv2.COLOR_GRAY2BGR)

                # Add labels to each image
                cv2.putText(input_color, 'Input Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(projected_color, 'Projected Image', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

                # Create a blank expected image and overlay for visualization
                expected_color = np.zeros_like(input_color)
                overlay = np.zeros_like(input_color)

                # Add labels
                cv2.putText(expected_color, 'No Match', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(overlay, 'No Overlay', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

                # Stack images side by side for visualization
                top_row = np.hstack((input_color, expected_color))
                bottom_row = np.hstack((projected_color, overlay))
                combined_image = np.vstack((top_row, bottom_row))

                # Resize combined image if it's too large
                screen_res = 1280, 720
                scale_width = screen_res[0] / combined_image.shape[1]
                scale_height = screen_res[1] / combined_image.shape[0]
                scale = min(scale_width, scale_height, 1)
                window_width = int(combined_image.shape[1] * scale)
                window_height = int(combined_image.shape[0] * scale)
                combined_image_resized = cv2.resize(combined_image, (window_width, window_height))

                cv2.imshow(f"Visualization for Input Image {i}", combined_image_resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return projected_images, best_ious, best_indexes

def estimate_scale(keypoints0, keypoints1, depths0, depths1, scales0, scales1, K0, K1, R, t, S=None):
    """
    Estimate the scale factor for the translation vector using depth maps.

    Parameters:
    - keypoints0: Nx2 array of keypoints in image 0 (resized pixel coordinates).
    - keypoints1: Nx2 array of keypoints in image 1 (resized pixel coordinates).
    - depths0: 2D numpy array of depth values for image 0 (original resolution).
    - depths1: 2D numpy array of depth values for image 1 (original resolution).
    - scales0: Tuple or list of scaling factors (scale_x0, scale_y0) for image 0.
    - scales1: Tuple or list of scaling factors (scale_x1, scale_y1) for image 1.
    - K0: Camera intrinsic matrix for image 0 (3x3 numpy array).
    - K1: Camera intrinsic matrix for image 1 (3x3 numpy array).
    - R: Estimated rotation matrix from pose estimation (3x3 numpy array).
    - t: Estimated translation vector from pose estimation (3-element numpy array).

    Returns:
    - s: Estimated scale factor (float).
    - t_scaled: Scaled translation vector (3-element numpy array).
    """

    # Ensure inputs are numpy arrays
    keypoints0 = np.asarray(keypoints0)
    keypoints1 = np.asarray(keypoints1)
    depths0 = np.asarray(depths0)
    depths1 = np.asarray(depths1)
    scales0 = np.asarray(scales0)
    scales1 = np.asarray(scales1)
    t = np.asarray(t).flatten()  # Ensure t is a 1D array of shape (3,)

    # Invert the intrinsic matrices
    K0_inv = np.linalg.inv(K0)
    K1_inv = np.linalg.inv(K1)

    numerator = 0.0
    denominator = 0.0
    valid_points = 0  # Count of valid points used in estimation
    N = keypoints0.shape[0]

    for i in range(N):
        # Keypoints in resized coordinate frame
        u0_resized, v0_resized = keypoints0[i]
        u1_resized, v1_resized = keypoints1[i]

        # Map keypoints to original image coordinate frame
        u0_orig = int(u0_resized * scales0[0])
        v0_orig = int(v0_resized * scales0[1])
        u1_orig = int(u1_resized * scales1[0])
        v1_orig = int(v1_resized * scales1[1])

        # Ensure pixel indices are within the image bounds
        h0, w0 = depths0.shape
        h1, w1 = depths1.shape
        if not (0 <= u0_orig < w0 and 0 <= v0_orig < h0 and
                0 <= u1_orig < w1 and 0 <= v1_orig < h1):
            continue  # Skip if the keypoint is outside the image bounds

        # Get depth values from the depth maps at the original resolution
        z0 = depths0[v0_orig, u0_orig]
        z1 = depths1[v1_orig, u1_orig]

        # Skip if depth values are invalid (e.g., zero or NaN)
        if z0 <= 0 or np.isnan(z0) or z1 <= 0 or np.isnan(z1):
            continue

        # Construct scaled homogeneous coordinates
        point0_h_scaled = np.array([u0_orig, v0_orig, 1.0])
        point1_h_scaled = np.array([u1_orig, v1_orig, 1.0])

        # Reconstruct 3D points in camera coordinate system
        X0 = K0_inv @ point0_h_scaled * z0
        X1 = K1_inv @ point1_h_scaled * z1

        # Transform X1 to the coordinate system of the first camera
        X1_transformed = R @ X1

        # Compute the difference vector
        delta_X = (X0 - X1_transformed)

        # Accumulate numerator and denominator for scale estimation
        numerator += np.dot(t, delta_X)
        denominator += np.dot(t, t)

        valid_points += 1

    # Check if denominator is zero to avoid division by zero
    if denominator == 0 or valid_points == 0:
        return None, None
        #raise ValueError("No valid points found; cannot compute scale factor.")

    # Compute the scale factor s
    s = numerator / denominator  # Result is a scalar
    if S:
        s = S

    # Compute the scaled translation vector
    t_scaled = (s) * t  # t_scaled is now a 1D array of shape (3,)
    #t_scaled = np.array([[0,0,1],[-1,0,0],[0,0,0]]) @ t_scaled 
    #print(np.linalg.norm(t))
    #print(np.linalg.norm(t_scaled))

    return s, t_scaled

def estimate_pose_3d(kpts0, kpts1, depth_map0, depth_map1, K0, K1, scales0, scales1, threshold=0.1):
    # Convert kpts0 and kpts1 to 3D points using respective depth maps
    points_3d_0 = []
    points_3d_1 = []
    valid_kpts1 = []
    mask = []

    for pt0, pt1 in zip(kpts0, kpts1):
        # Get (u, v) coordinates
        u0, v0 = int(pt0[0]*scales0[0]), int(pt0[1]*scales0[1])
        u1, v1 = int(pt1[0]*scales1[0]), int(pt1[1]*scales1[1])

        # Get depths for each point
        z0 = depth_map0[v0, u0]
        z1 = depth_map1[v1, u1]

        # Check for valid depth values
        if z0 > 0 and z1 > 0:
            # Convert to 3D coordinates using the depth maps and camera intrinsics
            x0 = (u0 - K0[0, 2]) * z0 / K0[0, 0]
            y0 = (v0 - K0[1, 2]) * z0 / K0[1, 1]
            points_3d_0.append([x0, y0, z0])

            x1 = (u1 - K1[0, 2]) * z1 / K1[0, 0]
            y1 = (v1 - K1[1, 2]) * z1 / K1[1, 1]
            points_3d_1.append([x1, y1, z1])

            valid_kpts1.append(pt1*scales1)

            # Assume initially that all points are inliers
            mask.append(1)
        else:
            # Mark as outlier if depth is missing
            mask.append(0)

    # Convert to NumPy arrays
    points_3d_0 = np.array(points_3d_0, dtype=np.float32)
    points_3d_1 = np.array(points_3d_1, dtype=np.float32)
    valid_kpts1 = np.array(valid_kpts1, dtype=np.float32)

    # Check for minimum number of points
    if len(points_3d_0) < 4:
        print("Not enough valid point correspondences to compute pose.")
        return None

    # Apply initial pose estimation using all valid points
    _, rvec, tvec = cv2.solvePnP(points_3d_0, valid_kpts1, K1, None)
    R, _ = cv2.Rodrigues(rvec)

    # Apply rotation and translation to points_3d_0 to project it to the second frame
    points_3d_0_transformed = (R @ points_3d_0.T).T + tvec.T

    # Check distances between transformed points_3d_0 and points_3d_1
    distances = np.linalg.norm(points_3d_0_transformed - points_3d_1, axis=1)

    # Update mask based on threshold for inliers
    mask = np.array(mask)
    inlier_mask = distances < threshold
    mask[mask == 1] = inlier_mask.astype(int)

    # Ensure there are enough inliers
    if len(points_3d_0) < 4:
        print("Not enough inliers after masking to compute pose.")
        return None

    # Re-estimate pose using inliers
    _, rvec, tvec = cv2.solvePnP(points_3d_0, valid_kpts1, K1, None)
    R, _ = cv2.Rodrigues(rvec)
    #visualize_3d_points(points_3d_0, points_3d_1, points_3d_0_transformed)
    ret = (R, tvec.flatten(), mask)
    return ret

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    #t[2] = 0.0 # Added by Rajitha, In GT z transition is forced to 0
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---

def plot_3d_vectors(t1, t2):
    """
    Plots two 3D vectors as arrows starting from the origin.
    
    Parameters:
    t1, t2: Lists or arrays with 3 elements each representing the vectors in 3D space.
    """
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define origin
    origin = [0, 0, 0]
    
    # Plot vector t1
    ax.quiver(*origin, t1[0], t1[1], t1[2], color='blue', label='t1', linewidth=2, arrow_length_ratio=0.1)
    
    # Plot vector t2
    ax.quiver(*origin, t2[0], t2[1], t2[2], color='red', label='t2', linewidth=2, arrow_length_ratio=0.1)
    
    # Set plot limits for better visualization
    max_range = max(np.linalg.norm(t1), np.linalg.norm(t2))
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Show the plot
    plt.show()

def plot_pointcloud_with_rgb(image, depth_map, K):
    """
    Plots a 3D point cloud with RGB colors from an RGB image and depth map.

    Parameters:
    - image: RGB image as a (H, W, 3) array.
    - depth_map: Depth map as a (H, W) array, with depth in meters.
    - K: Camera intrinsic matrix as a (3, 3) array.
    """
    # Prepare lists for storing point cloud data
    points_3d = []
    colors = []
    
    # Get image dimensions
    h, w = depth_map.shape
    
    # Intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]  # focal lengths
    cx, cy = K[0, 2], K[1, 2]  # principal point

    # Convert each pixel to a 3D point with RGB
    for v in range(h):
        for u in range(w):
            z = depth_map[v, u]  # Depth value at pixel (u, v)
            
            if z > 0:  # Only consider points with valid depth
                # Calculate 3D coordinates
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points_3d.append([x, y, z])
                colors.append(image[v, u] / 255.0)  # Normalize RGB values to [0, 1]

    # Convert to numpy arrays
    points_3d = np.array(points_3d)
    colors = np.array(colors)
    
    # Plot the 3D point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=colors, marker='o', s=1)
    
    # Labeling and viewing settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Point Cloud with RGB Colors")
    plt.show()

def visualize_3d_points(points_3d_0, points_3d_1=None, points_3d_0_transformed=None, mask=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the initial set of points
    ax.scatter(points_3d_0[:, 0], points_3d_0[:, 1], points_3d_0[:, 2], c='b', marker='o', label='Points 3D 0')
    
    # Plot the second set of points
    if points_3d_1 is not None:
        ax.scatter(points_3d_1[:, 0], points_3d_1[:, 1], points_3d_1[:, 2], c='r', marker='^', label='Points 3D 1')
    
    # Plot the transformed points if provided
    if points_3d_0_transformed is not None:
        ax.scatter(points_3d_0_transformed[:, 0], points_3d_0_transformed[:, 1], points_3d_0_transformed[:, 2], 
                   c='g', marker='x', label='Transformed Points 3D 0')
    
    # Optional: Highlight inliers
    if mask is not None:
        inliers_0 = points_3d_0[mask == 1]
        inliers_1 = points_3d_1[mask == 1]
        ax.scatter(inliers_0[:, 0], inliers_0[:, 1], inliers_0[:, 2], c='cyan', marker='o', s=50, label='Inliers 3D 0')
        ax.scatter(inliers_1[:, 0], inliers_1[:, 1], inliers_1[:, 2], c='magenta', marker='^', s=50, label='Inliers 3D 1')

    # Labeling and viewing settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Point Visualization")
    plt.show()

def plot_image_pair_horizontal(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*3/4, size*n*0.5) if size is not None else None  # Adjust for vertical layout
    _, ax = plt.subplots(n, 1, figsize=figsize, dpi=dpi)  # Change to n rows, 1 column
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H = H0 + H1 + margin  # Total height by stacking
    W = max(W0, W1)       # Width is the max of both

    # Create an empty output image with the correct dimensions
    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0  # Place image0 at the top
    out[H0+margin:H0+margin+H1, :W1] = image1  # Place image1 at the bottom
    out = np.stack([out]*3, -1)
    
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x, y + H0 + margin), 2, black, -1,  # Shift y-coordinates
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y + H0 + margin), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1, y1 + H0 + margin),  # Shift y1 to match vertical stacking
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # Display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1, y1 + H0 + margin), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(W / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # Text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # Text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

# Apply the mask with transparency and darkening
def apply_mask(image, mask, alpha):
    # Reshape the mask to 480x640
    mask_resized = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)

    # Convert mask from single-channel to three-channel for RGB image
    mask_3d = np.repeat(mask_resized[:, :, np.newaxis], 3, axis=2)

    # Darken areas where mask is 0, and leave areas where mask is 1 unchanged
    darkened_image = (image * alpha).astype(np.uint8)
    masked_image = np.where(mask_3d == 1, image, darkened_image)

    return masked_image

    
def make_matching_plot_fast_rgb(image0, image1, mask0, mask1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[], alpha=0.5):
    H0, W0, _ = image0.shape
    H1, W1, _ = image1.shape
    H = H0 + H1 + margin  # Total height by stacking
    W = max(W0, W1)       # Width is the max of both

    # Apply mask to both images
    image0 = apply_mask(image0, mask0, alpha)
    image1 = apply_mask(image1, mask1, alpha)

    # Create an empty output image with the correct dimensions
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0  # Place image0 at the top
    out[H0+margin:H0+margin+H1, :W1, :] = image1  # Place image1 at the bottom
    
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x, y + H0 + margin), 2, black, -1,  # Shift y-coordinates
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y + H0 + margin), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    #print(len(mkpts0), len(mkpts1))
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1, y1 + H0 + margin),  # Shift y1 to match vertical stacking
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # Display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1, y1 + H0 + margin), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(W / 640., 2.0)
    '''
    # Big text.
    Ht = int(30 * sc)  # Text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # Text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)
    '''
    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

def make_matching_plot_fast_horizontal(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0, _ = image0.shape
    H1, W1, _ = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1
    #out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out

def DimSqueeze(arr):
    if arr.dim() == 1:
        return arr.shape[0]
    else:
        return arr.shape[1]
    
def concatenate_dictionaries(dict1, dict2):
    """
    Concatenates values from two dictionaries. If a key is present in both, it concatenates the values.
    Supports lists, torch.Tensors, and strings. For other types, it raises a ValueError.

    :param dict1: First dictionary
    :param dict2: Second dictionary
    :return: A new dictionary with concatenated values for common keys
    """
    result = {}

    # Iterate over all unique keys from both dictionaries
    for key in set(dict1) | set(dict2):
        if key in dict1 and key in dict2:
            # Handle concatenation if the key exists in both dictionaries
            if isinstance(dict1[key], list) and isinstance(dict2[key], list):
                # Concatenate lists
                result[key] = dict1[key] + dict2[key]
            elif isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor):
                # Concatenate torch tensors
                result[key] = torch.cat((dict1[key], dict2[key]), dim=0)
            elif isinstance(dict1[key], str) and isinstance(dict2[key], str):
                # Concatenate strings
                result[key] = dict1[key] + dict2[key]
            else:
                # Raise an error for unsupported types
                raise ValueError(f"Unsupported type for key {key}: {type(dict1[key])}")
        elif key in dict1:
            # If the key only exists in dict1, just copy it
            result[key] = dict1[key]
        else:
            # If the key only exists in dict2, just copy it
            result[key] = dict2[key]

    return result
    
def reconstruct_predictions(pred, indexes0, indexes1, pred_background=None, pred_semantic=None):
    """
    Reconstruct the original prediction dictionary to have all the matches
    in the same format as the original keypoints, descriptors, and scores.
    Works even if pred_background or pred_semantic is missing (None).
    """
    # Initialize placeholders with the same size as the original data
    num_keypoints0 = indexes0.shape[0]
    num_keypoints1 = indexes1.shape[0]
    
    matches0 = -torch.ones(num_keypoints0, dtype=torch.long, device=pred['descriptors0'].device)
    matches1 = -torch.ones(num_keypoints1, dtype=torch.long, device=pred['descriptors1'].device)
    matching_scores0 = torch.zeros(num_keypoints0, device=pred['descriptors0'].device)
    matching_scores1 = torch.zeros(num_keypoints1, device=pred['descriptors1'].device)
    
    # Separate indexes for background and semantic components
    bg_indexes0 = torch.where(indexes0 == -1)[0]
    bg_indexes1 = torch.where(indexes1 == -1)[0]
    sem_indexes0 = torch.where(indexes0 != -1)[0]
    sem_indexes1 = torch.where(indexes1 != -1)[0]
    
    # If pred_background is provided, update matches and scores for background
    if pred_background is not None:
        if 'matches0' in pred_background: 
            matches0[bg_indexes0] = pred_background['matches0'].squeeze(0)
            matching_scores0[bg_indexes0] = pred_background['matching_scores0'].squeeze(0)
        if 'matches1' in pred_background:
            matches1[bg_indexes1] = pred_background['matches1'].squeeze(0)
            matching_scores1[bg_indexes1] = pred_background['matching_scores1'].squeeze(0)
    
    # If pred_semantic is provided, update matches and scores for semantic
    if pred_semantic is not None:
        if 'matches0' in pred_semantic:
            matches0[sem_indexes0] = pred_semantic['matches0'].squeeze(0)
            matching_scores0[sem_indexes0] = pred_semantic['matching_scores0'].squeeze(0)
        if 'matches1' in pred_semantic:
            matches1[sem_indexes1] = pred_semantic['matches1'].squeeze(0)
            matching_scores1[sem_indexes1] = pred_semantic['matching_scores1'].squeeze(0)
    
    # Add the reconstructed matches and matching scores back to the pred dictionary
    pred['matches0'] = matches0.unsqueeze(0)
    pred['matches1'] = matches1.unsqueeze(0)
    pred['matching_scores0'] = matching_scores0.unsqueeze(0)
    pred['matching_scores1'] = matching_scores1.unsqueeze(0)
    
    return pred

def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)

# --- OPERATIONS ---

def compute_sem_match_stat(
    keypoints0, keypoints1,
    indexes0, indexes1,
    iou_indexes,
    matches
):
    '''
    Computes semantic match statistics and returns confusion matrix.

    Returns:
        A dictionary with percentages and confusion matrix:
            'semantics_to_semantics_pct'
            'background_to_background_pct'
            'correct_mask_pct'
            'confusion_matrix'
    '''
    matches = np.array(matches)
    indexes0 = np.array(indexes0)
    indexes1 = np.array(indexes1)

    valid_matches_mask = matches != -1
    valid_indices0 = np.where(valid_matches_mask)[0]
    matched_indices1 = matches[valid_matches_mask]

    total_matches = len(valid_indices0)
    if total_matches == 0:
        print('No valid matches provided')
        return {
            'semantics_to_semantics_pct': 0.0,
            'background_to_background_pct': 0.0,
            'correct_mask_pct': 0.0,
            'confusion_matrix': np.array([[0, 0], [0, 0]])
        }

    idx0_labels = indexes0[valid_indices0]
    idx1_labels = indexes1[matched_indices1]

    # True labels: whether keypoints in image0 are foreground (1) or background (0)
    true_labels = (idx0_labels >= 0).astype(int)
    # Predicted labels: whether matched keypoints in image1 are foreground (1) or background (0)
    predicted_labels = (idx1_labels >= 0).astype(int)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    # Calculate percentages based on confusion matrix
    # True Positives (TP): semantics to semantics
    TP = conf_matrix[1, 1]
    # True Negatives (TN): background to background
    TN = conf_matrix[0, 0]
    # Total valid matches
    total = conf_matrix.sum()

    # Correct mask matches
    correct_mask = 0
    total_iou_considered = 0

    # Iterate through matched keypoints from image0 and image1
    for idx0, idx1 in zip(idx0_labels, idx1_labels):
        if idx0 >= 0 and idx0 < len(iou_indexes):  # Only consider keypoints on valid masks in image0
            corresponding_idx1 = iou_indexes[idx0]
            if corresponding_idx1 != -1:
                total_iou_considered += 1  # Track keypoints for which there is a valid corresponding mask
                if corresponding_idx1 == idx1:
                    correct_mask += 1  # Count the number of correct matches to the corresponding mask

    semantics_to_semantics_pct = (TP / total) * 100
    background_to_background_pct = (TN / total) * 100
    
    # Calculate correct mask percentage considering only valid IoU mappings
    if total_iou_considered > 0:
        correct_mask_pct = (correct_mask / total_iou_considered) * 100
    else:
        correct_mask_pct = 0.0  # Set to 0 if no valid IoU-mapped keypoints are considered to avoid division by zero
    print(semantics_to_semantics_pct,background_to_background_pct,correct_mask_pct)

    return {
        'semantics_to_semantics_pct': semantics_to_semantics_pct,
        'background_to_background_pct': background_to_background_pct,
        'correct_mask_pct': correct_mask_pct,
        'confusion_matrix': conf_matrix
    }