import cv2
import numpy as np
import os
import os.path as ops


def gen_train_sample(save_txt_path, b_gt_image_dir, image_dir):
    with open(save_txt_path, 'w') as file:
        for image_name in os.listdir(b_gt_image_dir):
            if not image_name.endswith('.jpg'):
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name)
            image_path = ops.join(image_dir, image_name)

            assert ops.exists(image_path), '{:s} not exist'.format(image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if b_gt_image is None or image is None is None:
                continue
            else:
                info = '{:s} {:s}'.format(image_path, binary_gt_image_path)
                file.write(info + '\n')
    return



save_image_path = 'binary_lane_bdd/Images/'
save_binary_path = 'binary_lane_bdd/Labels/'
save_txt_path = 'binary_lane_bdd/train.txt'