import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

# Import tensorflow dependencies - Functional API
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def data_aug(img):
    data = []
    for i in range(10):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100,
                                                     seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1,
                                                   seed=(np.random.randint(100), np.random.randint(100)))

        data.append(img)

    return data

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = os.path.join('data', 'data_faces_from_camera')

# Duyệt qua các thư mục của mỗi người
for person_folder in os.listdir(data_dir):
    person_folder_path = os.path.join(data_dir, person_folder)

    # Kiểm tra xem mục hiện tại có phải là thư mục không
    if os.path.isdir(person_folder_path):
        # Duyệt qua các ảnh trong thư mục của người này
        for image_file in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_file)
            img = cv2.imread(image_path)
            print(image_path)
            augmented_images = data_aug(img)
            # Tạo ra ảnh mới bằng cách áp dụng các phép biến đổi dữ liệu
            # augmented_images = data_aug(img)

            # Lưu các ảnh mới với tên tương ứng
            for idx, image in enumerate(augmented_images):
                new_image_path = os.path.join(person_folder_path, '{}{}.jpg'.format(image_file.split('.')[0], idx))
                cv2.imwrite(new_image_path, image.numpy())
