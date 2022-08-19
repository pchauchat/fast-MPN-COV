"""
Separates images into Train and Test folders, compatible with the ImageFolder class
"""
import csv
import os
import shutil

data_path = 'C:/Users/chauchatp/Documents/Data/CUB_200_2011/CUB_200_2011'
train_dir = os.path.join(data_path, 'train')
test_dir = os.path.join(data_path, 'test')
val_dir = os.path.join(data_path, 'val')

for dir in [train_dir, test_dir]:
    if not os.path.exists(dir):
                print('Creating directory "%s" ...' % (dir))
                os.makedirs(dir)



image_dir = os.path.join(data_path, 'images')

split_file_path = os.path.join(data_path, 'train_test_split.txt')
images_file_path = os.path.join(data_path, 'images.txt')
with open(split_file_path, 'r') as split_file:
    with open(images_file_path, 'r') as images_file:
        split_reader = csv.reader(split_file, delimiter=' ')
        image_reader = csv.reader(images_file, delimiter=' ')

        for img_id, is_train in split_reader:
            _, img_file_name = next(image_reader)
            img_dir_name = os.path.dirname(img_file_name)
            train_img_dir = os.path.join(train_dir, img_dir_name)
            val_img_dir = os.path.join(val_dir, img_dir_name)
            for dir in [train_img_dir, val_img_dir]:
                if not os.path.exists(dir):
                    print('Creating directory "%s" ...' % dir)
                    os.makedirs(dir)
                
            image_path = os.path.join(image_dir, img_file_name)
            if int(is_train):
                dest_path = os.path.join(train_dir, img_file_name)
            else:
                # dest_path = os.path.join(test_dir, os.path.basename(img_file_name))
                dest_path = os.path.join(val_dir, img_file_name)
            shutil.copy(image_path, dest_path)
                

