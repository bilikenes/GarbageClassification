import os
import shutil
import random

IMAGE_DIR = r'C:\Users\PC\Desktop\garbage_dataset\Dataset\images'
LABEL_DIR = r'C:\Users\PC\Desktop\garbage_dataset\Dataset\labels'
OUTPUT_DIR = r'C:\Users\PC\Desktop\garbage_dataset\yolo_dataset'

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

def create_dirs(output_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

def split_dataset():
    create_dirs(OUTPUT_DIR)

    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42)
    random.shuffle(images)

    total_images = len(images)
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    def copy_files(image_list, split_name):
        for img_name in image_list:
            src_img = os.path.join(IMAGE_DIR, img_name)
            dst_img = os.path.join(OUTPUT_DIR, 'images', split_name, img_name)

            txt_name = os.path.splitext(img_name)[0] + '.txt'
            src_txt = os.path.join(LABEL_DIR, txt_name)
            dst_txt = os.path.join(OUTPUT_DIR, 'labels', split_name, txt_name)

            shutil.copy(src_img, dst_img)

            if os.path.exists(src_txt):
                shutil.copy(src_txt, dst_txt)

    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')

if __name__ == '__main__':
    split_dataset()