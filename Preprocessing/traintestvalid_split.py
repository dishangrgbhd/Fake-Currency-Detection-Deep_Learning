import os
import shutil
import numpy as np


def create_dirs(base_path):
    """Create directories for train, val, test sets"""
    for split in ['train', 'val', 'test']:
        for category in ['Real', 'Fake']:
            os.makedirs(os.path.join(base_path, split, category), exist_ok=True)


def copy_files(files, src_dir, dst_dir, category, split):
    """Copy files to respective directories"""
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, split, category, file))


def split_data(src_dir, dst_dir, split_ratio, category):
    """Split the data into train, val, test and move to respective directories"""
    all_files = os.listdir(src_dir)
    np.random.shuffle(all_files)
    train_split = int(split_ratio[0] * len(all_files))
    val_split = int(split_ratio[1] * len(all_files)) + train_split

    train_files = all_files[:train_split]
    val_files = all_files[train_split:val_split]
    test_files = all_files[val_split:]

    copy_files(train_files, src_dir, dst_dir, category, 'train')
    copy_files(val_files, src_dir, dst_dir, category, 'val')
    copy_files(test_files, src_dir, dst_dir, category, 'test')


# Paths
dataset_path = r'D:/Dataset 500'
base_dst_path = r'D:/Dataset 500/split dataset'

real_path = os.path.join(dataset_path, 'Real')
fake_path = os.path.join(dataset_path, 'Fake')

# Split ratios
split_ratio = [0.7, 0.15, 0.15]  # 70% train, 15% val, 15% test

# Create directories
create_dirs(base_dst_path)

# Split and move Real data
split_data(real_path, base_dst_path, split_ratio, 'Real')

# Split and move Fake data
split_data(fake_path, base_dst_path, split_ratio, 'Fake')
