"""
Generates a JSON file organizing KITTI Depth Prediction dataset into train/val/test splits.

This script processes the KITTI dataset directory structure to create a JSON file
containing file paths for RGB images, sparse LiDAR depth maps (raw Velodyne LiDAR 
sensor measurements - very sparse (~5% of pixels have depth values)), ground truth 
depth (accumulated/projected LiDAR from multiple frames - semi-dense (~30-40% of 
pixels have depth values)), and camera calibration files.


The script handles two different use cases:

1. **Standard Mode** (default): Generates train/val/test splits for model training
   - Train: Raw driving sequences for training
   - Val: Raw driving sequences for validation  
   - Test: Hand-selected evaluation frames from val_selection_cropped
   
2. **Test Submission Mode** (--test_data flag): Generates test split for benchmark submission
   - Test: Anonymous test set (test_depth_prediction_anonymous)
   - Uses dummy placeholders for depth/gt (not provided in test set)
   - Only RGB images and calibration available for inference

Directory Structure Handled:
├── train/                          # Training driving sequences
│   └── 2011_09_26_drive_0001_sync/
│       ├── image_02/data/          # Left camera RGB images
│       ├── image_03/data/          # Right camera RGB images
│       ├── proj_depth/
│       │   ├── velodyne_raw/       # Sparse LiDAR depth (input)
│       │   └── groundtruth/        # Dense accumulated depth (target)
│       └── calib_cam_to_cam.txt    # Camera intrinsics
├── val/                            # Validation driving sequences (same structure)
└── depth_selection/
    ├── val_selection_cropped/      # Official test set (1000 frames)
    └── test_depth_prediction_anonymous/  # Benchmark submission test set

Output JSON Formats:

Standard Mode (kitti.json):
{
    "train": [
        {
            "rgb": "train/2011_09_26_drive_0001_sync/image_02/data/0000000005.png",
            "depth": "train/.../proj_depth/velodyne_raw/image_02/0000000005.png",
            "gt": "train/.../proj_depth/groundtruth/image_02/0000000005.png",
            "K": "train/2011_09_26_drive_0001_sync/calib_cam_to_cam.txt"
        }, ...  // ~85,898 samples  
    ],
    "val": [...],   // ~6,852 samples 
    "test": [...]   // 1,000 selected test frames
}

Test Submission Mode (kitti_test.json):
{
    "test": [
        {
            "rgb": "depth_selection/test_depth_prediction_anonymous/image/0000000000.png",
            "depth": "depth_selection/test_depth_completion_anonymous/velodyne_raw/0000000000.png",  // dummy
            "gt": "depth_selection/test_depth_completion_anonymous/velodyne_raw/0000000000.png",     // dummy
            "K": "depth_selection/test_depth_prediction_anonymous/intrinsics/0000000000.txt"
        }, ...  // 500 anonymous test frames for benchmark submission
    ]
}
"""

import os
import argparse
import random
import json

parser = argparse.ArgumentParser(description="KITTI Depth Prediction json generator")

parser.add_argument('--path_root', type=str, required=True,
                    help="Path to the KITTI Depth Prediction dataset")
parser.add_argument('--path_out', type=str, required=False,
                    default='src/datasets', help="Output path")
parser.add_argument('--name_out', type=str, required=False,
                    default='kitti.json', help="Output file name")
parser.add_argument('--num_train', type=int, required=False,
                    default=int(1e12), help="Maximum number of train data")
parser.add_argument('--num_val', type=int, required=False,
                    default=int(1e10), help="Maximum number of val data")
parser.add_argument('--num_test', type=int, required=False,
                    default=int(1e10), help="Maximum number of test data")
parser.add_argument('--seed', type=int, required=False,
                    default=7240, help='Random seed')
parser.add_argument('--test_data', action='store_true',
                    default=False, help='json for DP test set generation')

args = parser.parse_args()

random.seed(args.seed)


# Some miscellaneous functions
def check_dir_existence(path_dir):
    assert os.path.isdir(path_dir), "Directory does not exist : {}".format(path_dir)


def check_file_existence(path_file):
    assert os.path.isfile(path_file), "File does not exist : {}".format(path_file)


def generate_json():
    check_dir_existence(args.path_out)

    # For train/val splits
    dict_json = {}
    for split in ['train', 'val']:
        path_base = args.path_root + '/' + split

        list_seq = os.listdir(path_base)
        list_seq.sort()

        list_pairs = []
        for seq in list_seq:
            cnt_seq = 0

            for cam in ['image_02', 'image_03']:
                list_depth = os.listdir(path_base + '/' + seq + '/proj_depth/velodyne_raw/{}'.format(cam))
                list_depth.sort()

                for name in list_depth:
                    path_rgb = split + '/' + seq + '/' + cam + '/data/' + name
                    path_depth = split + '/' + seq + '/proj_depth/velodyne_raw/' + cam + '/' + name
                    path_gt = split + '/' + seq + '/proj_depth/groundtruth/' + cam + '/' + name
                    path_calib = split + '/' + seq + '/calib_cam_to_cam.txt'

                    dict_sample = {
                        'rgb': path_rgb,
                        'depth': path_depth,
                        'gt': path_gt,
                        'K': path_calib
                    }

                    flag_valid = True
                    for val in dict_sample.values():
                        flag_valid &= os.path.exists(args.path_root + '/' + val)
                        if not flag_valid:
                            break

                    if not flag_valid:
                        continue

                    list_pairs.append(dict_sample)
                    cnt_seq += 1

            print("{} : {} samples".format(seq, cnt_seq))

        dict_json[split] = list_pairs
        print("{} split : Total {} samples".format(split, len(list_pairs)))
    
    # For test split
    split = 'test'
    path_base = args.path_root + '/depth_selection/val_selection_cropped'

    list_depth = os.listdir(path_base + '/velodyne_raw')
    list_depth.sort()

    list_pairs = []
    for name in list_depth:
        name_base = name.split('velodyne_raw')
        name_base = name_base[0] + '{}' + name_base[1]

        path_rgb = 'depth_selection/val_selection_cropped/image/' + name_base.format('image')
        path_depth = 'depth_selection/val_selection_cropped/velodyne_raw/' + name
        path_gt = 'depth_selection/val_selection_cropped/groundtruth_depth/' + name_base.format('groundtruth_depth')
        path_calib = 'depth_selection/val_selection_cropped/intrinsics/' + name_base.format('image')[:-4] + '.txt'

        dict_sample = {
            'rgb': path_rgb,
            'depth': path_depth,
            'gt': path_gt,
            'K': path_calib
        }

        flag_valid = True
        for val in dict_sample.values():
            flag_valid &= os.path.exists(args.path_root + '/' + val)
            if not flag_valid:
                break

        if not flag_valid:
            continue

        list_pairs.append(dict_sample)

    dict_json[split] = list_pairs
    print("{} split : Total {} samples".format(split, len(list_pairs)))
    random.shuffle(dict_json['train'])

    print("\n ###### Summary statistics ######")
    print("{} split : Total {} samples".format("train", len(dict_json["train"])))
    print("{} split : Total {} samples".format('val', len(dict_json['val'])))
    print("{} split : Total {} samples".format('test', len(dict_json['test'])))

    # Cut if maximum is set
    for s in [('train', args.num_train), ('val', args.num_val), ('test', args.num_test)]:
        if len(dict_json[s[0]]) > s[1]:
            # Do shuffle
            random.shuffle(dict_json[s[0]])

            num_orig = len(dict_json[s[0]])
            dict_json[s[0]] = dict_json[s[0]][0:s[1]]
            print("{} split : {} -> {}".format(s[0], num_orig, len(dict_json[s[0]])))

    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("\n Json file generation finished.")


def generate_json_test():
    check_dir_existence(args.path_out)

    dict_json = {}

    # For test split
    split = 'test'
    path_base = args.path_root + '/depth_selection/test_depth_prediction_anonymous'

    list_images = os.listdir(path_base + '/image')
    list_images.sort()

    list_pairs = []
    dummpy_placeholder = 'depth_selection/test_depth_completion_anonymous/velodyne_raw/0000000000.png'
    for name in list_images:
        path_rgb = 'depth_selection/test_depth_prediction_anonymous/image/' + name
        path_depth = dummpy_placeholder
        path_gt = dummpy_placeholder
        path_calib = 'depth_selection/test_depth_prediction_anonymous/intrinsics/' + name[:-4] + '.txt'

        dict_sample = {
            'rgb': path_rgb,
            'depth': path_depth,
            'gt': path_gt,
            'K': path_calib
        }

        flag_valid = True
        for val in dict_sample.values():
            flag_valid &= os.path.exists(args.path_root + '/' + val)
            if not flag_valid:
                break

        if not flag_valid:
            continue

        list_pairs.append(dict_sample)

    dict_json[split] = list_pairs
    print("{} split : Total {} samples".format(split, len(list_pairs)))
    
    f = open(args.path_out + '/' + args.name_out, 'w')
    json.dump(dict_json, f, indent=4)
    f.close()

    print("Json file generation finished.")


if __name__ == '__main__':
    print('\nArguments :')
    for arg in vars(args):
        print(arg, ':',  getattr(args, arg))
    print('')

    if args.test_data:
        generate_json_test()
    else:
        generate_json()