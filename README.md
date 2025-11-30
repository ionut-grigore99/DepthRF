<div align="center">


# DepthRF
**Some Description Here.**

______________________________________________________________________

<p align="center">
  <a href="https://arxiv.org/abs/2303.05021">Arxiv</a> •
  <a href="#instalation">Installation</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#training">Training</a> •
  <a href="#testing">Testing</a> •
  <a href="#pre-trained-models-and-results">Pretrained Models</a> •
  <a href="#citation">Citation</a> •
  <a href="#contact">Contact</a><br> 
 </p>

______________________________________________________________________

<br>


</div>

Some description here 

### Citation

If you find our work useful in your research please consider citing our paper:

```
citation here
```


---------------------------------------------------------------------------------


### Instalation

Our released implementation is tested on:

- Ubuntu 24.04
- Python 3.8.x 
- PyTorch 1.10.0 / torchvision 0.11.0
- NVIDIA CUDA 12.8
- 1x NVIDIA RTX 4090 GPU
- mmdet3d  1.0.0rc4
- mmcv-full 1.6.2
- mmsegmentation 0.27.0
- mmdet 2.25.1

First create a conda environment called **depthrf**:
```bash
conda create -n depthrf  python=3.8
```

Activate the new enviroment:
```bash
conda activate depthrf
```

After that, run the following:
```bash
pip install -e .
```
or
```bash
pip install -e . && pip install -e ".[dev]"
```

Recommended to install the `[dev]` dependencies.


---------------------------------------------------------------------------------



### Dataset

We used two datasets for training and evaluation. 

#### NYU Depth V2 (NYUv2)

We used preprocessed NYUv2 HDF5 dataset.

```bash
$ cd PATH_TO_DOWNLOAD
$ wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
$ tar -xvf nyudepthv2.tar.gz
```
After preparing the dataset, you should generate a json file containing paths to individual images.

```bash
$ python -m src.data_prep.generate_json_NYUDepthV2 --path_root PATH_TO_NYUv2
```
After that, you will get a data structure as follows:

```
nyudepthv2
├── train
│    ├── basement_0001a
│    │    ├── 00001.h5
│    │    └── ...
│    ├── basement_0001b
│    │    ├── 00001.h5
│    │    └── ...
│    └── ...
└── val
    └── official
        ├── 00001.h5
        └── ...
```

#### KITTI Depth Prediction

KITTI Depth Prediction dataset is available at the [KITTI Website](http://www.cvlibs.net/datasets/kitti). We should choose depth prediction. 

For color images, KITTI Raw dataset (~180GB) from the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php) is also needed. You can download it using our provided script:

```bash
# Download KITTI Raw dataset (this will take a while)
$ bash src/data_prep/raw_data_downloader.sh /path/to/download/kitti_raw
```

After downloading dataset, you should first copy color images, poses, and calibrations from the KITTI Raw to the KITTI Depth Prediction dataset.

```bash
$ python -m src.data_prep.prepare_KITTI --path_root_dp PATH_TO_Dataset --path_root_raw PATH_TO_KITTI_RAW
```

```
.
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```

After preparing the dataset, you should generate a json file containing paths to individual images.

```bash
# For Train / Validation
$ python -m src.data_prep.generate_json_KITTI --path_root PATH_TO_KITTI

# For Online Evaluation Data
$ python -m src.data_prep.generate_json_KITTI --path_root PATH_TO_KITTI --name_out kitti_dp_test.json --test_data
```


### Training

```bash
$ python -m src.train 
```

Please refer to the config.yaml for more training options.

During the training, tensorboard logs are saved under the experiments directory. To run the tensorboard:

```bash
$ cd DepthRF/experiments/train
$ tensorboard --logdir=. --bind_all
```
The tensorboard visualization includes metric curves and depth map visualization as shown below:


Put image here




### Testing

```bash
$ python -m src.test
```

Please refer to the config.yaml for more testing options.

### Pre-trained Models and Results

We provide the pre-trained models and results on KITTI depth prediction test split.

<div align="center">


|    Model    | RMSE  | MAE | iRMSE     |  iMAE    |REL| D<sup>1</sup> |  D<sup>2</sup>    |   D<sup>3</sup>     |      
|----------|:-------------:|:----:|:----:|:----:|:-------------:|:----:|:----:|:----:|
|   [.....](https://huggingface.co/claudecc/diffusiondepth/blob/main/res50_model_00030.pt) | ...... | ...... | ......  | ...... | ...... | ...... | ......  | ...... |
|  [....](https://huggingface.co/claudecc/diffusiondepth/blob/main/swin_model_00035.pt) |     ......    | ...... | ...... | ...... |    ......    | ...... | ...... |......
| [.....](https://huggingface.co/claudecc/diffusiondepth/blob/main/mpvit_model_00037.pt)    |      ......     |......  | ...... | ...... |      ......     | ......| ...... | ...... |


</div>

### Contact

Contact us: ionut.grigore@cs.upt.ro