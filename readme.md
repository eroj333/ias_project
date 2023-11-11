# Panoptic Segmentation

This project performs panoptic segmentation using Detectron2. The instances are tracked using Hungarian Algorithm. 

**Completed tasks:**
- [x] Video processing pipeline
- [x] Hungarian Algorithm for tracking instances
- [x] Draw bounding boxes and semantic segmentation masks
- [x] Save output video and semantic segmentation masks


**Remaining tasks:**
- [ ] Fine tune model in KITTI dataset
- [ ] Fix bugs present in the code


Video processing pipeline for each frame of video:
1. Panoptic segmentation with Detectron2 ([details](docs/model.md)) 
2. Track instances using Hungarian Algorithm
3. Draw bounding boxes and semantic segmentation masks
4. Save output video and semantic segmentation masks

## Installation

[Numpy in windows has some data type issues. Please use WSL if on windows.](https://github.com/numpy/numpy/issues/17640)



Create conda env
```commandline
conda create -n panoptic python=3.10.13
```
Activate env
```commandline
conda activate panoptic
```
Install PyTorch following the instructions [here](https://pytorch.org/get-started/locally/)

Install Detectron2 following the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
or
```commandline
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Install panopticapi
```commandline
pip install git+https://github.com/cocodataset/panopticapi.git
```

Install openCV
```commandline
pip install opencv-python
```

Install the dependencies
```commandline
pip install -r requirements.txt
```

## Usage

```
usage: python main.py [-h] [--image IMAGE] [--video VIDEO] [--model_path MODEL_PATH] [--method {cv2,cocopan}] [--output OUTPUT]

Panoptic Segmentation

options:
  -h, --help            show this help message and exit
  --image IMAGE         path to image
  --video VIDEO         path to video
  --model_path MODEL_PATH
                        path to model
  --method {cv2,cocopan}
                        path to model
  --output OUTPUT       path to output

```

## Demo
Demo of segmentation on a video is included in the `demo` folder. 

`<demo_file>` input video file 

`<demo_file>_out.mp4` output with bounding box for things class

`<demo_file>_seg_out.mp4` semantic seg output for things class

Video source: pexels.com


## References
1. [Detectron2](https://detectron2.readthedocs.io/en/latest/index.html)
2. [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)

## Remaining Work
1. Fine tune model in KITTI dataset
    - [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015)
    - Problem: converting KITTI dataset to COCO format for Detectron2 is not straightforward
2. Fix bugs present in the code