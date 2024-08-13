# mmrotate-dcfl
Official method implementation for the paper: Oriented Tiny Object Detection:
A Dataset, Benchmark, and Dynamic Unbiased Learning 

## Introduction
DCFL is a learning framework that can be plugged into [one-stage](configs/dcfl) and [two-stage](configs/dcfl) architectures for detecting oriented tiny objects.

![demo image](static/images/pipeline_pami.png)

## Installation and Get Started

Required environments:
- Linux
- Python 3.7+
- PyTorch 1.10.0+
- CUDA 9.2+
- GCC 5+
- MMdet 2.23.0+
- [MMCV-DCFL](https://github.com/Chasel-Tsui/MMCV-DCFL) 


Install:
Note that this repository is based on the MMRotate. Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```
git clone https://github.com/Chasel-Tsui/AI-TOD-R.git
cd AI-TOD-R
pip install -r requirements/build.txt
python setup.py develop
```

## Visualization
Predictions of the RetinaNet-O are shown in the first row, predictions of the DCFL are shown in the second row. Note that the green box denotes the True Positive, the red box denotes the False Negative and the blue box denotes the False Positive predictions.
![demo_images](static/images/vis_pred.png)

## Citation
If you find this work helpful, please consider citing:
```bibtex
@article{xu2024oriented,
  title={Oriented Tiny Object Detection: A Dataset, Benchmark, and Dynamic Unbiased Learning},
  author={Xu, Chang and Zhang, Ruixiang and Yang, Wen and Zhu, Haoran and Xu, Fang and Ding, Jian and Xia, Gui-Song},
  journal={arXiv preprint},
  year={2024}
}
```