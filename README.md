# TTA-3D: Test-Time Adaptation for Brain Tumor Segmentation
> Heng Gao

All our experiments are conducted on one NVIDIA V100 GPU.

## Usage

### Installation

-python==3.8
-torch=1.11.0+cu113

```bash
pip install -r requirements.txt
```

### Data preparation

Downloading data
```bash
cd data
gdown https://drive.google.com/uc?id=1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU 
```
Note, one can also download data using the link provided in `./data`.

### Training

One can directly train a brain tumor segmentation model using codes given in `brats_segmentation_3d.ipynb`.

### Test-time Adaptation using Tent

```python
python tta_main.py -method tta -step 1
```
## Acknowledgments
Our code is developed based on [MONAI](https://github.com/Project-MONAI/) and [Tent](https://github.com/DequanWang/tent/).
