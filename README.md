# Test-Time Adaptation for Enhancing Brain Tumor Segmentation
> Heng Gao

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Data preparation

```bash
wget 
```

### Training

One can directly train a brain tumor segmentation model using codes given in brats_segmentation_3d.ipynb

### Test-time Adaptation using Tent

```python
python tta_main.py -method tta -step 1
```
## Acknowledgment
Our code is developed based on [MONAI](https://github.com/Project-MONAI/) and [Tent](https://github.com/DequanWang/tent/).
