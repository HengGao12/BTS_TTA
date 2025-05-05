import argparse
from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from transform import ConvertToMultiChannelBasedOnBratsClassesd
from monai.utils import set_determinism

import torch
torch.backends.cudnn.benchmark = True  # enable cuDNN benchmark
import logging
import tent_3d

# set seed
set_determinism(seed=0)


VAL_AMP = True

# define inference method
def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.autocast("cuda"):
            return _compute(input)
    else:
        return _compute(input)




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test-time Adaptation for Brain Tumor Segmentation") 
    parser.add_argument('-method', type=str, default='none', choices=['none', 'tta'], help='Method for TTA: tent or none')
    parser.add_argument('-root_dir', type=str, default='./data', help='Root directory for data')
    parser.add_argument('-steps', type=int, default=1, help='Number of steps for TTA')
    args = parser.parse_args()

    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    # here we don't cache any data in case out of memory issue
    train_ds = DecathlonDataset(
        root_dir=args.root_dir,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_ds = DecathlonDataset(
        root_dir=args.root_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    max_epochs = 100
    val_interval = 1
    

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    dice_metric = DiceMetric(include_background=True, reduction="mean")

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # load the pretrained model
    model.load_state_dict(torch.load("./model_output/best_metric_model.pth", map_location=device))
    
    if args.method == 'none':
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler('./log_file/baseline_log.txt')) # save the log
        # baseline model
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs, model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            
            logger.info(f"Mean Dice (MD) without TTA: {metric:.4f}")    
    else: 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler('./log_file/tta_log.txt')) # save the log   
        # Test time adaptation using Tent: https://github.com/DequanWang/tent
        optimizer = torch.optim.Adam(model.parameters(), 1e-6, weight_decay=1e-5) # re-initialize the optimizer
        params, param_names = tent_3d.collect_params(model)
        tented_model = tent_3d.Tent_3D(model, optimizer, steps=args.steps)  # forward and adapt
        
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = inference(val_inputs, tented_model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_labels)

        metric = dice_metric.aggregate().item()
        
        logger.info(f"Mean Dice (MD) after TTA: {metric:.4f}")