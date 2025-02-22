# SegFormer | SegFormer-Pytorch

Pretrained SegFormer and SegFormer  for CityScapes dataset.

## Quick Start 

### 1. Available Architectures
|SegFormer   |   SegFormer      | SegFormer with Few-Shot Learning   |
| :---: | :---:     |


### 2. Visualize segmentation outputs:
```python
outputs = model(images)
preds = outputs.max(1)[1].detach().cpu().numpy()
colorized_preds = val_dst.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# Do whatever you like here with the colorized segmentation maps
colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
```

Image folder:
```bash
python predict.py --input datasets/data/kitti_Road_Dataset/leftImg8bit/train/bremen  --dataset Kitti_Road --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_Kitti_Road_os16.pth --save_val_results_to test_results
```

### 3. New backbones

Please refer to [this commit (Xception)](https://github.com/VainF/DeepLabV3Plus-Pytorch/commit/c4b51e435e32b0deba5fc7c8ff106293df90590d) for more details about how to add new backbones.

### 4. New datasets

You can train deeplab models on your own datasets. Your ``torch.utils.data.Dataset`` should provide a decoding method that transforms your predictions to colorized images, just like the [VOC Dataset](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/bfe01d5fca5b6bb648e162d522eed1a9a8b324cb/datasets/voc.py#L156):
```python

class MyDataset(data.Dataset):
    ...
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]
```


## Results

### 1. Performance on Pascal VOC2012 Aug (21 classes, 513 x 513)

Training: 513x513 random crop  
validation: 513x513 center crop

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  | Tencent Weiyun  | 
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: | :----:   |
| DeepLabV3-MobileNet       | 16      |  6.0G      |   16/16  |  0.701     |    [Download](https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=0)       | [Download](https://share.weiyun.com/A4ubD1DD) |
### 2. Performance on Cityscapes (19 classes, 1024 x 2048)

Training: 768x768 random crop  
validation: 1024x2048

|  Model          | Batch Size  | FLOPs  | train/val OS   |  mIoU        | Dropbox  |  Tencent Weiyun  |
| :--------        | :-------------: | :----:   | :-----------: | :--------: | :--------: |  :----:   |
| DeepLabV3Plus-MobileNet   | 16      |  135G      |  16/16   |  0.721  |    [Download](https://www.dropbox.com/s/753ojyvsh3vdjol/best_deeplabv3plus_mobilenet_cityscapes_os16.pth?dl=0) | [Download](https://share.weiyun.com/aSKjdpbL) 
#### Segmentation Results on Pascal VOC2012 (DeepLabv3Plus-MobileNet)

<div>
<img src="samples/1_image.png"   width="20%">
<img src="samples/1_target.png"  width="20%">
<img src="samples/1_pred.png"    width="20%">
<img src="samples/1_overlay.png" width="20%">
</div>

<div>
<img src="samples/23_image.png"   width="20%">
<img src="samples/23_target.png"  width="20%">
<img src="samples/23_pred.png"    width="20%">
<img src="samples/23_overlay.png" width="20%">
</div>

<div>
<img src="samples/114_image.png"   width="20%">
<img src="samples/114_target.png"  width="20%">
<img src="samples/114_pred.png"    width="20%">
<img src="samples/114_overlay.png" width="20%">
</div>

#### Segmentation Results on Cityscapes (DeepLabv3Plus-MobileNet)

<div>
<img src="samples/city_1_target.png"   width="45%">
<img src="samples/city_1_overlay.png"  width="45%">
</div>

<div>
<img src="samples/city_6_target.png"   width="45%">
<img src="samples/city_6_overlay.png"  width="45%">
</div>

### 2. Prepare Datasets


#### 2.2  Kitti Road Segmentation (Recommended!!)

The dataset comprises two main parts: a training set with annotated ground truth and a testing set for performance evaluation.

#### 1. **Data Composition** 

The dataset comprises two main parts: a training set with annotated ground truth and a testing set for performance evaluation.

- **Training Set:**
  - **Images:**  
    The training images are stored in a folder typically named image_2 within the training directory. These images capture a variety of urban road scenes under different conditions.
  
  - **Ground Truth Annotations:**  
    Accompanying the images, the ground truth masks are provided in the gt_image_2 folder. These annotations are manually created and mark the road regions (and in some cases, individual lanes) with specific labels. The ground truth masks are often binary (or multi-class in more detailed annotations) where pixels corresponding to the road are labeled with a distinct value (for example, 1 or 255) and the background is labeled as 0.

- **Testing Set:**
  - The testing data is organized in a similar fashion but typically only includes the raw images (without ground truth annotations). This enables an unbiased evaluation of segmentation models on unseen data.

#### 3. **Image Categories**

The dataset contains images that represent diverse urban road conditions. The naming conventions of the files hint at different scene types:

- **Urban Unmarked (um):**  
  Images with the prefix um_ (e.g., um_000000.png to um_000094.png) represent road scenes where lanes are not marked with explicit line segments.

- **Urban Marked (uu):**  
  Images with the prefix uu_ (e.g., uu_000000.png to uu_000097.png) depict scenes where roads have clear lane markings.

- **Urban Multiple Marked Lanes (umm):**  
  Images with the prefix umm_ (e.g., umm_000000.png to umm_000095.png) contain more complex scenarios with multiple lane markings. These scenes challenge segmentation algorithms to correctly identify several drivable lanes.

#### 4. **File Naming Conventions and Structure**

The careful naming of files in the dataset supports easy pairing between images and their corresponding annotations:

- **Training Folder Structure:**
  - **Images (training/image_2):**  
    The images are sequentially numbered and grouped by type. For example:
    - um_000000.png to um_000094.png (urban unmarked)
    - umm_000000.png to umm_000095.png (urban multiple marked lanes)
    - uu_000000.png to uu_000097.png (urban marked)
  
  - **Annotations (training/gt_image_2):**  
    The ground truth masks follow a similar naming pattern but with identifiers that indicate the type of annotation:
    - Files such as um_lane_000000.png to um_lane_000094.png may provide lane-specific annotations for urban unmarked scenes.
    - Similarly, files like umm_road_000000.png to umm_road_000095.png and uu_road_000000.png to uu_000097.png are used for the other respective categories.
    
    This naming convention ensures that each image from image_2 has a corresponding mask in gt_image_2, thereby simplifying the process of dataset creation and model training.

- **Testing Folder Structure:**
  - **Images (testing/image_2/image_2):**  
    Test images are organized in a nested image_2 folder and follow a sequential naming similar to the training set:
    - For instance, um_000000.png to um_000095.png, umm_000000.png to umm_000093.png, and uu_000000.png to uu_000099.png.
  
  The testing folder is used during model evaluation, where the segmentation algorithm’s output can be compared against the ground truth (if available in a validation subset) or visually inspected.

### 2. Train your model on Kitti Road Segmentation

```bash
python main.py --model deeplabv3plus_mobilenet --dataset kitti_Road --enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 16 --output_stride 16 --data_root ./datasets/data/kitti_road 
```

## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
