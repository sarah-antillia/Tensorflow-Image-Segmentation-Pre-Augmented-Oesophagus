<h2>Tensorflow-Image-Segmentation-Pre-Augmented-Oesophagus (2025/01/07)</h2>

This is the first experiment of Image Segmentation for Human Oesophagus Nuclei
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a pre-augmented <a href="https://drive.google.com/file/d/1pas3Xya8mKTBU9qJq3dWLbKwjNh7wGr1/view?usp=sharing">
Oeophagus-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="https://www.kaggle.com/datasets/ipateam/nuinsseg">
NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H&E-Stained Images.
</a><br>
<br>

<b>Dataset Augmentation Strategy</b><br>
 To address the limited size of Oesophagus dataset, which contains 47 tissue images and their corresponding binary masks in human oesophagus of NuInsSeg dataset, 
 we employed <a href="./generator/ImageMaskDatasetGenerator.py">an offline augmentation tool</a> to generate a pre-augmented dataset, which supports the following augmentation methods.
<br>
<li>Vertical flip</li>
<li>Horizontal flip</li>
<li>Rotation</li>
<li>Shrinks</li>
<li>Shears</li> 
<li>Deformation</li>
<li>Distortion</li>
<li>Barrel distortion</li>
<li>Pincushion distortion</li>
<br>
Please see also the following tools <br>
<li><a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a></li>
<br>
<b>Color Space Conversion Strategy</b><br>
We also applied an RGB-to-HSV color space conversion to the train and valid images, and inference test images.  
<br>
<br>
<table>
<tr>
<th>RGB</th>
<th>HSV</th>
<th>Mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/18.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/hsvimage/18.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/18.jpg" width="320" height="auto"></td>

</tr>

</table>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/10.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/10.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/10.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/32.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/32.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/32.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/deformed_alpha_1300_sigmoid_8_26.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/deformed_alpha_1300_sigmoid_8_26.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/deformed_alpha_1300_sigmoid_8_26.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this OesophagusSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been take from the following kaggle web site 
<a href="https://www.kaggle.com/datasets/ipateam/nuinsseg/data">
NuInsSeg
</a><br><br>

<b>About Dataset</b><br>
NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H&E-Stained Histological Images<br>
<br>
<b>Citation</b><br>
@article{mahbod2023nuinsseg,<br>
  title={NuInsSeg: A Fully Annotated Dataset for Nuclei Instance Segmentation in H\&E-Stained Histological Images},<br>
  author={Mahbod, Amirreza and Polak, Christine and Feldmann, Katharina and Khan, Rumsha and Gelles, <br>
  Katharina and Dorffner, Georg and Woitek, Ramona and Hatamikia, Sepideh and Ellinger, Isabella},<br>
  journal={arXiv preprint arXiv:2308.01760},<br>
  year={2023}<br>
}
<br>
<br>

<br>
<h3>
<a id="2">
2 Oesophagus ImageMask Dataset
</a>
</h3>
 If you would like to train this Oesophagus Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1pas3Xya8mKTBU9qJq3dWLbKwjNh7wGr1/view?usp=sharing">
Oesophagus-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Oesophagus
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>

On the derivation of this dataset, please refer to the following Python scripts:
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
<br>
<b>Oesophagus Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/Oesophagus_Statistics.png" width="512" height="auto"><br>
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained OesophagusTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Oesophagus and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.3
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Enabled color space converter</b><br>
<pre>
[image]
color_converter = "cv2.COLOR_BGR2HSV"
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3) </b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (74,75,76) </b><br>

<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 76  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/train_console_output_at_epoch_76.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Oesophagus</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Oesophagus.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/evaluate_console_output_at_epoch_76.png" width="720" height="auto">
<br><br>Image-Segmentation-Oesophagus

<a href="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Oesophagus/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.1437
dice_coef,0.8713
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Oesophagus</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Oesophagus.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/17.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/17.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/17.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/18.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/18.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/18.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/barrdistorted_1002_0.3_0.3_8.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/barrdistorted_1002_0.3_0.3_8.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/barrdistorted_1002_0.3_0.3_8.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/barrdistorted_1004_0.3_0.3_17.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/barrdistorted_1004_0.3_0.3_17.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/barrdistorted_1004_0.3_0.3_17.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/deformed_alpha_1300_sigmoid_8_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/deformed_alpha_1300_sigmoid_8_12.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/deformed_alpha_1300_sigmoid_8_12.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/images/deformed_alpha_1300_sigmoid_8_36.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test/masks/deformed_alpha_1300_sigmoid_8_36.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Oesophagus/mini_test_output/deformed_alpha_1300_sigmoid_8_36.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. Improving generalization capability of deep learning-based nuclei instance segmentation <br>
by non-deterministic train time and deterministic test time stain normalization
</b><br>
Amirreza Mahbod, Georg Dorffner, Isabella Ellinger, Ramona Woitek, Sepideh Hatamikia<br>

<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10825317/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC10825317/
</a>
<br>
