# Image Super-Resolution Using Deep Convolutional Networks
Tensorflow implementation of SRCNN.

## Prerequisites
 * Python 3
 * Tensorflow
 * Numpy
 * Scipy
 * Opencv 3
 * h5py

## Usage

To train, uncomment the scripts in the bottom in net.py.
Then type `python net.py`
<br>
To test, set proper img_path, save_path and upscaling factor (multiplier) in the use_SRCNN.py.
Then type `python use_SRCNN.py`

## Results
The following results are based on 45 hours of training on my i7 CPU.
<br>

Bicubic interpolation:<br>
![bicubic](https://github.com/Edwardlzy/SRCNN/blob/master/Set5_result/bicubic_butterfly_GT.bmp)<br>
SRCNN:<br>
![srcnn](https://github.com/Edwardlzy/SRCNN/blob/master/Set5_result/butterfly_GT.bmp)

<br><br>

Bicubic interpolation:<br>
![bicubic](https://github.com/Edwardlzy/SRCNN/blob/master/Set5_result/bicubic_head_GT.bmp)<br>
SRCNN:<br>
![srcnn](https://github.com/Edwardlzy/SRCNN/blob/master/Set5_result/head_GT.bmp)

<br><br>

Bicubic interpolation:<br>
![bicubic](https://github.com/Edwardlzy/SRCNN/blob/master/Set5_result/bicubic_baby_GT.bmp)<br>
SRCNN:<br>
![srcnn](https://github.com/Edwardlzy/SRCNN/blob/master/Set5_result/baby_GT.bmp)
<br><br>

We can also feed any image to this model to get an upscaled version with interpolated details:<br>
Original image:<br>
![lenna](https://github.com/Edwardlzy/SRCNN/blob/master/result/lenna.bmp)<br>
SRCNN:<br>
![3xlenna](https://github.com/Edwardlzy/SRCNN/blob/master/result/lenna_3x.png)

  
Reference:

* [Dong, C., Loy, C.C., He, K., Tang, X.: Learning a Deep Convolutional Network for Image Super-Resolution](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html). <br>
* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)
  * - I have followed the loading and storing of h5 format files of this repository.
