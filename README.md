# efficient_LDCT

An efficient sinogram-domain fully convolutional interpolation network for sparse-view CT reconstruction
---------------
#Summary：

Inspired by super-resolution networks for natural images, the proposed model interpolates projection data directly in the sinogram domain with a fully convolutional neural network that consists of only four convolution layers. 
The proposed model can be used directly for sparse-view CT reconstruction by concatenating the classic filtered back-projection (FBP) module, or it can be incorporated into existing dual-domain reconstruction frameworks as a generic sinogram-domain module.

#he code was written based on python 3.8, tensorflow 2.4. 


