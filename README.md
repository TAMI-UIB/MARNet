# Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening

[![arXiv](https://img.shields.io/badge/arXiv-<arxiv_id>-B31B1B.svg)](https://arxiv.org/abs/<arxiv_id>)

This repository contains the implementation and additional resources for the paper:

**Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening**  
*Ivan Pereira-Sánchez, Eloi Sans, Julia Navarro, Joan Duran*  
Submmited to International Journal of Computer Vision 

## Abstract
The objective of pansharpening and hypersharpening is to accurately combine a high-resolution panchromatic (PAN) image with a low-resolution multispectral (MS) or hyperspectral (HS) image, respectively. Unfolding fusion methods integrate the powerful representation capabilities of deep learning with the robustness of model-based approaches. These techniques involve unrolling the steps of the optimization scheme derived from the minimization of an energy into a deep learning framework, resulting in efficient and highly interpretable architectures. In this paper, we propose a model-based deep unfolded method for satellite image fusion. Our approach is based on a variational formulation that incorporates the classic observation model for MS/HS data, a high-frequency injection constraint based on the PAN image, and an arbitrary convex prior. For the unfolding stage, we introduce upsampling and downsampling layers that use geometric information encoded in the PAN image through residual networks. The backbone of our method is a multi-head attention residual network (MARNet), which replaces the proximity operator in the optimization scheme and combines multiple head attentions with residual learning to exploit image self-similarities via nonlocal operators defined in terms of patches. Additionally, we incorporate a post-processing module based on the MARNet architecture to further enhance the quality of the fused images. Experimental results on PRISMA, Quickbird, and WorldView2 datasets demonstrate the superior performance of our method and its ability to generalize across different sensor configurations and varying spatial and spectral resolutions.

### Ongoing

We will soon upload the code for the training and test that we have used.

## arXiv Preprint

The paper is currently under revision, and the preprint is available on [arXiv](https://arxiv.org/abs/2409.02675).

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{pansharpening2024,
  title={Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening},
  author={Pereira-S{\'a}nchez, Ivan and Sans, Eloi and Navarro, Julia and Duran, Joan},
  journal={arXiv preprint arXiv:2409.02675},
  year={2024}
}
