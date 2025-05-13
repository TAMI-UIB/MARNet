# Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening

[![arXiv](https://img.shields.io/badge/arXiv-2409.02675-B31B1B.svg)](https://arxiv.org/abs/2409.02675)

This repository contains the implementation and additional resources for the paper:

**Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening**  
*Ivan Pereira-Sánchez, Eloi Sans, Julia Navarro, Joan Duran*  
Submmited to the International Journal of Computer Vision 

---

## 📄 Abstract
The objective of pansharpening and hypersharpening is to accurately fuse a high-resolution panchromatic (PAN) image with a low-resolution multispectral (MS) or hyperspectral (HS) image, respectively. Unfolding fusion methods integrate the powerful representation capabilities of deep learning with the robustness of model-based approaches. These techniques usually involve unrolling the steps of the optimization scheme derived from the minimization of a variational energy into a deep learning framework, resulting in efficient and highly interpretable architectures. In this paper, we present a model-based deep unfolded method for satellite image fusion. Our approach relies on a variational formulation that incorporates the classic observation model for MS/HS data, a high-frequency injection constraint, and a general prior. For the unfolding stage, we design upsampling and downsampling layers that leverage geometric information encoded in the PAN image through residual networks. The core of our method is a Multi-Head Attention Residual Network (MARNet), which combines multiple head attentions with residual learning to capture image self-similarities using nonlocal patch-based operators. Additionally, we include a post-processing module based on the MARNet architecture to further enhance the quality of the fused images. Experimental results on PRISMA, QuickBird, and WorldView2 datasets demonstrate the superior performance of our method, both at reduced and full-scale resolutions, along with its ability to generalize across different sensor configurations and varying spatial and spectral resolutions.

---

## 📚 arXiv Preprint

The paper is currently under revision, and the first preprint is available on [arXiv](https://arxiv.org/abs/2409.02675).


---

## 🛠️ Environment

You can set up the development environment using either **Conda** or **pip**.

#### 📦 Option 1: Using Conda (`environment.yml`)

1. Create the environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate MARNet
   ```

---

#### 💡 Option 2: Using pip (`requirements.txt`)

1. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
---

## ⚙️ Setup

To begin, create an .env file in the project root directory and define the `DATASET_PATH` variable, pointing to the directory where your dataset is stored.

We provide an example DataModule using WorldView-2 satellite imagery. This module requires the data to be preprocessed according to the Wald protocol and stored as cropped .h5 files.

Also, you can adapt the dataset class accordingly how you have the data stored. Please note that we are unable to share the dataset used for training due to data access restrictions.

---
## Train

Run the following command:
   ```bash
   python train.py 
   ```
---
## Test

For reduced resolution run following command:

   ```bash
   python test_ref.py +model.ckpt_path=${CKPT_PATH} 
   ```

For full resolution run following command:
```bash
python test_non_ref.py +model.ckpt_path=${CKPT_PATH} 
```

Make sure to replace `${CKPT_PATH}` with the actual path to your checkpoint file.

---
## 📌 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{pansharpening2024,
  title={Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening},
  author={Pereira-S{\'a}nchez, Ivan and Sans, Eloi and Navarro, Julia and Duran, Joan},
  journal={arXiv preprint arXiv:2409.02675},
  year={2024}
}
```