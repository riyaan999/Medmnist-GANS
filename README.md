# MedMNIST Image Generation GANs

## Overview
This project implements **LSGAN, WGAN, and WGAN-GP** to generate synthetic **ChestMNIST X-ray images** for medical AI applications. Using PyTorch, these GANs enhance training stability and image realism through techniques like **least squares loss**, **Wasserstein distance**, and **gradient penalty**. The dataset is preprocessed and trained over **50 epochs** with **Adam optimizer**. Generated images are evaluated using **loss metrics**, and results indicate **WGAN-GP produces the highest-quality outputs**. 
