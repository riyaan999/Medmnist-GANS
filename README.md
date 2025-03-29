# ChestMNIST-Image-Generation-GANs

### **Image Generation using LSGAN, WGAN, and WGAN-GP**  

## **Overview**  
This repository contains implementations of three Generative Adversarial Networks (GANs) for generating synthetic chest X-ray images from the **ChestMNIST** dataset:  

- **Least Squares GAN (LSGAN)** - Uses mean squared error (MSE) loss for stable training.  
- **Wasserstein GAN (WGAN)** - Uses the Wasserstein distance for improved gradient flow.  
- **Wasserstein GAN with Gradient Penalty (WGAN-GP)** - Adds gradient penalty to WGAN for better convergence.  

These models generate realistic medical images that can be used for data augmentation, anomaly detection, or deep learning research.  

---

## **Dataset**  
We use **ChestMNIST** from the MedMNIST dataset collection. It consists of **chest X-ray images** for binary classification (normal vs. pneumonia). The dataset is automatically downloaded using the `medmnist` package.  

**Preprocessing**:  
- Images are resized and normalized to **[−1,1]** for GAN training.  
- Torchvision transforms are used for standardization.  

---

## **Model Architectures**  
### **Generator**  
- Takes a **random noise vector** as input.  
- Passes through **fully connected layers and activation functions** (ReLU, Tanh).  
- Outputs a **28×28 grayscale image** matching ChestMNIST format.  

### **Discriminator**  
- A CNN-based model that **classifies real vs. generated images**.  
- Uses **binary cross-entropy loss (LSGAN)** or **Wasserstein loss (WGAN & WGAN-GP)**.  
- The WGAN-GP discriminator includes **gradient penalty** for stable training.  

---

## **Installation**  
First, install the required dependencies:  
```bash
pip install torch torchvision tensorboard medmnist
```

## **Training**  
### **Run the following functions to train different GANs:**  
```python
train_lsgan()   # Train LSGAN
train_wgan()    # Train WGAN
train_wgan_gp() # Train WGAN-GP
```
By default, the models train for 50 epochs using the Adam optimizer.

**Training Details:**  
- Batch size: 64  
- Learning rate: 0.0002  
- Optimizer: Adam (β1=0.5, β2=0.999)  

---

## **Generation & Visualization**  
### **Generate Fake Images**  
After training, you can generate synthetic chest X-ray images using:  
```python
z = torch.randn(1, 100).to(device)
fake_image = generator(z).detach().cpu()
```

### **Monitor Training with TensorBoard**  
To track losses and generated images, use TensorBoard:  
```bash
%load_ext tensorboard
tensorboard --logdir=runs/
```

---

## **Results**  
Each GAN produces different results:

### **LSGAN Results**  
LSGAN generates blurry but stable images.
<p align="center">
  <img src="https://github.com/user-attachments/assets/bece7f31-12e3-4165-8ff9-99dd8c4e3a45" width="45%">
  <img src="https://github.com/user-attachments/assets/9e1bbb65-605c-4f00-8ea8-852877e677db" width="45%">
</p>

### **WGAN Results**  
WGAN improves realism with better loss behavior.
<p align="center">
  <img src="https://github.com/user-attachments/assets/776bf295-6562-407e-82f9-d87365974a3c" width="45%">
  <img src="https://github.com/user-attachments/assets/b344a6ce-36c6-4754-9279-42fdab932369" width="45%">
</p>

### **WGAN-GP Results**  
WGAN-GP produces the highest quality images with stable gradients.
<p align="center">
  <img src="https://github.com/user-attachments/assets/56e51599-7d19-4e2f-b2da-60743c93c27f" width="45%">
  <img src="https://github.com/user-attachments/assets/ee9fc397-5507-4299-a46b-f4a69b2940c2" width="45%">
</p>

Example outputs are saved in the `results/` folder.

---

## **Evaluation**  
| Metric                          | LSGAN       | WGAN        | WGAN-GP     |
|---------------------------------|------------|------------|------------|
| **Discriminator Loss (Smoothed)** | 0.2321     | 0.0574     | 9.4314     |
| **Discriminator Loss (Value)**    | 0.21       | 0.2721     | 13.4622    |
| **Generator Loss (Smoothed)**     | 0.346      | -0.2897    | -27.9073   |
| **Generator Loss (Value)**        | 0.3777     | -0.0101    | -37.5647   |
| **Observations**                   | Blurry but stable images | Improved realism with better loss behavior | Highest quality images with stable gradients |

---

## **Future Scope**
- Implement **Progressive Growing GANs (PGGANs)** for better high-resolution image generation.
- Fine-tune **hyperparameters** to further improve image quality.
- Use **self-supervised learning** to enhance GAN performance.
- Apply **transfer learning** techniques to improve generator and discriminator efficiency.

---

## **Contributors**
- **[Your Name]** - Model Implementation & Training
- **[Contributor Name]** - Data Preprocessing & Evaluation
- **[Contributor Name]** - Documentation & Deployment

For any queries, feel free to raise an issue or contribute to this repository!

---

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
