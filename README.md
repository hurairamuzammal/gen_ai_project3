# Self-Supervised Learning with Masked Autoencoders (MAE)

This project implements a **Masked Autoencoder (MAE)** based on the architecture proposed by He et al. (2021), applied to the **TinyImageNet** dataset. MAE is a powerful self-supervised learning technique that enables Vision Transformers (ViT) to learn highly semantic features from unlabeled image data.

## 🚀 Key Features
- **Scalable Architecture**: Implements the ViT-Base encoder and a lightweight transformer decoder.
- **Asymmetric Design**: The encoder only processes visible patches (25% of the image), significantly reducing memory and compute overhead.
- **Efficient Pre-training**: Designed for masked patch reconstruction using Mean Squared Error (MSE) loss.
- **Mixed Precision Training**: Utilizes PyTorch `autocast` and `GradScaler` for optimized performance on NVIDIA GPUs.

## 🛠️ Architecture Overview
The model operates on image patches ($16 \times 16$):
1. **Masking**: 75% of the patches are randomly masked out.
2. **Encoder**: A standard Vision Transformer (ViT) that only receives the visible (unmasked) patches.
3. **Decoder**: A smaller Transformer that receives the latent representation from the encoder plus learned "mask tokens" to reconstruct the original image pixels.
4. **Reconstruction**: The goal is to predict the pixel values of the original masked patches.

## 📁 Dataset
The project uses the **TinyImageNet** dataset:
- **Classes**: 200
- **Training Images**: 100,000
- **Validation Images**: 10,000
- **Resolution**: Resized to $224 \times 224$ for standard ViT compatibility.

## 📈 Training Details
- **Optimizer**: AdamW ($lr=1.5e-4$, $weight\_decay=0.05$)
- **Scheduler**: Cosine Annealing over 30 epochs.
- **Loss Function**: Patch-normalized MSE loss.
- **Device Support**: Multi-GPU (DataParallel) and CUDA.

## 🖥️ Getting Started
1. **Prerequisites**:
   ```bash
   pip install torch torchvision numpy pandas matplotlib pillow
   ```
2. **Execution**:
   Open `genai(1).ipynb` in Jupyter or Kaggle and run the cells sequentially.

## 📊 Results
The model's progress is tracked via reconstruction loss across epochs. You can find the loss curves generated as `loss_curve.png` in the output directory.

## 📚 References
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2021).
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020).
