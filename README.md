

# 📦 Image Denoising with Autoencoder (PyTorch)

This repository implements an autoencoder-based image denoising pipeline in PyTorch, inspired by the Kaggle notebook *“Image Denoising Using Autoencoder (PyTorch)”* ([kaggle.com][1]).

---

## 🧠 Overview

* A **convolutional autoencoder** is trained to map noisy images → clean images.
* Useful for removing Gaussian, salt‑and‑pepper, or other synthetic noise.
* Designed for academic, educational, and research use.

---

## 🏗️ Architecture

The network is a symmetric encoder–decoder with intermediate latent space:

* **Encoder**: series of Conv2D → ReLU → MaxPool layers
* **Latent bottleneck**: captures image representation
* **Decoder**: ConvTranspose2D / Upsample layers to reconstruct the input
* **Loss**: MSE (L2) between reconstruction and clean image

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/image-denoising-autoencoder.git
cd image-denoising-autoencoder
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

*(Requirements include `torch`, `torchvision`, `matplotlib`, `Pillow`)*

---

## 🧪 Usage

### 1. Prepare Dataset

Place your clean images in `data/clean/` and noisy versions in `data/noisy/`, matching filenames one-to-one:

```
data/
 ├── clean/
 │    ├── img1.png
 │    ├── ...
 └── noisy/
      ├── img1.png
      ├── ...
```

### 2. Train the Model

```bash
python train.py \
  --clean_dir data/clean \
  --noisy_dir data/noisy \
  --epochs 50 \
  --batch_size 16 \
  --lr 1e-3
```

Monitor training loss, PSNR, and SSIM on validation sets.

### 3. Test / Evaluate

```bash
python evaluate.py \
  --checkpoint best_model.pth \
  --test_noisy_dir data/test_noisy \
  --output_dir results/
```

Generates denoised images in `results/` and prints quantitative metrics.

---

## 📌 Repository Structure

```
.
├── train.py           # Training script
├── evaluate.py        # Evaluation/inference script
├── models.py          # Autoencoder model definition
├── datasets.py        # Custom PyTorch dataset loader
├── requirements.txt   # Required Python packages
└── README.md          # This file
```

---

## 🧩 Example Performance

| Noise Type         | PSNR (dB) | SSIM |
| ------------------ | --------- | ---- |
| Gaussian σ = 25    | \~29.5    | 0.85 |
| Salt‑and‑pepper 5% | \~30.1    | 0.87 |

*Your results may vary depending on dataset and hyperparameters.*

---

## 🧠 Extensions & Ideas

* Use **denoising autoencoders** or variants like **skip connections**, **batch normalization**, or **residual learning**.
* Experiment with **different loss functions**: L1, SSIM, perceptual (VGG).
* Train on real-world noisy datasets like **SIDD**, **DND**, or **BSD68**.
* Convert to **ONNX or TensorFlow Lite** for deployment on edge devices.

---

## 🔗 References & Credits

* Original Kaggle notebook: *“Image Denoising Using Autoencoder (PyTorch)”* ([arxiv.org][2])
* Autoencoder fundamentals: \[Autoencoder — Wikipedia] ([en.wikipedia.org][3])

---

## 🎓 License

This project is released under the **MIT License**.


