# Retinal Fundus Image Preprocessing Pipeline for Lesion Segmentation

This repository contains an end-to-end, high-performance medical image preprocessing pipeline tailored specifically for the **IDRiD (Indian Diabetic Retinopathy Image Dataset)** segmentation task. Built on top of **PyTorch** and **Albumentations**, this engine streams raw high-resolution fundus photographs on-the-fly, isolates optimal diagnostic features, and outputs structurally normalized tensors perfectly optimized for deep learning models like U-Net, Attention U-Net, or TransUNet.

---

## 🌟 Key Features
* **Zero-Disk Streaming:** Processes images and masks dynamically in RAM via a custom PyTorch `Dataset` wrapper, saving disk storage and reducing I/O overhead.
* **Domain-Specific Enhancements:** Leverages retinal optics principles (Green-channel isolation + CLAHE) to dramatically boost microaneurysm, exudate, and vessel contrast.
* **Strict Mask Fidelity:** Uses nearest-neighbor spatial interpolation during resizing to completely prevent edge-blurring or label degradation on ground-truth segmentation masks.
* **Synchronized Augmentation:** Ensures spatial modifications (flips, rotations) are perfectly mirrored across both inputs and targets simultaneously using Albumentations.

---

## 🛠️ Pipeline Architecture & Steps

```text
[Step 1: Ingestion] ──────► Loads high-res Image (RGB) & Mask (Grayscale)
         │
[Step 2: Image Cutting] ──► Synchronized bounding box crop to strip out black background margins
         │
[Step 3: Uniform Resize] ─► Scale arrays to (512, 512) via Bilinear (Image) and Nearest (Mask)
         │
[Step 4: Contrast Opt] ───► Extract Green Channel + Apply CLAHE to flatten flash illumination
         │
[Step 5: Denoising] ──────► 3x3 Median Blur to erase sensor grain while keeping structures sharp
         │
[Step 6: Augmentation] ───► Synchronized geometric Flips & Rotations via Albumentations
         │
[Step 7: Tensor Format] ──► Normalize pixel values [0, 1] & cast to channel-first PyTorch tensors
