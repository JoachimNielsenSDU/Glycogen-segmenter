# Glycogen Segmenter

A deep learning model for automatically predicting glycogen within distinct subcellular regions of skeletal muscle fibers using transmission electron microscopy (TEM) images.

## Overview

This repository contains a U-Net based segmentation model trained to identify subcellular regions and quantify glycogen distribution in skeletal muscle TEM images. The model combines two separate neural networks: one for predicting subcellular regions (A-band, I-band, intermyofibrillar, mitochondria, and Z-disc) and another for glycogen detection.

The raw images, annotated masks, and pre-trained model weights are available at: https://zenodo.org/uploads/18390286 (DOI: 10.5281/zenodo.18390286)

## Getting Started

### Prerequisites

- Python 3.11 or later
- Conda or pip package manager

### Installation

1. **Clone or download this repository:**
   ```bash
   git clone https://github.com/JoachimNielsenSDU/Glycogen-segmenter.git
   cd glycogen-segmenter
   ```
   You can also download and unzip.

2. **Create and activate a Python environment:**
   ```bash
   conda create --name glyco python=3.11
   conda activate glyco
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *You must be in the repository directory for this step to work*

4. **Set up model weights:**
   Download the pre-trained model weights from the [Zenodo repository](https://zenodo.org/uploads/18390286) and place them in a `weights` directory within the repository (you must create this yourself):
   ```
   weights/
   ├── glycogen_model.pth
   └── region_model.pth
   ```

### Running the GUI

To launch the interactive GUI:
```bash
python gui.py
```

The application will open in your default web browser.

#### Custom Model Weights Paths

If your model weights are located in a non-standard location, you can specify the paths via command-line arguments:
```bash
python gui.py --glycogen_model /path/to/glycogen_model.pth --region_model /path/to/region_model.pth
```

## Model Capabilities

- **Subcellular region detection:** A-band, I-band, intermyofibrillar regions, mitochondria, and Z-disc identification
- **Glycogen quantification:** Automatic segmentation of glycogen deposits
- **Muscle fiber typing:** Z-disc width and mitochondrial distribution provide fiber type indicators

### Important Notes

- The model was trained exclusively on myofibrillar regions. When applied to subsarcolemmal images, subsarcolemmal glycogen is classified as intermyofibrillar.
- The model can additionally predict mitochondria and Z-disc width, which may serve as fiber type indicators.

## Training

The model consists of two separate Attention U-Net architectures trained independently:
- **Region model:** Predicts seven subcellular compartments (binary for Z-disc and mitochondria, multi-class for regions)
- **Glycogen model:** Binary glycogen detection

When executing the training scripts (`train.py` for region model and `train_glycogen.py` for glycogen model), the following modules are utilized:
- `unet.py` - Attention U-Net implementation
- `datagenerator.py` - Data loading and augmentation
- `loss.py` - Custom loss functions
- `nrrdreader.py` - NRRD file I/O

Training data should be organized as:
- Raw images in TIFF format
- Annotations in NRRD format (`.seg.nrrd` files)
- Separate datasets for regions and glycogen (available at the [Zenodo repository](https://zenodo.org/uploads/18390286))
