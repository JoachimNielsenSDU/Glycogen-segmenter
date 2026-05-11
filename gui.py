import os
import argparse
import re

# Work around mixed OpenMP runtimes (libomp/libiomp) on some Windows Conda setups.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import tifffile
import torch
import torch.optim as optim
from contextlib import nullcontext
from datetime import datetime
import json
from PIL import Image
from unet import AttUNet
import gradio as gr
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms.v2.functional import gaussian_blur

# Import training components
try:
    from datagenerator import ImageSegmentationDataset, ImagePatchBuffer
    from loss import CategoricalFocalLoss
    from nrrdreader import NRRDReader
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

default_label_mapping = """{
    "A-band": ("red", [0]),
    "I-band": ("blue", [1]),
    "intermyofibrillar": ("green", [2]),
    "mitochondria": ("orange", [5]),
    "z-disc": ("yellow", [6])
}"""

# ---------- Resize helpers (scale + snap to 8/16/32) ----------
def ceil_mult(x: int, mult: int = 128) -> int:
    if mult <= 0:
        return int(x)
    return int(np.ceil(x / float(mult))) * mult

def resize_with_snap(img: np.ndarray, scale: float = 1.0, mult: int = 16) -> np.ndarray:
    """
    Resizes image by 'scale' then snaps H/W to nearest multiple of 'mult'.
    Preserves original dtype range (no normalization here).
    Works for 2D or 3D arrays (H,W) or (H,W,C).
    """
    if img is None:
        return img
    if scale is None:
        scale = 1.0
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    H, W = img.shape[:2]

    # First resize by scale (if needed)
    if not np.isclose(scale, 1.0, rtol=1e-6):
        Hs = max(1, int(round(H * scale)))
        Ws = max(1, int(round(W * scale)))
        interp = cv2.INTER_LINEAR
        img = cv2.resize(img, (Ws, Hs), interpolation=interp)
        H, W = img.shape[:2]

    # Then snap to multiple (if needed)
    H2 = ceil_mult(H, mult)
    W2 = ceil_mult(W, mult)
    if (H2, W2) != (H, W):
        interp = cv2.INTER_LINEAR
        img = cv2.resize(img, (W2, H2), interpolation=interp)

    return img


def resolve_training_model_path(model_file, model_path_text):
    if model_file is not None:
        return model_file.name
    return (model_path_text or "").strip()


def parse_segment_names(segment_names_text):
    if not segment_names_text or not segment_names_text.strip():
        return ["Glycogen", "Background"]
    return [name.strip() for name in segment_names_text.split(",") if name.strip()]


def soften_binary_boundaries(labels, sigma):
    if sigma <= 0:
        return labels.float()

    valid_mask = labels >= 0
    if not torch.any(valid_mask):
        return labels.float()

    labels_float = torch.where(valid_mask, labels.float(), torch.zeros_like(labels, dtype=torch.float32))
    kernel_size = max(3, int(2 * round(3 * float(sigma)) + 1))

    blurred_labels = gaussian_blur(labels_float, kernel_size=[kernel_size, kernel_size], sigma=[float(sigma), float(sigma)])
    blurred_mask = gaussian_blur(valid_mask.float(), kernel_size=[kernel_size, kernel_size], sigma=[float(sigma), float(sigma)])
    softened = blurred_labels / blurred_mask.clamp_min(1e-6)
    softened = softened.clamp(0.0, 1.0)

    return torch.where(valid_mask, softened, torch.full_like(softened, -1.0))


def _normalize_training_stem(file_name):
    stem, _ = os.path.splitext(file_name)
    if stem.lower().endswith('.seg'):
        stem = stem[:-4]
    return stem.lower()


def _collect_stems_from_dir(dir_path, extensions):
    stems = set()
    for file_name in os.listdir(dir_path):
        if file_name.lower().endswith(extensions):
            stems.add(_normalize_training_stem(file_name))
    return stems


def collect_dataset_name_mismatches(images_dir, labels_dir):
    if not images_dir or not os.path.exists(images_dir):
        return None, None, [f"Images directory not found: {images_dir}"]
    if not labels_dir or not os.path.exists(labels_dir):
        return None, None, [f"Labels directory not found: {labels_dir}"]

    image_stems = _collect_stems_from_dir(images_dir, ('.tif', '.tiff'))
    label_stems = _collect_stems_from_dir(labels_dir, ('.nrrd',))

    errors = []
    if not image_stems:
        errors.append(f"No .tif/.tiff image files found in: {images_dir}")
    if not label_stems:
        errors.append(f"No .nrrd label files found in: {labels_dir}")

    image_only = sorted(image_stems - label_stems)
    label_only = sorted(label_stems - image_stems)
    return image_only, label_only, errors


def build_mismatch_report(dataset_pairs):
    lines = []
    has_mismatch = False

    for dataset_title, images_dir, labels_dir, enabled in dataset_pairs:
        if not enabled:
            continue

        image_only, label_only, errors = collect_dataset_name_mismatches(images_dir, labels_dir)

        lines.append(f"[{dataset_title}]")
        if errors:
            has_mismatch = True
            lines.extend(errors)
            lines.append("")
            continue

        if not image_only and not label_only:
            lines.append("No mismatches found. Image/label file names align.")
            lines.append("")
            continue

        has_mismatch = True
        if image_only:
            lines.append("Image files with no matching label:")
            lines.extend([f"  - {name}" for name in image_only])
        if label_only:
            lines.append("Label files with no matching image:")
            lines.extend([f"  - {name}" for name in label_only])
        lines.append("")

    if not lines:
        return "No datasets selected to validate.", False

    return "\n".join(lines).strip(), has_mismatch


def finetune_model(model_file, model_path_text, images_dir, labels_dir, output_dir, model_name,
                   learning_rate, batch_size, num_epochs, early_stopping_patience,
                   patch_size, augmentation, reconstruction_weight, training_scale_factor, device):
    """Fine-tune a model with the given parameters"""
    
    if not TRAINING_AVAILABLE:
        return "Error: Training modules not available. Please ensure datagenerator.py and loss.py are in the same directory."
    
    # Convert augmentation string to boolean
    augmentation_bool = augmentation != "None"
    
    # Determine model path
    model_path = resolve_training_model_path(model_file, model_path_text)
    
    # Validate inputs
    errors = []
    if not model_path or not os.path.exists(model_path):
        errors.append(f"Model file not found: {model_path}")
    if not images_dir or not os.path.exists(images_dir):
        errors.append(f"Images directory not found: {images_dir}")
    if not labels_dir or not os.path.exists(labels_dir):
        errors.append(f"Labels directory not found: {labels_dir}")
    if not output_dir:
        errors.append("Output directory is required")
    if not model_name or not model_name.strip():
        errors.append("Model name is required")
    
    if errors:
        return "\n".join(errors)
    
    try:
        def infer_segment_names_from_nrrd(label_directory):
            label_files = sorted([f for f in os.listdir(label_directory) if f.lower().endswith('.nrrd')])
            if not label_files:
                raise ValueError(f"No .nrrd label files found in: {label_directory}")

            first_label_path = os.path.join(label_directory, label_files[0])
            seg_info = NRRDReader(first_label_path).get_segments()
            ordered_segment_names = [
                name for name, _ in sorted(seg_info.items(), key=lambda kv: kv[1]["index"])
            ]

            if not ordered_segment_names:
                raise ValueError(
                    f"Label file '{label_files[0]}' contains no segments. "
                    f"Please ensure NRRD files are properly annotated with segment metadata from Slicer."
                )

            return ordered_segment_names

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Log training parameters
        log_file = os.path.join(output_dir, f"{model_name}_training_log.txt")
        status_msg = f"Starting fine-tuning at {datetime.now()}\n\n"
        status_msg += f"Configuration:\n"
        status_msg += f"  Model: {model_path}\n"
        status_msg += f"  Images: {images_dir}\n"
        status_msg += f"  Labels: {labels_dir}\n"
        status_msg += f"  Learning Rate: {learning_rate}\n"
        status_msg += f"  Batch Size: {batch_size}\n"
        status_msg += f"  Epochs: {num_epochs}\n"
        status_msg += f"  Patch Size: {patch_size}\n"
        status_msg += f"  Training Scale Factor: {training_scale_factor}\n"
        status_msg += f"  Augmentation: {augmentation}\n"
        status_msg += f"  Augmentation: {augmentation}\n\n"
        
        with open(log_file, 'w') as f:
            f.write(status_msg)
        
        print(status_msg)
        status_msg += "Loading data...\n"
        
        # Load dataset - infer classes from NRRD
        segment_names = infer_segment_names_from_nrrd(labels_dir)
        num_classes = len(segment_names)
        status_msg += f"Detected {num_classes} classes from NRRD files: {segment_names}\n"

        imagebuffer = ImagePatchBuffer(
            images_dir,
            labels_dir,
            segment_names=segment_names,
            patch_size=patch_size,
            scale_factor=training_scale_factor
        )
        
        # Split dataset
        val_split = 0.2
        n = len(imagebuffer)
        indices = np.arange(n)
        split = int(np.floor(val_split * n))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_index, test_index = indices[split:], indices[:split]
        
        status_msg += f"Dataset: {n} total patches, {len(train_index)} train, {len(test_index)} validation\n"
        
        # Create datasets and dataloaders
        train_dataset = ImageSegmentationDataset(imagebuffer, augment=augmentation_bool, indices=train_index)
        validation_dataset = ImageSegmentationDataset(imagebuffer, augment=False, indices=test_index)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        
        status_msg += "Loading pre-trained model...\n"
        
        # Load model architecture and weights
        filters = [8, 16, 32, 64, 128, 256, 512, 1024]
        model = AttUNet(1, num_classes, filters, final_activation='softmax').to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        
        # Setup optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = CategoricalFocalLoss(num_classes=num_classes)
        scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
        
        training_history = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        status_msg += "Starting training...\n\n"
        print(status_msg)
        
        # Training loop
        for epoch in range(num_epochs):
            if epochs_without_improvement >= early_stopping_patience:
                status_msg += f'Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)\n'
                break
            
            model.train()
            total_loss = 0
            num_batches = len(dataloader)
            
            for batch_idx, (images, labels) in enumerate(dataloader):
                # AMP (Automatic Mixed Precision)
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == 'cuda' else nullcontext()
                with autocast_context:
                    images, labels = images.to(device), labels.to(device)
                    seg, recon = model.forward(images)
                    
                    # Combined loss: segmentation + reconstruction
                    loss = criterion(seg, labels) + reconstruction_weight * torch.mean((images - recon) ** 2)
                
                # Backward pass
                if device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                
                optimizer.zero_grad()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for images, labels in dataloader_val:
                    images, labels = images.to(device), labels.to(device)
                    seg, recon = model.forward(images)
                    loss = criterion(seg, labels) + reconstruction_weight * torch.mean((images - recon) ** 2)
                    total_val_loss += loss.item()
            
            avg_train_loss = total_loss / len(dataloader)
            avg_val_loss = total_val_loss / len(dataloader_val)
            
            training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            epoch_msg = f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n'
            status_msg += epoch_msg
            print(epoch_msg.strip())
            
            # Early stopping and checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")
                torch.save(model.state_dict(), best_model_path)
                status_msg += f'  ✓ Best model saved (val_loss: {avg_val_loss:.6f})\n'
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch+1}: val_loss={avg_val_loss:.6f} (BEST)\n")
            else:
                epochs_without_improvement += 1
                with open(log_file, 'a') as f:
                    f.write(f"Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}\n")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
        
        # Save history
        history_path = os.path.join(output_dir, f"{model_name}_history.npy")
        np.save(history_path, training_history)
        
        # Save config
        config = {
            'model_name': model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs_trained': len(training_history),
            'patch_size': patch_size,
            'num_classes': num_classes,
            'augmentation': augmentation,
            'reconstruction_weight': reconstruction_weight
        }
        config_path = os.path.join(output_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        status_msg += f"\n✓ Fine-tuning completed successfully!\n\n"
        status_msg += f"Best model: {os.path.join(output_dir, f'{model_name}_best.pth')}\n"
        status_msg += f"Training log: {log_file}\n"
        status_msg += f"History: {history_path}\n"
        status_msg += f"Config: {config_path}\n"
        status_msg += f"Best validation loss: {best_val_loss:.6f}\n"
        status_msg += f"Total epochs: {len(training_history)}\n"
        
        return status_msg
        
    except Exception as e:
        error_msg = f"Error during training: {str(e)}\n"
        import traceback
        error_msg += traceback.format_exc()
        print(error_msg)
        return error_msg


def finetune_glycogen_model(model_file, model_path_text, images_dir, labels_dir, output_dir, model_name,
                            learning_rate, batch_size, num_epochs, patch_size, augmentation,
                            training_scale_factor, save_frequency, segment_names_text,
                            use_boundary_softening, boundary_softening_sigma, device):
    """Fine-tune a glycogen model using the binary training flow from train_glycogen.py."""

    if not TRAINING_AVAILABLE:
        return "Error: Training modules not available. Please ensure datagenerator.py and loss.py are in the same directory."

    augmentation_bool = augmentation != "None"
    model_path = resolve_training_model_path(model_file, model_path_text)
    segment_names = parse_segment_names(segment_names_text)

    errors = []
    if not model_path or not os.path.exists(model_path):
        errors.append(f"Model file not found: {model_path}")
    if not images_dir or not os.path.exists(images_dir):
        errors.append(f"Images directory not found: {images_dir}")
    if not labels_dir or not os.path.exists(labels_dir):
        errors.append(f"Labels directory not found: {labels_dir}")
    if not output_dir:
        errors.append("Output directory is required")
    if not model_name or not model_name.strip():
        errors.append("Model name is required")
    if not segment_names:
        errors.append("At least one segment name is required")

    if errors:
        return "\n".join(errors)

    try:
        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, f"{model_name}_training_log.txt")
        status_msg = f"Starting glycogen fine-tuning at {datetime.now()}\n\n"
        status_msg += "Configuration:\n"
        status_msg += f"  Model: {model_path}\n"
        status_msg += f"  Images: {images_dir}\n"
        status_msg += f"  Labels: {labels_dir}\n"
        status_msg += f"  Segment Names: {segment_names}\n"
        status_msg += f"  Learning Rate: {learning_rate}\n"
        status_msg += f"  Batch Size: {batch_size}\n"
        status_msg += f"  Epochs: {num_epochs}\n"
        status_msg += f"  Patch Size: {patch_size}\n"
        status_msg += f"  Training Scale Factor: {training_scale_factor}\n"
        status_msg += f"  Augmentation: {augmentation}\n"
        status_msg += f"  Save Frequency: {save_frequency}\n\n"
        status_msg += f"  Boundary Softening: {bool(use_boundary_softening)}\n"
        status_msg += f"  Boundary Softening Sigma: {float(boundary_softening_sigma):.3f}\n\n"

        with open(log_file, 'w') as f:
            f.write(status_msg)

        print(status_msg)
        status_msg += "Loading data...\n"

        imagebuffer = ImagePatchBuffer(
            images_dir,
            labels_dir,
            segment_names=segment_names,
            patch_size=patch_size,
            scale_factor=training_scale_factor
        )

        val_split = 0.2
        n = len(imagebuffer)
        indices = np.arange(n)
        split = int(np.floor(val_split * n))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_index, test_index = indices[split:], indices[:split]

        status_msg += f"Dataset: {n} total patches, {len(train_index)} train, {len(test_index)} validation\n"

        train_dataset = ImageSegmentationDataset(imagebuffer, augment=augmentation_bool, indices=train_index)
        validation_dataset = ImageSegmentationDataset(imagebuffer, augment=False, indices=test_index)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        status_msg += "Loading pre-trained glycogen model...\n"

        filters = [8, 16, 32, 64, 128, 256, 512]
        model = AttUNet(1, 1, filters, final_activation='sigmoid', include_reconstruction=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = CategoricalFocalLoss(num_classes=len(segment_names))
        scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

        training_history = []
        best_val_loss = float('inf')
        best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")

        status_msg += "Starting glycogen training...\n\n"
        print(status_msg)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for images, labels in dataloader:
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == 'cuda' else nullcontext()
                with autocast_context:
                    images, labels = images.to(device), labels.to(device)

                    unlabeled = torch.sum(labels, axis=1) < 0
                    labels = torch.argmax(labels, axis=1)
                    labels[unlabeled] = -1
                    labels = labels.unsqueeze(1)
                    if use_boundary_softening and float(boundary_softening_sigma) > 0:
                        labels = soften_binary_boundaries(labels, float(boundary_softening_sigma))

                    seg = model.forward(images)
                    loss = criterion(seg, labels)

                if device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                optimizer.zero_grad()
                total_loss += loss.item()

            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for images, labels in dataloader_val:
                    images, labels = images.to(device), labels.to(device)

                    unlabeled = torch.sum(labels, axis=1) < 0
                    labels = torch.argmax(labels, axis=1)
                    labels[unlabeled] = -1
                    labels = labels.unsqueeze(1)

                    seg = model.forward(images)
                    loss = criterion(seg, labels)
                    total_val_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader)
            avg_val_loss = total_val_loss / len(dataloader_val)

            training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            epoch_msg = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n"
            status_msg += epoch_msg
            print(epoch_msg.strip())

            with open(log_file, 'a') as f:
                f.write(epoch_msg)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)

            if (epoch + 1) % int(save_frequency) == 0:
                checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)

        history_path = os.path.join(output_dir, f"{model_name}_history.npy")
        np.save(history_path, training_history)

        final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
        torch.save(model.state_dict(), final_model_path)

        config = {
            'model_name': model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs_trained': len(training_history),
            'patch_size': patch_size,
            'segment_names': segment_names,
            'augmentation': augmentation,
            'training_scale_factor': training_scale_factor,
            'save_frequency': int(save_frequency),
            'use_boundary_softening': bool(use_boundary_softening),
            'boundary_softening_sigma': float(boundary_softening_sigma)
        }
        config_path = os.path.join(output_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        status_msg += "\n✓ Glycogen fine-tuning completed successfully!\n\n"
        status_msg += f"Best model: {best_model_path}\n"
        status_msg += f"Final model: {final_model_path}\n"
        status_msg += f"Training log: {log_file}\n"
        status_msg += f"History: {history_path}\n"
        status_msg += f"Config: {config_path}\n"
        status_msg += f"Best validation loss: {best_val_loss:.6f}\n"
        status_msg += f"Total epochs: {len(training_history)}\n"

        return status_msg

    except Exception as e:
        error_msg = f"Error during glycogen training: {str(e)}\n"
        import traceback
        error_msg += traceback.format_exc()
        print(error_msg)
        return error_msg


def train_glycogen_model_from_scratch(images_dir, labels_dir, output_dir, model_name,
                                      learning_rate, batch_size, num_epochs, save_frequency,
                                      segment_names_text, init_model_file, init_model_path_text,
                                      patch_size, augmentation, training_scale_factor,
                                      primary_resample_mode,
                                      include_second_dataset, images_dir_2, labels_dir_2,
                                      second_resample_mode, second_scale_value,
                                      include_third_dataset, images_dir_3, labels_dir_3,
                                      third_resample_mode, third_scale_value,
                                      use_boundary_softening, boundary_softening_sigma,
                                      device):
    """Train a glycogen model from scratch (or optional weight initialization) using train_glycogen.py flow."""

    if not TRAINING_AVAILABLE:
        return "Error: Training modules not available. Please ensure datagenerator.py and loss.py are in the same directory."

    segment_names = parse_segment_names(segment_names_text)
    augmentation_bool = augmentation != "None"
    init_model_path = resolve_training_model_path(init_model_file, init_model_path_text)

    def resolve_resample_scale(mode, value):
        v = float(value)
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid resampling factor: {value}. Must be > 0.")
        mode = (mode or "Direct scale factor").strip()
        if mode == "Up-sample by factor":
            return v
        if mode == "Down-sample by factor":
            return 1.0 / v
        return v

    errors = []
    if not images_dir or not os.path.exists(images_dir):
        errors.append(f"Images directory not found: {images_dir}")
    if not labels_dir or not os.path.exists(labels_dir):
        errors.append(f"Labels directory not found: {labels_dir}")
    if not output_dir:
        errors.append("Output directory is required")
    if not model_name or not model_name.strip():
        errors.append("Model name is required")
    if not segment_names:
        errors.append("At least one segment name is required")
    if init_model_path and not os.path.exists(init_model_path):
        errors.append(f"Initialization model file not found: {init_model_path}")

    include_second = bool(include_second_dataset)
    include_third = bool(include_third_dataset)

    if include_second:
        if not images_dir_2 or not os.path.exists(images_dir_2):
            errors.append(f"Second dataset images directory not found: {images_dir_2}")
        if not labels_dir_2 or not os.path.exists(labels_dir_2):
            errors.append(f"Second dataset labels directory not found: {labels_dir_2}")

    if include_third:
        if not images_dir_3 or not os.path.exists(images_dir_3):
            errors.append(f"Third dataset images directory not found: {images_dir_3}")
        if not labels_dir_3 or not os.path.exists(labels_dir_3):
            errors.append(f"Third dataset labels directory not found: {labels_dir_3}")

    if errors:
        return "\n".join(errors)

    try:
        primary_scale_factor = resolve_resample_scale(primary_resample_mode, training_scale_factor)
        second_scale_factor = None
        third_scale_factor = None
        if include_second:
            second_scale_factor = resolve_resample_scale(second_resample_mode, second_scale_value)
        if include_third:
            third_scale_factor = resolve_resample_scale(third_resample_mode, third_scale_value)

        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, f"{model_name}_training_log.txt")
        status_msg = f"Starting glycogen training from scratch at {datetime.now()}\n\n"
        status_msg += "Configuration:\n"
        status_msg += f"  Images: {images_dir}\n"
        status_msg += f"  Labels: {labels_dir}\n"
        status_msg += f"  Segment Names: {segment_names}\n"
        status_msg += f"  Learning Rate: {learning_rate}\n"
        status_msg += f"  Batch Size: {batch_size}\n"
        status_msg += f"  Epochs: {num_epochs}\n"
        status_msg += f"  Save Frequency: {save_frequency}\n"
        status_msg += f"  Patch Size: {patch_size}\n"
        status_msg += f"  Augmentation: {augmentation}\n"
        status_msg += f"  Primary Resample Mode: {primary_resample_mode}\n"
        status_msg += f"  Primary Scale Factor Applied: {primary_scale_factor}\n"
        status_msg += f"  Include Second Dataset: {include_second}\n"
        status_msg += f"  Include Third Dataset: {include_third}\n"
        status_msg += f"  Boundary Softening: {bool(use_boundary_softening)}\n"
        status_msg += f"  Boundary Softening Sigma: {float(boundary_softening_sigma):.3f}\n"
        if include_second:
            status_msg += f"  Images (Dataset 2): {images_dir_2}\n"
            status_msg += f"  Labels (Dataset 2): {labels_dir_2}\n"
            status_msg += f"  Second Resample Mode: {second_resample_mode}\n"
            status_msg += f"  Second Scale Factor Applied: {second_scale_factor}\n"
        if include_third:
            status_msg += f"  Images (Dataset 3): {images_dir_3}\n"
            status_msg += f"  Labels (Dataset 3): {labels_dir_3}\n"
            status_msg += f"  Third Resample Mode: {third_resample_mode}\n"
            status_msg += f"  Third Scale Factor Applied: {third_scale_factor}\n"
        if init_model_path:
            status_msg += f"  Initialization Weights: {init_model_path}\n"
        else:
            status_msg += "  Initialization Weights: None (pure scratch training)\n"
        status_msg += "\n"

        with open(log_file, 'w') as f:
            f.write(status_msg)

        print(status_msg)
        status_msg += "Loading data...\n"

        imagebuffer = ImagePatchBuffer(
            images_dir,
            labels_dir,
            segment_names=segment_names,
            patch_size=int(patch_size),
            scale_factor=primary_scale_factor
        )

        if include_second:
            imagebuffer_2 = ImagePatchBuffer(
                images_dir_2,
                labels_dir_2,
                segment_names=segment_names,
                patch_size=int(patch_size),
                scale_factor=second_scale_factor
            )
            imagebuffer.image_buffer.extend(imagebuffer_2.image_buffer)
            imagebuffer.label_buffer.extend(imagebuffer_2.label_buffer)
            status_msg += f"Merged dataset 2 patches: {len(imagebuffer_2)}\n"

        if include_third:
            imagebuffer_3 = ImagePatchBuffer(
                images_dir_3,
                labels_dir_3,
                segment_names=segment_names,
                patch_size=int(patch_size),
                scale_factor=third_scale_factor
            )
            imagebuffer.image_buffer.extend(imagebuffer_3.image_buffer)
            imagebuffer.label_buffer.extend(imagebuffer_3.label_buffer)
            status_msg += f"Merged dataset 3 patches: {len(imagebuffer_3)}\n"

        n = len(imagebuffer)
        if n < 2:
            return "Error: Need at least 2 extracted patches to create train/validation splits."

        val_split = 0.2
        indices = np.arange(n)
        split = max(1, int(np.floor(val_split * n)))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_index, test_index = indices[split:], indices[:split]

        if len(train_index) == 0 or len(test_index) == 0:
            return "Error: Unable to create non-empty train/validation split. Increase data amount or adjust settings."

        status_msg += f"Dataset: {n} total patches, {len(train_index)} train, {len(test_index)} validation\n"

        train_dataset = ImageSegmentationDataset(imagebuffer, augment=augmentation_bool, indices=train_index)
        validation_dataset = ImageSegmentationDataset(imagebuffer, augment=False, indices=test_index)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(validation_dataset, batch_size=int(batch_size), shuffle=False)

        filters = [8, 16, 32, 64, 128, 256, 512]
        model = AttUNet(1, 1, filters, final_activation='sigmoid', include_reconstruction=False).to(device)

        if init_model_path:
            model.load_state_dict(torch.load(init_model_path, map_location=device, weights_only=True))
            status_msg += "Loaded initialization weights.\n"

        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))
        criterion = CategoricalFocalLoss(num_classes=len(segment_names))
        scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

        history = []
        best_val_loss = float('inf')
        best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")

        status_msg += "Starting glycogen scratch training...\n\n"
        print(status_msg)

        for epoch in range(int(num_epochs)):
            total_loss = 0.0
            model.train()

            for images, labels in dataloader:
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == 'cuda' else nullcontext()
                with autocast_context:
                    images, labels = images.to(device), labels.to(device)

                    unlabeled = torch.sum(labels, axis=1) < 0
                    labels = torch.argmax(labels, axis=1)
                    labels[unlabeled] = -1
                    labels = labels.unsqueeze(1)
                    if use_boundary_softening and float(boundary_softening_sigma) > 0:
                        labels = soften_binary_boundaries(labels, float(boundary_softening_sigma))

                    seg = model.forward(images)
                    loss = criterion(seg, labels)

                if device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                optimizer.zero_grad()
                total_loss += loss.item()

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for images, labels in dataloader_val:
                    images, labels = images.to(device), labels.to(device)

                    unlabeled = torch.sum(labels, axis=1) < 0
                    labels = torch.argmax(labels, axis=1)
                    labels[unlabeled] = -1
                    labels = labels.unsqueeze(1)

                    seg = model.forward(images)
                    loss = criterion(seg, labels)
                    total_val_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader)
            avg_val_loss = total_val_loss / len(dataloader_val)

            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            epoch_msg = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n"
            status_msg += epoch_msg
            print(epoch_msg.strip())

            with open(log_file, 'a') as f:
                f.write(epoch_msg)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)

            if (epoch + 1) % int(save_frequency) == 0:
                checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)

        history_path = os.path.join(output_dir, f"{model_name}_history.npy")
        np.save(history_path, history)

        final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
        torch.save(model.state_dict(), final_model_path)

        config = {
            'model_name': model_name,
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'epochs_trained': len(history),
            'patch_size': int(patch_size),
            'segment_names': segment_names,
            'augmentation': augmentation,
            'primary_resample_mode': primary_resample_mode,
            'primary_scale_factor_applied': float(primary_scale_factor),
            'include_second_dataset': include_second,
            'second_dataset_images_dir': images_dir_2 if include_second else '',
            'second_dataset_labels_dir': labels_dir_2 if include_second else '',
            'second_resample_mode': second_resample_mode if include_second else '',
            'second_scale_factor_applied': float(second_scale_factor) if second_scale_factor is not None else 0.0,
            'include_third_dataset': include_third,
            'third_dataset_images_dir': images_dir_3 if include_third else '',
            'third_dataset_labels_dir': labels_dir_3 if include_third else '',
            'third_resample_mode': third_resample_mode if include_third else '',
            'third_scale_factor_applied': float(third_scale_factor) if third_scale_factor is not None else 0.0,
            'save_frequency': int(save_frequency),
            'initial_weights': init_model_path if init_model_path else '',
            'use_boundary_softening': bool(use_boundary_softening),
            'boundary_softening_sigma': float(boundary_softening_sigma)
        }
        config_path = os.path.join(output_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        status_msg += "\n✓ Glycogen training from scratch completed successfully!\n\n"
        status_msg += f"Best model: {best_model_path}\n"
        status_msg += f"Final model: {final_model_path}\n"
        status_msg += f"Training log: {log_file}\n"
        status_msg += f"History: {history_path}\n"
        status_msg += f"Config: {config_path}\n"
        status_msg += f"Best validation loss: {best_val_loss:.6f}\n"
        status_msg += f"Total epochs: {len(history)}\n"

        return status_msg

    except Exception as e:
        error_msg = f"Error during glycogen scratch training: {str(e)}\n"
        import traceback
        error_msg += traceback.format_exc()
        print(error_msg)
        return error_msg


def train_region_model_from_scratch(images_dir, labels_dir, output_dir, model_name,
                                    learning_rate, batch_size, num_epochs, early_stopping_patience,
                                    save_frequency, patch_size, augmentation, reconstruction_weight,
                                    primary_resample_mode, primary_scale_value,
                                    include_second_dataset, images_dir_2, labels_dir_2,
                                    second_resample_mode, second_scale_value,
                                    include_third_dataset, images_dir_3, labels_dir_3,
                                    third_resample_mode, third_scale_value,
                                    include_fourth_dataset, images_dir_4, labels_dir_4,
                                    fourth_resample_mode, fourth_scale_value,
                                    device):
    """Train a region model from scratch with optional additional datasets at different magnifications."""

    if not TRAINING_AVAILABLE:
        return "Error: Training modules not available. Please ensure datagenerator.py and loss.py are in the same directory."

    augmentation_bool = augmentation != "None"

    def resolve_resample_scale(mode, value):
        v = float(value)
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid resampling factor: {value}. Must be > 0.")
        mode = (mode or "Direct scale factor").strip()
        if mode == "Up-sample by factor":
            return v
        if mode == "Down-sample by factor":
            return 1.0 / v
        return v

    def infer_segment_names_from_nrrd(label_directory):
        label_files = sorted([f for f in os.listdir(label_directory) if f.lower().endswith('.nrrd')])
        if not label_files:
            raise ValueError(f"No .nrrd label files found in: {label_directory}")

        first_label_path = os.path.join(label_directory, label_files[0])
        seg_info = NRRDReader(first_label_path).get_segments()
        ordered_segment_names = [
            name for name, _ in sorted(seg_info.items(), key=lambda kv: kv[1]["index"])
        ]

        if not ordered_segment_names:
            raise ValueError(
                f"Label file '{label_files[0]}' contains no segments. "
                f"Please ensure NRRD files are properly annotated with segment metadata from Slicer."
            )

        return ordered_segment_names

    errors = []
    if not images_dir or not os.path.exists(images_dir):
        errors.append(f"Images directory not found: {images_dir}")
    if not labels_dir or not os.path.exists(labels_dir):
        errors.append(f"Labels directory not found: {labels_dir}")
    if not output_dir:
        errors.append("Output directory is required")
    if not model_name or not model_name.strip():
        errors.append("Model name is required")

    include_second = bool(include_second_dataset)
    include_third = bool(include_third_dataset)
    include_fourth = bool(include_fourth_dataset)

    if include_second:
        if not images_dir_2 or not os.path.exists(images_dir_2):
            errors.append(f"Second dataset images directory not found: {images_dir_2}")
        if not labels_dir_2 or not os.path.exists(labels_dir_2):
            errors.append(f"Second dataset labels directory not found: {labels_dir_2}")

    if include_third:
        if not images_dir_3 or not os.path.exists(images_dir_3):
            errors.append(f"Third dataset images directory not found: {images_dir_3}")
        if not labels_dir_3 or not os.path.exists(labels_dir_3):
            errors.append(f"Third dataset labels directory not found: {labels_dir_3}")

    if include_fourth:
        if not images_dir_4 or not os.path.exists(images_dir_4):
            errors.append(f"Fourth dataset images directory not found: {images_dir_4}")
        if not labels_dir_4 or not os.path.exists(labels_dir_4):
            errors.append(f"Fourth dataset labels directory not found: {labels_dir_4}")

    if errors:
        return "\n".join(errors)

    try:
        primary_scale_factor = resolve_resample_scale(primary_resample_mode, primary_scale_value)
        second_scale_factor = resolve_resample_scale(second_resample_mode, second_scale_value) if include_second else None
        third_scale_factor = resolve_resample_scale(third_resample_mode, third_scale_value) if include_third else None
        fourth_scale_factor = resolve_resample_scale(fourth_resample_mode, fourth_scale_value) if include_fourth else None

        segment_names = infer_segment_names_from_nrrd(labels_dir)
        num_classes = len(segment_names)

        os.makedirs(output_dir, exist_ok=True)

        log_file = os.path.join(output_dir, f"{model_name}_training_log.txt")
        status_msg = f"Starting region training from scratch at {datetime.now()}\n\n"
        status_msg += "Configuration:\n"
        status_msg += f"  Images (Dataset 1): {images_dir}\n"
        status_msg += f"  Labels (Dataset 1): {labels_dir}\n"
        status_msg += f"  Segment Names: {segment_names}\n"
        status_msg += f"  Num Classes: {num_classes}\n"
        status_msg += f"  Learning Rate: {float(learning_rate)}\n"
        status_msg += f"  Batch Size: {int(batch_size)}\n"
        status_msg += f"  Epochs: {int(num_epochs)}\n"
        status_msg += f"  Early Stopping Patience: {int(early_stopping_patience)}\n"
        status_msg += f"  Save Frequency: {int(save_frequency)}\n"
        status_msg += f"  Patch Size: {int(patch_size)}\n"
        status_msg += f"  Augmentation: {augmentation}\n"
        status_msg += f"  Reconstruction Weight: {float(reconstruction_weight)}\n"
        status_msg += f"  Primary Resample Mode: {primary_resample_mode}\n"
        status_msg += f"  Primary Scale Factor Applied: {primary_scale_factor}\n"
        status_msg += f"  Include Second Dataset: {include_second}\n"
        status_msg += f"  Include Third Dataset: {include_third}\n"
        status_msg += f"  Include Fourth Dataset: {include_fourth}\n"

        if include_second:
            status_msg += f"  Images (Dataset 2): {images_dir_2}\n"
            status_msg += f"  Labels (Dataset 2): {labels_dir_2}\n"
            status_msg += f"  Second Resample Mode: {second_resample_mode}\n"
            status_msg += f"  Second Scale Factor Applied: {second_scale_factor}\n"

        if include_third:
            status_msg += f"  Images (Dataset 3): {images_dir_3}\n"
            status_msg += f"  Labels (Dataset 3): {labels_dir_3}\n"
            status_msg += f"  Third Resample Mode: {third_resample_mode}\n"
            status_msg += f"  Third Scale Factor Applied: {third_scale_factor}\n"

        if include_fourth:
            status_msg += f"  Images (Dataset 4): {images_dir_4}\n"
            status_msg += f"  Labels (Dataset 4): {labels_dir_4}\n"
            status_msg += f"  Fourth Resample Mode: {fourth_resample_mode}\n"
            status_msg += f"  Fourth Scale Factor Applied: {fourth_scale_factor}\n"

        status_msg += "\n"

        with open(log_file, 'w') as f:
            f.write(status_msg)

        print(status_msg)
        status_msg += "Loading data...\n"

        imagebuffer = ImagePatchBuffer(
            images_dir,
            labels_dir,
            segment_names=segment_names,
            patch_size=int(patch_size),
            scale_factor=primary_scale_factor
        )

        if include_second:
            imagebuffer_2 = ImagePatchBuffer(
                images_dir_2,
                labels_dir_2,
                segment_names=segment_names,
                patch_size=int(patch_size),
                scale_factor=second_scale_factor
            )
            imagebuffer.image_buffer.extend(imagebuffer_2.image_buffer)
            imagebuffer.label_buffer.extend(imagebuffer_2.label_buffer)
            status_msg += f"Merged dataset 2 patches: {len(imagebuffer_2)}\n"

        if include_third:
            imagebuffer_3 = ImagePatchBuffer(
                images_dir_3,
                labels_dir_3,
                segment_names=segment_names,
                patch_size=int(patch_size),
                scale_factor=third_scale_factor
            )
            imagebuffer.image_buffer.extend(imagebuffer_3.image_buffer)
            imagebuffer.label_buffer.extend(imagebuffer_3.label_buffer)
            status_msg += f"Merged dataset 3 patches: {len(imagebuffer_3)}\n"

        if include_fourth:
            imagebuffer_4 = ImagePatchBuffer(
                images_dir_4,
                labels_dir_4,
                segment_names=segment_names,
                patch_size=int(patch_size),
                scale_factor=fourth_scale_factor
            )
            imagebuffer.image_buffer.extend(imagebuffer_4.image_buffer)
            imagebuffer.label_buffer.extend(imagebuffer_4.label_buffer)
            status_msg += f"Merged dataset 4 patches: {len(imagebuffer_4)}\n"

        n = len(imagebuffer)
        if n < 2:
            return "Error: Need at least 2 extracted patches to create train/validation splits."

        val_split = 0.2
        indices = np.arange(n)
        split = max(1, int(np.floor(val_split * n)))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_index, test_index = indices[split:], indices[:split]

        if len(train_index) == 0 or len(test_index) == 0:
            return "Error: Unable to create non-empty train/validation split. Increase data amount or adjust settings."

        status_msg += f"Dataset: {n} total patches, {len(train_index)} train, {len(test_index)} validation\n"

        train_dataset = ImageSegmentationDataset(imagebuffer, augment=augmentation_bool, indices=train_index)
        validation_dataset = ImageSegmentationDataset(imagebuffer, augment=False, indices=test_index)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
        dataloader_val = torch.utils.data.DataLoader(validation_dataset, batch_size=int(batch_size), shuffle=False)

        filters = [8, 16, 32, 64, 128, 256, 512, 1024]
        model = AttUNet(1, num_classes, filters, final_activation='softmax').to(device)

        optimizer = optim.Adam(model.parameters(), lr=float(learning_rate))
        criterion = CategoricalFocalLoss(num_classes=num_classes)
        scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

        history = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        best_model_path = os.path.join(output_dir, f"{model_name}_best.pth")

        status_msg += "Starting region scratch training...\n\n"
        print(status_msg)

        for epoch in range(int(num_epochs)):
            if epochs_without_improvement >= int(early_stopping_patience):
                status_msg += (
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {int(early_stopping_patience)} epochs)\n"
                )
                break

            total_loss = 0.0
            model.train()

            for images, labels in dataloader:
                autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == 'cuda' else nullcontext()
                with autocast_context:
                    images, labels = images.to(device), labels.to(device)
                    seg, recon = model.forward(images)
                    loss = criterion(seg, labels) + float(reconstruction_weight) * torch.mean((images - recon) ** 2)

                if device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                optimizer.zero_grad()
                total_loss += loss.item()

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for images, labels in dataloader_val:
                    images, labels = images.to(device), labels.to(device)
                    seg, recon = model.forward(images)
                    loss = criterion(seg, labels) + float(reconstruction_weight) * torch.mean((images - recon) ** 2)
                    total_val_loss += loss.item()

            avg_train_loss = total_loss / len(dataloader)
            avg_val_loss = total_val_loss / len(dataloader_val)

            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            epoch_msg = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n"
            status_msg += epoch_msg
            print(epoch_msg.strip())

            with open(log_file, 'a') as f:
                f.write(epoch_msg)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % int(save_frequency) == 0:
                checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)

        history_path = os.path.join(output_dir, f"{model_name}_history.npy")
        np.save(history_path, history)

        final_model_path = os.path.join(output_dir, f"{model_name}_final.pth")
        torch.save(model.state_dict(), final_model_path)

        config = {
            'model_name': model_name,
            'learning_rate': float(learning_rate),
            'batch_size': int(batch_size),
            'epochs_trained': len(history),
            'early_stopping_patience': int(early_stopping_patience),
            'patch_size': int(patch_size),
            'segment_names': segment_names,
            'num_classes': int(num_classes),
            'augmentation': augmentation,
            'reconstruction_weight': float(reconstruction_weight),
            'primary_resample_mode': primary_resample_mode,
            'primary_scale_factor_applied': float(primary_scale_factor),
            'include_second_dataset': include_second,
            'second_dataset_images_dir': images_dir_2 if include_second else '',
            'second_dataset_labels_dir': labels_dir_2 if include_second else '',
            'second_resample_mode': second_resample_mode if include_second else '',
            'second_scale_factor_applied': float(second_scale_factor) if second_scale_factor is not None else 0.0,
            'include_third_dataset': include_third,
            'third_dataset_images_dir': images_dir_3 if include_third else '',
            'third_dataset_labels_dir': labels_dir_3 if include_third else '',
            'third_resample_mode': third_resample_mode if include_third else '',
            'third_scale_factor_applied': float(third_scale_factor) if third_scale_factor is not None else 0.0,
            'save_frequency': int(save_frequency)
        }
        config_path = os.path.join(output_dir, f"{model_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        status_msg += "\n✓ Region training from scratch completed successfully!\n\n"
        status_msg += f"Best model: {best_model_path}\n"
        status_msg += f"Final model: {final_model_path}\n"
        status_msg += f"Training log: {log_file}\n"
        status_msg += f"History: {history_path}\n"
        status_msg += f"Config: {config_path}\n"
        status_msg += f"Best validation loss: {best_val_loss:.6f}\n"
        status_msg += f"Total epochs: {len(history)}\n"

        return status_msg

    except Exception as e:
        error_msg = f"Error during region scratch training: {str(e)}\n"
        import traceback
        error_msg += traceback.format_exc()
        print(error_msg)
        return error_msg


def _normalize_imf_region_name(region):
    return "".join(ch.lower() for ch in str(region) if ch.isalnum())


def _canonical_image_value(image_value):
    if pd.isna(image_value):
        return ""
    return str(image_value).strip()


def _image_sort_key(image_value):
    text = _canonical_image_value(image_value)
    try:
        return (0, float(text), text)
    except (TypeError, ValueError):
        pass

    nums = re.findall(r"\d+", text)
    if nums:
        return (1, tuple(int(n) for n in nums), text.lower())

    return (2, text.lower())


def _build_image_location_map(df):
    location_map = {}
    for subfolder, grp in df.groupby("Subfolder", sort=False):
        image_values = sorted(grp["Image"].tolist(), key=_image_sort_key)
        unique_images = []
        seen = set()
        for image_value in image_values:
            key = _canonical_image_value(image_value)
            if key in seen:
                continue
            seen.add(key)
            unique_images.append(key)

        n = len(unique_images)
        if n == 0:
            continue

        superficial = set(unique_images[:min(6, n)])
        central = set(unique_images[max(0, n - 6):])

        for image_key in unique_images:
            is_central = image_key in central
            is_superficial = image_key in superficial
            if is_central and is_superficial:
                location = "central/superficial"
            elif is_central:
                location = "central"
            elif is_superficial:
                location = "superficial"
            else:
                location = ""
            location_map[(subfolder, image_key)] = location

    return location_map


def _safe_div(numerator, denominator):
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def _compute_imf_region_set(glycogen_area, mean_feret_diameter, total_area, thickness_um,
                            aa_slope=None, aa_intercept=None, pixel_size_nm=0.7698,
                            mean_particle_area=None, diameter_method="feret"):
    """
    Compute IMF/intra stereology metrics.
    
    diameter_method: "feret" (default) uses mean_feret_diameter, 
                    "area_circle" uses particle_area assuming circular profiles
    """
    aa = _safe_div(glycogen_area, total_area)
    if (
        np.isfinite(aa)
        and aa_slope is not None
        and aa_intercept is not None
        and np.isfinite(aa_slope)
        and np.isfinite(aa_intercept)
    ):
        aa = aa - ((aa_slope * aa) + aa_intercept)

    # Compute particle diameter based on selected method
    if diameter_method == "area_circle" and mean_particle_area is not None and np.isfinite(mean_particle_area):
        # Compute diameter from area assuming circular profile
        # For a circle: A = π * (d/2)² => d = 2 * sqrt(A / π)
        # Convert from pixels to micrometers (1 um = 1000 nm)
        diameter_um = 2.0 * np.sqrt(mean_particle_area / np.pi) * pixel_size_nm / 1000.0
        h = diameter_um
    else:
        # Default: use mean feret diameter
        h = (mean_feret_diameter * pixel_size_nm) / 1000.0
    
    s = np.pi * (h ** 2)
    na = _safe_div(aa, np.pi * ((0.5 * h) ** 2))
    nv = _safe_div(na, thickness_um + h)
    sv = nv * s if np.isfinite(nv) and np.isfinite(s) else np.nan

    ba = np.nan
    if np.isfinite(sv) and np.isfinite(nv) and np.isfinite(h):
        ba = ((np.pi * sv) / 4.0) + (thickness_um * nv * np.pi * h)

    vv = np.nan
    if np.isfinite(aa) and np.isfinite(ba) and np.isfinite(na) and np.isfinite(h):
        vv = 100.0 * (
            aa - (thickness_um * ((ba / np.pi) - (_safe_div(na * thickness_um * h, (thickness_um + h)))))
        )

    numerical_density = np.nan
    if np.isfinite(vv) and np.isfinite(h):
        numerical_density = _safe_div(vv / 100, (3.0 / 4.0) * np.pi * ((h / 2.0) ** 3))

    return {
        "aa": aa,
        "h": h,
        "s": s,
        "na": na,
        "nv": nv,
        "sv": sv,
        "ba": ba,
        "vv_pct": vv,
        "numerical_density": numerical_density,
    }


def _build_imf_output_row(subfolder, image_value, region_map, total_area, pixel_size_nm, thickness_um,
                          imf_aa_slope, imf_aa_intercept, intra_aa_slope, intra_aa_intercept,
                          include_image_location=False, image_location="", diameter_method="feret"):
    out_row = {
        "Subfolder": subfolder,
        "Image": image_value,
        "sum_area_imf_mito_zdisc_intra": total_area,
    }
    if include_image_location:
        out_row["Image Location"] = image_location

    if "intermyofibrillar" in region_map:
        imf_mean_particle_area = float(region_map["intermyofibrillar"].get("Mean Particle Area", np.nan))
        imf = _compute_imf_region_set(
            glycogen_area=float(region_map["intermyofibrillar"]["Glycogen Area"]),
            mean_feret_diameter=float(region_map["intermyofibrillar"]["Mean Feret Diameter"]),
            total_area=total_area,
            thickness_um=float(thickness_um),
            aa_slope=float(imf_aa_slope),
            aa_intercept=float(imf_aa_intercept),
            pixel_size_nm=float(pixel_size_nm),
            mean_particle_area=imf_mean_particle_area,
            diameter_method=diameter_method,
        )
        out_row.update({f"IMF_{k}": v for k, v in imf.items()})
    else:
        out_row.update({f"IMF_{k}": np.nan for k in ["aa", "h", "s", "na", "nv", "sv", "ba", "vv_pct", "numerical_density"]})

    if "intra" in region_map:
        intra_mean_particle_area = float(region_map["intra"].get("Mean Particle Area", np.nan))
        intra = _compute_imf_region_set(
            glycogen_area=float(region_map["intra"]["Glycogen Area"]),
            mean_feret_diameter=float(region_map["intra"]["Mean Feret Diameter"]),
            total_area=total_area,
            thickness_um=float(thickness_um),
            aa_slope=float(intra_aa_slope),
            aa_intercept=float(intra_aa_intercept),
            pixel_size_nm=float(pixel_size_nm),
            mean_particle_area=intra_mean_particle_area,
            diameter_method=diameter_method,
        )
        out_row.update({f"intra_{k}": v for k, v in intra.items()})
    else:
        out_row.update({f"intra_{k}": np.nan for k in ["aa", "h", "s", "na", "nv", "sv", "ba", "vv_pct", "numerical_density"]})

    intra_area = float(region_map["intra"]["Area"]) if "intra" in region_map else np.nan
    zdisc_area = float(region_map["zdisc"]["Area"]) if "zdisc" in region_map else np.nan
    out_row["intra_zdisc_area_per_total_area"] = _safe_div(
        sum(area for area in [intra_area, zdisc_area] if np.isfinite(area)),
        total_area,
    )
    out_row["intra_vv_pct_per_intra_zdisc_area"] = _safe_div(
        out_row.get("intra_vv_pct", np.nan),
        out_row["intra_zdisc_area_per_total_area"],
    )

    mito_area = float(region_map["mitochondria"]["Area"]) if "mitochondria" in region_map else np.nan
    out_row["mito_vv_pct"] = 100.0 * _safe_div(mito_area, total_area)

    if "zdisc" in region_map:
        zdisc_max_feret = float(region_map["zdisc"]["Z-disc max feret"])
        out_row["zdiscwidth"] = _safe_div(zdisc_area, zdisc_max_feret) * float(pixel_size_nm)
    else:
        out_row["zdiscwidth"] = np.nan

    return out_row


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid_mask = np.isfinite(values) & np.isfinite(weights)
    if not np.any(valid_mask):
        return np.nan
    return float(np.average(values[valid_mask], weights=weights[valid_mask]))


def _compute_subfolder_outputs_from_raw_inputs(raw_df, pixel_size_nm, thickness_um,
                                               imf_aa_slope, imf_aa_intercept,
                                               intra_aa_slope, intra_aa_intercept,
                                               include_image_location=False,
                                               use_weighted_ratio=False,
                                               weight_ratio=3.0,
                                               diameter_method="feret"):
    if raw_df.empty or "Subfolder" not in raw_df.columns:
        return pd.DataFrame()

    if use_weighted_ratio and not include_image_location:
        raise ValueError("Weighted subfolder averaging requires Image Location data.")

    if "_region" not in raw_df.columns:
        working_df = raw_df.copy()
        working_df["_region"] = working_df["Region"].map(_normalize_imf_region_name)
    else:
        working_df = raw_df

    if include_image_location:
        image_location_map = _build_image_location_map(working_df)
    else:
        image_location_map = {}

    wanted_regions = ["intermyofibrillar", "mitochondria", "zdisc", "intra"]
    numeric_input_cols = ["Area", "Glycogen Area", "Mean Feret Diameter", "Z-disc max feret"]
    if "Mean Particle Area" in working_df.columns:
        numeric_input_cols.append("Mean Particle Area")

    rows = []
    for subfolder, subgrp in working_df.groupby("Subfolder", sort=False):
        region_map = {}
        for region in wanted_regions:
            region_rows = subgrp.loc[subgrp["_region"] == region].copy()
            if region_rows.empty:
                continue

            if use_weighted_ratio:
                weights = [
                    float(weight_ratio)
                    if "superficial" in str(image_location_map.get((subfolder, _canonical_image_value(image)), "")).lower()
                    else 1.0
                    for image in region_rows["Image"]
                ]
            else:
                weights = np.ones(len(region_rows), dtype=float)

            region_data = {}
            for col in numeric_input_cols:
                if col in region_rows.columns:
                    region_data[col] = _weighted_mean(region_rows[col].to_numpy(), weights)

            if region_data:
                region_map[region] = pd.Series(region_data)

        total_area = float(
            sum(float(region_map[r]["Area"]) for r in wanted_regions if r in region_map)
        )

        if include_image_location:
            image_location = "subfolder-level (from raw inputs)"
        else:
            image_location = ""

        rows.append(
            _build_imf_output_row(
                subfolder=subfolder,
                image_value="subfolder-avg (raw inputs)",
                region_map=region_map,
                total_area=total_area,
                pixel_size_nm=pixel_size_nm,
                thickness_um=thickness_um,
                imf_aa_slope=imf_aa_slope,
                imf_aa_intercept=imf_aa_intercept,
                intra_aa_slope=intra_aa_slope,
                intra_aa_intercept=intra_aa_intercept,
                include_image_location=include_image_location,
                image_location=image_location,
                diameter_method=diameter_method,
            )
        )

    return pd.DataFrame(rows).sort_values(["Subfolder", "Image"]).reset_index(drop=True)


def _compute_subfolder_averages(out_df, use_weighted_ratio=False, weight_ratio=3.0):
    """
    Compute weighted subfolder-level averages from image-level metrics.
    
    If use_weighted_ratio=True, superficial images are weighted at weight_ratio (default 3),
    and central images are weighted at 1.0. Otherwise, all images are weighted equally (1.0).
    
    Returns a DataFrame with one row per Subfolder, aggregating numeric columns from the image-level metrics.
    """
    if out_df.empty or "Subfolder" not in out_df.columns:
        return pd.DataFrame()
    
    # Identify numeric columns (excluding Subfolder, Image, Image Location)
    exclude_cols = {"Subfolder", "Image", "Image Location"}
    numeric_cols = [
        col for col in out_df.columns 
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(out_df[col])
    ]
    
    rows = []
    for subfolder, grp in out_df.groupby("Subfolder", sort=False):
        out_row = {"Subfolder": subfolder}
        
        # Determine weights for each image
        if use_weighted_ratio and "Image Location" in out_df.columns:
            weights = []
            for location in grp["Image Location"]:
                if "superficial" in str(location).lower():
                    weights.append(float(weight_ratio))
                else:
                    weights.append(1.0)
            weights = np.array(weights)
        else:
            # Equal weighting
            weights = np.ones(len(grp))
        
        # Compute weighted means for numeric columns
        for col in numeric_cols:
            col_data = grp[col].values
            valid_mask = np.isfinite(col_data)
            
            if np.any(valid_mask):
                valid_data = col_data[valid_mask]
                valid_weights = weights[valid_mask]
                weighted_mean = np.average(valid_data, weights=valid_weights)
                out_row[col] = weighted_mean
            else:
                out_row[col] = np.nan
        
        rows.append(out_row)
    
    avg_df = pd.DataFrame(rows)
    return avg_df.sort_values("Subfolder").reset_index(drop=True)


def compute_imf_stereology_metrics(input_csv_path, output_xlsx_path,
                                   pixel_size_nm, thickness_um,
                                   imf_aa_slope, imf_aa_intercept,
                                   intra_aa_slope, intra_aa_intercept,
                                   include_image_location=False,
                                   include_subfolder_avg=False,
                                   use_weighted_ratio=False,
                                   weight_ratio=3.0,
                                   diameter_method="feret",
                                   subfolder_calc_mode="average_metrics"):
    df = pd.read_csv(input_csv_path)
    required_cols = {
        "Subfolder",
        "Image",
        "Region",
        "Area",
        "Glycogen Area",
        "Mean Feret Diameter",
        "Z-disc max feret",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric_cols = ["Area", "Glycogen Area", "Mean Feret Diameter", "Z-disc max feret"]
    if "Mean Particle Area" in df.columns:
        numeric_cols.append("Mean Particle Area")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_region"] = df["Region"].map(_normalize_imf_region_name)
    wanted_regions = ["intermyofibrillar", "mitochondria", "zdisc", "intra"]
    image_location_map = _build_image_location_map(df) if include_image_location else {}

    rows = []
    for (subfolder, image), grp in df.groupby(["Subfolder", "Image"], sort=True):
        region_map = {
            region: grp.loc[grp["_region"] == region].iloc[0]
            for region in wanted_regions
            if not grp.loc[grp["_region"] == region].empty
        }

        total_area = float(
            sum(float(region_map[r]["Area"]) for r in wanted_regions if r in region_map)
        )

        out_row = _build_imf_output_row(
            subfolder=subfolder,
            image_value=image,
            region_map=region_map,
            total_area=total_area,
            pixel_size_nm=pixel_size_nm,
            thickness_um=thickness_um,
            imf_aa_slope=imf_aa_slope,
            imf_aa_intercept=imf_aa_intercept,
            intra_aa_slope=intra_aa_slope,
            intra_aa_intercept=intra_aa_intercept,
            include_image_location=include_image_location,
            image_location=image_location_map.get((subfolder, _canonical_image_value(image)), ""),
            diameter_method=diameter_method,
        )

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["Subfolder", "Image"]).reset_index(drop=True)

    output_path = (output_xlsx_path or "").strip()
    if not output_path:
        input_stem = os.path.splitext(os.path.abspath(input_csv_path))[0]
        output_path = f"{input_stem}_stereology.xlsx"

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    options_rows = [
        ("Input CSV Path", os.path.abspath(input_csv_path)),
        ("Output Workbook Path", os.path.abspath(output_path)),
        ("Pixel Size (nm)", float(pixel_size_nm)),
        ("Section Thickness (um)", float(thickness_um)),
        ("IMF AA Correction Slope", float(imf_aa_slope)),
        ("IMF AA Correction Intercept", float(imf_aa_intercept)),
        ("Intra AA Correction Slope", float(intra_aa_slope)),
        ("Intra AA Correction Intercept", float(intra_aa_intercept)),
        ("Diameter Method", str(diameter_method)),
        ("Include Image Location", bool(include_image_location)),
        ("Include Subfolder Averages", bool(include_subfolder_avg)),
        ("Subfolder Calculation Mode", str(subfolder_calc_mode)),
        ("Use Weighted Ratio (3:1 superficial:central)", bool(use_weighted_ratio)),
        ("Superficial Weight Ratio", float(weight_ratio)),
        ("Generated At", datetime.now().isoformat(timespec="seconds")),
    ]
    options_df = pd.DataFrame(options_rows, columns=["Option", "Value"])

    def _write_imf_workbook(path_to_write):
        with pd.ExcelWriter(path_to_write, engine='openpyxl') as writer:
            out_df.to_excel(writer, sheet_name='Image-level', index=False)

            if include_subfolder_avg:
                if subfolder_calc_mode == "average_inputs":
                    avg_df = _compute_subfolder_outputs_from_raw_inputs(
                        raw_df=df,
                        pixel_size_nm=pixel_size_nm,
                        thickness_um=thickness_um,
                        imf_aa_slope=imf_aa_slope,
                        imf_aa_intercept=imf_aa_intercept,
                        intra_aa_slope=intra_aa_slope,
                        intra_aa_intercept=intra_aa_intercept,
                        include_image_location=include_image_location,
                        use_weighted_ratio=use_weighted_ratio,
                        weight_ratio=weight_ratio,
                        diameter_method=diameter_method,
                    )
                else:
                    avg_df = _compute_subfolder_averages(
                        out_df,
                        use_weighted_ratio=use_weighted_ratio,
                        weight_ratio=weight_ratio
                    )
                avg_df.to_excel(writer, sheet_name='Subfolder-avg', index=False)

            options_df.to_excel(writer, sheet_name='Options', index=False)

    try:
        _write_imf_workbook(output_path)
    except PermissionError:
        alt_output_path = os.path.splitext(output_path)[0] + "_v2.xlsx"
        _write_imf_workbook(alt_output_path)
        output_path = alt_output_path

    return out_df, output_path


def main():
    parser = argparse.ArgumentParser(description='Open GUI to run inference')
    parser.add_argument('--glycogen_model', type=str, help='Path to model weights',
                        default='weights_glyco/model_epoch_50.pth')
    parser.add_argument('--region_model', type=str, help='Path to model weights',
                        default='weights/model_epoch_50.pth')
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def resolve_model_path(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(base_dir, path)

    args.glycogen_model = resolve_model_path(args.glycogen_model)
    args.region_model = resolve_model_path(args.region_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_cache = {}
    region_model_cache = {}

    def _load_state_dict(path: str):
        raw = torch.load(path, map_location=device, weights_only=True)
        if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = raw["state_dict"]
        else:
            state_dict = raw

        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint does not contain a valid state_dict: {path}")
        return state_dict

    def _infer_attunet_config(state_dict):
        if "final.weight" not in state_dict:
            raise ValueError("Checkpoint missing 'final.weight' and does not look like an AttUNet model.")

        num_classes = int(state_dict["final.weight"].shape[0])

        encoder_indices = []
        for key in state_dict.keys():
            if key.startswith("encoder."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    encoder_indices.append(int(parts[1]))

        if not encoder_indices:
            raise ValueError("Could not infer encoder depth from checkpoint keys.")

        filters = []
        for i in range(max(encoder_indices) + 1):
            conv_key = f"encoder.{i}.conv.conv.0.weight"
            if conv_key not in state_dict:
                raise ValueError(f"Checkpoint missing expected key: {conv_key}")
            filters.append(int(state_dict[conv_key].shape[0]))

        include_reconstruction = "recon_final.weight" in state_dict and "recon_final.bias" in state_dict
        return filters, num_classes, include_reconstruction

    def _build_model_from_checkpoint(path: str, expected_task: str):
        state_dict = _load_state_dict(path)
        filters, num_classes, include_reconstruction = _infer_attunet_config(state_dict)

        if expected_task == "glycogen" and num_classes != 1:
            raise ValueError(
                f"The selected glycogen model has {num_classes} output classes and appears to be a region model: {path}. "
                f"Use a 1-class glycogen model in 'Glycogen Model Path' and put this file in 'Region Model Path'."
            )

        if expected_task == "region" and num_classes < 2:
            raise ValueError(
                f"The selected region model has {num_classes} output class: {path}. "
                f"Region model should have multiple classes (typically 7)."
            )

        final_activation = 'sigmoid' if num_classes == 1 else 'softmax'
        model = AttUNet(
            1,
            num_classes,
            filters,
            final_activation=final_activation,
            include_reconstruction=include_reconstruction
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def get_inference_models(glycogen_model_path, region_model_path):
        glycogen_model_path = (glycogen_model_path or '').strip()
        region_model_path = (region_model_path or '').strip()

        glyco_path = resolve_model_path(glycogen_model_path) if glycogen_model_path else args.glycogen_model
        region_path = resolve_model_path(region_model_path) if region_model_path else args.region_model

        if not os.path.exists(glyco_path):
            raise ValueError(f"Glycogen model file not found: {glyco_path}")
        if not os.path.exists(region_path):
            raise ValueError(f"Region model file not found: {region_path}")

        cache_key = (glyco_path, region_path)
        if cache_key not in model_cache:
            glyco_model = _build_model_from_checkpoint(glyco_path, expected_task="glycogen")
            region_model = _build_model_from_checkpoint(region_path, expected_task="region")
            model_cache[cache_key] = (glyco_model, region_model)

        return model_cache[cache_key]

    def get_region_model_for_comparison(region_model_path):
        region_model_path = (region_model_path or '').strip()
        resolved_path = resolve_model_path(region_model_path) if region_model_path else args.region_model

        if not os.path.exists(resolved_path):
            raise ValueError(f"Region model file not found: {resolved_path}")

        if resolved_path not in region_model_cache:
            region_model_cache[resolved_path] = _build_model_from_checkpoint(resolved_path, expected_task="region")

        return region_model_cache[resolved_path]

    def colorize_index_map(index_map, palette_rgb, unlabeled_color=(0, 0, 0)):
        h, w = index_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i, color in enumerate(palette_rgb):
            rgb[index_map == i] = color
        rgb[index_map < 0] = np.array(unlabeled_color, dtype=np.uint8)
        return Image.fromarray(rgb)

    def resize_label_channels(label_channels, target_h, target_w):
        if label_channels.shape[1] == target_h and label_channels.shape[2] == target_w:
            return label_channels
        resized = np.stack([
            cv2.resize(label_channels[c].astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            for c in range(label_channels.shape[0])
        ], axis=0).astype(np.float32)
        return resized

    def _normalize_region_name(name):
        return (name or "").strip().lower().replace("_", "-").replace(" ", "-")

    def _find_required_region_indices(class_names):
        normalized = [_normalize_region_name(n) for n in class_names]

        def find_idx(candidates, label_for_error):
            for i, name in enumerate(normalized):
                if name in candidates:
                    return i
            raise ValueError(
                f"Could not find required class '{label_for_error}' in annotation/model classes: {class_names}"
            )

        idx_a = find_idx({"a-band", "aband", "a"}, "A-band")
        idx_i = find_idx({"i-band", "iband", "i"}, "I-band")
        idx_z = find_idx({"z-disc", "zdisc", "z"}, "z-disc")
        idx_inter = find_idx({"intermyofibrillar", "inter-myofibrillar", "inter"}, "intermyofibrillar")
        idx_mito = find_idx({"mitochondria", "mito", "mitochondrion"}, "mitochondria")
        return idx_a, idx_i, idx_z, idx_inter, idx_mito

    def _format_pct(numerator, denominator, parenthesize=False):
        if denominator <= 0:
            text = "0"
        else:
            text = str(int(np.rint((100.0 * float(numerator)) / float(denominator))))
        return f"({text})" if parenthesize else text

    def build_confusion_matrices(gt_labels, pred_labels, ordered_segment_names):
        idx_a, idx_i, idx_z, idx_inter, idx_mito = _find_required_region_indices(ordered_segment_names)
        intra_indices = [idx_a, idx_i, idx_z]

        rows = ["A-band", "I-band", "Z-disc", "Intra", "Inter", "Mito"]
        cols = ["A-band", "I-band", "Z-disc", "Intra", "Inter", "Mito"]

        gt_masks = {
            "A-band": gt_labels == idx_a,
            "I-band": gt_labels == idx_i,
            "Z-disc": gt_labels == idx_z,
            "Intra": np.isin(gt_labels, intra_indices),
            "Inter": gt_labels == idx_inter,
            "Mito": gt_labels == idx_mito,
        }
        pred_masks = {
            "A-band": pred_labels == idx_a,
            "I-band": pred_labels == idx_i,
            "Z-disc": pred_labels == idx_z,
            "Intra": np.isin(pred_labels, intra_indices),
            "Inter": pred_labels == idx_inter,
            "Mito": pred_labels == idx_mito,
        }

        valid_union = (
            gt_masks["A-band"] | gt_masks["I-band"] | gt_masks["Z-disc"] |
            gt_masks["Inter"] | gt_masks["Mito"]
        )

        gt_counts = {r: int(np.sum(gt_masks[r] & valid_union)) for r in rows}
        pred_counts = {c: int(np.sum(pred_masks[c] & valid_union)) for c in cols}

        table_a = []
        table_b = []

        for r in rows:
            row_a = {"Ground truth": r}
            row_b = {"Ground truth": r}
            for c in cols:
                overlap = int(np.sum(gt_masks[r] & pred_masks[c] & valid_union))

                # Hide redundant Intra decomposition cells.
                if (r in {"A-band", "I-band", "Z-disc"} and c == "Intra") or (r == "Intra" and c in {"A-band", "I-band", "Z-disc"}):
                    row_a[c] = "-"
                    row_b[c] = "-"
                    continue

                # Table A: percentage of GT-positive pixels (agreement / false negatives)
                a_parenthesize = (r in {"Inter", "Mito"} and c in {"A-band", "I-band", "Z-disc"})
                row_a[c] = _format_pct(overlap, gt_counts[r], parenthesize=a_parenthesize)

                # Table B: percentage of predicted-positive pixels (PPV / false positives)
                b_parenthesize = (r in {"A-band", "I-band", "Z-disc"} and c in {"Inter", "Mito"})
                row_b[c] = _format_pct(overlap, pred_counts[c], parenthesize=b_parenthesize)

            table_a.append(row_a)
            table_b.append(row_b)

        return pd.DataFrame(table_a), pd.DataFrame(table_b)

    def render_confusion_matrix_panel(df, panel_letter, panel_title):
        columns = ["A-band", "I-band", "Z-disc", "Intra", "Inter", "Mito"]
        rows = df["Ground truth"].tolist()
        values = df[columns].astype(str).values.tolist()

        fig, ax = plt.subplots(figsize=(8.8, 4.2), dpi=220)
        ax.axis('off')

        table = ax.table(
            cellText=values,
            rowLabels=rows,
            colLabels=columns,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.55)

        # Header style
        for c in range(len(columns)):
            table[(0, c)].set_facecolor('#efefef')
            table[(0, c)].set_edgecolor('black')
            table[(0, c)].set_linewidth(1.0)

        # Row-label style
        for r in range(1, len(rows) + 1):
            table[(r, -1)].set_facecolor('#f7f7f7')
            table[(r, -1)].set_edgecolor('black')
            table[(r, -1)].set_linewidth(1.0)

        # Data cells and diagonal emphasis (agreement/PPV cells)
        for r in range(len(rows)):
            for c in range(len(columns)):
                cell = table[(r + 1, c)]
                cell.set_edgecolor('black')
                cell.set_linewidth(0.9)
                cell.set_facecolor('#d9d9d9' if r == c else 'white')

        fig.text(0.015, 0.98, panel_letter, fontsize=16, fontweight='bold', va='top', ha='left')
        fig.text(0.52, 0.98, panel_title, fontsize=12, va='top', ha='center')
        fig.text(0.007, 0.52, "Ground truth", fontsize=11, rotation=90, va='center', ha='left')
        fig.text(0.52, 0.90, "Model", fontsize=11, va='center', ha='center')

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        pil_img = Image.fromarray(img[..., :3])
        plt.close(fig)
        return pil_img

    def make_confusion_export_paths(image_path, export_dir):
        export_root = (export_dir or "").strip()
        if not export_root:
            export_root = os.path.dirname(os.path.abspath(image_path))
        os.makedirs(export_root, exist_ok=True)

        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{image_stem}_region_confusion_{ts}"

        paths = {
            "png_a": os.path.join(export_root, f"{prefix}_A.png"),
            "png_b": os.path.join(export_root, f"{prefix}_B.png"),
            "png_ab": os.path.join(export_root, f"{prefix}_AB.png"),
            "csv_a": os.path.join(export_root, f"{prefix}_A.csv"),
            "csv_b": os.path.join(export_root, f"{prefix}_B.csv"),
        }
        return paths

    def concat_panels_horizontally(panel_a, panel_b, gap_px=24):
        a = np.array(panel_a)
        b = np.array(panel_b)
        h = max(a.shape[0], b.shape[0])

        def pad_to_height(img, target_h):
            if img.shape[0] == target_h:
                return img
            pad_top = (target_h - img.shape[0]) // 2
            pad_bottom = target_h - img.shape[0] - pad_top
            return np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=255)

        a2 = pad_to_height(a, h)
        b2 = pad_to_height(b, h)
        gap = np.full((h, int(gap_px), 3), 255, dtype=np.uint8)
        combined = np.concatenate([a2, gap, b2], axis=1)
        return Image.fromarray(combined)

    def run_region_annotation_comparison(image_file, annotation_file, region_model_path,
                                         region_scale_value, region_scale_mode, snap_multiple,
                                         export_dir, label_mapping):
        def resolve_model_scale(scale_value, scale_mode):
            v = float(scale_value)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"Scale value must be > 0, got: {scale_value}")
            if scale_mode == "Up-sample by factor":
                return v
            if scale_mode == "Down-sample by factor":
                return 1.0 / v
            return v

        try:
            if image_file is None:
                raise ValueError("Input image file is required (.tif/.tiff).")
            if annotation_file is None:
                raise ValueError("Annotation file is required (.nrrd).")

            region_model = get_region_model_for_comparison(region_model_path)
            region_scale = resolve_model_scale(region_scale_value, region_scale_mode)

            image_path = image_file.name
            annotation_path = annotation_file.name

            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            if not os.path.exists(annotation_path):
                raise ValueError(f"Annotation file not found: {annotation_path}")

            img = tifffile.imread(image_path)
            if img.ndim == 3:
                img = img[..., 0]
            img = img.astype(np.float32)

            img_region = resize_with_snap(img, scale=region_scale, mult=int(snap_multiple))
            h, w = img_region.shape[:2]

            vmin, vmax = np.min(img_region), np.max(img_region)
            if vmax > vmin:
                img_region01 = (img_region - vmin) / (vmax - vmin)
            else:
                img_region01 = np.zeros_like(img_region, dtype=np.float32)

            tensor_region = torch.from_numpy(img_region01).float().unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                region_pred = region_model(tensor_region)
                if isinstance(region_pred, tuple):
                    region_pred = region_pred[0]
                region_output = region_pred.squeeze().detach().cpu().numpy()

            if region_output.ndim != 3:
                raise ValueError(
                    f"Region model output is not multi-class (shape={region_output.shape}). "
                    f"Please select a multi-class region model."
                )

            pred_labels = np.argmax(region_output, axis=0).astype(np.int32)

            nrrd_reader = NRRDReader(annotation_path)
            segment_info = nrrd_reader.get_segments()
            ordered_segment_names = [
                name for name, _ in sorted(segment_info.items(), key=lambda kv: kv[1]["index"])
            ]

            if not ordered_segment_names:
                raise ValueError("No segments found in annotation NRRD metadata.")

            if len(ordered_segment_names) != int(region_output.shape[0]):
                raise ValueError(
                    f"Class mismatch: annotation has {len(ordered_segment_names)} segments, "
                    f"but model predicts {int(region_output.shape[0])} classes."
                )

            gt_hwc = nrrd_reader.extract_segments(ordered_segment_names)
            gt_chw = np.moveaxis(gt_hwc, -1, 0).astype(np.float32)
            gt_chw = resize_label_channels(gt_chw, h, w)

            unlabeled = np.sum(gt_chw, axis=0) <= 0
            gt_labels = np.argmax(gt_chw, axis=0).astype(np.int32)
            gt_labels[unlabeled] = -1

            misclassified = (pred_labels != gt_labels) & (~unlabeled)

            # Build palette from the same label_mapping used by the single-image tab.
            label_dict_colors = eval(label_mapping)
            # name -> matplotlib color string, normalised for lookup
            _name_to_color = {
                _normalize_region_name(k): v[0]
                for k, v in label_dict_colors.items()
            }
            palette = np.array([
                tuple(int(c * 255) for c in colors.to_rgb(
                    _name_to_color.get(_normalize_region_name(n), "#b4b4b4")
                ))
                for n in ordered_segment_names
            ], dtype=np.uint8)
            annotation_image = colorize_index_map(gt_labels, palette)
            prediction_image = colorize_index_map(pred_labels, palette)

            misclassified_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            misclassified_rgb[misclassified] = np.array([255, 0, 0], dtype=np.uint8)
            misclassified_image = Image.fromarray(misclassified_rgb)

            confusion_a_df, confusion_b_df = build_confusion_matrices(gt_labels, pred_labels, ordered_segment_names)

            panel_a = render_confusion_matrix_panel(
                confusion_a_df,
                panel_letter="A",
                panel_title="Agreement (%) of GT-positive pixels"
            )
            panel_b = render_confusion_matrix_panel(
                confusion_b_df,
                panel_letter="B",
                panel_title="Positive Predicted Pixels (%)"
            )

            export_paths = make_confusion_export_paths(image_path, export_dir)
            panel_a.save(export_paths["png_a"])
            panel_b.save(export_paths["png_b"])
            concat_panels_horizontally(panel_a, panel_b).save(export_paths["png_ab"])
            confusion_a_df.to_csv(export_paths["csv_a"], index=False)
            confusion_b_df.to_csv(export_paths["csv_b"], index=False)

            export_msg = (
                "Saved confusion-matrix exports:\n"
                f"A image: {export_paths['png_a']}\n"
                f"B image: {export_paths['png_b']}\n"
                f"A+B image: {export_paths['png_ab']}\n"
                f"A CSV: {export_paths['csv_a']}\n"
                f"B CSV: {export_paths['csv_b']}"
            )

            return (
                annotation_image,
                prediction_image,
                misclassified_image,
                panel_a,
                panel_b,
                confusion_a_df,
                confusion_b_df,
                export_msg
            )

        except Exception as e:
            raise gr.Error(str(e))

    def apply_viridis(image01):
        cmap = cm.viridis
        colored_image = cmap(image01)
        rgb_image = (colored_image[:, :, :3] * 255).astype('uint8')
        image = Image.fromarray(rgb_image)
        return image

    def apply_label_mapping(predictions, label_dict):
        # Create a new image with new number of channels, then we add the predicted labels for each of the old channels
        new_labels = np.zeros((predictions.shape[1], predictions.shape[2], len(label_dict)), dtype=np.float32)
        for new_idx, (label, (col, ids)) in enumerate(label_dict.items()):
            for i in ids:
                if i >= predictions.shape[0]:
                    raise ValueError(
                        f"Label mapping references channel {i}, but region model only outputs {predictions.shape[0]} channels."
                    )
                probs = predictions[i, :, :]
                new_labels[:, :, new_idx] += probs

        # Argmax to get the most likely label
        new_labels = np.argmax(new_labels, axis=2)
        cmap = colors.ListedColormap([label_dict[label][0] for label in label_dict.keys()])
        colored_image = cmap(new_labels)
        rgb_image = (colored_image[:, :, :3] * 255).astype('uint8')
        image = Image.fromarray(rgb_image)
        return image, new_labels

    def calculate_maximal_feret(contours):
        max_diameter = 0
        max_diameter_contour = None
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            diameter = max(width, height)
            if diameter > max_diameter:
                max_diameter = diameter
                max_diameter_contour = cnt
        return max_diameter if max_diameter > 0 else 0, max_diameter_contour

    def draw_maximal_feret(image, contour):
        if contour is not None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)  # Draw the rectangle in green

    def compute_particle_shape_metrics(cnt):
        if len(cnt) < 3:
            return None
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        form_factor = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0.0
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        major = max(width, height)
        minor = min(width, height)
        aspect_ratio = (major / minor) if minor > 0 else 0.0
        roundness = (4 * area) / (np.pi * (major ** 2)) if major > 0 else 0.0
        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "form_factor": float(form_factor),
            "feret_diameter": float(major),
            "aspect_ratio": float(aspect_ratio),
            "roundness": float(roundness),
            "major_axis": float(major),
            "minor_axis": float(minor)
        }

    # ------------- SINGLE IMAGE -------------
    def run_segmentation(image, label_mapping, threshold, min_area, max_area, circularity_threshold,
                         glycogen_scale_value, glycogen_scale_mode,
                         region_scale_value, region_scale_mode,
                         snap_multiple, glycogen_model_path, region_model_path):
        """
        image: numpy (from gr.Image, grayscale float)
        """
        def resolve_model_scale(scale_value, scale_mode):
            v = float(scale_value)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"Scale value must be > 0, got: {scale_value}")
            if scale_mode == "Up-sample by factor":
                return v
            if scale_mode == "Down-sample by factor":
                return 1.0 / v
            return v

        try:
            glyco_model, region_model = get_inference_models(glycogen_model_path, region_model_path)
            glyco_scale = resolve_model_scale(glycogen_scale_value, glycogen_scale_mode)
            region_scale = resolve_model_scale(region_scale_value, region_scale_mode)
        except Exception as e:
            raise gr.Error(str(e))

        # Parse label dict safely
        label_dict = eval(label_mapping)

        # Ensure float32
        img = np.array(image)
        if img.ndim == 3:
            # if it comes as (H,W,1)
            img = img[..., 0]
        img = img.astype(np.float32)

        # Region preprocessing (reference grid)
        img_region = resize_with_snap(img, scale=region_scale, mult=int(snap_multiple))
        H, W = img_region.shape[:2]

        # Normalize region input to 0-1
        vmin, vmax = np.min(img_region), np.max(img_region)
        if vmax > vmin:
            img_region01 = (img_region - vmin) / (vmax - vmin)
        else:
            img_region01 = np.zeros_like(img_region, dtype=np.float32)

        # Glycogen preprocessing (can use different scale)
        img_glyco = resize_with_snap(img, scale=glyco_scale, mult=int(snap_multiple))
        vgmin, vgmax = np.min(img_glyco), np.max(img_glyco)
        if vgmax > vgmin:
            img_glyco01 = (img_glyco - vgmin) / (vgmax - vgmin)
        else:
            img_glyco01 = np.zeros_like(img_glyco, dtype=np.float32)

        tensor_region = torch.from_numpy(img_region01).float().unsqueeze(0).unsqueeze(0).to(device)
        tensor_glyco = torch.from_numpy(img_glyco01).float().unsqueeze(0).unsqueeze(0).to(device)

        # Glycogen segmentation
        with torch.no_grad():
            glyco_pred = glyco_model(tensor_glyco)
            if isinstance(glyco_pred, tuple):
                glyco_pred = glyco_pred[0]
            glyco_output = 1 - glyco_pred.squeeze().detach().cpu().numpy()

        # Align glycogen output to region grid for consistent overlays/metrics.
        if glyco_output.shape != (H, W):
            glyco_output = cv2.resize(glyco_output, (W, H), interpolation=cv2.INTER_LINEAR)

        glyco_viri = apply_viridis(glyco_output)

        # Region segmentation
        with torch.no_grad():
            region_pred = region_model(tensor_region)
            if isinstance(region_pred, tuple):
                region_pred = region_pred[0]
            region_output = region_pred.squeeze().detach().cpu().numpy()
        region_colored, new_labels = apply_label_mapping(region_output, label_dict)

        # Quantify glycogen
        glyco_thresholded = glyco_output > float(threshold)

        # Canvases based on resized image size (no more fixed 2048x2048)
        filtered_image_with_feret_line = np.zeros((H, W, 3), dtype=np.uint8)

        z_disc_maximal_feret_sum = 0

        # Accumulators
        label_areas = {}
        glycogen_areas = {}
        mean_particle_size = {}

        for label in label_dict.keys():
            label_areas[label] = 0
            glycogen_areas[label] = 0
            mean_particle_size[label] = 0

        # Per-label z-disc feret lines
        for i, label in enumerate(label_dict.keys()):
            label_area = np.sum(new_labels == i)
            label_areas[label] += label_area

            contours, _ = cv2.findContours((new_labels == i).astype(np.uint8) * 255,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if label == "z-disc":
                for cnt in contours:
                    max_feret_diameter, max_feret_contour = calculate_maximal_feret([cnt])
                    z_disc_maximal_feret_sum += max_feret_diameter
                    draw_maximal_feret(filtered_image_with_feret_line, max_feret_contour)

        # Create an empty image to draw filtered areas as circles
        filtered_image = np.zeros((H, W, 3), dtype=np.uint8)

        amounts = []
        for i, label in enumerate(label_dict.keys()):
            label_area = np.sum(new_labels == i)
            label_areas[label] += label_area
            glycogen_area = np.sum(np.logical_and(glyco_thresholded, new_labels == i))
            glycogen_areas[label] += glycogen_area

            # contours on glycogen mask
            contours, _ = cv2.findContours(glyco_thresholded.astype(np.uint8) * 255,
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            filtered_areas = []
            filtered_form_factors = []
            filtered_feret_diameters = []

            for cnt in contours:
                metrics = compute_particle_shape_metrics(cnt)
                if metrics is not None:
                    area = metrics["area"]
                    circularity = metrics["circularity"]
                    if float(min_area) <= area <= float(max_area) and circularity >= float(circularity_threshold):
                        # centroid
                        M = cv2.moments(cnt)
                        if M["m00"] == 0:
                            continue
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if 0 <= cy < H and 0 <= cx < W and new_labels[cy, cx] == i:
                            filtered_areas.append(area)
                            filtered_form_factors.append(metrics["form_factor"])
                            filtered_feret_diameters.append(metrics["feret_diameter"])
                            color_bgr = tuple(int(c * 255) for c in colors.to_rgb(label_dict[label][0]))
                            rad = max(1, int(np.sqrt(area / np.pi)))
                            cv2.circle(filtered_image, (cx, cy), rad, color_bgr[::-1], thickness=-1)

            mean_particle_area = np.mean(filtered_areas) if filtered_areas else 0
            mean_form_factor = np.mean(filtered_form_factors) if filtered_form_factors else 0
            mean_feret_diameter = np.mean(filtered_feret_diameters) if filtered_feret_diameters else 0
            amounts.append((label, label_area, glycogen_area, mean_particle_area, mean_form_factor, mean_feret_diameter))

        df = pd.DataFrame(amounts, columns=[
            "Region", "Area", "Glycogen Area", "Particle area", "Form factor", "Feret diameter"
        ])
        df["z-disc maximal feret sum"] = z_disc_maximal_feret_sum

        return (glyco_viri, region_colored, Image.fromarray(filtered_image),
                Image.fromarray(filtered_image_with_feret_line), df)

    # ------------- BATCH -------------
    def run_batch(folder_input, folder_output, label_mapping, threshold, thresholds_text, min_area, max_area,
                  circularity_threshold, save_region, save_region_individual, save_glyco, save_glyco_png_transparent,
                  glyco_png_threshold, save_filtered, save_zdisc,
                  glycogen_scale_value, glycogen_scale_mode,
                  region_scale_value, region_scale_mode,
                  snap_multiple, glycogen_model_path, region_model_path, progress=gr.Progress()):
        """
        Supports independent glycogen/region model scaling in batch processing.
        Region output size is used as the reference grid for quantification.
        """
        def resolve_model_scale(scale_value, scale_mode):
            v = float(scale_value)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"Scale value must be > 0, got: {scale_value}")
            if scale_mode == "Up-sample by factor":
                return v
            if scale_mode == "Down-sample by factor":
                return 1.0 / v
            return v

        def parse_thresholds(default_threshold, threshold_text):
            text = (threshold_text or "").strip()
            if not text:
                values = [float(default_threshold)]
            else:
                chunks = [c.strip() for c in text.replace(",", ";").split(";") if c.strip()]
                if not chunks:
                    values = [float(default_threshold)]
                else:
                    values = [float(c) for c in chunks]
            unique = []
            seen = set()
            for v in values:
                if not np.isfinite(v) or v < 0 or v > 1:
                    raise ValueError(f"Invalid threshold {v}. Thresholds must be in [0, 1].")
                if v not in seen:
                    seen.add(v)
                    unique.append(v)
            return unique

        def threshold_tag(v):
            text = f"{float(v):.6f}".rstrip('0').rstrip('.')
            if text == "":
                text = "0"
            return text.replace('.', 'p')

        glycogen_model_path_input = (glycogen_model_path or '').strip()
        region_model_path_input = (region_model_path or '').strip()
        resolved_glyco_model_path = resolve_model_path(glycogen_model_path_input) if glycogen_model_path_input else args.glycogen_model
        resolved_region_model_path = resolve_model_path(region_model_path_input) if region_model_path_input else args.region_model

        try:
            glyco_model, region_model = get_inference_models(glycogen_model_path, region_model_path)
            glyco_scale = resolve_model_scale(glycogen_scale_value, glycogen_scale_mode)
            region_scale = resolve_model_scale(region_scale_value, region_scale_mode)
            thresholds = parse_thresholds(threshold, thresholds_text)
            glyco_png_threshold = float(glyco_png_threshold)
            if not np.isfinite(glyco_png_threshold) or glyco_png_threshold < 0 or glyco_png_threshold > 1:
                raise ValueError(f"Invalid transparent PNG threshold {glyco_png_threshold}. Threshold must be in [0, 1].")
        except Exception as e:
            progress(0, desc=f"Batch setup failed: {str(e)}")
            return 0

        label_dict_combined = eval(label_mapping)
        label_dict_combined["intra"] = ("purple", label_dict_combined["A-band"][1] + label_dict_combined["I-band"][1])
        original_label_dict_individuals = eval(label_mapping)

        if folder_input is None or not os.path.isdir(folder_input):
            progress(0, desc="Invalid input folder.")
            return 0

        subfolders = [os.path.join(folder_input, subfolder)
                      for subfolder in os.listdir(folder_input)
                      if os.path.isdir(os.path.join(folder_input, subfolder))]
        total_subfolders = len(subfolders)

        batch_log_path = None
        if folder_output and folder_output.strip() != "":
            os.makedirs(folder_output, exist_ok=True)
            batch_log_path = os.path.join(
                folder_output,
                f"batch_settings_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            log_lines = [
                f"Batch run started: {datetime.now().isoformat()}",
                "",
                "Models:",
                f"  Glycogen model: {resolved_glyco_model_path}",
                f"  Region model: {resolved_region_model_path}",
                "",
                "Folders:",
                f"  Input folder: {folder_input}",
                f"  Output folder: {folder_output}",
                "",
                "Thresholds:",
                f"  Default threshold slider: {float(threshold)}",
                f"  Multiple thresholds text: {thresholds_text if thresholds_text else '(empty)'}",
                f"  Effective thresholds: {', '.join(str(float(t)) for t in thresholds)}",
                "",
                "Particle filtering settings:",
                f"  Min area: {float(min_area)}",
                f"  Max area: {float(max_area)}",
                f"  Circularity threshold: {float(circularity_threshold)}",
                "",
                "Batch output options:",
                f"  Save region image: {bool(save_region)}",
                f"  Save region image (individual labels): {bool(save_region_individual)}",
                f"  Save glycogen image: {bool(save_glyco)}",
                f"  Save glycogen transparent PNG: {bool(save_glyco_png_transparent)}",
                f"  Glycogen transparent PNG threshold: {float(glyco_png_threshold)}",
                f"  Save filtered image: {bool(save_filtered)}",
                f"  Save z-disc image: {bool(save_zdisc)}",
                "",
                "Model scaling and resize settings:",
                f"  Glycogen scale value: {float(glycogen_scale_value)}",
                f"  Glycogen scale mode: {glycogen_scale_mode}",
                f"  Effective glycogen scale factor: {float(glyco_scale)}",
                f"  Region scale value: {float(region_scale_value)}",
                f"  Region scale mode: {region_scale_mode}",
                f"  Effective region scale factor: {float(region_scale)}",
                f"  Snap multiple: {int(snap_multiple)}",
                "",
                "Label mapping:",
                str(label_mapping),
                "",
                f"Detected top-level subfolders to process: {total_subfolders}",
                ""
            ]
            with open(batch_log_path, 'w') as f:
                f.write("\n".join(log_lines))

        csv_rows_combined_by_threshold = {t: [] for t in thresholds}
        csv_rows_individuals_by_threshold = {t: [] for t in thresholds}
        particle_data_by_threshold = {t: [] for t in thresholds}
        processed_images = 0

        for idx, subfolder in enumerate(subfolders):
            progress((idx + 1) / max(1, total_subfolders), desc=f"Processing {subfolder}...")
            for root, dirs, files in os.walk(subfolder):
                for imgpath in files:
                    if not imgpath.endswith(('.tif', '.tiff', '.TIF', '.TIFF')):
                        continue

                    processed_images += 1

                    img_full_path = os.path.join(root, imgpath)
                    subfolder_name = os.path.basename(root)
                    base_filename = os.path.splitext(imgpath)[0]

                    img = tifffile.imread(img_full_path)
                    if img.ndim == 3:
                        img = img[..., 0]
                    img = img.astype(np.float32)

                    img_region = resize_with_snap(img, scale=region_scale, mult=int(snap_multiple))
                    H, W = img_region.shape[:2]

                    vmin, vmax = np.min(img_region), np.max(img_region)
                    if vmax > vmin:
                        img_region01 = (img_region - vmin) / (vmax - vmin)
                    else:
                        img_region01 = np.zeros_like(img_region, dtype=np.float32)
                    tensor_region = torch.from_numpy(img_region01).float().unsqueeze(0).unsqueeze(0).to(device)

                    img_glyco = resize_with_snap(img, scale=glyco_scale, mult=int(snap_multiple))
                    vgmin, vgmax = np.min(img_glyco), np.max(img_glyco)
                    if vgmax > vgmin:
                        img_glyco01 = (img_glyco - vgmin) / (vgmax - vgmin)
                    else:
                        img_glyco01 = np.zeros_like(img_glyco, dtype=np.float32)
                    tensor_glyco = torch.from_numpy(img_glyco01).float().unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        glyco_pred = glyco_model(tensor_glyco)
                        if isinstance(glyco_pred, tuple):
                            glyco_pred = glyco_pred[0]
                        glyco_output = 1 - glyco_pred.squeeze().detach().cpu().numpy()

                    if glyco_output.shape != (H, W):
                        glyco_output = cv2.resize(glyco_output, (W, H), interpolation=cv2.INTER_LINEAR)

                    with torch.no_grad():
                        region_pred = region_model(tensor_region)
                        if isinstance(region_pred, tuple):
                            region_pred = region_pred[0]
                        region_output = region_pred.squeeze().detach().cpu().numpy()

                    region_colored_combined, new_labels_combined = apply_label_mapping(region_output, label_dict_combined)
                    region_colored_individuals, new_labels_individuals = apply_label_mapping(region_output, original_label_dict_individuals)

                    if folder_output and folder_output.strip() != "" and save_region:
                        region_output_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_region.tif")
                        tifffile.imwrite(region_output_path, np.array(region_colored_combined), photometric='rgb')

                    if folder_output and folder_output.strip() != "" and save_region_individual:
                        region_output_individual_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_region_individual.tif")
                        tifffile.imwrite(region_output_individual_path, np.array(region_colored_individuals), photometric='rgb')

                    if folder_output and folder_output.strip() != "" and save_glyco_png_transparent:
                        png_thr_tag = threshold_tag(glyco_png_threshold)
                        glyco_png_mask = glyco_output > float(glyco_png_threshold)
                        glyco_rgb = np.where(glyco_png_mask[..., np.newaxis], np.array([255, 255, 255], dtype=np.uint8), np.array([0, 0, 0], dtype=np.uint8)).astype(np.uint8)
                        glyco_alpha = (glyco_png_mask.astype(np.uint8) * 255)
                        glyco_rgba = np.dstack((glyco_rgb, glyco_alpha))
                        glyco_png_path = os.path.join(
                            folder_output,
                            f"{subfolder_name}_{base_filename}_glyco_transparent_thr_{png_thr_tag}.png"
                        )
                        Image.fromarray(glyco_rgba, mode='RGBA').save(glyco_png_path)

                    for thr in thresholds:
                        thr_tag = threshold_tag(thr)
                        glyco_thresholded = glyco_output > float(thr)

                        if folder_output and folder_output.strip() != "" and save_glyco:
                            glyco_output_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_glyco_thr_{thr_tag}.tif")
                            tifffile.imwrite(glyco_output_path, glyco_thresholded.astype(np.uint8))

                        contours_mask, _ = cv2.findContours(glyco_thresholded.astype(np.uint8) * 255,
                                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        filtered_image_combined = np.zeros((H, W, 3), dtype=np.uint8)
                        filtered_image_with_feret_line_combined = np.zeros((H, W, 3), dtype=np.uint8)
                        z_disc_maximal_feret_sum_combined = 0

                        for i_lbl, label in enumerate(label_dict_combined.keys()):
                            label_area_combined = int(np.sum(new_labels_combined == i_lbl))
                            glycogen_area_combined = int(np.sum(np.logical_and(glyco_thresholded, new_labels_combined == i_lbl)))

                            filtered_areas_combined = []
                            filtered_form_factors_combined = []
                            filtered_feret_diameters_combined = []

                            for cnt in contours_mask:
                                metrics = compute_particle_shape_metrics(cnt)
                                if metrics is None:
                                    continue
                                area = metrics["area"]
                                circularity = metrics["circularity"]
                                if not (float(min_area) <= area <= float(max_area) and circularity >= float(circularity_threshold)):
                                    continue

                                M = cv2.moments(cnt)
                                if M["m00"] == 0:
                                    continue
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                if not (0 <= cy < H and 0 <= cx < W and new_labels_combined[cy, cx] == i_lbl):
                                    continue

                                filtered_areas_combined.append(area)
                                filtered_form_factors_combined.append(metrics["form_factor"])
                                filtered_feret_diameters_combined.append(metrics["feret_diameter"])
                                particle_data_by_threshold[thr].append({
                                    "Threshold": float(thr),
                                    "Subfolder": subfolder_name,
                                    "Image": imgpath,
                                    "Region": label,
                                    "Particle_Area": float(area),
                                    "Perimeter": metrics["perimeter"],
                                    "Circularity": metrics["circularity"],
                                    "Form_Factor": metrics["form_factor"],
                                    "Feret_Diameter": metrics["feret_diameter"],
                                    "Aspect_Ratio": metrics["aspect_ratio"],
                                    "Roundness": metrics["roundness"],
                                    "Major_Axis": metrics["major_axis"],
                                    "Minor_Axis": metrics["minor_axis"],
                                    "X_Coordinate": int(cx),
                                    "Y_Coordinate": int(cy)
                                })
                                color_bgr = tuple(int(c * 255) for c in colors.to_rgb(label_dict_combined[label][0]))
                                rad = max(1, int(np.sqrt(area / np.pi)))
                                cv2.circle(filtered_image_combined, (cx, cy), rad, color_bgr[::-1], thickness=-1)

                            contours_lbl, _ = cv2.findContours((new_labels_combined == i_lbl).astype(np.uint8) * 255,
                                                               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            if label == "z-disc":
                                for cnt in contours_lbl:
                                    max_feret_diameter, max_feret_contour = calculate_maximal_feret([cnt])
                                    z_disc_maximal_feret_sum_combined += max_feret_diameter
                                    draw_maximal_feret(filtered_image_with_feret_line_combined, max_feret_contour)

                            csv_rows_combined_by_threshold[thr].append((
                                subfolder_name,
                                imgpath,
                                label,
                                label_area_combined,
                                glycogen_area_combined,
                                float(np.mean(filtered_areas_combined) if filtered_areas_combined else 0.0),
                                float(np.mean(filtered_form_factors_combined) if filtered_form_factors_combined else 0.0),
                                float(np.mean(filtered_feret_diameters_combined) if filtered_feret_diameters_combined else 0.0),
                                float(z_disc_maximal_feret_sum_combined if label == "z-disc" else 0.0)
                            ))

                        if folder_output and folder_output.strip() != "" and save_filtered:
                            filtered_image_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_filtered_thr_{thr_tag}.tif")
                            tifffile.imwrite(filtered_image_path, filtered_image_combined, photometric='rgb')

                        if folder_output and folder_output.strip() != "" and save_zdisc:
                            zdisc_image_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_zdisc_thr_{thr_tag}.tif")
                            tifffile.imwrite(zdisc_image_path, filtered_image_with_feret_line_combined, photometric='rgb')

                        for i_lbl, label in enumerate(original_label_dict_individuals.keys()):
                            label_area_individual = int(np.sum(new_labels_individuals == i_lbl))
                            glycogen_area_individual = int(np.sum(np.logical_and(glyco_thresholded, new_labels_individuals == i_lbl)))

                            filtered_areas_individuals = []
                            filtered_form_factors_individuals = []
                            filtered_feret_diameters_individuals = []

                            for cnt in contours_mask:
                                metrics = compute_particle_shape_metrics(cnt)
                                if metrics is None:
                                    continue
                                area = metrics["area"]
                                circularity = metrics["circularity"]
                                if not (float(min_area) <= area <= float(max_area) and circularity >= float(circularity_threshold)):
                                    continue
                                M = cv2.moments(cnt)
                                if M["m00"] == 0:
                                    continue
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                if 0 <= cy < H and 0 <= cx < W and new_labels_individuals[cy, cx] == i_lbl:
                                    filtered_areas_individuals.append(area)
                                    filtered_form_factors_individuals.append(metrics["form_factor"])
                                    filtered_feret_diameters_individuals.append(metrics["feret_diameter"])

                            csv_rows_individuals_by_threshold[thr].append((
                                subfolder_name,
                                imgpath,
                                label,
                                label_area_individual,
                                glycogen_area_individual,
                                float(np.mean(filtered_areas_individuals) if filtered_areas_individuals else 0.0),
                                float(np.mean(filtered_form_factors_individuals) if filtered_form_factors_individuals else 0.0),
                                float(np.mean(filtered_feret_diameters_individuals) if filtered_feret_diameters_individuals else 0.0)
                            ))

        if folder_output and folder_output.strip() != "":
            for thr in thresholds:
                thr_tag = threshold_tag(thr)

                df_combined = pd.DataFrame(csv_rows_combined_by_threshold[thr], columns=[
                    "Subfolder", "Image", "Region", "Area", "Glycogen Area", "Mean Particle Area", "Mean Form Factor",
                    "Mean Feret Diameter", "Z-disc max feret"
                ])
                df_combined.to_csv(
                    os.path.join(folder_output, f"glycogen_distribution_combined_threshold_{thr_tag}.csv"),
                    index=False
                )

                df_individuals = pd.DataFrame(csv_rows_individuals_by_threshold[thr], columns=[
                    "Subfolder", "Image", "Region", "Area", "Glycogen Area", "Mean Particle Area", "Mean Form Factor",
                    "Mean Feret Diameter"
                ])
                df_individuals.to_csv(
                    os.path.join(folder_output, f"glycogen_distribution_individuals_threshold_{thr_tag}.csv"),
                    index=False
                )

                df_particles = pd.DataFrame(particle_data_by_threshold[thr])
                df_particles.to_csv(
                    os.path.join(folder_output, f"Particle size raw_threshold_{thr_tag}.csv"),
                    index=False
                )

            if batch_log_path is not None:
                with open(batch_log_path, 'a') as f:
                    f.write(f"Batch run completed: {datetime.now().isoformat()}\n")
                    f.write(f"Processed images: {processed_images}\n")
                    f.write(f"Generated threshold-specific CSV sets: {len(thresholds)}\n")
                    f.write("CSV outputs per threshold:\n")
                    for thr in thresholds:
                        thr_tag = threshold_tag(thr)
                        f.write(f"  - glycogen_distribution_combined_threshold_{thr_tag}.csv\n")
                        f.write(f"  - glycogen_distribution_individuals_threshold_{thr_tag}.csv\n")
                        f.write(f"  - Particle size raw_threshold_{thr_tag}.csv\n")

        return 100  # for the progress bar

    def open_tif(file):
        img = tifffile.imread(file.name)
        return img

    def resolve_uploaded_or_text_path(uploaded_file, path_text):
        if uploaded_file is not None:
            return uploaded_file.name
        return (path_text or "").strip()

    def run_imf_metrics(csv_file, csv_path_text, output_xlsx_path,
                        pixel_size_nm, thickness_um,
                        imf_aa_slope, imf_aa_intercept,
                        intra_aa_slope, intra_aa_intercept,
                        include_image_location, include_subfolder_avg,
                        use_weighted_ratio, diameter_method, subfolder_calc_mode):
        input_csv_path = resolve_uploaded_or_text_path(csv_file, csv_path_text)
        if not input_csv_path:
            raise gr.Error("Select a batch CSV file or provide its path.")
        if not os.path.exists(input_csv_path):
            raise gr.Error(f"CSV file not found: {input_csv_path}")

        metrics_df, saved_path = compute_imf_stereology_metrics(
            input_csv_path=input_csv_path,
            output_xlsx_path=output_xlsx_path,
            pixel_size_nm=float(pixel_size_nm),
            thickness_um=float(thickness_um),
            imf_aa_slope=float(imf_aa_slope),
            imf_aa_intercept=float(imf_aa_intercept),
            intra_aa_slope=float(intra_aa_slope),
            intra_aa_intercept=float(intra_aa_intercept),
            include_image_location=bool(include_image_location),
            include_subfolder_avg=bool(include_subfolder_avg),
            use_weighted_ratio=bool(use_weighted_ratio),
            weight_ratio=3.0,
            diameter_method=str(diameter_method),
            subfolder_calc_mode=str(subfolder_calc_mode),
        )

        status = (
            f"Loaded CSV: {input_csv_path}\n"
            f"Rows computed: {len(metrics_df)}\n"
            f"Saved workbook: {saved_path}\n"
            f"Image location added: {bool(include_image_location)}\n"
            f"Subfolder averaging: {bool(include_subfolder_avg)}\n"
            f"Diameter method: {diameter_method}\n"
            f"Subfolder calculation mode: {subfolder_calc_mode}"
        )
        if include_subfolder_avg:
            status += f"\nWeighted ratio (3:1 superficial:central): {bool(use_weighted_ratio)}"
        return metrics_df, status, saved_path

    # Create the GUI
    with gr.Blocks() as gui:
        # Main layout
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Batch processing"):
                    # Folder selection (as text input)
                    with gr.Row():
                        folder_input = gr.Textbox(label="Input Folder", info="Full path of directory containing images")
                        folder_output = gr.Textbox(label="Output Folder", info="Full path of directory to save results")

                    with gr.Row():
                        glycogen_model_batch = gr.Textbox(
                            label="Glycogen Model Path",
                            value=args.glycogen_model,
                            placeholder="Full path to glycogen .pth model"
                        )
                        region_model_batch = gr.Textbox(
                            label="Region Model Path",
                            value=args.region_model,
                            placeholder="Full path to region .pth model"
                        )

                    with gr.Row():
                        threshold_b = gr.Slider(label="Threshold", minimum=0, maximum=1, step=0.01, value=0.3)
                        circularity_threshold_b = gr.Slider(label="Single Particle circularity threshold", minimum=0, maximum=1, step=0.01, value=0.80)

                    thresholds_b = gr.Textbox(
                        label="Multiple Thresholds (optional)",
                        placeholder="Semicolon-separated values, e.g. 0.01;0.02;0.05",
                        value=""
                    )

                    with gr.Row():
                        min_area_b = gr.Slider(label="Single Particle min area (with 0.7698 nm/pixel, 133 pixels = 10 nm diameter)", minimum=0, maximum=500, step=1, value=133)
                        max_area_b = gr.Slider(label="Single Particle max area (with 0.7698 nm/pixel, 2338 pixels = 42 nm diameter)", minimum=0, maximum=4000, step=1, value=2338)

                    label_mapping_b = gr.Textbox(label="Label dictionary", info="Python dictionary describing the included labels",
                                                 value=default_label_mapping)

                    with gr.Row():
                        save_region_b = gr.Checkbox(label="Save Region Image (Warning 10 images = 1 GB)", value=False)
                        save_region_individual_b = gr.Checkbox(
                            label="Save Region Image (I-band blue, A-band red)",
                            value=False
                        )
                        save_glyco_b = gr.Checkbox(label="Save Glycogen Image (Warning 2000 images = 1 GB)", value=False)
                        save_glyco_png_transparent_b = gr.Checkbox(
                            label="Save Glycogen Transparent PNG (thresholded)",
                            value=False
                        )
                        save_filtered_b = gr.Checkbox(label="Save Filtered Image (Warning 80 images = 1 GB)", value=False)
                        save_zdisc_b = gr.Checkbox(label="Save Z-disc Image (Warning 80 images = 1 GB)", value=False)

                    glyco_png_threshold_b = gr.Slider(
                        label="Glycogen Transparent PNG Threshold",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.3
                    )

                    # Per-model scale & snap for batch
                    with gr.Row():
                        glycogen_scale_value_b = gr.Number(value=1.0, precision=6, label="Glycogen Model Scale Value")
                        glycogen_scale_mode_b = gr.Dropdown(
                            choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                            value="Direct scale factor",
                            label="Glycogen Model Scaling Mode"
                        )

                    with gr.Row():
                        region_scale_value_b = gr.Number(value=1.0, precision=6, label="Region Model Scale Value")
                        region_scale_mode_b = gr.Dropdown(
                            choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                            value="Direct scale factor",
                            label="Region Model Scaling Mode"
                        )

                    with gr.Row():
                        snap_multiple_b = gr.Dropdown(choices=[32, 64, 128], value=128, label="Snap to multiple (batch)")
                        gr.Markdown("Tip: each model can use its own up/down scaling before inference; outputs are aligned to the region model grid for quantification.")

                    with gr.Row():
                        run_batch_button = gr.Button("Run")
                        progress_bar = gr.Slider(0, 100, label="Progress", interactive=False)

                    run_batch_button.click(
                        run_batch,
                        inputs=[folder_input, folder_output, label_mapping_b, threshold_b, thresholds_b, min_area_b, max_area_b,
                            circularity_threshold_b, save_region_b, save_region_individual_b, save_glyco_b, save_glyco_png_transparent_b,
                            glyco_png_threshold_b, save_filtered_b, save_zdisc_b,
                                glycogen_scale_value_b, glycogen_scale_mode_b,
                                region_scale_value_b, region_scale_mode_b,
                                snap_multiple_b, glycogen_model_batch, region_model_batch],
                        outputs=[progress_bar]
                    )

                with gr.Tab("IMF metrics"):
                    gr.Markdown("Compute IMF and intra stereology metrics from a batch CSV exported by the Batch processing tab.")

                    with gr.Row():
                        batch_csv_file = gr.File(
                            label="Batch CSV File (.csv)",
                            file_types=[".csv"]
                        )
                        batch_csv_path = gr.Textbox(
                            label="Or CSV Path",
                            placeholder="Full path to glycogen_distribution_combined_threshold_*.csv"
                        )

                    output_xlsx_path = gr.Textbox(
                        label="Output Workbook Path",
                        placeholder="If blank, saves next to the CSV as *_stereology.xlsx"
                    )

                    include_image_location_imf = gr.Checkbox(
                        label="Add Image Location (central/superficial from Image ranking)",
                        value=False
                    )

                    include_subfolder_avg_imf = gr.Checkbox(
                        label="Include Subfolder-level Averages (requires Image Location)",
                        value=False
                    )

                    use_weighted_ratio_imf = gr.Checkbox(
                        label="Use 3:1 Weighting (superficial:central)",
                        value=False
                    )

                    subfolder_calc_mode_imf = gr.Dropdown(
                        choices=["average_metrics", "average_inputs"],
                        value="average_metrics",
                        label="Subfolder Calculation Mode",
                        info="average_metrics: average image-level metrics | average_inputs: average raw inputs then compute metrics"
                    )

                    diameter_method_imf = gr.Dropdown(
                        choices=["feret", "area_circle"],
                        value="feret",
                        label="Particle Diameter Calculation Method",
                        info="feret: from Mean Feret Diameter | area_circle: from Area assuming circular profile"
                    )

                    with gr.Row():
                        pixel_size_nm_imf = gr.Number(
                            label="Pixel Size (nm)",
                            value=0.7698,
                            precision=4
                        )
                        thickness_um_imf = gr.Number(
                            label="Section Thickness (um)",
                            value=0.06,
                            precision=4
                        )

                    with gr.Row():
                        imf_aa_slope = gr.Number(
                            label="IMF AA Correction Slope",
                            value=-0.1767,
                            precision=6
                        )
                        imf_aa_intercept = gr.Number(
                            label="IMF AA Correction Intercept",
                            value=0.00299,
                            precision=6
                        )

                    with gr.Row():
                        intra_aa_slope = gr.Number(
                            label="Intra AA Correction Slope",
                            value=-0.1561,
                            precision=6
                        )
                        intra_aa_intercept = gr.Number(
                            label="Intra AA Correction Intercept",
                            value=0.001728,
                            precision=6
                        )

                    gr.Markdown("The CSV must contain Subfolder, Image, Region, Area, Glycogen Area, Mean Feret Diameter, and Z-disc max feret columns.")

                    run_imf_metrics_button = gr.Button("Compute IMF Metrics", variant="primary")

                    imf_metrics_output = gr.DataFrame(
                        label="Computed IMF Metrics",
                        interactive=False
                    )

                    imf_metrics_status = gr.Textbox(
                        label="Status",
                        lines=4,
                        interactive=False
                    )

                    run_imf_metrics_button.click(
                        run_imf_metrics,
                        inputs=[
                            batch_csv_file,
                            batch_csv_path,
                            output_xlsx_path,
                            pixel_size_nm_imf,
                            thickness_um_imf,
                            imf_aa_slope,
                            imf_aa_intercept,
                            intra_aa_slope,
                            intra_aa_intercept,
                            include_image_location_imf,
                            include_subfolder_avg_imf,
                            use_weighted_ratio_imf,
                            diameter_method_imf,
                            subfolder_calc_mode_imf,
                        ],
                        outputs=[imf_metrics_output, imf_metrics_status, output_xlsx_path]
                    )

                with gr.Tab("Single image"):
                    with gr.Row():
                        glycogen_model_single = gr.Textbox(
                            label="Glycogen Model Path",
                            value=args.glycogen_model,
                            placeholder="Full path to glycogen .pth model"
                        )
                        region_model_single = gr.Textbox(
                            label="Region Model Path",
                            value=args.region_model,
                            placeholder="Full path to region .pth model"
                        )

                    with gr.Row():
                        tif_input = gr.File(label="Input Image", file_types=[".tif", ".tiff", ".TIF", ".TIFF"])
                        tif_prewiew = gr.Image(image_mode="F", height=1024, width=1024)

                    with gr.Row():
                        with gr.Column(scale=1):
                            threshold_s = gr.Slider(label="Threshold", minimum=0, maximum=1, step=0.01, value=0.3)
                            circularity_threshold_s = gr.Slider(label="Single Particle circularity threshold", minimum=0, maximum=1, step=0.01, value=0.80)
                            min_area_s = gr.Slider(label="Single Particle min area (with 0.7698 nm/pixel, 133 pixels = 10 nm diameter)", minimum=0, maximum=500, step=1, value=133)
                            max_area_s = gr.Slider(label="Single Particle max area (with 0.7698 nm/pixel, 2338 pixels = 42 nm diameter)", minimum=0, maximum=4000, step=1, value=2338)
                        
                        with gr.Column(scale=1):
                            label_mapping_s = gr.Textbox(
                                label="Label dictionary", 
                                info="Python dictionary describing the included labels",
                                value=default_label_mapping,
                                lines=10
                            )

                    # Per-model scale & snap for single image
                    with gr.Row():
                        glycogen_scale_value_s = gr.Number(value=1.0, precision=6, label="Glycogen Model Scale Value")
                        glycogen_scale_mode_s = gr.Dropdown(
                            choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                            value="Direct scale factor",
                            label="Glycogen Model Scaling Mode"
                        )

                    with gr.Row():
                        region_scale_value_s = gr.Number(value=1.0, precision=6, label="Region Model Scale Value")
                        region_scale_mode_s = gr.Dropdown(
                            choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                            value="Direct scale factor",
                            label="Region Model Scaling Mode"
                        )

                    with gr.Row():
                        snap_multiple_s = gr.Dropdown(choices=[32, 64, 128], value=128, label="Snap to multiple")
                        gr.Markdown("Tip: each model can use independent up/down scaling; outputs are aligned to the region model grid.")


                    with gr.Row():
                        glyco_output = gr.Image(interactive=False, height=256, width=256)
                        region_output = gr.Image(interactive=False, height=256, width=256)
                        filtered_output = gr.Image(interactive=False, height=256, width=256)
                        feret_output = gr.Image(interactive=False, height=256, width=256)

                    stats = gr.DataFrame(headers=[
                        "Region", "Area", "Glycogen Area", "Particle area", "Form factor", "Feret diameter",
                        "z-disc maximal feret sum"
                    ])

                    run_button = gr.Button("Run")
                    tif_input.upload(open_tif, inputs=tif_input, outputs=tif_prewiew)

                    run_button.click(
                        run_segmentation,
                        inputs=[tif_prewiew, label_mapping_s, threshold_s, min_area_s, max_area_s, circularity_threshold_s,
                            glycogen_scale_value_s, glycogen_scale_mode_s,
                            region_scale_value_s, region_scale_mode_s,
                            snap_multiple_s, glycogen_model_single, region_model_single],
                        outputs=[glyco_output, region_output, filtered_output, feret_output, stats]
                    )

                with gr.Tab("Region Annotation Comparison"):
                    gr.Markdown("Compare a single fully-annotated region NRRD against region model predictions.")

                    with gr.Row():
                        region_model_compare = gr.Textbox(
                            label="Region Model Path",
                            value=args.region_model,
                            placeholder="Full path to region .pth model"
                        )

                    with gr.Row():
                        region_image_compare = gr.File(
                            label="Input Image (.tif/.tiff)",
                            file_types=[".tif", ".tiff", ".TIF", ".TIFF"]
                        )
                        region_annotation_compare = gr.File(
                            label="Annotated Regions (.nrrd)",
                            file_types=[".nrrd", ".NRRD"]
                        )

                    with gr.Row():
                        region_scale_value_cmp = gr.Number(value=1.0, precision=6, label="Region Model Scale Value")
                        region_scale_mode_cmp = gr.Dropdown(
                            choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                            value="Direct scale factor",
                            label="Region Model Scaling Mode"
                        )
                        snap_multiple_cmp = gr.Dropdown(choices=[32, 64, 128], value=128, label="Snap to multiple")

                    label_mapping_cmp = gr.Textbox(
                        label="Label dictionary",
                        info="Must match the label dictionary used in the Single image tab",
                        value=default_label_mapping,
                        lines=8
                    )

                    export_dir_cmp = gr.Textbox(
                        label="Export Folder (optional)",
                        placeholder="If blank, files are saved next to the input image"
                    )

                    run_compare_button = gr.Button("Run Comparison")

                    with gr.Row():
                        annotation_output_cmp = gr.Image(label="Annotation (Regions)", interactive=False, height=256, width=256)
                        prediction_output_cmp = gr.Image(label="Model Prediction", interactive=False, height=256, width=256)
                        misclassified_output_cmp = gr.Image(label="Misclassified Pixels", interactive=False, height=256, width=256)

                    confusion_a_output = gr.DataFrame(
                        label="Confusion Matrix A: Agreement (%) relative to Ground Truth Positives",
                        interactive=False
                    )
                    confusion_b_output = gr.DataFrame(
                        label="Confusion Matrix B: Positive Predictive Values (%) relative to Predicted Positives",
                        interactive=False
                    )

                    with gr.Row():
                        confusion_a_panel = gr.Image(
                            label="Panel A (Image-Style)",
                            interactive=False,
                            height=350
                        )
                        confusion_b_panel = gr.Image(
                            label="Panel B (Image-Style)",
                            interactive=False,
                            height=350
                        )

                    export_status_cmp = gr.Textbox(
                        label="Export Status",
                        lines=6,
                        interactive=False
                    )

                    run_compare_button.click(
                        run_region_annotation_comparison,
                        inputs=[
                            region_image_compare,
                            region_annotation_compare,
                            region_model_compare,
                            region_scale_value_cmp,
                            region_scale_mode_cmp,
                            snap_multiple_cmp,
                            export_dir_cmp,
                            label_mapping_cmp
                        ],
                        outputs=[
                            annotation_output_cmp,
                            prediction_output_cmp,
                            misclassified_output_cmp,
                            confusion_a_panel,
                            confusion_b_panel,
                            confusion_a_output,
                            confusion_b_output,
                            export_status_cmp
                        ]
                    )

                with gr.Tab("Region Fine-tune"):
                    gr.Markdown("## Fine-tune Model on New Data")
                    gr.Markdown("Train an existing model on your own annotated images and masks")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Model Configuration")
                            
                            model_file_ft = gr.File(
                                label="Pre-trained Model (.pth)",
                                file_types=[".pth"]
                            )
                            
                            model_path_text_ft = gr.Textbox(
                                label="Or Model Path",
                                placeholder="Full path to .pth file"
                            )
                            
                            gr.Markdown("""
                            **Class Information:**  
                            Classes are automatically detected from your NRRD label files.  
                            Expected format from [Slicer](https://www.slicer.org):
                            - "A-band": ("red", [0])
                            - "I-band": ("blue", [1])
                            - "intermyofibrillar": ("green", [2])
                            - "mitochondria": ("orange", [5])
                            - "z-disc": ("yellow", [6])
                            
                            The number of classes will be inferred from segment metadata in your .nrrd files.
                            """)
                            
                        with gr.Column():
                            gr.Markdown("### Data Configuration")
                            
                            images_dir_ft = gr.Textbox(
                                label="Training Images Directory",
                                placeholder="Full path to folder with .tif images"
                            )
                            
                            labels_dir_ft = gr.Textbox(
                                label="Training Labels Directory",
                                placeholder="Full path to folder with .nrrd masks"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Output Configuration")
                            
                            output_dir_ft = gr.Textbox(
                                label="Output Directory",
                                placeholder="Full path to save trained models"
                            )
                            
                            model_name_ft = gr.Textbox(
                                label="Model Name",
                                placeholder="e.g., glycogen_finetuned_v2",
                                value="model_finetuned"
                            )
                            
                        with gr.Column():
                            gr.Markdown("### Image Processing")
                            
                            patch_size_ft = gr.Dropdown(
                                label="Patch Size",
                                choices=[256, 512, 1024, 2048],
                                value=1024
                            )
                            
                            augmentation_ft = gr.Dropdown(
                                label="Data Augmentation",
                                choices=["None", "Basic (Flips + Rotations)"],
                                value="Basic (Flips + Rotations)"
                            )

                            training_scale_factor_ft = gr.Number(
                                label="Training Image/Mask Scale Factor",
                                value=1.0,
                                precision=4
                            )

                            gr.Markdown("""
                            **Training Scale Factor (default 1.0):**
                            Resizes both training images and NRRD masks before patch extraction.
                            - Use this to match resolution between source data and the pre-trained model.
                            - Formula: `scale = input_nm_per_px / target_nm_per_px`
                            - Example: 3.7 nm/px to 0.7 nm/px -> scale = 3.7 / 0.7 = 5.29 (start with ~5.0)

                            Interpolation:
                            - Images: linear interpolation
                            - Masks: nearest-neighbor interpolation (preserves class labels)
                            """)
                            
                            gr.Markdown("""
                            **Data Augmentation Details:**
                            
                            - **None**: No augmentation. Use for large, diverse datasets.
                            
                            - **Basic (Flips + Rotations)**: Applied to training data only (not validation).
                              - **Horizontal flip** (50% probability): Mirrors image left-to-right
                              - **Vertical flip** (50% probability): Mirrors image top-to-bottom  
                              - **Random rotation** (50% probability): Rotates by 0°, 90°, 180°, or 270°
                              
                            *Recommended*: Enable augmentation for smaller datasets (<50 images) to prevent overfitting.
                            """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Training Hyperparameters")
                            
                            learning_rate_ft = gr.Number(
                                label="Learning Rate",
                                value=0.00005,
                                precision=6
                            )
                            
                            gr.Markdown("""
                            **Learning Rate (0.00005 = 5e-5):**  
                            Controls how much the model weights change with each training step.
                            - **Too high (>1e-3)**: Model may fail to converge, loss oscillates or explodes. Training unstable.
                            - **5e-5** (default): Optimal for fine-tuning pre-trained models. Preserves learned features while adapting.
                            - **1e-4 to 1e-5**: Safe range for fine-tuning. Use 1e-4 for faster adaptation on very different data.
                            - **Too low (<1e-6)**: Training is too slow, may not converge within reasonable epochs.
                            
                            *For fine-tuning*: Keep learning rate low (1e-5 to 1e-4) to avoid destroying pre-trained knowledge.  
                            *For training from scratch*: Higher rates (1e-3 to 1e-4) are typically needed.
                            """)
                            
                            batch_size_ft = gr.Slider(
                                label="Batch Size",
                                minimum=1,
                                maximum=32,
                                value=8,
                                step=1
                            )
                            
                            num_epochs_ft = gr.Slider(
                                label="Number of Epochs",
                                minimum=5,
                                maximum=500,
                                value=50,
                                step=5
                            )
                            
                            early_stopping_patience_ft = gr.Slider(
                                label="Early Stopping Patience",
                                minimum=5,
                                maximum=100,
                                value=20,
                                step=5
                            )
                            
                        with gr.Column():
                            gr.Markdown("### Advanced Options")
                            
                            reconstruction_weight_ft = gr.Number(
                                label="Reconstruction Loss Weight",
                                value=0.1,
                                precision=2
                            )
                            
                            gr.Markdown("""
                            **Reconstruction Loss Weight (0.1):**  
                            Controls the balance between segmentation accuracy and image reconstruction.
                            - **0.1** (default): 10% reconstruction + 90% segmentation. Good for general use.
                            - **Lower (0.01-0.05)**: Focus more on segmentation accuracy.
                            - **Higher (0.2-0.5)**: Emphasize image fidelity and feature learning.
                            
                            **Batch Size (8):**  
                            Number of image patches processed simultaneously before updating model weights.
                            - **1-4**: Safe for limited GPU/RAM (6-8 GB). Slower but always works.
                            - **8** (default): Balanced - efficient training with moderate memory use (12-16 GB).
                            - **16-32**: Faster training, needs powerful GPU (24+ GB). May cause out-of-memory errors.
                            
                            *Tip*: If you get "out of memory" errors, reduce batch size to 4 or 2.
                            
                            **Other Tips:**
                            - Enable augmentation for small datasets (<50 images)
                            - Patch size must be compatible with your images
                            - Larger patch sizes require more memory (reduce batch size if needed)
                            """)
                    
                    # Training button and output
                    train_button = gr.Button("Start Fine-tuning", variant="primary")
                    
                    training_output = gr.Textbox(
                        label="Training Status",
                        lines=15,
                        max_lines=25
                    )
                    
                    # Wrapper function to pass device
                    def run_finetuning(model_file, model_path_text, images_dir, labels_dir, output_dir, model_name,
                                      learning_rate, batch_size, num_epochs, early_stopping_patience,
                                  patch_size, augmentation, reconstruction_weight, training_scale_factor):
                        return finetune_model(model_file, model_path_text, images_dir, labels_dir, output_dir, 
                                            model_name, learning_rate, batch_size, num_epochs, 
                                            early_stopping_patience, patch_size, augmentation, 
                                      reconstruction_weight, training_scale_factor, device)
                    
                    train_button.click(
                        run_finetuning,
                        inputs=[model_file_ft, model_path_text_ft, images_dir_ft, labels_dir_ft,
                               output_dir_ft, model_name_ft, learning_rate_ft, batch_size_ft,
                               num_epochs_ft, early_stopping_patience_ft, patch_size_ft,
                               augmentation_ft, reconstruction_weight_ft, training_scale_factor_ft],
                        outputs=[training_output]
                    )

                with gr.Tab("Glycogen Fine-tune"):
                    gr.Markdown("## Fine-tune Glycogen Model")
                    gr.Markdown("Train an existing glycogen model on your own annotated images and NRRD masks")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Model Configuration")

                            model_file_glyco_ft = gr.File(
                                label="Pre-trained Glycogen Model (.pth)",
                                file_types=[".pth"]
                            )

                            model_path_text_glyco_ft = gr.Textbox(
                                label="Or Glycogen Model Path",
                                placeholder="Full path to glycogen .pth file",
                                value=args.glycogen_model
                            )

                            segment_names_glyco_ft = gr.Textbox(
                                label="Segment Names",
                                placeholder="Comma-separated segment names",
                                value="Glycogen, Background"
                            )

                            gr.Markdown("""
                            **Glycogen Training Setup:**  
                            This follows the binary training flow in `train_glycogen.py`:
                            - single-channel sigmoid output
                            - no reconstruction branch
                            - labels converted to binary targets with ignored unlabeled pixels

                            Default segment names are `Glycogen, Background`.
                            """)

                        with gr.Column():
                            gr.Markdown("### Data Configuration")

                            images_dir_glyco_ft = gr.Textbox(
                                label="Training Images Directory",
                                placeholder="Full path to folder with .tif images"
                            )

                            labels_dir_glyco_ft = gr.Textbox(
                                label="Training Labels Directory",
                                placeholder="Full path to folder with .nrrd masks"
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Output Configuration")

                            output_dir_glyco_ft = gr.Textbox(
                                label="Output Directory",
                                placeholder="Full path to save trained glycogen models",
                                value=os.path.join(base_dir, "weights_glyco")
                            )

                            model_name_glyco_ft = gr.Textbox(
                                label="Model Name",
                                placeholder="e.g., model_glycogen_finetuned",
                                value="model_glycogen_finetuned"
                            )

                        with gr.Column():
                            gr.Markdown("### Image Processing")

                            patch_size_glyco_ft = gr.Dropdown(
                                label="Patch Size",
                                choices=[256, 512, 1024, 2048],
                                value=1024
                            )

                            augmentation_glyco_ft = gr.Dropdown(
                                label="Data Augmentation",
                                choices=["None", "Basic (Flips + Rotations)"],
                                value="Basic (Flips + Rotations)"
                            )

                            training_scale_factor_glyco_ft = gr.Number(
                                label="Training Image/Mask Scale Factor",
                                value=1.0,
                                precision=4
                            )

                            use_boundary_softening_glyco_ft = gr.Checkbox(
                                label="Enable Boundary Softening",
                                value=False
                            )

                            boundary_softening_sigma_glyco_ft = gr.Number(
                                label="Boundary Softening Sigma (pixels)",
                                value=1.0,
                                precision=2
                            )

                            gr.Markdown("""
                            **Training Scale Factor (default 1.0):**
                            Resizes both training images and NRRD masks before patch extraction.
                            Use the same scale convention as the inference tabs when matching microscope resolutions.

                            **Boundary Softening Tip:**
                            Use this when mask edges are a bit uncertain or hand-drawn. Start with `1.0` to `2.0` pixels.
                            Keep it at `0` or leave the checkbox off for crisp boundaries. Values above `3` can oversoften thin structures.
                            """)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Training Hyperparameters")

                            learning_rate_glyco_ft = gr.Number(
                                label="Learning Rate",
                                value=0.00005,
                                precision=6
                            )

                            batch_size_glyco_ft = gr.Slider(
                                label="Batch Size",
                                minimum=1,
                                maximum=32,
                                value=8,
                                step=1
                            )

                            num_epochs_glyco_ft = gr.Slider(
                                label="Number of Epochs",
                                minimum=5,
                                maximum=500,
                                value=50,
                                step=5
                            )

                        with gr.Column():
                            gr.Markdown("### Advanced Options")

                            save_frequency_glyco_ft = gr.Slider(
                                label="Checkpoint Save Frequency",
                                minimum=1,
                                maximum=25,
                                value=5,
                                step=1
                            )

                            gr.Markdown("""
                            **Checkpoint Save Frequency:**  
                            Matches the glycogen training script by saving model checkpoints every `N` epochs.

                            **Notes:**
                            - This glycogen path does not use reconstruction loss.
                            - No early stopping is applied.
                            - If GPU memory is tight, reduce batch size before reducing patch size.
                            """)

                    train_button_glyco = gr.Button("Start Glycogen Fine-tuning", variant="primary")

                    training_output_glyco = gr.Textbox(
                        label="Training Status",
                        lines=15,
                        max_lines=25
                    )

                    def run_glycogen_finetuning(model_file, model_path_text, images_dir, labels_dir, output_dir, model_name,
                                                learning_rate, batch_size, num_epochs, patch_size, augmentation,
                                                training_scale_factor, save_frequency, segment_names_text,
                                                use_boundary_softening, boundary_softening_sigma):
                        return finetune_glycogen_model(
                            model_file,
                            model_path_text,
                            images_dir,
                            labels_dir,
                            output_dir,
                            model_name,
                            learning_rate,
                            batch_size,
                            num_epochs,
                            patch_size,
                            augmentation,
                            training_scale_factor,
                            save_frequency,
                            segment_names_text,
                            use_boundary_softening,
                            boundary_softening_sigma,
                            device
                        )

                    train_button_glyco.click(
                        run_glycogen_finetuning,
                        inputs=[model_file_glyco_ft, model_path_text_glyco_ft, images_dir_glyco_ft, labels_dir_glyco_ft,
                                output_dir_glyco_ft, model_name_glyco_ft, learning_rate_glyco_ft, batch_size_glyco_ft,
                                num_epochs_glyco_ft, patch_size_glyco_ft, augmentation_glyco_ft,
                            training_scale_factor_glyco_ft, save_frequency_glyco_ft, segment_names_glyco_ft,
                            use_boundary_softening_glyco_ft, boundary_softening_sigma_glyco_ft],
                        outputs=[training_output_glyco]
                    )

                with gr.Tab("Region Train From Scratch"):
                    gr.Markdown("## Train Region Model From Scratch")
                    gr.Markdown("Train a region model without pre-trained weights, with optional dataset 2 and dataset 3 for mixed magnifications.")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Primary Dataset (required)")

                            images_dir_region_sc = gr.Textbox(
                                label="Dataset 1 Images Directory",
                                placeholder="Full path to folder with .tif images"
                            )

                            labels_dir_region_sc = gr.Textbox(
                                label="Dataset 1 Labels Directory",
                                placeholder="Full path to folder with .nrrd masks"
                            )

                        with gr.Column():
                            gr.Markdown("### Output Configuration")

                            output_dir_region_sc = gr.Textbox(
                                label="Output Directory",
                                placeholder="Full path to save trained region models",
                                value=os.path.join(base_dir, "weights")
                            )

                            model_name_region_sc = gr.Textbox(
                                label="Model Name",
                                placeholder="e.g., model_region_scratch",
                                value="model_region_scratch"
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Core Hyperparameters")

                            learning_rate_region_sc = gr.Number(
                                label="Learning Rate",
                                value=0.001,
                                precision=6
                            )

                            batch_size_region_sc = gr.Slider(
                                label="Batch Size",
                                minimum=1,
                                maximum=32,
                                value=8,
                                step=1
                            )

                            num_epochs_region_sc = gr.Slider(
                                label="Number of Epochs",
                                minimum=5,
                                maximum=500,
                                value=100,
                                step=5
                            )

                            early_stopping_patience_region_sc = gr.Slider(
                                label="Early Stopping Patience",
                                minimum=5,
                                maximum=100,
                                value=20,
                                step=5
                            )

                            save_frequency_region_sc = gr.Slider(
                                label="Checkpoint Save Frequency",
                                minimum=1,
                                maximum=50,
                                value=5,
                                step=1
                            )

                        with gr.Column():
                            gr.Markdown("### Patch / Loss Settings")

                            patch_size_region_sc = gr.Dropdown(
                                label="Patch Size",
                                choices=[256, 512, 1024, 2048],
                                value=1024
                            )

                            augmentation_region_sc = gr.Dropdown(
                                label="Data Augmentation",
                                choices=["None", "Basic (Flips + Rotations)"],
                                value="Basic (Flips + Rotations)"
                            )

                            reconstruction_weight_region_sc = gr.Number(
                                label="Reconstruction Loss Weight",
                                value=0.1,
                                precision=3
                            )

                            gr.Markdown("Segment classes are inferred automatically from NRRD metadata in Dataset 1 labels.")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Dataset 1 Magnification / Resampling")

                            primary_scale_value_region_sc = gr.Number(
                                label="Dataset 1 Scale Value",
                                value=1.0,
                                precision=4
                            )

                            primary_resample_mode_region_sc = gr.Dropdown(
                                label="Dataset 1 Resampling Mode",
                                choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                                value="Direct scale factor"
                            )

                        with gr.Column():
                            gr.Markdown("### Optional Dataset 2")

                            include_second_dataset_region_sc = gr.Checkbox(
                                label="Include Dataset 2",
                                value=False
                            )

                            images_dir_2_region_sc = gr.Textbox(
                                label="Dataset 2 Images Directory",
                                placeholder="Full path to folder with .tif images (optional)"
                            )

                            labels_dir_2_region_sc = gr.Textbox(
                                label="Dataset 2 Labels Directory",
                                placeholder="Full path to folder with .nrrd masks (optional)"
                            )

                            second_scale_value_region_sc = gr.Number(
                                label="Dataset 2 Scale Value",
                                value=1.0,
                                precision=4
                            )

                            second_resample_mode_region_sc = gr.Dropdown(
                                label="Dataset 2 Resampling Mode",
                                choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                                value="Direct scale factor"
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Optional Dataset 3")

                            include_third_dataset_region_sc = gr.Checkbox(
                                label="Include Dataset 3",
                                value=False
                            )

                            images_dir_3_region_sc = gr.Textbox(
                                label="Dataset 3 Images Directory",
                                placeholder="Full path to folder with .tif images (optional)"
                            )

                            labels_dir_3_region_sc = gr.Textbox(
                                label="Dataset 3 Labels Directory",
                                placeholder="Full path to folder with .nrrd masks (optional)"
                            )

                            third_scale_value_region_sc = gr.Number(
                                label="Dataset 3 Scale Value",
                                value=1.0,
                                precision=4
                            )

                            third_resample_mode_region_sc = gr.Dropdown(
                                label="Dataset 3 Resampling Mode",
                                choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                                value="Direct scale factor"
                            )

                        with gr.Column():
                            gr.Markdown("### Notes")
                            gr.Markdown("""
                            Use 1 dataset (only Dataset 1), 2 datasets (enable Dataset 2), or 3 datasets (enable Dataset 2 and Dataset 3).

                            Each dataset can use its own resampling mode and scale value to normalize magnification differences before patch extraction.
                            """)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Optional Dataset 4")

                            include_fourth_dataset_region_sc = gr.Checkbox(
                                label="Include Dataset 4",
                                value=False
                            )

                            images_dir_4_region_sc = gr.Textbox(
                                label="Dataset 4 Images Directory",
                                placeholder="Full path to folder with .tif images (optional)"
                            )

                            labels_dir_4_region_sc = gr.Textbox(
                                label="Dataset 4 Labels Directory",
                                placeholder="Full path to folder with .nrrd masks (optional)"
                            )

                            fourth_scale_value_region_sc = gr.Number(
                                label="Dataset 4 Scale Value",
                                value=1.0,
                                precision=4
                            )

                            fourth_resample_mode_region_sc = gr.Dropdown(
                                label="Dataset 4 Resampling Mode",
                                choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                                value="Direct scale factor"
                            )

                        with gr.Column():
                            pass

                    train_button_region_sc = gr.Button("Start Region Scratch Training", variant="primary")

                    check_mismatch_button_region_sc = gr.Button("Check Image/Label Filename Mismatches")

                    mismatch_output_region_sc = gr.Textbox(
                        label="Filename Mismatch Report",
                        lines=12,
                        max_lines=20
                    )

                    training_output_region_sc = gr.Textbox(
                        label="Training Status",
                        lines=15,
                        max_lines=25
                    )

                    def run_region_filename_mismatch_check(images_dir, labels_dir,
                                                           include_second_dataset, images_dir_2, labels_dir_2,
                                                           include_third_dataset, images_dir_3, labels_dir_3,
                                                           include_fourth_dataset, images_dir_4, labels_dir_4):
                        report, _ = build_mismatch_report([
                            ("Dataset 1", images_dir, labels_dir, True),
                            ("Dataset 2", images_dir_2, labels_dir_2, bool(include_second_dataset)),
                            ("Dataset 3", images_dir_3, labels_dir_3, bool(include_third_dataset)),
                            ("Dataset 4", images_dir_4, labels_dir_4, bool(include_fourth_dataset)),
                        ])
                        return report

                    def run_region_scratch_training(images_dir, labels_dir, output_dir, model_name,
                                                    learning_rate, batch_size, num_epochs, early_stopping_patience,
                                                    save_frequency, patch_size, augmentation, reconstruction_weight,
                                                    primary_resample_mode, primary_scale_value,
                                                    include_second_dataset, images_dir_2, labels_dir_2,
                                                    second_resample_mode, second_scale_value,
                                                    include_third_dataset, images_dir_3, labels_dir_3,
                                                    third_resample_mode, third_scale_value,
                                                    include_fourth_dataset, images_dir_4, labels_dir_4,
                                                    fourth_resample_mode, fourth_scale_value):
                        mismatch_report, has_mismatch = build_mismatch_report([
                            ("Dataset 1", images_dir, labels_dir, True),
                            ("Dataset 2", images_dir_2, labels_dir_2, bool(include_second_dataset)),
                            ("Dataset 3", images_dir_3, labels_dir_3, bool(include_third_dataset)),
                            ("Dataset 4", images_dir_4, labels_dir_4, bool(include_fourth_dataset)),
                        ])

                        if has_mismatch:
                            return (
                                "Error: Filename mismatches detected between image and label files. "
                                "Fix mismatches before training.\n\n"
                                + mismatch_report
                            )

                        return train_region_model_from_scratch(
                            images_dir,
                            labels_dir,
                            output_dir,
                            model_name,
                            learning_rate,
                            batch_size,
                            num_epochs,
                            early_stopping_patience,
                            save_frequency,
                            patch_size,
                            augmentation,
                            reconstruction_weight,
                            primary_resample_mode,
                            primary_scale_value,
                            include_second_dataset,
                            images_dir_2,
                            labels_dir_2,
                            second_resample_mode,
                            second_scale_value,
                            include_third_dataset,
                            images_dir_3,
                            labels_dir_3,
                            third_resample_mode,
                            third_scale_value,
                            include_fourth_dataset,
                            images_dir_4,
                            labels_dir_4,
                            fourth_resample_mode,
                            fourth_scale_value,
                            device
                        )

                    check_mismatch_button_region_sc.click(
                        run_region_filename_mismatch_check,
                        inputs=[images_dir_region_sc, labels_dir_region_sc,
                                include_second_dataset_region_sc, images_dir_2_region_sc, labels_dir_2_region_sc,
                                include_third_dataset_region_sc, images_dir_3_region_sc, labels_dir_3_region_sc,
                                include_fourth_dataset_region_sc, images_dir_4_region_sc, labels_dir_4_region_sc],
                        outputs=[mismatch_output_region_sc]
                    )

                    train_button_region_sc.click(
                        run_region_scratch_training,
                        inputs=[images_dir_region_sc, labels_dir_region_sc, output_dir_region_sc, model_name_region_sc,
                                learning_rate_region_sc, batch_size_region_sc, num_epochs_region_sc, early_stopping_patience_region_sc,
                                save_frequency_region_sc, patch_size_region_sc, augmentation_region_sc, reconstruction_weight_region_sc,
                                primary_resample_mode_region_sc, primary_scale_value_region_sc,
                                include_second_dataset_region_sc, images_dir_2_region_sc, labels_dir_2_region_sc,
                                second_resample_mode_region_sc, second_scale_value_region_sc,
                                include_third_dataset_region_sc, images_dir_3_region_sc, labels_dir_3_region_sc,
                                third_resample_mode_region_sc, third_scale_value_region_sc,
                                include_fourth_dataset_region_sc, images_dir_4_region_sc, labels_dir_4_region_sc,
                                fourth_resample_mode_region_sc, fourth_scale_value_region_sc],
                        outputs=[training_output_region_sc]
                    )

                with gr.Tab("Glycogen Train From Scratch"):
                    gr.Markdown("## Train Glycogen Model From Scratch")
                    gr.Markdown("Training flow and core options follow train_glycogen.py, with optional initialization weights.")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Data Configuration")

                            images_dir_glyco_sc = gr.Textbox(
                                label="Training Images Directory",
                                placeholder="Full path to folder with .tif images"
                            )

                            labels_dir_glyco_sc = gr.Textbox(
                                label="Training Labels Directory",
                                placeholder="Full path to folder with .nrrd masks"
                            )

                            segment_names_glyco_sc = gr.Textbox(
                                label="Segment Names",
                                placeholder="Comma-separated segment names",
                                value="Glycogen, Background"
                            )

                            include_second_dataset_glyco_sc = gr.Checkbox(
                                label="Include Second Dataset (different magnification)",
                                value=False
                            )

                            include_third_dataset_glyco_sc = gr.Checkbox(
                                label="Include Third Dataset (different magnification)",
                                value=False
                            )

                        with gr.Column():
                            gr.Markdown("### Optional Initialization")

                            init_model_file_glyco_sc = gr.File(
                                label="Optional Init Weights (.pth)",
                                file_types=[".pth"]
                            )

                            init_model_path_glyco_sc = gr.Textbox(
                                label="Or Init Weights Path",
                                placeholder="Full path to .pth file (optional)"
                            )

                            gr.Markdown("Leave both empty for true training from scratch.")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Output Configuration")

                            output_dir_glyco_sc = gr.Textbox(
                                label="Output Directory",
                                placeholder="Full path to save trained glycogen models",
                                value=os.path.join(base_dir, "weights_glyco")
                            )

                            model_name_glyco_sc = gr.Textbox(
                                label="Model Name",
                                placeholder="e.g., model_glycogen_scratch",
                                value="model_glycogen_scratch"
                            )

                        with gr.Column():
                            gr.Markdown("### Core Hyperparameters")

                            learning_rate_glyco_sc = gr.Number(
                                label="Learning Rate",
                                value=0.001,
                                precision=6
                            )

                            batch_size_glyco_sc = gr.Slider(
                                label="Batch Size",
                                minimum=1,
                                maximum=64,
                                value=32,
                                step=1
                            )

                            num_epochs_glyco_sc = gr.Slider(
                                label="Number of Epochs",
                                minimum=5,
                                maximum=500,
                                value=50,
                                step=5
                            )

                            save_frequency_glyco_sc = gr.Slider(
                                label="Checkpoint Save Frequency",
                                minimum=1,
                                maximum=50,
                                value=5,
                                step=1
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Data Loader / Patch Settings")

                            patch_size_glyco_sc = gr.Dropdown(
                                label="Patch Size",
                                choices=[256, 512, 1024, 2048],
                                value=1024
                            )

                            augmentation_glyco_sc = gr.Dropdown(
                                label="Data Augmentation",
                                choices=["None", "Basic (Flips + Rotations)"],
                                value="Basic (Flips + Rotations)"
                            )

                            training_scale_factor_glyco_sc = gr.Number(
                                label="Primary Dataset Scale Value",
                                value=1.0,
                                precision=4
                            )

                            primary_resample_mode_glyco_sc = gr.Dropdown(
                                label="Primary Dataset Resampling Mode",
                                choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                                value="Direct scale factor"
                            )

                            use_boundary_softening_glyco_sc = gr.Checkbox(
                                label="Enable Boundary Softening",
                                value=False
                            )

                            boundary_softening_sigma_glyco_sc = gr.Number(
                                label="Boundary Softening Sigma (pixels)",
                                value=1.0,
                                precision=2
                            )

                            gr.Markdown("""
                            Use up/down mode to adapt data from a different magnification before training.

                            **Boundary Softening Tip:**
                            Start with `1.0` to `2.0` pixels when mask edges are uncertain.
                            Leave it disabled, or use `0`, for crisp boundaries. Values above `3` can oversoften thin structures.
                            """)

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Optional Second Dataset")

                            images_dir_2_glyco_sc = gr.Textbox(
                                label="Second Dataset Images Directory",
                                placeholder="Full path to folder with .tif images (optional)"
                            )

                            labels_dir_2_glyco_sc = gr.Textbox(
                                label="Second Dataset Labels Directory",
                                placeholder="Full path to folder with .nrrd masks (optional)"
                            )

                        with gr.Column():
                            second_scale_value_glyco_sc = gr.Number(
                                label="Second Dataset Scale Value",
                                value=1.0,
                                precision=4
                            )

                            second_resample_mode_glyco_sc = gr.Dropdown(
                                label="Second Dataset Resampling Mode",
                                choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                                value="Direct scale factor"
                            )

                            gr.Markdown("Defaults match train_glycogen.py (lr=0.001, batch=32, epochs=50, save_freq=5, patch_size=1024).")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Optional Third Dataset")

                            images_dir_3_glyco_sc = gr.Textbox(
                                label="Third Dataset Images Directory",
                                placeholder="Full path to folder with .tif images (optional)"
                            )

                            labels_dir_3_glyco_sc = gr.Textbox(
                                label="Third Dataset Labels Directory",
                                placeholder="Full path to folder with .nrrd masks (optional)"
                            )

                        with gr.Column():
                            third_scale_value_glyco_sc = gr.Number(
                                label="Third Dataset Scale Value",
                                value=1.0,
                                precision=4
                            )

                            third_resample_mode_glyco_sc = gr.Dropdown(
                                label="Third Dataset Resampling Mode",
                                choices=["Direct scale factor", "Up-sample by factor", "Down-sample by factor"],
                                value="Direct scale factor"
                            )

                            gr.Markdown("Use Dataset 1 only, or enable Dataset 2 and/or Dataset 3 as needed.")

                    train_button_glyco_sc = gr.Button("Start Glycogen Scratch Training", variant="primary")

                    training_output_glyco_sc = gr.Textbox(
                        label="Training Status",
                        lines=15,
                        max_lines=25
                    )

                    def run_glycogen_scratch_training(images_dir, labels_dir, output_dir, model_name,
                                                      learning_rate, batch_size, num_epochs, save_frequency,
                                                      segment_names_text, init_model_file, init_model_path_text,
                                                      patch_size, augmentation, training_scale_factor,
                                                      primary_resample_mode,
                                                      include_second_dataset, images_dir_2, labels_dir_2,
                                                      second_resample_mode, second_scale_value,
                                                      include_third_dataset, images_dir_3, labels_dir_3,
                                                      third_resample_mode, third_scale_value,
                                                      use_boundary_softening, boundary_softening_sigma):
                        return train_glycogen_model_from_scratch(
                            images_dir,
                            labels_dir,
                            output_dir,
                            model_name,
                            learning_rate,
                            batch_size,
                            num_epochs,
                            save_frequency,
                            segment_names_text,
                            init_model_file,
                            init_model_path_text,
                            patch_size,
                            augmentation,
                            training_scale_factor,
                            primary_resample_mode,
                            include_second_dataset,
                            images_dir_2,
                            labels_dir_2,
                            second_resample_mode,
                            second_scale_value,
                            include_third_dataset,
                            images_dir_3,
                            labels_dir_3,
                            third_resample_mode,
                            third_scale_value,
                            use_boundary_softening,
                            boundary_softening_sigma,
                            device
                        )

                    train_button_glyco_sc.click(
                        run_glycogen_scratch_training,
                        inputs=[images_dir_glyco_sc, labels_dir_glyco_sc, output_dir_glyco_sc, model_name_glyco_sc,
                                learning_rate_glyco_sc, batch_size_glyco_sc, num_epochs_glyco_sc, save_frequency_glyco_sc,
                                segment_names_glyco_sc, init_model_file_glyco_sc, init_model_path_glyco_sc,
                            patch_size_glyco_sc, augmentation_glyco_sc, training_scale_factor_glyco_sc,
                            primary_resample_mode_glyco_sc,
                            include_second_dataset_glyco_sc, images_dir_2_glyco_sc, labels_dir_2_glyco_sc,
                            second_resample_mode_glyco_sc, second_scale_value_glyco_sc,
                            include_third_dataset_glyco_sc, images_dir_3_glyco_sc, labels_dir_3_glyco_sc,
                            third_resample_mode_glyco_sc, third_scale_value_glyco_sc,
                            use_boundary_softening_glyco_sc, boundary_softening_sigma_glyco_sc],
                        outputs=[training_output_glyco_sc]
                    )

    gui.queue()
    gui.launch(inbrowser=True)

if __name__ == '__main__':
    main()