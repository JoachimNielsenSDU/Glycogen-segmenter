# Runs the model using the given weights
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, colors
from skimage import io
from tqdm import tqdm

from datagenerator import ImagePatchBuffer, ImageSegmentationDataset
from unet import AttUNet


def main():
    # Parge arguments
    parser = argparse.ArgumentParser(description='Run inference on a U-Net model')
    parser.add_argument('--input', '-i', type=str, help='Path to directory or image file to run inference on',
                        default='data/glycogen_test')
    parser.add_argument('--output', '-o', type=str, help='Path to save output images', default='output')
    parser.add_argument('--weights', '-w', type=str, help='Path to model weights',
                        default='weights_glyco/glycogen_model.pth')
    parser.add_argument('--show', action='store_true', help='Display output images')
    parser.add_argument('--include_ground_truth', action='store_true', help='Include ground truth in output images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    filters = [8, 16, 32, 64, 128, 256, 512]
    model = AttUNet(1, 1, filters, final_activation='sigmoid', include_reconstruction=False).to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    # Load the test dataset
    buffer = ImagePatchBuffer(args.input, args.input,
                              segment_names=["Glycogen", "Background"],
                              patch_size=2048)
    dataset = ImageSegmentationDataset(buffer, augment=False)

    df_rows = []

    # Run inference on each image
    for i, (image, label) in tqdm(enumerate(dataset)):

        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            seg = model(image)

        seg = seg.squeeze().cpu().numpy()

        img_path = buffer.image_files[i]
        print(img_path)
        # print the proportion of glycogen in the image at various thresholds
        area_fractions = []
        for threshold in np.linspace(0, 1, 11):
            area_fraction = 1 - (np.sum(seg > threshold) / seg.size)
            area_fractions.append(area_fraction)

        df_rows.append([img_path] + area_fractions)

        if i%5 == 0:
            plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
            plt.imshow(1 - seg.squeeze(), alpha=0.5, cmap='viridis')
            # colorbar
            cbar = plt.colorbar()
            cbar.set_label("Glycogen")
            plt.tight_layout()
            plt.savefig(f"examples/glyco_predseg_{i}.png")
            plt.show()


    df = pd.DataFrame(df_rows, columns=["image"] + [f"threshold_{i}" for i in range(11)])
    df.to_csv("glycogen_area_fraction.csv", index=False)

if __name__ == "__main__":
    main()
