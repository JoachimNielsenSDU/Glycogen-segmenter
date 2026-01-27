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
                        default='data/test')
    parser.add_argument('--output', '-o', type=str, help='Path to save output images', default='output')
    parser.add_argument('--weights', '-w', type=str, help='Path to model weights',
                        default='weights/region_model.pth')
    parser.add_argument('--show', action='store_true', help='Display output images')
    parser.add_argument('--include_ground_truth', action='store_true', help='Include ground truth in output images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    model = AttUNet(1, 7, filters, final_activation='softmax').to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    # Load the test dataset
    buffer = ImagePatchBuffer(args.input, args.input,
                              segment_names=["A-band", "I-band", "intermyofibrillar", "myofibril", "sarco",
                                             "mitochondria", "z-disc"],
                              patch_size=2048)
    dataset = ImageSegmentationDataset(buffer, augment=False)

    confusion_matrix = np.zeros((4, 7, 7))
    df_rows = []

    image_groups = pd.read_csv("groups.csv", header=0)

    # Run inference on each image
    for i, (image, label) in tqdm(enumerate(dataset)):

        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            seg, recon = model(image)

        seg = seg.squeeze().cpu().numpy()
        recon = recon.squeeze().cpu().numpy()
        image = image.squeeze().cpu().numpy()
        label = label.squeeze().cpu().numpy()

        # Reduce to single channel

        seg = np.argmax(seg, axis=0)

        unlabeled = np.sum(label, axis=0) < 0
        label = np.argmax(label, axis=0)
        label[unlabeled] = -1

        # Convert all instances of label 4 to label 2
        label[label == 4] = 2
        label[label == 3] = 1

        # Determine image group
        img_path = buffer.image_files[i]
        pattern = r"_([0-9]+).\w+"
        img_id = int(re.search(pattern, img_path).group(1))
        image_group = image_groups[image_groups['billede navn'] == img_id]['gruppe'].values[0]
        image_group = int(image_group) - 1

        # Add to df rows (image id, image group, class, gt proportion, pred proportion)
        for j in range(7):
            gt_proportion = np.sum(label == j) / np.sum(label != -1)
            pred_proportion = np.sum(seg == j) / np.sum(seg != -1)
            df_rows.append([i, image_group, j, gt_proportion, pred_proportion])

        # Update confusion matrix
        for j in range(7):
            for k in range(7):
                confusion_matrix[image_group, j, k] += np.sum((seg == j) & (label == k))

        label += 1

        if args.show or True:
            if args.include_ground_truth or True:

                # set figsize to 3x1 ratio
                plt.figure(figsize=(20, 5))

                # Pred
                plt.subplot(1, 5, 1)
                plt.imshow(image, cmap='gray')
                cmap = colors.ListedColormap(['teal', 'red', 'green', 'blue', 'yellow', 'purple', 'orange'])
                plt.imshow(seg, alpha=0.4, cmap=cmap, vmin=0, vmax=6)

                # GT
                plt.subplot(1, 5, 2)
                plt.imshow(image, cmap='gray')
                cmap = colors.ListedColormap(['black', 'teal', 'red', 'green', 'blue', 'yellow', 'purple', 'orange'])
                plt.imshow(label, alpha=0.4, cmap=cmap, vmin=0, vmax=7)

                # Plot green where the prediction and ground truth match
                # and red where they do not
                # and black where there is no ground truth
                plt.subplot(1, 5, 3)
                plt.imshow(image, cmap='gray')
                cmap_green = colors.ListedColormap(['black', 'green'])
                cmap_red = colors.ListedColormap(['black', 'red'])
                match = (seg + 1 == label) & (label != 0)
                mismatch = (seg + 1 != label) & (label != 0)
                plt.imshow(match, alpha=0.4, cmap=cmap_green, vmin=0, vmax=1)
                plt.imshow(mismatch * 2, alpha=0.4, cmap=cmap_red, vmin=0, vmax=1)

                # Recon
                plt.subplot(1, 5, 4)
                plt.imshow(recon, cmap='gray')

                # confusion matrix
                plt.subplot(1, 5, 5)
                plt.imshow(np.log(confusion_matrix[image_group]), cmap='viridis')
                plt.colorbar()

            else:
                plt.imshow(image, cmap='gray')
                cmap = colors.ListedColormap(['teal', 'red', 'green', 'blue', 'yellow', 'purple', 'orange'])
                plt.imshow(seg, alpha=0.4, cmap=cmap, vmin=0, vmax=6)

            plt.tight_layout()
            plt.savefig(f"examples/prediction_{i}.png")
            plt.show()

    class_names = ["A-band", "I-band", "intermyofibrillar", "myofibril", "sarco", "mitochondria", "z-disc"]

    # Remove empty classes (3 and 4) from confusion matrices
    selection = np.ones(7, dtype=bool)
    selection[3] = False
    selection[4] = False
    confusion_matrix_used = confusion_matrix[:, selection, :][:, :, selection]
    class_names_used = ["A-band","I-band","intermyo","mito","z-disc"]
    from sklearn.metrics import ConfusionMatrixDisplay
    for i in range(4):
        disp = ConfusionMatrixDisplay(confusion_matrix_used[i], display_labels=class_names_used)
        disp.plot()
        plt.savefig(f"examples/confusion_matrix_{i}.png")
        plt.tight_layout()
        plt.show()

    # Common confusion matrix (sum of all confusion matrices)
    common_confusion_matrix = np.sum(confusion_matrix_used, axis=0)
    disp = ConfusionMatrixDisplay(common_confusion_matrix, display_labels=class_names_used)
    disp.plot()
    plt.savefig(f"examples/common_confusion_matrix.png")
    plt.tight_layout()
    plt.show()

    # Save the confusion matrices in a csv file
    # The columns are the predicted classes and the rows are the ground truth classes
    for i in range(4):
        df = pd.DataFrame(confusion_matrix_used[i], columns=class_names_used, index=class_names_used)
        df.to_csv(f"examples/confusion_matrix_{i}.csv")


    dataframe = pd.DataFrame(df_rows, columns=['image_id', 'image_group', 'class', 'gt_proportion', 'pred_proportion'])
    # Remove class 3 and 4 (empty)
    dataframe = dataframe[dataframe['class'] != 3]
    dataframe = dataframe[dataframe['class'] != 4]
    # Convert class ids to class names
    class_names = ["A-band", "I-band", "intermyofibrillar", "myofibril", "sarco", "mitochondria", "z-disc"]
    dataframe['class'] = dataframe['class'].apply(lambda x: class_names[x])
    dataframe.to_csv('examples/prediction.csv', index=False)
    print(f"Done!")


if __name__ == "__main__":
    main()
