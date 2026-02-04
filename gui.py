import argparse
import cv2
import numpy as np
import tifffile
import torch
from PIL import Image
from unet import AttUNet
import gradio as gr
from matplotlib import cm, colors
import pandas as pd
import os

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


def main():
    parser = argparse.ArgumentParser(description='Open GUI to run inference')
    parser.add_argument('--glycogen_model', type=str, help='Path to model weights',
                        default='weights/glycogen_model.pth')
    parser.add_argument('--region_model', type=str, help='Path to model weights',
                        default='weights/region_model.pth')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the models
    filters = [8, 16, 32, 64, 128, 256, 512]
    glyco_model = AttUNet(1, 1, filters, final_activation='sigmoid', include_reconstruction=False).to(device)
    glyco_model.load_state_dict(torch.load(args.glycogen_model, map_location=device, weights_only=True))
    glyco_model.eval()

    filters = [8, 16, 32, 64, 128, 256, 512, 1024]
    region_model = AttUNet(1, 7, filters, final_activation='softmax').to(device)
    region_model.load_state_dict(torch.load(args.region_model, map_location=device, weights_only=True))
    region_model.eval()

    def glycogen_segmentation(image_tensor):
        with torch.no_grad():
            seg = glyco_model(image_tensor)
        return seg.squeeze().detach().cpu().numpy()

    def region_segmentation(image_tensor):
        with torch.no_grad():
            seg, _ = region_model(image_tensor)
        return seg.squeeze().detach().cpu().numpy()

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

    # ------------- SINGLE IMAGE -------------
    def run_segmentation(image, label_mapping, threshold, min_area, max_area, circularity_threshold,
                         scale, snap_multiple):
        """
        image: numpy (from gr.Image, grayscale float)
        """
        # Parse label dict safely
        label_dict = eval(label_mapping)

        # Ensure float32
        img = np.array(image)
        if img.ndim == 3:
            # if it comes as (H,W,1)
            img = img[..., 0]
        img = img.astype(np.float32)

        # Resize by scale & snap
        img_rs = resize_with_snap(img, scale=float(scale), mult=int(snap_multiple))
        H, W = img_rs.shape[:2]

        # Normalize to 0-1 robustly
        vmin, vmax = np.min(img_rs), np.max(img_rs)
        if vmax > vmin:
            img01 = (img_rs - vmin) / (vmax - vmin)
        else:
            img01 = np.zeros_like(img_rs, dtype=np.float32)

        # Torch tensor (1,1,H,W)
        tensor = torch.from_numpy(img01).float().unsqueeze(0).unsqueeze(0).to(device)

        # Glycogen segmentation (note: original inverts predictions)
        glyco_output = 1 - glycogen_segmentation(tensor)  # (H,W) float
        glyco_viri = apply_viridis(glyco_output)

        # Region segmentation
        region_output = region_segmentation(tensor)  # (C=7,H,W)
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

            for cnt in contours:
                if len(cnt) >= 3:
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    if float(min_area) <= area <= float(max_area) and circularity >= float(circularity_threshold):
                        # centroid
                        M = cv2.moments(cnt)
                        if M["m00"] == 0:
                            continue
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if 0 <= cy < H and 0 <= cx < W and new_labels[cy, cx] == i:
                            filtered_areas.append(area)
                            color_bgr = tuple(int(c * 255) for c in colors.to_rgb(label_dict[label][0]))
                            rad = max(1, int(np.sqrt(area / np.pi)))
                            cv2.circle(filtered_image, (cx, cy), rad, color_bgr[::-1], thickness=-1)

            mean_particle_area = np.mean(filtered_areas) if filtered_areas else 0
            amounts.append((label, label_area, glycogen_area, mean_particle_area))

        df = pd.DataFrame(amounts, columns=["Region", "Area", "Glycogen Area", "Particle area"])
        df["z-disc maximal feret sum"] = z_disc_maximal_feret_sum

        return (glyco_viri, region_colored, Image.fromarray(filtered_image),
                Image.fromarray(filtered_image_with_feret_line), df)

    # ------------- BATCH -------------
    def run_batch(folder_input, folder_output, label_mapping, threshold, min_area, max_area,
                  circularity_threshold, save_region, save_glyco, save_filtered, save_zdisc,
                  scale, snap_multiple, progress=gr.Progress()):
        """
        Adds scale & snap to the existing batch pipeline. All canvases now match image size.
        """
        csv_rows_combined = []
        csv_rows_individuals = []
        particle_data = []
        total_area = 0
        total_glyco_area = 0
        label_areas_combined = {}
        glycogen_areas_combined = {}
        mean_particle_size_combined = {}

        label_areas_individuals = {}
        glycogen_areas_individuals = {}
        mean_particle_size_individuals = {}

        # Combined mapping (adds "intra")
        label_dict_combined = eval(label_mapping)
        label_dict_combined["intra"] = ("purple", label_dict_combined["A-band"][1] + label_dict_combined["I-band"][1])

        for label in label_dict_combined.keys():
            label_areas_combined[label] = 0
            glycogen_areas_combined[label] = 0
            mean_particle_size_combined[label] = 0

        original_label_dict_individuals = eval(label_mapping)
        for label in original_label_dict_individuals.keys():
            label_areas_individuals[label] = 0
            glycogen_areas_individuals[label] = 0
            mean_particle_size_individuals[label] = 0

        if folder_input is None or not os.path.isdir(folder_input):
            progress(0, desc="Invalid input folder.")
            return 0

        subfolders = [os.path.join(folder_input, subfolder)
                      for subfolder in os.listdir(folder_input)
                      if os.path.isdir(os.path.join(folder_input, subfolder))]
        total_subfolders = len(subfolders)

        # Ensure output
        if folder_output and folder_output.strip() != "":
            os.makedirs(folder_output, exist_ok=True)

        for idx, subfolder in enumerate(subfolders):
            progress((idx + 1) / max(1, total_subfolders), desc=f"Processing {subfolder}...")
            for root, dirs, files in os.walk(subfolder):
                for imgpath in files:
                    if imgpath.endswith(('.tif', '.tiff', '.TIF', '.TIFF')):
                        img_full_path = os.path.join(root, imgpath)
                        subfolder_name = os.path.basename(root)

                        # Read image
                        img = tifffile.imread(img_full_path)
                        if img.ndim == 3:
                            img = img[..., 0]
                        img = img.astype(np.float32)

                        # Resize (scale + snap)
                        img_rs = resize_with_snap(img, scale=float(scale), mult=int(snap_multiple))
                        H, W = img_rs.shape[:2]

                        # Normalize to 0-1
                        vmin, vmax = np.min(img_rs), np.max(img_rs)
                        if vmax > vmin:
                            img01 = (img_rs - vmin) / (vmax - vmin)
                        else:
                            img01 = np.zeros_like(img_rs, dtype=np.float32)

                        tensor = torch.from_numpy(img01).float().unsqueeze(0).unsqueeze(0).to(device)

                        img_size = H * W
                        total_area += img_size

                        # Glycogen segmentation
                        glyco_output = 1 - glycogen_segmentation(tensor)
                        glyco_thresholded_combined = glyco_output > float(threshold)
                        img_glyco = int(np.sum(glyco_thresholded_combined))
                        total_glyco_area += img_glyco

                        # Region segmentation (C,H,W)
                        region_output = region_segmentation(tensor)

                        # Apply mappings
                        region_colored_combined, new_labels_combined = apply_label_mapping(region_output, label_dict_combined)
                        _, new_labels_individuals = apply_label_mapping(region_output, original_label_dict_individuals)

                        # Save region / glyco outputs if requested
                        if folder_output and folder_output.strip() != "":
                            base_filename = os.path.splitext(imgpath)[0]
                            if save_region:
                                region_output_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_region.tif")
                                # save RGB uint8
                                region_colored_arr = np.array(region_colored_combined)
                                tifffile.imwrite(region_output_path, region_colored_arr, photometric='rgb')
                            if save_glyco:
                                glyco_output_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_glyco.tif")
                                tifffile.imwrite(glyco_output_path, glyco_thresholded_combined.astype(np.uint8))

                        # Canvases based on resized image
                        filtered_image_combined = np.zeros((H, W, 3), dtype=np.uint8)

                        # Combined labels accumulation
                        for i_lbl, label in enumerate(label_dict_combined.keys()):
                            label_area_combined = int(np.sum(new_labels_combined == i_lbl))
                            glycogen_area_combined = int(np.sum(np.logical_and(glyco_thresholded_combined, new_labels_combined == i_lbl)))
                            label_areas_combined[label] += label_area_combined
                            glycogen_areas_combined[label] += glycogen_area_combined
                            csv_rows_combined.append((subfolder_name, imgpath, label, label_area_combined, glycogen_area_combined))

                            contours_combined, _ = cv2.findContours(glyco_thresholded_combined.astype(np.uint8) * 255,
                                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            filtered_areas_combined = []

                            for cnt in contours_combined:
                                if len(cnt) >= 3:
                                    area = cv2.contourArea(cnt)
                                    perimeter = cv2.arcLength(cnt, True)
                                    if perimeter == 0:
                                        continue
                                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                                    if float(min_area) <= area <= float(max_area) and circularity >= float(circularity_threshold):
                                        # centroid
                                        M = cv2.moments(cnt)
                                        if M["m00"] == 0:
                                            continue
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        if 0 <= cy < H and 0 <= cx < W and new_labels_combined[cy, cx] == i_lbl:
                                            filtered_areas_combined.append(area)
                                            particle_data.append({
                                                "Subfolder": subfolder_name,
                                                "Image": imgpath,
                                                "Region": label,
                                                "Particle_Area": float(area),
                                                "X_Coordinate": int(cx),
                                                "Y_Coordinate": int(cy)
                                            })
                                            color_bgr = tuple(int(c * 255) for c in colors.to_rgb(label_dict_combined[label][0]))
                                            rad = max(1, int(np.sqrt(area / np.pi)))
                                            cv2.circle(filtered_image_combined, (cx, cy), rad, color_bgr[::-1], thickness=-1)

                            # Save the filtered areas image
                            if save_filtered and folder_output and folder_output.strip() != "":
                                filtered_image_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_filtered.tif")
                                tifffile.imwrite(filtered_image_path, filtered_image_combined, photometric='rgb')

                            mean_particle_area_by_label_combined = float(np.mean(filtered_areas_combined) if filtered_areas_combined else 0.0)
                            csv_rows_combined[-1] += (mean_particle_area_by_label_combined,)

                            # z-disc feret (combined)
                            filtered_image_with_feret_line_combined = np.zeros((H, W, 3), dtype=np.uint8)
                            z_disc_maximal_feret_sum_combined = 0

                            contours_lbl, _ = cv2.findContours((new_labels_combined == i_lbl).astype(np.uint8) * 255,
                                                               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            if label == "z-disc":
                                for cnt in contours_lbl:
                                    max_feret_diameter, max_feret_contour = calculate_maximal_feret([cnt])
                                    z_disc_maximal_feret_sum_combined += max_feret_diameter
                                    draw_maximal_feret(filtered_image_with_feret_line_combined, max_feret_contour)

                            if save_zdisc and folder_output and folder_output.strip() != "":
                                zdisc_image_path = os.path.join(folder_output, f"{subfolder_name}_{base_filename}_zdisc.tif")
                                tifffile.imwrite(zdisc_image_path, filtered_image_with_feret_line_combined, photometric='rgb')

                            # Add zdisc max feret to rows
                            if label == "z-disc":
                                csv_rows_combined[-1] += (z_disc_maximal_feret_sum_combined,)
                            else:
                                csv_rows_combined[-1] += (0.0,)

                        # Individuals
                        glyco_thresholded_individuals = glyco_output > float(threshold)
                        for i_lbl, label in enumerate(original_label_dict_individuals.keys()):
                            label_area_individual = int(np.sum(new_labels_individuals == i_lbl))
                            glycogen_area_individual = int(np.sum(np.logical_and(glyco_thresholded_individuals, new_labels_individuals == i_lbl)))
                            label_areas_individuals[label] += label_area_individual
                            glycogen_areas_individuals[label] += glycogen_area_individual
                            csv_rows_individuals.append((subfolder_name, imgpath, label, label_area_individual, glycogen_area_individual))

                            contours_individuals, _ = cv2.findContours(glyco_thresholded_individuals.astype(np.uint8) * 255,
                                                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            filtered_areas_individuals = []

                            for cnt in contours_individuals:
                                if len(cnt) >= 3:
                                    area = cv2.contourArea(cnt)
                                    perimeter = cv2.arcLength(cnt, True)
                                    if perimeter == 0:
                                        continue
                                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                                    if float(min_area) <= area <= float(max_area) and circularity >= float(circularity_threshold):
                                        M = cv2.moments(cnt)
                                        if M["m00"] == 0:
                                            continue
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        if 0 <= cy < H and 0 <= cx < W and new_labels_individuals[cy, cx] == i_lbl:
                                            filtered_areas_individuals.append(area)
                                            particle_data.append({
                                                "Subfolder": subfolder_name,
                                                "Image": imgpath,
                                                "Region": label,
                                                "Particle_Area": float(area),
                                                "X_Coordinate": int(cx),
                                                "Y_Coordinate": int(cy)
                                            })

                            mean_particle_area_by_label_individual = float(np.mean(filtered_areas_individuals) if filtered_areas_individuals else 0.0)
                            csv_rows_individuals[-1] += (mean_particle_area_by_label_individual,)

        # Save the CSV files
        if folder_output and folder_output.strip() != "":
            df_combined = pd.DataFrame(csv_rows_combined, columns=[
                "Subfolder", "Image", "Region", "Area", "Glycogen Area", "Mean Particle Area", "Z-disc max feret"
            ])
            df_combined.to_csv(os.path.join(folder_output, "glycogen_distribution_combined.csv"), index=False)

            df_individuals = pd.DataFrame(csv_rows_individuals, columns=[
                "Subfolder", "Image", "Region", "Area", "Glycogen Area", "Mean Particle Area"
            ])
            df_individuals.to_csv(os.path.join(folder_output, "glycogen_distribution_individuals.csv"), index=False)

            df2 = pd.DataFrame(particle_data)
            df2.to_csv(os.path.join(folder_output, "Particle size raw.csv"), index=False)

        return 100  # for the progress bar

    def open_tif(file):
        img = tifffile.imread(file.name)
        return img

    # Create the GUI
    with gr.Blocks() as gui:
        # Main layout
        with gr.Row():
            main = gr.Column(scale=3)
            with gr.Column():  # global settings
                threshold = gr.Slider(label="Threshold", minimum=0, maximum=1, step=0.01, value=0.3)
                label_mapping = gr.Textbox(label="Label dictionary", info="Python dictionary describing the included labels",
                                           value=default_label_mapping)
                min_area = gr.Slider(label="Single Particle min area (with 0.7698 nm/pixel, 133 pixels = 10 nm diameter)", minimum=0, maximum=500, step=1, value=133)
                max_area = gr.Slider(label="Single Particle max area (with 0.7698 nm/pixel, 2338 pixels = 42 nm diameter)", minimum=0, maximum=4000, step=1, value=2338)
                circularity_threshold = gr.Slider(label="Single Particle circularity threshold", minimum=0, maximum=1, step=0.01, value=0.80)
                save_region = gr.Checkbox(label="Save Region Image (Warning 10 images = 1 GB)", value=False)
                save_glyco = gr.Checkbox(label="Save Glycogen Image (Warning 2000 images = 1 GB)", value=False)
                save_filtered = gr.Checkbox(label="Save Filtered Image (Warning 80 images = 1 GB)", value=False)
                save_zdisc = gr.Checkbox(label="Save Z-disc Image (Warning 80 images = 1 GB)", value=False)

            with main:  # main column
                with gr.Tab("Batch processing"):
                    # Folder selection (as text input)
                    with gr.Row():
                        folder_input = gr.Textbox(label="Input Folder", info="Full path of directory containing images")
                        folder_output = gr.Textbox(label="Output Folder", info="Full path of directory to save results")

                    # NEW: scale & snap for batch
                    with gr.Row():
                        scale_b = gr.Number(value=1.0, precision=6, label="Scale (pixel multiplier) (batch)")
                        snap_multiple_b = gr.Dropdown(choices=[32, 64, 128], value=128, label="Snap to multiple (batch)")
                        gr.Markdown("Tip: set scale = input_nm_per_px / target_nm_per_px. E.g., x33000 (CFIM): 1.53 / 0.7698 ≈ 2")

                    with gr.Row():
                        run_batch_button = gr.Button("Run")
                        progress_bar = gr.Slider(0, 100, label="Progress", interactive=False)

                    run_batch_button.click(
                        run_batch,
                        inputs=[folder_input, folder_output, label_mapping, threshold, min_area, max_area,
                                circularity_threshold, save_region, save_glyco, save_filtered, save_zdisc,
                                scale_b, snap_multiple_b],
                        outputs=[progress_bar]
                    )

                with gr.Tab("Single image"):
                    with gr.Row():
                        tif_input = gr.File(label="Input Image", file_types=[".tif", ".tiff", ".TIF", ".TIFF"])
                        tif_prewiew = gr.Image(image_mode="F", height=1024, width=1024)

                    # NEW: scale & snap for single
                    with gr.Row():
                        scale_s = gr.Number(value=1.0, precision=6, label="Scale (pixel multiplier)")
                        snap_multiple_s = gr.Dropdown(choices=[32, 64, 128], value=128, label="Snap to multiple")
                        gr.Markdown("Tip: set scale = input_nm_per_px / target_nm_per_px. E.g., x33000 (CFIM): 1.53 / 0.7698 ≈ 2")


                    with gr.Row():
                        glyco_output = gr.Image(interactive=False, height=256, width=256)
                        region_output = gr.Image(interactive=False, height=256, width=256)
                        filtered_output = gr.Image(interactive=False, height=256, width=256)
                        feret_output = gr.Image(interactive=False, height=256, width=256)

                    stats = gr.DataFrame(headers=["Region", "Area", "Glycogen Area", "Particle area", "z-disc maximal feret sum"])

                    run_button = gr.Button("Run")
                    tif_input.upload(open_tif, inputs=tif_input, outputs=tif_prewiew)

                    run_button.click(
                        run_segmentation,
                        inputs=[tif_prewiew, label_mapping, threshold, min_area, max_area, circularity_threshold,
                                scale_s, snap_multiple_s],
                        outputs=[glyco_output, region_output, filtered_output, feret_output, stats]
                    )

    gui.launch(inbrowser=True)

if __name__ == '__main__':
    main()
