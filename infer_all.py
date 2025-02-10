import os
import time
import argparse
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import your model-building function
from matcher.GFSAM import build_model  # Ensure that the import path is correct


def process_image(image, desired_size, interpolation=Image.BILINEAR):
    """
    Resizes the input image so that its shorter side equals desired_size,
    then center-crops the resized image to a square of (desired_size, desired_size).

    Returns:
        cropped_img (PIL.Image): The center-cropped image.
        resize_info (tuple): (new_w, new_h, left, top) where (new_w, new_h)
                             are the dimensions after resizing and (left, top) is the
                             upper-left corner of the crop in the resized image.
    """
    w, h = image.size
    scale = desired_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized_img = image.resize((new_w, new_h), interpolation)
    # Compute center crop coordinates
    left = (new_w - desired_size) // 2
    top = (new_h - desired_size) // 2
    right = left + desired_size
    bottom = top + desired_size
    cropped_img = resized_img.crop((left, top, right, bottom))
    return cropped_img, (new_w, new_h, left, top)


def run_inference_for_folder(data_folder, model, args):
    """
    Given a data folder with the following structure:
       ├── masks/
       ├── refs/
       ├── target.jpg
    This function:
      - Updates the folder-specific paths in args,
      - Processes the reference images and their masks,
      - Processes the target image,
      - Runs the inference pipeline, and
      - Saves the output mask in the same folder.
      
    Returns:
      The number of reference images processed.
    """
    # Set folder-specific paths
    args.ref_dir = os.path.join(data_folder, "refs")
    args.mask_dir = os.path.join(data_folder, "masks")
    args.target_img = os.path.join(data_folder, "target.jpg")
    args.output_path = os.path.join(data_folder, "output.png")

    ref_images = []
    ref_masks = []

    # Process each reference image (assumed to be .jpg) in the refs folder.
    ref_files = sorted(os.listdir(args.ref_dir))
    for img_name in ref_files:
        if not img_name.lower().endswith('.jpg'):
            continue
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        img_path = os.path.join(args.ref_dir, img_name)
        mask_path = os.path.join(args.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image {img_name}: expected {mask_name}")

        # Process reference image (using bilinear interpolation)
        img = Image.open(img_path).convert('RGB')
        img_processed, _ = process_image(img, args.img_size, interpolation=Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img_processed)
        ref_images.append(img_tensor)

        # Process corresponding mask (using nearest neighbor to preserve binary values)
        mask = Image.open(mask_path).convert('L')
        mask_processed, _ = process_image(mask, args.img_size, interpolation=Image.NEAREST)
        mask_np = np.array(mask_processed)
        mask_tensor = torch.tensor(mask_np).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()  # binarize
        ref_masks.append(mask_tensor)

    # Create batched tensors for reference images and masks
    ref_imgs = torch.stack(ref_images, dim=0).unsqueeze(0).to(args.device)  # shape: [1, N, 3, H, W]
    ref_masks = torch.stack(ref_masks, dim=0).unsqueeze(0).to(args.device)   # shape: [1, N, H, W]

    # Process target image
    target_img = Image.open(args.target_img).convert('RGB')
    original_size = target_img.size  # (width, height)
    target_processed, target_resize_info = process_image(target_img, args.img_size, interpolation=Image.BILINEAR)
    target_tensor = transforms.ToTensor()(target_processed).unsqueeze(0).to(args.device)

    # Run inference using your model
    with torch.no_grad():
        model.clear()
        model.set_reference(ref_imgs, ref_masks)
        model.set_target(target_tensor)
        pred_mask, _ = model.predict()

    # Post-process the predicted mask
    pred_mask = pred_mask.squeeze().cpu().numpy()  # shape: (args.img_size, args.img_size)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    # Undo the cropping by padding the predicted mask back into the resized target image dimensions
    new_w, new_h, left, top = target_resize_info
    padded_mask = Image.new('L', (new_w, new_h), 0)
    pred_mask_img = Image.fromarray(pred_mask)
    padded_mask.paste(pred_mask_img, (left, top))

    # Finally, resize the padded mask back to the original target image resolution
    final_mask = padded_mask.resize(original_size, Image.NEAREST)
    final_mask.save(args.output_path)

    return len(ref_images)


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference over multiple data folders and plot inference time vs. number of reference images"
    )
    parser.add_argument(
        '--samples_dir',
        type=str,
        default='samples',
        help="Directory containing all data subfolders (each with refs/, masks/, target.jpg, etc.)"
    )
    parser.add_argument('--img_size', type=int, default=1024, help="Desired resolution for processing")
    parser.add_argument('--dinov2_size', type=str, default='vit_large', choices=['vit_small', 'vit_base', 'vit_large'])
    parser.add_argument('--sam_size', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--dinov2_weights', type=str, default='models/dinov2_vitl14_pretrain.pth')
    parser.add_argument('--sam_weights', type=str, default='models/sam_vit_h_4b8939.pth')
    args = parser.parse_args()

    # Set device and build model once (to reuse across data folders)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Using device: {device}")

    model = build_model(args)

    # Get a list of subdirectories inside samples_dir
    data_folders = [
        os.path.join(args.samples_dir, d)
        for d in os.listdir(args.samples_dir)
        if os.path.isdir(os.path.join(args.samples_dir, d))
    ]

    inference_times = []  # list to hold inference time per folder
    ref_counts = []       # list to hold number of reference images per folder
    folder_names = []     # folder names (for annotating the plot)

    # Process each data folder
    for folder in data_folders:
        target_path = os.path.join(folder, "target.jpg")
        if not os.path.exists(target_path):
            print(f"Skipping folder {folder}: target.jpg not found.")
            continue

        print(f"Processing folder: {folder}")
        start_time = time.time()
        try:
            num_refs = run_inference_for_folder(folder, model, args)
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue
        elapsed = time.time() - start_time
        print(f"Inference on {folder} took {elapsed:.2f} seconds with {num_refs} reference image(s).")

        inference_times.append(elapsed)
        ref_counts.append(num_refs)
        folder_names.append(os.path.basename(folder))

    # Generate scatter plot: x-axis = number of reference images, y-axis = inference time (s)
    plt.figure(figsize=(8, 6))
    plt.scatter(ref_counts, inference_times, color='blue')
    for i, folder in enumerate(folder_names):
        plt.annotate(folder, (ref_counts[i], inference_times[i]), textcoords="offset points", xytext=(5, 5), ha='center')
    plt.xlabel("Number of Reference Images")
    plt.ylabel("Inference Time (seconds)")
    plt.title("Inference Time vs. Number of Reference Images")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("inference_time_scatter.png")
    plt.show()


if __name__ == '__main__':
    main()
