import os
import argparse
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from matcher.GFSAM import build_model  # Ensure correct import path

def process_image(image, desired_size, interpolation=Image.BILINEAR):
    """
    Resizes the input image so that the shorter side equals desired_size,
    then center-crops the resized image to a square of (desired_size, desired_size).

    Args:
        image (PIL.Image): The input image.
        desired_size (int): The target size for the shorter side (and for the crop).
        interpolation: Interpolation method for resizing (e.g. Image.BILINEAR or Image.NEAREST).

    Returns:
        cropped_img (PIL.Image): The center-cropped image.
        resize_info (tuple): A tuple (new_w, new_h, left, top) where (new_w, new_h)
                             are the dimensions after resizing and (left, top) is the
                             upper-left corner of the crop in the resized image.
    """
    w, h = image.size
    scale = desired_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized_img = image.resize((new_w, new_h), interpolation)
    # Calculate the coordinates for a center crop of size (desired_size, desired_size)
    left = (new_w - desired_size) // 2
    top = (new_h - desired_size) // 2
    right = left + desired_size
    bottom = top + desired_size
    cropped_img = resized_img.crop((left, top, right, bottom))
    return cropped_img, (new_w, new_h, left, top)

def main():
    parser = argparse.ArgumentParser(description='Inference script for GFSAM with multiple references')
    parser.add_argument('--ref_dir', type=str, required=True, help='Directory containing reference images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing reference masks')
    parser.add_argument('--target_img', type=str, required=True, help='Path to target image')
    parser.add_argument('--output_path', type=str, default='output.png', help='Path to save the output mask')
    parser.add_argument('--img_size', type=int, default=1024, help='Desired resolution for processing')
    parser.add_argument('--dinov2_size', type=str, default='vit_large', choices=['vit_small', 'vit_base', 'vit_large'])
    parser.add_argument('--sam_size', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'])
    parser.add_argument('--dinov2_weights', type=str, default='models/dinov2_vitl14_pretrain.pth')
    parser.add_argument('--sam_weights', type=str, default='models/sam_vit_h_4b8939.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Build model
    GFSAM = build_model(args)

    # Process reference images and masks
    ref_images = []
    ref_masks = []

    # Pair images and masks based on filenames (assumes image: 'img.jpg', mask: 'img_mask.png')
    for img_name in os.listdir(args.ref_dir):
        base_name = os.path.splitext(img_name)[0]
        mask_name = f"{base_name}_mask.png"
        img_path = os.path.join(args.ref_dir, img_name)
        mask_path = os.path.join(args.mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image {img_name}: expected {mask_name}")
        
        # Process the reference image with bilinear interpolation
        img = Image.open(img_path).convert('RGB')
        img_processed, _ = process_image(img, args.img_size, interpolation=Image.BILINEAR)
        img_tensor = transforms.ToTensor()(img_processed)  # [3, H, W]
        ref_images.append(img_tensor)
        
        # Process the reference mask using nearest neighbor interpolation
        mask = Image.open(mask_path).convert('L')
        mask_processed, _ = process_image(mask, args.img_size, interpolation=Image.NEAREST)
        mask_np = np.array(mask_processed)
        mask_tensor = torch.tensor(mask_np).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()  # Binarize mask
        ref_masks.append(mask_tensor)

    # Stack reference images and masks into batch tensors of shape [1, N, ...]
    ref_imgs = torch.stack(ref_images, dim=0).unsqueeze(0).to(device)
    ref_masks = torch.stack(ref_masks, dim=0).unsqueeze(0).to(device)

    # Load and process target image
    target_img = Image.open(args.target_img).convert('RGB')
    original_size = target_img.size  # Save original (width, height)
    target_processed, target_resize_info = process_image(target_img, args.img_size, interpolation=Image.BILINEAR)
    target_tensor = transforms.ToTensor()(target_processed).unsqueeze(0).to(device)  # [1, 3, H, W]

    # Run inference
    with torch.no_grad():
        GFSAM.clear()
        GFSAM.set_reference(ref_imgs, ref_masks)
        GFSAM.set_target(target_tensor)
        pred_mask, _ = GFSAM.predict()

    # Post-process the predicted mask:
    # 1. The model produces a mask of size (args.img_size, args.img_size).
    # 2. We pad this mask back into the full resized target dimensions.
    # 3. Finally, we resize it to the original target resolution.
    pred_mask = pred_mask.squeeze().cpu().numpy()  # shape: (args.img_size, args.img_size)
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Binarize and scale to 0-255

    # Unpack the resizing information for the target image: (new_w, new_h, left, top)
    new_w, new_h, left, top = target_resize_info

    # Create a blank canvas with the resized dimensions
    padded_mask = Image.new('L', (new_w, new_h), 0)
    pred_mask_img = Image.fromarray(pred_mask)
    padded_mask.paste(pred_mask_img, (left, top))  # Place the predicted mask at the correct location

    # Resize the padded mask back to the original target image size
    final_mask = padded_mask.resize(original_size, Image.NEAREST)

    # Save the final mask
    final_mask.save(args.output_path)
    print(f"Output saved to {args.output_path}")

if __name__ == '__main__':
    main()
