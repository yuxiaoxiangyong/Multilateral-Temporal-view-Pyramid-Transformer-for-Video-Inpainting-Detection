import numpy as np
import os
import glob
from PIL import Image
import cv2
import argparse


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate segmentation results.")
    parser.add_argument('--input', '-i', required=True,
                        help='Directory of input images.')
    parser.add_argument('--mask_dir', '-d', required=True,
                        help='Directory of ground truth masks.')
    parser.add_argument('--im_dir', '-l', required=False,
                        help='Directory of original images.')
    return parser.parse_args()


def load_image(filepath, target_size=None, normalize=False):
    """
    Load an image, resize it, and optionally normalize pixel values.

    Args:
        filepath (str): Path to the image.
        target_size (tuple): Desired output size (width, height).
        normalize (bool): Whether to normalize pixel values to [0, 1].

    Returns:
        np.ndarray: Processed image array.
    """
    try:
        img = Image.open(filepath).convert('L')  # Convert to grayscale
        if target_size:
            img = img.resize(target_size, Image.BILINEAR)
        img_array = np.asarray(img)
        if normalize:
            img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None


def iou_score(output, target):
    """
    Compute Intersection over Union (IoU) score.

    Args:
        output (np.ndarray): Predicted mask.
        target (np.ndarray): Ground truth mask.

    Returns:
        float: IoU score.
    """
    smooth = 1e-5
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    return (intersection + smooth) / (union + smooth)


def evaluate_image(input_path, mask_path, crop_size):
    """
    Evaluate a single image against its ground truth.

    Args:
        input_path (str): Path to the predicted mask.
        mask_path (str): Path to the ground truth mask.
        crop_size (tuple): Size to which images will be resized.

    Returns:
        tuple: F1 score, IoU score.
    """
    gt_mask = load_image(mask_path, target_size=crop_size) > 0
    if gt_mask is None:
        return None, None

    result = load_image(input_path, target_size=crop_size, normalize=True)
    if result is None:
        return None, None

    result_binary = result > 0.5
    recall = np.sum(gt_mask & result_binary) / np.sum(gt_mask + 1e-6)
    precision = np.sum(gt_mask & result_binary) / (np.sum(result_binary) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    iou = iou_score(result, gt_mask)

    return f1, iou


def main():
    args = get_args()
    crop_size = (224, 224)
    input_images = glob.glob(os.path.join(args.input, '*/*.png')) or glob.glob(os.path.join(args.input, '*/*.jpg'))
    if not input_images:
        print("No input images found.")
        return

    f1_scores = []
    iou_scores = []

    for i, img_path in enumerate(input_images):
        print(f"\nProcessing image {i + 1}/{len(input_images)}: {img_path}")
        file_name = "{:05d}".format(int(os.path.splitext(img_path.split('/')[-1])[0].split('_')[0]))
        mask_file = os.path.join(args.mask_dir, img_path.split('/')[-2], file_name + '.png')

        if not os.path.exists(mask_file):
            print(f"Ground truth mask not found: {mask_file}")
            continue

        input_path = img_path

        if not os.path.exists(input_path):
            print(f"Predicted mask not found: {input_path}")
            continue

        f1, iou = evaluate_image(input_path, mask_file, crop_size)
        if f1 is not None and iou is not None and f1 <= 1 and iou <= 1:
            f1_scores.append(f1)
            iou_scores.append(iou)
            print(f"IoU: {iou:.4f}, F1: {f1:.4f}")
        else:
            print(f"Invalid scores for {img_path}: IoU={iou}, F1={f1}")

    print("\nEvaluation Complete")
    print(f"Average F1: {np.mean(f1_scores):.4f}")
    print(f"Average IoU: {np.mean(iou_scores):.4f}")


if __name__ == "__main__":
    main()
