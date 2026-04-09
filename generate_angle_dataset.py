"""
=============================================================================
 Angle Classification Dataset Generator
=============================================================================
 Purpose: Simulate camera tilt angles on handwritten Hindi paragraph images
          and generate a labeled dataset for angle classification.

 Pipeline:
   1. Load original images from dataSet2/
   2. Apply perspective transformation to simulate camera tilt (-70° to +70°)
   3. Apply data augmentation (brightness, blur, noise, shadows)
   4. Organize output into angle-range folders
   5. Generate 1000+ augmented images

 Dependencies: OpenCV, NumPy
=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import io
import random
import shutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = "dataSet2"                    # Source images folder
OUTPUT_DIR = "OCR_dataset"                # Output root folder
ANGLE_START = -70                         # Start angle (degrees)
ANGLE_END = 70                            # End angle (degrees)
ANGLE_STEP = 5                            # Angle interval
MIN_TOTAL_IMAGES = 1000                   # Minimum images to generate
AUGMENTATION_ROUNDS = 3                   # Extra augmented copies per angle image
RANDOM_SEED = 42                          # For reproducibility

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =============================================================================
# STEP 1: Perspective Transformation (Camera Tilt Simulation)
# =============================================================================
def simulate_camera_tilt(image, angle_degrees):
    """
    Simulate a camera tilt by applying a perspective (homography) transformation.
    
    How it works:
    - A positive angle tilts the camera upward (top of image appears farther away)
    - A negative angle tilts the camera downward (bottom appears farther away)
    - We compute a perspective warp that mimics a real camera viewing the page
      at the specified tilt angle.
    
    The math:
    - We define 4 source corners of the image
    - We compute 4 destination corners by shrinking the top or bottom edge
      based on the tilt angle (simulating foreshortening due to perspective)
    
    Args:
        image: Input image (numpy array)
        angle_degrees: Tilt angle in degrees (-70 to +70)
    
    Returns:
        Warped image simulating the camera tilt
    """
    h, w = image.shape[:2]
    
    # Source corners: full image rectangle
    src_pts = np.float32([
        [0, 0],          # Top-left
        [w, 0],          # Top-right
        [w, h],          # Bottom-right
        [0, h]           # Bottom-left
    ])
    
    # Calculate foreshortening factor based on angle
    # At 0°, no foreshortening. At 70°, significant foreshortening.
    # We use cos(angle) to simulate depth compression.
    angle_rad = np.radians(abs(angle_degrees))
    
    # Foreshortening: the farther edge appears narrower
    # Scale factor ranges from 1.0 (0°) to ~0.34 (70°)
    scale = np.cos(angle_rad)
    
    # Amount to shrink the far edge (in pixels)
    shrink = int(w * (1 - scale) / 2)
    
    # Also simulate vertical compression on far side
    v_shrink = int(h * (1 - scale) * 0.3)
    
    if angle_degrees > 0:
        # Positive angle: camera tilts up → top edge appears farther (narrower)
        dst_pts = np.float32([
            [shrink, v_shrink],          # Top-left moves inward
            [w - shrink, v_shrink],      # Top-right moves inward
            [w, h],                      # Bottom-right stays
            [0, h]                       # Bottom-left stays
        ])
    elif angle_degrees < 0:
        # Negative angle: camera tilts down → bottom edge appears farther (narrower)
        dst_pts = np.float32([
            [0, 0],                          # Top-left stays
            [w, 0],                          # Top-right stays
            [w - shrink, h - v_shrink],      # Bottom-right moves inward
            [shrink, h - v_shrink]           # Bottom-left moves inward
        ])
    else:
        # 0° — no transformation needed
        return image.copy()
    
    # Compute the perspective transformation matrix (3x3 homography)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply the warp with white background fill
    warped = cv2.warpPerspective(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # White background
    )
    
    return warped


# =============================================================================
# STEP 2: Data Augmentation Functions
# =============================================================================

def augment_brightness(image, factor_range=(0.6, 1.4)):
    """
    Randomly adjust image brightness.
    
    How: Convert to HSV color space, multiply the V (value/brightness) channel
    by a random factor, then convert back to BGR.
    
    Args:
        image: Input BGR image
        factor_range: (min, max) brightness multiplier
    
    Returns:
        Brightness-adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(*factor_range)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_gaussian_blur(image, kernel_range=(3, 7)):
    """
    Apply random Gaussian blur to simulate camera focus issues.
    
    How: Use a random odd-sized kernel for Gaussian blur.
    
    Args:
        image: Input image
        kernel_range: (min, max) kernel size (will be made odd)
    
    Returns:
        Blurred image
    """
    k = random.randint(*kernel_range)
    if k % 2 == 0:
        k += 1  # Kernel size must be odd
    return cv2.GaussianBlur(image, (k, k), 0)


def augment_gaussian_noise(image, mean=0, std_range=(5, 25)):
    """
    Add random Gaussian noise to simulate sensor noise.
    
    How: Generate a noise matrix with the same shape as the image,
    then add it to the original image.
    
    Args:
        image: Input image
        mean: Noise mean (usually 0)
        std_range: (min, max) standard deviation of noise
    
    Returns:
        Noisy image
    """
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def augment_shadow(image):
    """
    Simulate a shadow falling across the image.
    
    How: Create a random polygon mask, darken pixels inside it by
    multiplying with a factor < 1. This simulates a hand or object
    casting a shadow on the page.
    
    Args:
        image: Input image
    
    Returns:
        Image with simulated shadow
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Generate random shadow polygon (3-5 vertices)
    num_vertices = random.randint(3, 5)
    vertices = []
    for _ in range(num_vertices):
        x = random.randint(0, w)
        y = random.randint(0, h)
        vertices.append([x, y])
    
    # Create shadow mask
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    
    # Darken shadow region (multiply by 0.4-0.7)
    shadow_factor = random.uniform(0.4, 0.7)
    result[mask == 255] *= shadow_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_random_augmentation(image):
    """
    Apply a random combination of augmentations to an image.
    
    Randomly selects 1-3 augmentations from the available pool
    and applies them sequentially.
    
    Args:
        image: Input image
    
    Returns:
        Augmented image
    """
    augmentations = [
        ("brightness", augment_brightness),
        ("blur", augment_gaussian_blur),
        ("noise", augment_gaussian_noise),
        ("shadow", augment_shadow),
    ]
    
    # Pick 1-3 random augmentations
    num_augs = random.randint(1, 3)
    selected = random.sample(augmentations, num_augs)
    
    result = image.copy()
    for name, func in selected:
        result = func(result)
    
    return result


# =============================================================================
# STEP 3: Angle Range Classification
# =============================================================================

def get_angle_folder(angle_degrees):
    """
    Determine which angle-range folder an image belongs to.
    
    Categorizes angles into 10-degree buckets:
      0-10°, 10-20°, 20-30°, 30-40°, 40-50°, 50-60°, 60-70°
    
    We use absolute angle since + and - produce similar visual distortion.
    
    Args:
        angle_degrees: The tilt angle (-70 to +70)
    
    Returns:
        Folder name string, e.g., "20_30_degree"
    """
    abs_angle = abs(angle_degrees)
    
    if abs_angle == 0:
        return "0_10_degree"
    
    # Find the 10-degree bucket
    lower = (abs_angle // 10) * 10
    upper = lower + 10
    
    return f"{int(lower)}_{int(upper)}_degree"


# =============================================================================
# STEP 4: Output Directory Setup
# =============================================================================

def setup_output_dirs(output_dir):
    """
    Create the output directory structure:
    
    OCR_dataset/
    ├── original/          ← Copy of source images (unmodified)
    ├── 0_10_degree/       ← Images with 0-10° tilt
    ├── 10_20_degree/      ← Images with 10-20° tilt
    ├── 20_30_degree/      ← Images with 20-30° tilt
    ├── 30_40_degree/      ← Images with 30-40° tilt
    ├── 40_50_degree/      ← Images with 40-50° tilt
    ├── 50_60_degree/      ← Images with 50-60° tilt
    └── 60_70_degree/      ← Images with 60-70° tilt
    
    Args:
        output_dir: Root output directory path
    """
    # Remove existing output to start fresh
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create original folder
    os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
    
    # Create angle-range folders
    for lower in range(0, 70, 10):
        upper = lower + 10
        folder = f"{lower}_{upper}_degree"
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    
    print(f"  Created output structure in: {output_dir}/")


# =============================================================================
# STEP 5: Main Pipeline
# =============================================================================

def generate_dataset(input_dir, output_dir, angle_start, angle_end, angle_step,
                     augmentation_rounds, min_total_images):
    """
    Main pipeline that orchestrates the entire dataset generation:
    
    1. Load all source images
    2. Copy originals to output/original/
    3. For each image × each angle:
       a. Apply perspective transformation
       b. Save the base transformed image
       c. Generate augmented copies
    4. Print summary statistics
    
    Args:
        input_dir: Path to source images
        output_dir: Root output directory
        angle_start: First angle (e.g., -70)
        angle_end: Last angle (e.g., 70)
        angle_step: Angle increment (e.g., 5)
        augmentation_rounds: Number of augmented versions per angle image
        min_total_images: Target minimum dataset size
    """
    
    # --- Load source images ---
    print("\n" + "=" * 60)
    print("  ANGLE CLASSIFICATION DATASET GENERATOR")
    print("=" * 60)
    
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))],
        key=lambda x: int(x.split('.')[0])
    )
    
    if not image_files:
        print(f"ERROR: No images found in {input_dir}/")
        return
    
    print(f"\n  Source images: {len(image_files)} files in {input_dir}/")
    
    # --- Generate angle list ---
    angles = list(range(angle_start, angle_end + 1, angle_step))
    print(f"  Angles: {angle_start}° to {angle_end}° (step {angle_step}°) → {len(angles)} angles")
    
    # Calculate expected output
    base_images = len(image_files) * len(angles)
    aug_images = base_images * augmentation_rounds
    total_expected = base_images + aug_images + len(image_files)  # + originals
    print(f"  Expected output: {base_images} base + {aug_images} augmented + {len(image_files)} originals = {total_expected} images")
    
    # If not enough, increase augmentation rounds
    if total_expected < min_total_images:
        augmentation_rounds = max(augmentation_rounds,
                                  (min_total_images - base_images) // base_images + 1)
        aug_images = base_images * augmentation_rounds
        total_expected = base_images + aug_images + len(image_files)
        print(f"  Adjusted augmentation rounds to {augmentation_rounds} → {total_expected} images")
    
    # --- Setup output directories ---
    print("\n  Setting up output directories...")
    setup_output_dirs(output_dir)
    
    # --- Process each image ---
    total_generated = 0
    folder_counts = {}
    
    for img_idx, fname in enumerate(image_files):
        img_path = os.path.join(input_dir, fname)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  WARNING: Could not read {fname}, skipping.")
            continue
        
        base_name = fname.split('.')[0]  # e.g., "1" from "1.jpeg"
        
        # Copy original to original/ folder
        orig_dst = os.path.join(output_dir, "original", fname)
        cv2.imwrite(orig_dst, image)
        total_generated += 1
        
        # Process each angle
        for angle in angles:
            # STEP A: Apply perspective transformation
            warped = simulate_camera_tilt(image, angle)
            
            # STEP B: Determine output folder based on angle range
            folder = get_angle_folder(angle)
            folder_path = os.path.join(output_dir, folder)
            
            # STEP C: Save base transformed image
            # Naming: page{N}_angle_{angle}.jpg
            sign = "neg" if angle < 0 else ""
            out_name = f"page{base_name}_angle_{angle}.jpg"
            out_path = os.path.join(folder_path, out_name)
            cv2.imwrite(out_path, warped)
            total_generated += 1
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            # STEP D: Generate augmented copies
            for aug_round in range(augmentation_rounds):
                augmented = apply_random_augmentation(warped)
                aug_name = f"page{base_name}_angle_{angle}_aug{aug_round + 1}.jpg"
                aug_path = os.path.join(folder_path, aug_name)
                cv2.imwrite(aug_path, augmented)
                total_generated += 1
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
        
        # Progress update
        print(f"  [{img_idx + 1}/{len(image_files)}] Processed {fname} → {len(angles)} angles × {augmentation_rounds + 1} versions")
    
    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("  DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\n  Total images generated: {total_generated}")
    print(f"  Output directory: {output_dir}/")
    print(f"\n  {'Folder':<25} {'Images':>8}")
    print(f"  {'-' * 35}")
    print(f"  {'original':<25} {len(image_files):>8}")
    for folder in sorted(folder_counts.keys()):
        print(f"  {folder:<25} {folder_counts[folder]:>8}")
    print(f"  {'-' * 35}")
    print(f"  {'TOTAL':<25} {total_generated:>8}")
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    generate_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        angle_start=ANGLE_START,
        angle_end=ANGLE_END,
        angle_step=ANGLE_STEP,
        augmentation_rounds=AUGMENTATION_ROUNDS,
        min_total_images=MIN_TOTAL_IMAGES,
    )
