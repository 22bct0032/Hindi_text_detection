"""
Unified Detection Script — run_detection.py
Runs final_improved.py's word detection on any dataset folder.
Supports flat directories (dataSet, dataSet2) and nested directories (OCR_dataset).
Automatically applies inverse perspective correction for angle-tilted images.

Usage:
    python run_detection.py <dataset_folder>
    python run_detection.py dataSet
    python run_detection.py dataSet2
    python run_detection.py OCR_dataset
"""
import os
import sys
import io
import json
import time
import re
import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import process_image from final_improved.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_improved import process_image


# ─────────────────────────────────────────────────────────────
# Inverse Perspective Correction (for angle-tilted images)
# ─────────────────────────────────────────────────────────────
def inverse_perspective_correction(image, angle_degrees):
    """Undo perspective tilt to flatten image back to top-down view."""
    if angle_degrees == 0:
        return image.copy()

    h, w = image.shape[:2]
    angle_rad = np.radians(abs(angle_degrees))
    scale = np.cos(angle_rad)
    shrink = int(w * (1 - scale) / 2)
    v_shrink = int(h * (1 - scale) * 0.3)

    src_full = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    if angle_degrees > 0:
        dst_tilted = np.float32([
            [shrink, v_shrink], [w - shrink, v_shrink],
            [w, h], [0, h]
        ])
    else:
        dst_tilted = np.float32([
            [0, 0], [w, 0],
            [w - shrink, h - v_shrink], [shrink, h - v_shrink]
        ])

    matrix = cv2.getPerspectiveTransform(dst_tilted, src_full)
    corrected = cv2.warpPerspective(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return corrected


def extract_angle_from_filename(filename):
    """Extract tilt angle from filenames like 'page3_angle_-15.jpg'."""
    m = re.match(r'page\d+_angle_([-]?\d+)', filename)
    return int(m.group(1)) if m else 0


# ─────────────────────────────────────────────────────────────
# Collect all images from a dataset folder
# ─────────────────────────────────────────────────────────────
def collect_images(dataset_path):
    """
    Collect all images from a dataset folder.
    Supports flat directories and nested subdirectories.
    Returns list of (image_path, relative_output_dir, angle)
    """
    image_exts = ('.jpg', '.jpeg', '.png')
    images = []

    # Check if dataset has subdirectories with images
    subdirs = [d for d in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, d))]

    has_nested = False
    for sd in subdirs:
        sd_path = os.path.join(dataset_path, sd)
        if any(f.lower().endswith(image_exts) for f in os.listdir(sd_path)):
            has_nested = True
            break

    if has_nested:
        # Nested directory structure (e.g., OCR_dataset)
        for sd in sorted(subdirs):
            sd_path = os.path.join(dataset_path, sd)
            for f in sorted(os.listdir(sd_path)):
                if f.lower().endswith(image_exts):
                    angle = extract_angle_from_filename(f)
                    images.append((os.path.join(sd_path, f), sd, angle))
    else:
        # Flat directory structure (e.g., dataSet, dataSet2)
        for f in sorted(os.listdir(dataset_path), key=lambda x: (len(x), x)):
            if f.lower().endswith(image_exts):
                images.append((os.path.join(dataset_path, f), "", 0))

    return images


# ─────────────────────────────────────────────────────────────
# Process a single image with optional angle correction
# ─────────────────────────────────────────────────────────────
def detect_image(img_path, angle, output_dir):
    """Run detection on one image. Apply inverse correction if angle >= 30."""
    fname = os.path.basename(img_path)
    name_no_ext = os.path.splitext(fname)[0]
    # Ensure output extension is .jpg
    out_fname = name_no_ext + ".jpg"

    needs_correction = abs(angle) >= 30

    if needs_correction:
        img = cv2.imread(img_path)
        if img is None:
            return None, None, f"Cannot read {img_path}"

        corrected = inverse_perspective_correction(img, angle)

        # Save corrected to temp, run detection, clean up
        temp_path = os.path.join(output_dir, f"_temp_{fname}")
        cv2.imwrite(temp_path, corrected)
        finalimg, boxes, error = process_image(temp_path, skip_deskew=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if error:
            return None, None, error

        # Draw boxes on original tilted image
        original_img = img.copy()
        h, w = img.shape[:2]
        angle_rad = np.radians(abs(angle))
        scale = np.cos(angle_rad)
        shrink_val = int(w * (1 - scale) / 2)
        v_shrink_val = int(h * (1 - scale) * 0.3)

        src_full = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if angle > 0:
            dst_tilted = np.float32([
                [shrink_val, v_shrink_val], [w - shrink_val, v_shrink_val],
                [w, h], [0, h]
            ])
        else:
            dst_tilted = np.float32([
                [0, 0], [w, 0],
                [w - shrink_val, h - v_shrink_val], [shrink_val, h - v_shrink_val]
            ])

        fwd_matrix = cv2.getPerspectiveTransform(src_full, dst_tilted)
        for box in boxes:
            bx, by, bw, bh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            corners = np.float32([
                [bx, by], [bx + bw, by],
                [bx + bw, by + bh], [bx, by + bh]
            ]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, fwd_matrix)
            pts = transformed.reshape(-1, 2).astype(int)
            for k in range(4):
                cv2.line(original_img, tuple(pts[k]), tuple(pts[(k + 1) % 4]), (0, 0, 0), 1)

        cv2.imwrite(os.path.join(output_dir, out_fname), original_img)

    else:
        # Always skip deskew — it incorrectly rotates images causing twisted boxes
        finalimg, boxes, error = process_image(img_path, skip_deskew=True)
        if error:
            return None, None, error
        cv2.imwrite(os.path.join(output_dir, out_fname), finalimg)

    # Save JSON
    json_path = os.path.join(output_dir, f"{name_no_ext}.json")
    json_data = {
        "source_image": img_path,
        "angle": angle,
        "total_boxes": len(boxes),
        "detections": [
            {"box_id": i + 1, "x": int(b[0]), "y": int(b[1]),
             "width": int(b[2]), "height": int(b[3])}
            for i, b in enumerate(boxes)
        ]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    return boxes, len(boxes), None


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python run_detection.py <dataset_folder>")
        print("  e.g. python run_detection.py dataSet")
        print("       python run_detection.py dataSet2")
        print("       python run_detection.py OCR_dataset")
        sys.exit(1)

    dataset = sys.argv[1]
    if not os.path.isdir(dataset):
        print(f"Error: '{dataset}' is not a directory")
        sys.exit(1)

    dataset_name = os.path.basename(os.path.normpath(dataset))
    output_root = os.path.join("outputs", f"final_improved_{dataset_name}")

    print("=" * 60)
    print(f"  Word Detection — {dataset_name}")
    print(f"  Output: {output_root}/")
    print("=" * 60)

    images = collect_images(dataset)
    total = len(images)
    print(f"  Found {total} images\n")

    start_time = time.time()
    processed = 0
    errors = 0
    current_subdir = None

    for idx, (img_path, subdir, angle) in enumerate(images):
        if subdir != current_subdir:
            current_subdir = subdir
            label = subdir if subdir else "root"
            sub_count = sum(1 for _, s, _ in images if s == subdir)
            print(f"\n[{label}] ({sub_count} images)")

        output_dir = os.path.join(output_root, subdir) if subdir else output_root
        os.makedirs(output_dir, exist_ok=True)

        fname = os.path.basename(img_path)

        try:
            _, n_boxes, error = detect_image(img_path, angle, output_dir)

            if error:
                print(f"  [{processed+1}/{total}] {fname}: ERROR - {error}")
                errors += 1
            else:
                processed += 1
                elapsed = time.time() - start_time
                rate = (processed + errors) / elapsed if elapsed > 0 else 0
                eta = (total - processed - errors) / rate if rate > 0 else 0

                if (idx + 1) % 10 == 0 or idx == total - 1:
                    print(f"  [{processed}/{total}] {fname}: {n_boxes} boxes | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        except Exception as e:
            print(f"  [{processed+1}/{total}] {fname}: EXCEPTION - {str(e)[:80]}")
            errors += 1

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  COMPLETE!")
    print(f"  Processed: {processed}/{total} images in {total_time:.1f}s")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_root}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
