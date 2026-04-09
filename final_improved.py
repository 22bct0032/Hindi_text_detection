"""
Improved Text Detection with Post-Processing
Based on final.py with additional filters to reduce over/under-detection
"""

import cv2
import numpy as np
import os
import sys
import json

def oddize(x):
    x = int(round(x))
    if x <= 1:
        return 3
    return x if x % 2 == 1 else x + 1

def compute_band_kernel_sizes(stats, centroids, img_h,
                              num_bands,
                              area_thresh,
                              scale_factor,
                              min_k, max_k,
                              density_shrink_coeff):
    comp_heights = []
    comp_centroids_y = []
    n = stats.shape[0]
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_thresh:
            continue
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cy = centroids[i][1]
        comp_heights.append(h)
        comp_centroids_y.append(cy)

    if len(comp_heights) == 0:
        return [oddize(min(max(min_k, 5), max_k))] * num_bands

    comp_heights = np.array(comp_heights)
    comp_centroids_y = np.array(comp_centroids_y)

    band_kernel_sizes = []
    band_h = img_h / num_bands
    global_med = np.median(comp_heights)
    densities = []

    for b in range(num_bands):
        y0 = b * band_h
        y1 = (b + 1) * band_h
        mask = (comp_centroids_y >= y0) & (comp_centroids_y < y1)
        n_in_band = int(mask.sum())

        if n_in_band >= 2:
            med_h = np.median(comp_heights[mask])
            density = n_in_band / band_h
        else:
            med_h = global_med
            density = 0.0

        densities.append(density)
        k = float(med_h) * scale_factor
        shrink_scale = 1.0 / (1.0 + density_shrink_coeff * density * 50.0)
        k = k * shrink_scale
        k = max(min_k, min(k, max_k))
        k = oddize(k)
        band_kernel_sizes.append(k)

    smoothed = []
    for i in range(num_bands):
        w_sum, k_sum = 0, 0
        for j in range(max(0, i - 2), min(num_bands, i + 3)):
            w = densities[j] + 0.2
            k_sum += w * band_kernel_sizes[j]
            w_sum += w
        smoothed.append(oddize(k_sum / w_sum))

    return smoothed

def dynamic_closing_by_bands(gray_img, band_kernel_sizes, overlap=0.30):
    h, w = gray_img.shape[:2]
    num_bands = len(band_kernel_sizes)
    band_h = h / num_bands
    out = np.zeros_like(gray_img)

    for b in range(num_bands):
        ksize = int(band_kernel_sizes[b])
        local_overlap = overlap + 0.15 * (ksize / max(band_kernel_sizes))
        start = int(round(max(0, (b - local_overlap) * band_h)))
        end = int(round(min(h, (b + 1 + local_overlap) * band_h)))
        if end <= start:
            continue
        band_slice = gray_img[start:end].copy()

        k_w = oddize(max(3, ksize // 2))
        k_h = oddize(max(3, ksize))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_w, k_h))
        closed = cv2.morphologyEx(band_slice, cv2.MORPH_CLOSE, kernel)
        out[start:end] = np.maximum(out[start:end], closed)

    return out


# ============================================================
# POST-PROCESSING FILTERS (NEW)
# ============================================================

def filter_by_aspect_ratio(boxes, min_ratio=0.3, max_ratio=15.0):
    """Remove boxes with abnormal aspect ratios (likely lines or borders)"""
    filtered = []
    for box in boxes:
        x, y, w, h = box
        if h == 0:
            continue
        ratio = w / h
        if min_ratio <= ratio <= max_ratio:
            filtered.append(box)
    return filtered

def filter_by_relative_size(boxes, min_factor=0.05, max_factor=8.0):
    """Remove boxes much larger or smaller than median size"""
    if len(boxes) < 3:
        return boxes
    
    areas = [b[2] * b[3] for b in boxes]
    median_area = np.median(areas)
    
    filtered = []
    for box in boxes:
        area = box[2] * box[3]
        if median_area * min_factor <= area <= median_area * max_factor:
            filtered.append(box)
    return filtered

def filter_page_borders(boxes, img_w, img_h, margin_ratio=0.02, span_ratio=0.7):
    """Remove boxes near page edges that span most of width/height (borders)"""
    filtered = []
    margin_x = img_w * margin_ratio
    margin_y = img_h * margin_ratio
    
    for box in boxes:
        x, y, w, h = box
        
        # Check if box is at page edge AND spans most of that dimension
        at_left = x <= margin_x
        at_right = (x + w) >= (img_w - margin_x)
        at_top = y <= margin_y
        at_bottom = (y + h) >= (img_h - margin_y)
        
        spans_width = w >= img_w * span_ratio
        spans_height = h >= img_h * span_ratio
        
        # Remove if touching edge and spanning most of width/height
        if (at_left or at_right) and spans_height:
            continue
        if (at_top or at_bottom) and spans_width:
            continue
        # Remove if box covers nearly the entire page
        if spans_width and spans_height:
            continue
            
        filtered.append(box)
    return filtered

def non_maximum_suppression(boxes, overlap_thresh=0.5):
    """Remove overlapping boxes, keeping the one with area closer to median"""
    if len(boxes) < 2:
        return boxes
    
    areas = [b[2] * b[3] for b in boxes]
    median_area = np.median(areas)
    
    # Sort by how close area is to median (best first)
    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: abs(areas[i] - median_area))
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        keep.append(i)
        
        bx, by, bw, bh = boxes[i]
        
        for j in indices:
            if j in suppressed or j == i or j in [k for k in keep]:
                continue
            
            ox, oy, ow, oh = boxes[j]
            
            # Calculate intersection
            ix1 = max(bx, ox)
            iy1 = max(by, oy)
            ix2 = min(bx + bw, ox + ow)
            iy2 = min(by + bh, oy + oh)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                smaller_area = min(areas[i], areas[j])
                
                if smaller_area > 0 and intersection / smaller_area >= overlap_thresh:
                    suppressed.add(j)
    
    return [boxes[i] for i in keep]

def filter_line_like_boxes(boxes, min_height=8, max_aspect_for_line=25):
    """Remove very thin horizontal boxes (likely notebook rulings)"""
    filtered = []
    for box in boxes:
        x, y, w, h = box
        if h < min_height and w / max(h, 1) > max_aspect_for_line:
            continue
        filtered.append(box)
    return filtered

def filter_contained_boxes(boxes, containment_thresh=0.7):
    """Remove smaller boxes that are mostly contained inside larger boxes"""
    if len(boxes) < 2:
        return boxes
    
    # Sort by area (largest first)
    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][2] * boxes[i][3], reverse=True)
    
    keep = set(range(len(boxes)))
    
    for idx_i, i in enumerate(sorted_indices):
        if i not in keep:
            continue
        bx, by, bw, bh = boxes[i]
        b_area = bw * bh
        
        for j in sorted_indices[idx_i+1:]:
            if j not in keep:
                continue
            ox, oy, ow, oh = boxes[j]
            o_area = ow * oh
            
            # Calculate intersection
            ix1 = max(bx, ox)
            iy1 = max(by, oy)
            ix2 = min(bx + bw, ox + ow)
            iy2 = min(by + bh, oy + oh)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                # If smaller box is mostly inside larger box, remove it
                if o_area > 0 and intersection / o_area >= containment_thresh:
                    keep.discard(j)
    
    return [boxes[i] for i in sorted(keep)]

def second_pass_merge(boxes, gap_threshold=20, line_tolerance=0.5):
    """Second pass merge: group by horizontal lines and merge close boxes"""
    if len(boxes) < 2:
        return boxes
    
    # Group into lines by y-center
    lines = []
    for box in sorted(boxes, key=lambda b: b[1]):
        x, y, w, h = box
        center_y = y + h / 2
        placed = False
        
        for line in lines:
            line_centers = [b[1] + b[3]/2 for b in line]
            line_heights = [b[3] for b in line]
            avg_center = sum(line_centers) / len(line_centers)
            avg_height = sum(line_heights) / len(line_heights)
            
            if abs(center_y - avg_center) <= avg_height * line_tolerance:
                line.append(box)
                placed = True
                break
        
        if not placed:
            lines.append([box])
    
    # Merge close boxes within each line
    merged = []
    for line in lines:
        if len(line) == 0:
            continue
        line = sorted(line, key=lambda b: b[0])
        current = list(line[0])
        
        for next_box in line[1:]:
            cx, cy, cw, ch = current
            nx, ny, nw, nh = next_box
            cx2 = cx + cw
            nx2 = nx + nw
            
            gap = nx - cx2
            
            if gap <= gap_threshold:
                # Merge
                new_x = min(cx, nx)
                new_y = min(cy, ny)
                new_x2 = max(cx2, nx2)
                new_y2 = max(cy + ch, ny + nh)
                current = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
            else:
                merged.append(current)
                current = list(next_box)
        merged.append(current)
    
    return merged


def deskew_image(img):
    """
    Automatically detect and correct image rotation/skew.
    Returns: (deskewed_image, angle, paper_mask)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    angle = 0.0
    method_used = "none"
    
    # Strategy 1: Detect paper boundary via contour
    border_mean = np.mean([
        np.mean(gray[:10, :]), np.mean(gray[-10:, :]),
        np.mean(gray[:, :10]), np.mean(gray[:, -10:])
    ])
    center_mean = np.mean(gray[h//4:3*h//4, w//4:3*w//4])
    
    if abs(border_mean - center_mean) > 30:
        edges = cv2.Canny(blurred, 30, 100)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, k, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area_ratio = cv2.contourArea(largest) / (h * w)
            if area_ratio > 0.15:
                rect = cv2.minAreaRect(largest)
                rect_angle = rect[2]
                rect_w, rect_h = rect[1]
                if rect_w < rect_h:
                    rect_angle = rect_angle + 90
                while rect_angle > 90: rect_angle -= 180
                while rect_angle < -90: rect_angle += 180
                angle = rect_angle
                method_used = f"paper boundary ({area_ratio:.0%})"
    
    # Strategy 2: Hough lines histogram peak (always run to cross-validate)
    edges2 = cv2.Canny(blurred, 50, 150, apertureSize=3)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    dilated = cv2.dilate(edges2, k2, iterations=2)
    lines = cv2.HoughLinesP(dilated, 1, np.pi/180, 30,
                             minLineLength=w//12, maxLineGap=30)
    
    if lines is not None and len(lines) >= 5:
        all_angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < 20:
                continue
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            all_angles.append(a)
        
        if len(all_angles) >= 5:
            hist, bin_edges = np.histogram(all_angles, bins=36, range=(-90, 90))
            peak_idx = np.argmax(hist)
            peak_center = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
            nearby = [a for a in all_angles if abs(a - peak_center) < 10]
            hough_angle = np.median(nearby) if len(nearby) >= 3 else peak_center
            
            # Require minimum peak support: at least 15% of lines must agree
            peak_support = len(nearby) / len(all_angles) if len(all_angles) > 0 else 0
            
            # Sanity check: if paper boundary says small angle but hough says large,
            # trust paper boundary (hough may be confused by notebook edges/binding)
            hough_trustworthy = True
            if method_used != "none" and abs(angle) < 5 and abs(hough_angle) > 20:
                hough_trustworthy = False
                print(f"   [Deskew] Hough says {hough_angle:.1f}° but paper boundary says {angle:.1f}°, keeping paper boundary")
            
            # Reject near-90° angles (vertical lines from notebook binding)
            if abs(abs(hough_angle) - 90) < 10:
                hough_trustworthy = False
                print(f"   [Deskew] Hough angle {hough_angle:.1f}° is near 90°, rejecting (notebook binding)")
            
            # Require higher support for large angles (>30° needs >50% agreement)
            if abs(hough_angle) > 30 and peak_support < 0.50:
                hough_trustworthy = False
                print(f"   [Deskew] Large angle {hough_angle:.1f}° with only {peak_support:.0%} support, rejecting")
            
            if hough_trustworthy and peak_support >= 0.15 and (method_used == "none" or hist[peak_idx] > 20):
                angle = hough_angle
                method_used = f"hough peak ({len(nearby)}/{len(all_angles)} lines, {peak_center:.1f}°, support={peak_support:.0%})"
    
    # Skip if angle is tiny
    if abs(angle) < 2.0:
        print(f"   [Deskew] Angle {angle:.1f}° is small, skipping ({method_used})")
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return img, angle, mask, None
    
    print(f"   [Deskew] Detected rotation: {angle:.1f}° via {method_used}, correcting...")
    
    # Rotate the image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    # Use gray fill so borders don't create high-contrast edges
    deskewed = cv2.warpAffine(img, M, (new_w, new_h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(180, 180, 180))
    
    # Create paper mask by rotating a white rectangle
    white = np.ones_like(img) * 255
    rot_white = cv2.warpAffine(white, M, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))
    paper_mask = cv2.cvtColor(rot_white, cv2.COLOR_BGR2GRAY)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    paper_mask = cv2.erode(paper_mask, erode_k)
    _, paper_mask = cv2.threshold(paper_mask, 128, 255, cv2.THRESH_BINARY)
    
    return deskewed, angle, paper_mask, M


def process_image(image_path, skip_deskew=False):
    """Process a single image with improved detection and post-processing"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, f"Failed to read '{image_path}'"
    # Keep original image for output
    original_img = img.copy()

    # Step 0: Automatic deskewing (skip for perspective-transformed images)
    if skip_deskew:
        h, w = img.shape[:2]
        skew_angle = 0.0
        paper_mask = np.ones((h, w), dtype=np.uint8) * 255
        rotation_matrix = None
    else:
        img, skew_angle, paper_mask, rotation_matrix = deskew_image(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Dynamic Banding Logic
    num_bands = max(10, min(100, h // 200))

    # CLAHE Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    kernel_init = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    bg_init = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel_init)
    diff = cv2.absdiff(gray_eq, bg_init)
    norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    blurred = cv2.GaussianBlur(norm_img, (5, 5), 0)
    otsu_thresh, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If Otsu threshold is too high (missing faint text), use a lower threshold
    white_ratio = np.count_nonzero(binarized) / binarized.size
    if white_ratio < 0.05:  # Less than 5% detected as text - faint ink
        _, binarized = cv2.threshold(blurred, max(15, int(otsu_thresh * 0.4)), 255, cv2.THRESH_BINARY)

    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binarized = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, small_k)

    # Apply paper mask to exclude non-paper regions (gray border from rotation)
    if paper_mask is not None:
        pm_h, pm_w = paper_mask.shape[:2]
        bin_h, bin_w = binarized.shape[:2]
        if pm_h == bin_h and pm_w == bin_w:
            binarized = cv2.bitwise_and(binarized, paper_mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, connectivity=8)

    if num_labels <= 1:
        return img.copy(), [], None

    median_area = np.median(stats[1:, cv2.CC_STAT_AREA])
    area_thresh = max(5, median_area * 0.05)

    median_height = np.median(stats[1:, cv2.CC_STAT_HEIGHT])
    
    # Quality check: if median CC height is abnormally small, the binarization
    # produced mostly noise + merged blobs. Fall back to adaptive thresholding.
    if median_height <= 8 and num_labels > 50:
        max_cc_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        if max_cc_area > (h * w * 0.02):  # Largest CC covers >2% of image
            print(f"   [Binarization] Quality check failed: median_h={median_height:.0f}px, max_cc={max_cc_area}, re-binarizing...")
            
            # Try larger background subtraction kernel first
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
            bg_large = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel_large)
            diff2 = cv2.absdiff(gray_eq, bg_large)
            norm2 = cv2.normalize(diff2, None, 0, 255, cv2.NORM_MINMAX)
            blur2 = cv2.GaussianBlur(norm2, (5, 5), 0)
            _, binarized2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binarized2 = cv2.morphologyEx(binarized2, cv2.MORPH_OPEN, small_k)
            
            if paper_mask is not None:
                pm_h2, pm_w2 = paper_mask.shape[:2]
                bin_h2, bin_w2 = binarized2.shape[:2]
                if pm_h2 == bin_h2 and pm_w2 == bin_w2:
                    binarized2 = cv2.bitwise_and(binarized2, paper_mask)
            
            num_labels2b, _, stats2b, _ = cv2.connectedComponentsWithStats(binarized2)
            if num_labels2b > 1:
                med_h2 = np.median(stats2b[1:, cv2.CC_STAT_HEIGHT])
                max_cc2 = np.max(stats2b[1:, cv2.CC_STAT_AREA])
                
                if med_h2 > median_height and max_cc2 < max_cc_area * 0.8:
                    print(f"   [Binarization] Larger kernel improved: median_h={med_h2:.0f}px")
                    binarized = binarized2
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, connectivity=8)
                    median_area = np.median(stats[1:, cv2.CC_STAT_AREA])
                    median_height = np.median(stats[1:, cv2.CC_STAT_HEIGHT])
                else:
                    # Fallback to adaptive thresholding
                    adaptive = cv2.adaptiveThreshold(gray_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 31, 10)
                    # Invert (text should be white)
                    adaptive = cv2.bitwise_not(adaptive)
                    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, small_k)
                    
                    if paper_mask is not None:
                        pm_ha, pm_wa = paper_mask.shape[:2]
                        bin_ha, bin_wa = adaptive.shape[:2]
                        if pm_ha == bin_ha and pm_wa == bin_wa:
                            adaptive = cv2.bitwise_and(adaptive, paper_mask)
                    
                    num_labels_a, _, stats_a, _ = cv2.connectedComponentsWithStats(adaptive)
                    if num_labels_a > 1:
                        med_ha = np.median(stats_a[1:, cv2.CC_STAT_HEIGHT])
                        max_cc_a = np.max(stats_a[1:, cv2.CC_STAT_AREA])
                        # Accept if max CC dropped significantly (no more giant merged blob)
                        if max_cc_a < max_cc_area * 0.5 or med_ha > median_height:
                            print(f"   [Binarization] Adaptive threshold: median_h={med_ha:.0f}px, max_cc={max_cc_a} (was {max_cc_area})")
                            binarized = adaptive
                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, connectivity=8)
                            median_area = np.median(stats[1:, cv2.CC_STAT_AREA])
                            median_height = np.median(stats[1:, cv2.CC_STAT_HEIGHT])

    scale_factor = min(0.2, max(0.08, median_height / 130.0))

    density_shrink_coeff = 0.15
    min_k = max(3, median_height // 15)
    max_k = max(9, int(median_height * 0.5))
    overlap = 0.30
    min_area_keep = max(50, median_area * 0.08)

    band_k = compute_band_kernel_sizes(stats, centroids, binarized.shape[0],
                                      num_bands=num_bands,
                                      area_thresh=area_thresh,
                                      scale_factor=scale_factor,
                                      min_k=min_k, max_k=max_k,
                                      density_shrink_coeff=density_shrink_coeff)

    closed_dynamic = dynamic_closing_by_bands(binarized, band_k, overlap=overlap)

    num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(closed_dynamic, connectivity=8)
    refined = closed_dynamic.copy()
    for i in range(1, num_labels2):
        if stats2[i, cv2.CC_STAT_AREA] < min_area_keep:
            refined[labels2 == i] = 0

    boxes = []
    for i in range(1, num_labels2):
        x, y, bw, bh, area = stats2[i, cv2.CC_STAT_LEFT], stats2[i, cv2.CC_STAT_TOP], stats2[i, cv2.CC_STAT_WIDTH], stats2[i, cv2.CC_STAT_HEIGHT], stats2[i, cv2.CC_STAT_AREA]
        if area >= min_area_keep:
            boxes.append([x, y, bw, bh])

    initial_count = len(boxes)

    # ============================================================
    # MERGING (with tighter parameters)
    # ============================================================
    horizontal_gap_threshold = 3  # Very tight - only merge truly adjacent boxes
    vertical_overlap_ratio_min = 0.1  # Require some vertical overlap
    line_grouping_tolerance = 0.5  # Tight line grouping

    lines = []
    for box in boxes:
        x, y, bw, bh = box
        box_center_y = y + bh / 2
        placed = False
        
        for line in lines:
            line_y_centers = [b[1] + b[3]/2 for b in line]
            line_heights = [b[3] for b in line]
            line_center_y = sum(line_y_centers) / len(line_y_centers)
            avg_line_height = sum(line_heights) / len(line_heights)
            
            vertical_distance = abs(box_center_y - line_center_y)
            tolerance = avg_line_height * line_grouping_tolerance
            
            if vertical_distance <= tolerance:
                line.append(box)
                placed = True
                break
        
        if not placed:
            lines.append([box])

    final_boxes = []
    for line in lines:
        if len(line) == 0:
            continue
        line = sorted(line, key=lambda b: b[0])
        current = line[0]
        
        for next_box in line[1:]:
            x, y, bw, bh = current
            nx, ny, nw, nh = next_box
            x2, nx2 = x + bw, nx + nw

            horizontal_gap = nx - x2
            horizontal_overlap = max(0, x2 - nx)

            vertical_overlap = max(0, min(y + bh, ny + nh) - max(y, ny))
            min_h = min(bh, nh)
            vertical_overlap_ratio = vertical_overlap / float(min_h) if min_h > 0 else 0

            vertical_ok = vertical_overlap_ratio > vertical_overlap_ratio_min
            horizontal_ok = (horizontal_overlap > 0 or 
                             horizontal_gap <= 0 or 
                             horizontal_gap <= horizontal_gap_threshold)
            
            if vertical_ok and horizontal_ok:
                new_x = min(x, nx)
                new_y = min(y, ny)
                new_x2 = max(x2, nx2)
                new_y2 = max(y + bh, ny + nh)
                current = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
            else:
                final_boxes.append(current)
                current = next_box
        final_boxes.append(current)

    merged_count = len(final_boxes)

    # ============================================================
    # POST-PROCESSING PIPELINE (NEW)
    # ============================================================
    
    # Step 1: Filter small boxes by average area
    if len(final_boxes) > 0:
        final_box_areas = [b[2] * b[3] for b in final_boxes]
        avg_area = np.mean(final_box_areas)
        min_area_ratio = 0.06
        min_final_area = avg_area * min_area_ratio
        final_boxes = [b for b in final_boxes if (b[2] * b[3]) >= min_final_area]
    
    after_small_filter = len(final_boxes)

    # Step 2: Filter by aspect ratio (remove notebook lines, borders)
    final_boxes = filter_by_aspect_ratio(final_boxes, min_ratio=0.3, max_ratio=12.0)
    after_aspect = len(final_boxes)

    # Step 3: Filter line-like boxes (very thin horizontal bars)
    final_boxes = filter_line_like_boxes(final_boxes, min_height=10, max_aspect_for_line=20)
    after_lines = len(final_boxes)

    # Step 4: Filter page borders
    final_boxes = filter_page_borders(final_boxes, w, h, margin_ratio=0.01, span_ratio=0.5)
    after_borders = len(final_boxes)

    # Step 4b: Filter boxes outside the paper region using paper_mask
    if paper_mask is not None:
        mask_h, mask_w = paper_mask.shape[:2]
        on_paper = []
        for box in final_boxes:
            cx = min(box[0] + box[2]//2, mask_w - 1)
            cy = min(box[1] + box[3]//2, mask_h - 1)
            if paper_mask[cy, cx] > 0:
                on_paper.append(box)
        final_boxes = on_paper
    after_paper_mask = len(final_boxes)

    # Step 5: Filter by relative size (remove outliers)
    final_boxes = filter_by_relative_size(final_boxes, min_factor=0.08, max_factor=8.0)
    after_size = len(final_boxes)

    # Step 6: Containment filter - remove small boxes inside larger ones
    final_boxes = filter_contained_boxes(final_boxes, containment_thresh=0.7)
    after_contain = len(final_boxes)

    # Step 7: Adaptive second-pass merge
    # Only merge aggressively if box count seems too high (over-segmentation)
    # Estimate expected boxes from image size: ~1 word per 50px of height for Hindi text
    expected_word_count = max(20, h // 50)
    fragmentation_ratio = len(final_boxes) / expected_word_count if expected_word_count > 0 else 1.0
    
    if fragmentation_ratio > 6.0 and len(final_boxes) > 10:
        # Over-segmentation detected, apply aggressive merge
        heights = [b[3] for b in final_boxes]
        med_h = np.median(heights)
        second_gap = max(10, int(med_h * 0.4))
        final_boxes = second_pass_merge(final_boxes, gap_threshold=second_gap, 
                                         line_tolerance=0.5)
        print(f"   [Adaptive] Over-segmentation detected (ratio={fragmentation_ratio:.1f}), applied 2nd merge (gap={second_gap})")
    after_second_merge = len(final_boxes)

    # Step 8: Non-maximum suppression (remove overlapping duplicates)
    final_boxes = non_maximum_suppression(final_boxes, overlap_thresh=0.5)
    after_nms = len(final_boxes)

    # Step 8b: Split over-merged boxes (wider than 2.5x median width)
    if len(final_boxes) > 3:
        widths = [b[2] for b in final_boxes]
        med_w = np.median(widths)
        split_threshold = med_w * 2.5
        
        split_boxes = []
        for box in final_boxes:
            x, y, bw, bh = box
            if bw > split_threshold and bw > 50:
                # Use vertical projection to find split points
                roi = binarized[max(0,y):min(y+bh, binarized.shape[0]), 
                                max(0,x):min(x+bw, binarized.shape[1])]
                if roi.size > 0:
                    v_proj = np.sum(roi, axis=0) / 255
                    # Smooth projection
                    kernel_size = max(3, bw // 20)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    smooth_proj = cv2.GaussianBlur(v_proj.reshape(1, -1).astype(np.float32), 
                                                    (kernel_size, 1), 0).flatten()
                    
                    # Find valleys (potential split points)
                    margin = int(bw * 0.15)  # Don't split at edges
                    search_zone = smooth_proj[margin:-margin] if margin > 0 and len(smooth_proj) > 2*margin else smooth_proj
                    
                    if len(search_zone) > 10:
                        threshold = np.median(search_zone) * 0.3
                        # Find consecutive low regions
                        low_mask = search_zone < threshold
                        splits = []
                        i_start = None
                        for i_px in range(len(low_mask)):
                            if low_mask[i_px]:
                                if i_start is None:
                                    i_start = i_px
                            else:
                                if i_start is not None and (i_px - i_start) >= 3:
                                    split_x = margin + (i_start + i_px) // 2
                                    splits.append(split_x)
                                i_start = None
                        
                        if splits:
                            # Create sub-boxes at split points
                            prev_x = 0
                            for sx in splits:
                                sub_w = sx - prev_x
                                if sub_w > med_w * 0.3:
                                    split_boxes.append([x + prev_x, y, sub_w, bh])
                                prev_x = sx
                            # Last segment
                            sub_w = bw - prev_x
                            if sub_w > med_w * 0.3:
                                split_boxes.append([x + prev_x, y, sub_w, bh])
                            continue
                
                # Fallback: couldn't split, keep original
                split_boxes.append(box)
            else:
                split_boxes.append(box)
        
        if len(split_boxes) > len(final_boxes):
            print(f"   [Split] Split {len(split_boxes) - len(final_boxes)} over-merged boxes")
        final_boxes = split_boxes
    after_split = len(final_boxes)


    # Step 9: Isolation filter - remove noise in blank regions
    if len(final_boxes) > 3:
        box_heights = [b[3] for b in final_boxes]
        med_box_h = np.median(box_heights)
        med_box_area = np.median([b[2]*b[3] for b in final_boxes])
        
        # Find the main text region boundary
        y_bottoms = sorted([b[1] + b[3] for b in final_boxes])
        text_bottom = y_bottoms[int(len(y_bottoms) * 0.85)]  # 85th percentile
        
        non_isolated = []
        for i, box in enumerate(final_boxes):
            cx, cy = box[0] + box[2]//2, box[1] + box[3]//2
            box_area = box[2] * box[3]
            
            # Filter 1: Remove small boxes far below main text region
            if box[1] > text_bottom + med_box_h * 2 and box_area < med_box_area * 0.4:
                continue
            
            # Filter 2: Remove tiny isolated boxes with no neighbors
            if box_area < med_box_area * 0.2:
                isolation_dist = med_box_h * 5
                neighbors = sum(1 for j, other in enumerate(final_boxes) if i != j and
                    np.sqrt((cx - (other[0]+other[2]//2))**2 + (cy - (other[1]+other[3]//2))**2) < isolation_dist)
                if neighbors == 0:
                    continue
            
            non_isolated.append(box)
        
        final_boxes = non_isolated
    after_isolation = len(final_boxes)

    # Print filter summary
    print(f"   Initial boxes: {initial_count}")
    print(f"   After merging: {merged_count}")
    print(f"   After small filter: {after_small_filter}")
    print(f"   After aspect ratio: {after_aspect}")
    print(f"   After line filter: {after_lines}")
    print(f"   After border filter: {after_borders}")
    print(f"   After paper mask: {after_paper_mask}")
    print(f"   After size filter: {after_size}")
    print(f"   After containment: {after_contain}")
    print(f"   After 2nd merge: {after_second_merge}")
    print(f"   After NMS: {after_nms}")
    print(f"   After isolation: {after_isolation}")

    # Sort boxes in reading order
    sorted_boxes = sorted(final_boxes, key=lambda box: box[1])
    
    reading_lines = []
    if len(sorted_boxes) > 0:
        current_line = [sorted_boxes[0]]
        
        for box in sorted_boxes[1:]:
            current_y = box[1]
            last_y = current_line[-1][1]
            last_height = current_line[-1][3]
            
            if abs(current_y - last_y) < last_height * 0.5:
                current_line.append(box)
            else:
                current_line.sort(key=lambda b: b[0])
                reading_lines.append(current_line)
                current_line = [box]
        
        if current_line:
            current_line.sort(key=lambda b: b[0])
            reading_lines.append(current_line)
    
    reading_order_boxes = []
    for line in reading_lines:
        reading_order_boxes.extend(line)

    # Draw boxes on the ORIGINAL image (not the deskewed one)
    finalimg = original_img.copy()
    
    if rotation_matrix is not None:
        # Inverse-transform box corners from deskewed space to original space
        M_inv = cv2.invertAffineTransform(rotation_matrix)
        for (x, y, bw, bh) in final_boxes:
            # 4 corners of the box in deskewed image
            corners = np.array([
                [x, y],
                [x + bw, y],
                [x + bw, y + bh],
                [x, y + bh]
            ], dtype=np.float64)
            # Transform back to original image coordinates
            ones = np.ones((4, 1))
            corners_h = np.hstack([corners, ones])  # [x, y, 1]
            orig_corners = (M_inv @ corners_h.T).T  # Apply inverse transform
            orig_corners = orig_corners.astype(np.int32)
            # Draw the rotated box as a polygon
            cv2.polylines(finalimg, [orig_corners], True, (0, 0, 0), 1)
    else:
        # No rotation was applied, draw regular rectangles
        for (x, y, bw, bh) in final_boxes:
            cv2.rectangle(finalimg, (x, y), (x + bw, y + bh), (0, 0, 0), 1)

    return finalimg, reading_order_boxes, None


def main():
    # Accept image path from command-line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python final_improved.py <image_path>")
        sys.exit(1)
    
    print(f"\nProcessing: {image_path}")
    
    finalimg, boxes, error = process_image(image_path)
    
    if error:
        print(f"Error: {error}")
        sys.exit(1)
    
    # Save output
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/final_improved"
    os.makedirs(output_dir, exist_ok=True)
    
    input_filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(input_filename)[0]
    
    # Save image
    output_img_path = os.path.join(output_dir, input_filename)
    cv2.imwrite(output_img_path, finalimg)
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{name_without_ext}.json")
    json_data = {
        "source_image": image_path,
        "total_boxes": len(boxes),
        "detections": [
            {
                "box_id": i + 1,
                "x": int(box[0]),
                "y": int(box[1]),
                "width": int(box[2]),
                "height": int(box[3])
            }
            for i, box in enumerate(boxes)
        ]
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Image saved to: {output_img_path}")
    print(f"✓ JSON saved to: {json_path}")
    print(f"✓ Total boxes detected: {len(boxes)}")


if __name__ == "__main__":
    main()
