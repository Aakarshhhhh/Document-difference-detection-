import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

def _apply_clahe_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """Apply CLAHE on L channel in LAB space and return enhanced RGB image."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def _align_images_ecc(base_gray: np.ndarray, moving_rgb: np.ndarray) -> np.ndarray:
    """Try to align moving image to base using ECC; fallback returns original on failure."""
    try:
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)
        (cc, warp_matrix) = cv2.findTransformECC(base_gray, moving_gray, warp_matrix, warp_mode, criteria)
        h, w = base_gray.shape
        aligned = cv2.warpAffine(moving_rgb, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except Exception:
        return moving_rgb

def _align_images_orb(base_rgb: np.ndarray, moving_rgb: np.ndarray) -> np.ndarray:
    """ORB feature-based homography alignment as a fallback."""
    try:
        orb = cv2.ORB_create(1500)
        kp1, des1 = orb.detectAndCompute(cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY), None)
        kp2, des2 = orb.detectAndCompute(cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY), None)
        if des1 is None or des2 is None:
            return moving_rgb
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:200]
        if len(matches) < 10:
            return moving_rgb
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return moving_rgb
        h, w = base_rgb.shape[:2]
        aligned = cv2.warpPerspective(moving_rgb, H, (w, h))
        return aligned
    except Exception:
        return moving_rgb

def find_image_differences(img1: Image.Image, img2: Image.Image, threshold=0.95, min_change_size: int = 10, min_change_area: int = 200, enable_edge_detection: bool = True, enable_color_detection: bool = True, roi: tuple | None = None, return_mask: bool = False):
    """
    Compares two PIL Image objects and returns a difference image with numbered annotations.
    Returns the annotated image, list of changes, and SSIM score.
    Uses intelligent detection that focuses on meaningful changes while filtering noise.
    If roi is provided as (x, y, w, h), detection inside ROI is made more permissive.
    """
    # Ensure images are the same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    # Convert PIL Images to aligned, normalized RGB then to grayscale
    base_rgb = np.array(img1.convert('RGB'))
    moving_rgb = np.array(img2.convert('RGB'))

    # Lighting normalization on both images
    base_rgb = _apply_clahe_rgb(base_rgb)
    moving_rgb = _apply_clahe_rgb(moving_rgb)

    # Alignment (ECC with ORB fallback)
    base_gray_for_ecc = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    aligned_rgb = _align_images_ecc(base_gray_for_ecc, moving_rgb)
    if aligned_rgb is moving_rgb:
        aligned_rgb = _align_images_orb(base_rgb, moving_rgb)

    img1_cv = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    img2_cv = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)
    # Light denoising to reduce speckle noise
    img1_cv = cv2.GaussianBlur(img1_cv, (5, 5), 0)
    img2_cv = cv2.GaussianBlur(img2_cv, (5, 5), 0)

    # Compute SSIM between the two images
    score, diff = ssim(img1_cv, img2_cv, full=True)
    diff = (diff * 255).astype("uint8")
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # Create annotated image
    # Use aligned moving image for annotation background
    img_annotated = Image.fromarray(aligned_rgb)
    draw = ImageDraw.Draw(img_annotated)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    changes = []
    change_number = 1
    
    # Use a more intelligent approach: combine multiple methods but with better filtering
    all_contours = []
    
    # Method 1: SSIM-based detection with adaptive thresholding
    # Use the threshold parameter to determine sensitivity
    if threshold < 0.8:  # Sensitive mode
        thresh_vals = [25, 20, 15]
    elif threshold < 0.9:  # Balanced mode
        thresh_vals = [30, 25, 20]
    else:  # Conservative mode
        thresh_vals = [35, 30]
    
    mask_accumulate = np.zeros_like(img1_cv, dtype=np.uint8)
    for thresh_val in thresh_vals:
        thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Stronger morphological operations to suppress noise
        kernel = np.ones((5,5),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        all_contours.extend(contours)
        mask_accumulate = cv2.bitwise_or(mask_accumulate, thresh)

    # Method 2: Absolute difference with adaptive thresholding
    abs_diff = cv2.absdiff(img1_cv, img2_cv)
    abs_diff = cv2.GaussianBlur(abs_diff, (5, 5), 0)
    if threshold < 0.8:  # Sensitive mode
        abs_thresh_vals = [30, 25, 20, 15]
    elif threshold < 0.9:  # Balanced mode
        abs_thresh_vals = [35, 30, 25]
    else:  # Conservative mode
        abs_thresh_vals = [40, 35]
    
    for thresh_val in abs_thresh_vals:
        abs_thresh = cv2.threshold(abs_diff, thresh_val, 255, cv2.THRESH_BINARY)[1]
        
        # Stronger cleaning
        kernel = np.ones((5,5),np.uint8)
        abs_thresh = cv2.morphologyEx(abs_thresh, cv2.MORPH_CLOSE, kernel)
        abs_thresh = cv2.morphologyEx(abs_thresh, cv2.MORPH_OPEN, kernel)
        
        abs_contours = cv2.findContours(abs_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        abs_contours = abs_contours[0] if len(abs_contours) == 2 else abs_contours[1]
        all_contours.extend(abs_contours)
        mask_accumulate = cv2.bitwise_or(mask_accumulate, abs_thresh)

    # Method 3: Edge-based detection (only if significant differences)
    if enable_edge_detection:
        edges1 = cv2.Canny(img1_cv, 50, 150)
        edges2 = cv2.Canny(img2_cv, 50, 150)
        edge_diff = cv2.absdiff(edges1, edges2)
        
        # Only use edge detection if there are substantial edge differences
        edge_pixels = np.sum(edge_diff > 0)
        if edge_pixels > 200:  # Only if there are substantial edge differences
            edge_thresh = cv2.threshold(edge_diff, 20, 255, cv2.THRESH_BINARY)[1]
            # Moderate cleaning for edge detection
            kernel = np.ones((3,3),np.uint8)
            edge_thresh = cv2.morphologyEx(edge_thresh, cv2.MORPH_CLOSE, kernel)
            edge_thresh = cv2.morphologyEx(edge_thresh, cv2.MORPH_OPEN, kernel)
            
            edge_contours = cv2.findContours(edge_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge_contours = edge_contours[0] if len(edge_contours) == 2 else edge_contours[1]
            all_contours.extend(edge_contours)
            mask_accumulate = cv2.bitwise_or(mask_accumulate, edge_thresh)

    # Method 4: Color-based detection (LAB deltaE approximation)
    if enable_color_detection:
        img1_color = np.array(img1.convert('RGB'))
        img2_color = np.array(img2.convert('RGB'))

        img1_lab = cv2.cvtColor(img1_color, cv2.COLOR_RGB2LAB)
        img2_lab = cv2.cvtColor(img2_color, cv2.COLOR_RGB2LAB)

        dL = cv2.absdiff(img1_lab[:,:,0], img2_lab[:,:,0]).astype(np.float32)
        dA = cv2.absdiff(img1_lab[:,:,1], img2_lab[:,:,1]).astype(np.float32)
        dB = cv2.absdiff(img1_lab[:,:,2], img2_lab[:,:,2]).astype(np.float32)
        # Approximate deltaE (CIE76)
        deltaE = np.sqrt(dL**2 + dA**2 + dB**2)

        # Trigger if enough pixels exceed a modest color change
        if np.sum(deltaE > 12) > 80:
            color_thresh = (deltaE > 14).astype(np.uint8) * 255
            kernel = np.ones((3,3),np.uint8)
            color_thresh = cv2.morphologyEx(color_thresh, cv2.MORPH_CLOSE, kernel)
            color_thresh = cv2.morphologyEx(color_thresh, cv2.MORPH_OPEN, kernel)

            color_contours = cv2.findContours(color_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color_contours = color_contours[0] if len(color_contours) == 2 else color_contours[1]
            all_contours.extend(color_contours)
            mask_accumulate = cv2.bitwise_or(mask_accumulate, color_thresh)

    # Intelligent filtering: focus on meaningful changes while reducing noise
    filtered_contours = []
    img_area = img1_cv.shape[0] * img1_cv.shape[1]
    
    height, width = img1_cv.shape[:2]
    margin = max(10, int(min(height, width) * 0.02))
    # Minimum perimeter based on image size
    min_perimeter = int(np.sqrt(img_area) * (0.02 if threshold < 0.8 else 0.03))
    # Solidity requirement
    min_solidity = 0.55 if threshold < 0.8 else 0.65

    # ROI unpack
    roi_x, roi_y, roi_w, roi_h = (None, None, None, None)
    if roi and len(roi) == 4:
        roi_x, roi_y, roi_w, roi_h = roi

    for c in all_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)

        # Check if contour lies within ROI
        in_roi = False
        if roi_x is not None:
            in_roi = (x >= roi_x and y >= roi_y and x + w <= roi_x + roi_w and y + h <= roi_y + roi_h)

        # Ignore near page margins unless fairly large
        if not in_roi:
            if (x < margin or y < margin or x + w > width - margin or y + h > height - margin) and area < 5 * min_change_area:
                continue

        # Adaptive thresholds with ROI relaxation
        if threshold < 0.7:
            size_factor = 0.008
            area_factor = 0.00015
        elif threshold < 0.85:
            size_factor = 0.012
            area_factor = 0.0004
        else:
            size_factor = 0.02
            area_factor = 0.001

        adaptive_min_size = max(20, int(np.sqrt(img_area) * size_factor))
        adaptive_min_area = max(300, int(img_area * area_factor))
        min_size = max(min_change_size, adaptive_min_size)
        min_area = max(min_change_area, adaptive_min_area)

        # Relax requirements inside ROI
        if in_roi:
            min_size = max(5, int(min_size * 0.6))
            min_area = max(50, int(min_area * 0.4))
            min_perimeter_local = max(20, int(min_perimeter * 0.7))
            min_solidity_local = max(0.45, min_solidity - 0.1)
        else:
            min_perimeter_local = min_perimeter
            min_solidity_local = min_solidity

        if w > min_size and h > min_size and area > min_area and peri > min_perimeter_local:
            aspect_ratio = max(w, h) / max(1, min(w, h))
            if aspect_ratio < 12:
                local_area = w * h
                if area > local_area * 0.1:
                    hull = cv2.convexHull(c)
                    hull_area = cv2.contourArea(hull)
                    solidity = (area / hull_area) if hull_area > 0 else 0
                    if solidity >= min_solidity_local:
                        filtered_contours.append(c)
    
    # Smart merging: merge nearby contours but preserve distinct changes
    merged_contours = []
    for c in filtered_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        merged = False
        
        for i, existing in enumerate(merged_contours):
            (ex, ey, ew, eh) = cv2.boundingRect(existing)
            
            # Calculate overlap
            overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
            overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
            overlap_area = overlap_x * overlap_y
            
            # Merge if there's significant overlap or they're very close
            if overlap_area > 0 or (abs(x - ex) < 40 and abs(y - ey) < 40):
                # Merge by expanding the bounding box
                new_x = min(x, ex)
                new_y = min(y, ey)
                new_w = max(x + w, ex + ew) - new_x
                new_h = max(y + h, ey + eh) - new_y
                
                # Create a simple rectangular contour
                merged_contours[i] = np.array([[[new_x, new_y]], [[new_x + new_w, new_y]], 
                                            [[new_x + new_w, new_y + new_h]], [[new_x, new_y + new_h]]])
                merged = True
                break
        
        if not merged:
            merged_contours.append(c)
    
    # Sort by area (largest first) and limit to reasonable number
    merged_contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    
    # Adaptive limit based on image size and threshold
    if threshold < 0.8:  # Sensitive mode
        max_changes = min(15, int(np.sqrt(img_area) * 0.01))
    elif threshold < 0.9:  # Balanced mode
        max_changes = min(10, int(np.sqrt(img_area) * 0.008))
    else:  # Conservative mode
        max_changes = min(8, int(np.sqrt(img_area) * 0.006))
    
    merged_contours = merged_contours[:max_changes]
    
    # Final quality check: ensure changes are meaningful
    final_contours = []
    for c in merged_contours:
        area = cv2.contourArea(c)
        if area > img_area * 0.001:  # Must be at least 0.1% of image area
            final_contours.append(c)
    
    merged_contours = final_contours

    # Fallback: if too few changes detected, run a more sensitive pass
    if len(merged_contours) == 0:
        # Use a low threshold absolute difference on grayscale
        abs_diff_fallback = cv2.absdiff(img1_cv, img2_cv)
        fb_thresh = cv2.threshold(abs_diff_fallback, 10, 255, cv2.THRESH_BINARY)[1]
        kernel_fb = np.ones((3,3), np.uint8)
        fb_thresh = cv2.morphologyEx(fb_thresh, cv2.MORPH_CLOSE, kernel_fb)
        fb_thresh = cv2.dilate(fb_thresh, kernel_fb, iterations=1)
        fb_contours = cv2.findContours(fb_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fb_contours = fb_contours[0] if len(fb_contours) == 2 else fb_contours[1]

        # Relaxed filtering for fallback
        fallback_filtered = []
        fb_min_size = max(5, int(max(1, min_change_size) * 0.6))
        fb_min_area = max(50, int(max(1, min_change_area) * 0.25))
        for c in fb_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if w > fb_min_size and h > fb_min_size and area > fb_min_area:
                fallback_filtered.append(c)

        # Merge fallback contours similarly
        fb_merged = []
        for c in fallback_filtered:
            (x, y, w, h) = cv2.boundingRect(c)
            merged = False
            for i, existing in enumerate(fb_merged):
                (ex, ey, ew, eh) = cv2.boundingRect(existing)
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                if overlap_area > 0 or (abs(x - ex) < 30 and abs(y - ey) < 30):
                    new_x = min(x, ex)
                    new_y = min(y, ey)
                    new_w = max(x + w, ex + ew) - new_x
                    new_h = max(y + h, ex + eh) - new_y
                    fb_merged[i] = np.array([[[new_x, new_y]], [[new_x + new_w, new_y]],
                                             [[new_x + new_w, new_y + new_h]], [[new_x, new_y + new_h]]])
                    merged = True
                    break
            if not merged:
                fb_merged.append(c)

        # Use at most a reasonable number from fallback
        fb_merged.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        merged_contours = fb_merged[:10]
    
    # Process the merged contours
    for c in merged_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # Determine change type based on analysis
        change_type = determine_change_type(img1, img2, x, y, w, h)
        
        # Draw colored rectangle based on change type
        if change_type == "removal":
            color = (255, 0, 0)  # Red
            # Draw dashed line for removal
            draw_dashed_rectangle(draw, x, y, x + w, y + h, color, width=3)
        elif change_type == "addition":
            color = (0, 255, 0)  # Green
            # Draw solid rectangle using PIL so it persists
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        else:  # modification
            color = (255, 165, 0)  # Orange
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Draw numbered circle
        circle_radius = 15
        circle_x = x + w + 5
        circle_y = y - 5
        
        # Ensure circle is within image bounds
        circle_x = min(circle_x, img_annotated.width - circle_radius - 5)
        circle_y = max(circle_y, circle_radius + 5)
        
        draw.ellipse([circle_x - circle_radius, circle_y - circle_radius,
                    circle_x + circle_radius, circle_y + circle_radius], 
                   fill=color, outline=(0, 0, 0), width=2)
        
        # Draw change number
        text_bbox = draw.textbbox((0, 0), str(change_number), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = circle_x - text_width // 2
        text_y = circle_y - text_height // 2
        draw.text((text_x, text_y), str(change_number), fill=(255, 255, 255), font=font)

        changes.append({
            'number': change_number,
            'type': change_type,
            'bbox': (x, y, w, h),
            'description': get_change_description(change_type, change_number)
        })
        
        change_number += 1

    if return_mask:
        return img_annotated, changes, score, (mask_accumulate > 0).astype(np.uint8)
    return img_annotated, changes, score

def determine_change_type(img1, img2, x, y, w, h):
    """
    Determines the type of change based on analysis of the region.
    Enhanced to better detect object additions/removals.
    """
    # Extract the region from both images
    region1 = img1.crop((x, y, x + w, y + h))
    region2 = img2.crop((x, y, x + w, y + h))
    
    # Convert to grayscale for analysis
    region1_gray = region1.convert('L')
    region2_gray = region2.convert('L')
    
    # Calculate multiple metrics
    mean1 = np.mean(np.array(region1_gray))
    mean2 = np.mean(np.array(region2_gray))
    
    # Calculate variance to detect structural changes
    var1 = np.var(np.array(region1_gray))
    var2 = np.var(np.array(region2_gray))
    
    # Calculate edge density (more edges = more complex object)
    edges1 = cv2.Canny(np.array(region1_gray), 50, 150)
    edges2 = cv2.Canny(np.array(region2_gray), 50, 150)
    edge_density1 = np.sum(edges1 > 0) / (w * h)
    edge_density2 = np.sum(edges2 > 0) / (w * h)
    
    # Enhanced heuristics
    brightness_diff = abs(mean2 - mean1)
    variance_diff = abs(var2 - var1)
    edge_diff = abs(edge_density2 - edge_density1)
    
    # More sensitive detection for object additions/removals
    # If region2 has significantly more structure (edges), it's likely an addition
    if edge_density2 > edge_density1 + 0.05 and mean2 > mean1 + 10:
        return "addition"
    # If region1 has more structure, it's likely a removal
    elif edge_density1 > edge_density2 + 0.05 and mean1 > mean2 + 10:
        return "removal"
    # If brightness difference is significant, classify based on brightness
    elif brightness_diff > 15:
        if mean2 > mean1:
            return "addition"
        else:
            return "removal"
    # If variance difference is significant, it's likely a modification
    elif variance_diff > 100:
        return "modification"
    else:
        return "modification"

def draw_dashed_rectangle(draw, x1, y1, x2, y2, color, width=2, dash_length=5):
    """Draws a dashed rectangle."""
    # Top
    for x in range(x1, x2, dash_length * 2):
        draw.line([x, y1, min(x + dash_length, x2), y1], fill=color, width=width)
    # Bottom
    for x in range(x1, x2, dash_length * 2):
        draw.line([x, y2, min(x + dash_length, x2), y2], fill=color, width=width)
    # Left
    for y in range(y1, y2, dash_length * 2):
        draw.line([x1, y, x1, min(y + dash_length, y2)], fill=color, width=width)
    # Right
    for y in range(y1, y2, dash_length * 2):
        draw.line([x2, y, x2, min(y + dash_length, y2)], fill=color, width=width)

def get_change_description(change_type, number):
    """Returns a detailed description for the change type."""
    descriptions = {
        "addition": f"New content added (Change #{number})",
        "removal": f"Content removed (Change #{number})", 
        "modification": f"Content modified (Change #{number})"
    }
    return descriptions.get(change_type, f"Visual change detected (Change #{number})")

def compare_images_pixel_wise(img1, img2, threshold=30):
    """
    Performs pixel-wise comparison between two images.
    Returns a mask showing pixel-level differences.
    """
    # Convert to numpy arrays
    arr1 = np.array(img1.convert('RGB'))
    arr2 = np.array(img2.convert('RGB'))
    
    # Ensure same size
    if arr1.shape != arr2.shape:
        img2_resized = img2.resize(img1.size)
        arr2 = np.array(img2_resized.convert('RGB'))
    
    # Calculate absolute difference
    diff = np.abs(arr1.astype(int) - arr2.astype(int))
    
    # Create mask where differences exceed threshold
    mask = np.any(diff > threshold, axis=2)
    
    return mask