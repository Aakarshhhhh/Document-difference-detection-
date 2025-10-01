import cv2
from skimage.metrics import structural_similarity as ssim
from skimage import feature, measure
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from scipy import ndimage
from sklearn.cluster import DBSCAN

def _apply_clahe_rgb(rgb_img: np.ndarray) -> np.ndarray:
    """Apply enhanced CLAHE preprocessing for better change detection."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhanced CLAHE with better parameters for document analysis
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    l_eq = clahe.apply(l)
    
    # Apply gentle denoising to reduce artifacts
    l_eq = cv2.bilateralFilter(l_eq, 9, 75, 75)
    
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

def _enhance_image_quality(img: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing for better document analysis."""
    # Convert to LAB for better perceptual processing
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Enhance contrast adaptively
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))
    l = clahe.apply(l)
    
    # Sharpen slightly for better text/edge detection
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    l = cv2.filter2D(l, -1, kernel * 0.1 + np.array([[0,0,0], [0,1,0], [0,0,0]]))
    
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def _align_images_ecc(base_gray: np.ndarray, moving_rgb: np.ndarray) -> np.ndarray:
    """Enhanced ECC alignment with multiple warp modes and better preprocessing."""
    try:
        moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)
        
        # Try multiple alignment methods in order of complexity
        warp_modes = [
            (cv2.MOTION_TRANSLATION, np.eye(2, 3, dtype=np.float32)),
            (cv2.MOTION_EUCLIDEAN, np.eye(2, 3, dtype=np.float32)),
            (cv2.MOTION_AFFINE, np.eye(2, 3, dtype=np.float32))
        ]
        
        # Enhanced preprocessing for better alignment
        base_proc = cv2.GaussianBlur(base_gray, (3, 3), 0)
        moving_proc = cv2.GaussianBlur(moving_gray, (3, 3), 0)
        
        # Enhance edges for better feature matching
        base_proc = cv2.addWeighted(base_proc, 0.8, cv2.Laplacian(base_proc, cv2.CV_8U), 0.2, 0)
        moving_proc = cv2.addWeighted(moving_proc, 0.8, cv2.Laplacian(moving_proc, cv2.CV_8U), 0.2, 0)
        
        best_cc = -1
        best_aligned = moving_rgb
        
        for warp_mode, initial_warp in warp_modes:
            try:
                warp_matrix = initial_warp.copy()
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
                
                (cc, warp_matrix) = cv2.findTransformECC(
                    base_proc, moving_proc, warp_matrix, warp_mode, criteria
                )
                
                if cc > best_cc:
                    h, w = base_gray.shape
                    if warp_mode == cv2.MOTION_HOMOGRAPHY:
                        aligned = cv2.warpPerspective(moving_rgb, warp_matrix, (w, h))
                    else:
                        aligned = cv2.warpAffine(moving_rgb, warp_matrix, (w, h), 
                                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                    best_cc = cc
                    best_aligned = aligned
                    
                    # If we get good alignment, use it
                    if cc > 0.8:
                        break
                        
            except Exception:
                continue
                
        return best_aligned
        
    except Exception:
        return moving_rgb

def _align_images_orb(base_rgb: np.ndarray, moving_rgb: np.ndarray) -> np.ndarray:
    """Enhanced ORB feature-based alignment with better feature detection."""
    try:
        # Use enhanced ORB parameters for document images
        orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31
        )
        
        # Preprocess images for better feature detection
        base_gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
        moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)
        
        # Enhance for feature detection
        base_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(base_gray)
        moving_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(moving_gray)
        
        kp1, des1 = orb.detectAndCompute(base_enhanced, None)
        kp2, des2 = orb.detectAndCompute(moving_enhanced, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return moving_rgb
            
        # Use FLANN matcher for better matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=12,
                           key_size=20,
                           multi_probe_level=2)
        search_params = dict(checks=50)
        
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test for good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                        
        except Exception:
            # Fallback to brute force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:min(300, len(matches))]
        
        if len(good_matches) < 15:
            return moving_rgb
            
        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Try different transformation models
        transformations = [
            ('affine', cv2.estimateAffinePartial2D),
            ('homography', lambda s, d, *args: (cv2.findHomography(d, s, cv2.RANSAC, 3.0), None))
        ]
        
        best_aligned = moving_rgb
        best_inliers = 0
        
        for transform_name, transform_func in transformations:
            try:
                if transform_name == 'affine':
                    transformation_matrix, inliers = transform_func(
                        dst_pts, src_pts, cv2.RANSAC, 3.0, 2000, 0.99
                    )
                    if transformation_matrix is not None:
                        h, w = base_rgb.shape[:2]
                        aligned = cv2.warpAffine(moving_rgb, transformation_matrix, (w, h))
                        inlier_count = np.sum(inliers) if inliers is not None else 0
                else:
                    transformation_matrix, mask = transform_func(dst_pts, src_pts)
                    if transformation_matrix is not None:
                        h, w = base_rgb.shape[:2]
                        aligned = cv2.warpPerspective(moving_rgb, transformation_matrix, (w, h))
                        inlier_count = np.sum(mask) if mask is not None else 0
                        
                if transformation_matrix is not None and inlier_count > best_inliers:
                    best_aligned = aligned
                    best_inliers = inlier_count
                    
            except Exception:
                continue
                
        return best_aligned
        
    except Exception:
        return moving_rgb

def find_image_differences(img1: Image.Image, img2: Image.Image, threshold=0.95, min_change_size: int = 10, min_change_area: int = 200, enable_edge_detection: bool = True, enable_color_detection: bool = True, roi: tuple | None = None, return_mask: bool = False):
    """
    Enhanced image comparison with improved accuracy and multiple detection methods.
    Returns annotated image, changes list, and SSIM score.
    """
    # Ensure images are the same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    # Convert PIL Images to numpy arrays
    base_rgb = np.array(img1.convert('RGB'))
    moving_rgb = np.array(img2.convert('RGB'))
    
    # Enhanced preprocessing
    base_rgb = _enhance_image_quality(base_rgb)
    moving_rgb = _enhance_image_quality(moving_rgb)

    # Improved alignment with multiple methods
    base_gray_for_ecc = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    aligned_rgb = _align_images_ecc(base_gray_for_ecc, moving_rgb)
    
    # Check alignment quality and try ORB if needed
    alignment_check = cv2.matchTemplate(base_gray_for_ecc, 
                                      cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY), 
                                      cv2.TM_CCOEFF_NORMED)[0, 0]
    
    if alignment_check < 0.7:  # Poor alignment, try ORB
        orb_aligned = _align_images_orb(base_rgb, moving_rgb)
        orb_check = cv2.matchTemplate(base_gray_for_ecc, 
                                    cv2.cvtColor(orb_aligned, cv2.COLOR_RGB2GRAY), 
                                    cv2.TM_CCOEFF_NORMED)[0, 0]
        if orb_check > alignment_check:
            aligned_rgb = orb_aligned

    # Convert to grayscale for analysis with adaptive denoising
    img1_cv = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    img2_cv = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2GRAY)
    
    # Adaptive denoising based on image characteristics
    noise_level = np.std(cv2.Laplacian(img1_cv, cv2.CV_64F))
    if noise_level > 50:  # Noisy image
        img1_cv = cv2.bilateralFilter(img1_cv, 9, 75, 75)
        img2_cv = cv2.bilateralFilter(img2_cv, 9, 75, 75)
    else:  # Clean image, light smoothing only
        img1_cv = cv2.GaussianBlur(img1_cv, (3, 3), 0)
        img2_cv = cv2.GaussianBlur(img2_cv, (3, 3), 0)

    # Enhanced multi-method change detection
    img_height, img_width = img1_cv.shape
    img_area = img_height * img_width
    
    # Compute enhanced SSIM with windowing
    score, diff = ssim(img1_cv, img2_cv, full=True, win_size=7, data_range=255)
    diff = (diff * 255).astype("uint8")
    
    # Create annotated image
    img_annotated = Image.fromarray(aligned_rgb)
    draw = ImageDraw.Draw(img_annotated)
    
    # Load font with better fallback
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
        except:
            font = ImageFont.load_default()

    changes = []
    change_number = 1
    
    # === ENHANCED DETECTION PIPELINE ===
    
    # 1. Multi-scale SSIM analysis
    change_masks = []
    scales = [1.0, 0.75, 0.5] if img_area > 500000 else [1.0, 0.8]
    
    for scale in scales:
        if scale < 1.0:
            h_scaled = int(img_height * scale)
            w_scaled = int(img_width * scale)
            img1_scaled = cv2.resize(img1_cv, (w_scaled, h_scaled))
            img2_scaled = cv2.resize(img2_cv, (w_scaled, h_scaled))
        else:
            img1_scaled = img1_cv
            img2_scaled = img2_cv
            
        _, diff_scaled = ssim(img1_scaled, img2_scaled, full=True, win_size=min(7, min(img1_scaled.shape)//4*2+1))
        diff_scaled = (diff_scaled * 255).astype("uint8")
        
        if scale < 1.0:
            diff_scaled = cv2.resize(diff_scaled, (img_width, img_height))
            
        # Adaptive thresholding based on threshold parameter
        sensitivity_factor = 1.0 - threshold  # Higher threshold = lower sensitivity
        base_thresh = int(20 + sensitivity_factor * 40)  # 20-60 range
        
        diff_thresh = cv2.threshold(diff_scaled, base_thresh, 255, cv2.THRESH_BINARY_INV)[1]
        change_masks.append(diff_thresh)
    
    # 2. Absolute difference with adaptive parameters
    abs_diff = cv2.absdiff(img1_cv, img2_cv)
    abs_diff_smooth = cv2.GaussianBlur(abs_diff, (3, 3), 0)
    
    # Dynamic threshold based on image characteristics
    abs_mean = np.mean(abs_diff_smooth)
    abs_std = np.std(abs_diff_smooth)
    abs_threshold = max(15, int(abs_mean + 2 * abs_std * (1.0 - threshold)))
    
    abs_mask = cv2.threshold(abs_diff_smooth, abs_threshold, 255, cv2.THRESH_BINARY)[1]
    change_masks.append(abs_mask)
    
    # 3. Enhanced edge detection
    if enable_edge_detection:
        # Multi-scale edge detection
        edge_masks = []
        for sigma in [1.0, 2.0]:
            img1_smooth = cv2.GaussianBlur(img1_cv, (0, 0), sigma)
            img2_smooth = cv2.GaussianBlur(img2_cv, (0, 0), sigma)
            
            edges1 = cv2.Canny(img1_smooth, 50, 150)
            edges2 = cv2.Canny(img2_smooth, 50, 150)
            edge_diff = cv2.absdiff(edges1, edges2)
            
            if np.sum(edge_diff > 0) > 100:  # Sufficient edge changes
                edge_masks.append(edge_diff)
        
        if edge_masks:
            combined_edges = np.maximum.reduce(edge_masks)
            change_masks.append(combined_edges)
    
    # 4. Enhanced color detection
    if enable_color_detection:
        # Use enhanced images for color analysis
        img1_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB)
        img2_aligned_lab = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2LAB)
        
        # Compute Delta E with better precision
        dL = img1_lab[:,:,0].astype(np.float32) - img2_aligned_lab[:,:,0].astype(np.float32)
        dA = img1_lab[:,:,1].astype(np.float32) - img2_aligned_lab[:,:,1].astype(np.float32)
        dB = img1_lab[:,:,2].astype(np.float32) - img2_aligned_lab[:,:,2].astype(np.float32)
        
        deltaE = np.sqrt(dL**2 + dA**2 + dB**2)
        
        # Adaptive color threshold
        color_threshold = 10 + (1.0 - threshold) * 20  # 10-30 range
        color_mask = (deltaE > color_threshold).astype(np.uint8) * 255
        
        if np.sum(color_mask > 0) > img_area * 0.001:  # At least 0.1% pixels changed
            change_masks.append(color_mask)
    
    # 6. Enhanced morphological processing
    if change_masks:
        combined_mask = np.maximum.reduce(change_masks)
        
        # Adaptive morphological operations
        kernel_size = max(3, min(7, int(np.sqrt(img_area) / 200)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Clean up noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove very small noise
        if threshold > 0.8:  # Conservative mode
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, 
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    else:
        combined_mask = np.zeros_like(img1_cv, dtype=np.uint8)
    
    # 7. Smart contour detection and filtering
    contours = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Enhanced filtering with multiple criteria
    filtered_contours = []
    margin = max(5, int(min(img_height, img_width) * 0.01))
    
    # ROI setup
    roi_x, roi_y, roi_w, roi_h = (None, None, None, None)
    if roi and len(roi) == 4:
        roi_x, roi_y, roi_w, roi_h = roi
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Check if in ROI
        in_roi = False
        if roi_x is not None:
            in_roi = (x >= roi_x and y >= roi_y and x + w <= roi_x + roi_w and y + h <= roi_y + roi_h)
        
        # Adaptive thresholds based on sensitivity and image characteristics
        sensitivity_factor = 1.0 - threshold
        
        # Base requirements
        base_min_area = max(min_change_area, int(img_area * (0.0001 + sensitivity_factor * 0.0005)))
        base_min_size = max(min_change_size, int(min(img_height, img_width) * (0.005 + sensitivity_factor * 0.01)))
        
        # Relax for ROI
        if in_roi:
            min_area_thresh = max(50, int(base_min_area * 0.3))
            min_size_thresh = max(3, int(base_min_size * 0.4))
        else:
            min_area_thresh = base_min_area
            min_size_thresh = base_min_size
            
            # Skip margin areas unless significant
            if (x < margin or y < margin or x + w > img_width - margin or y + h > img_height - margin):
                if area < base_min_area * 3:
                    continue
        
        # Size and area filters
        if w < min_size_thresh or h < min_size_thresh or area < min_area_thresh:
            continue
            
        # Aspect ratio filter (avoid very thin lines unless in sensitive mode)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        max_aspect_ratio = 20 if threshold < 0.8 else 15
        if aspect_ratio > max_aspect_ratio:
            continue
            
        # Solidity filter (avoid very irregular shapes)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        min_solidity = 0.3 if threshold < 0.7 else 0.5
        if solidity < min_solidity:
            continue
            
        # Density check (avoid speckled regions)
        roi_mask = np.zeros_like(combined_mask)
        cv2.drawContours(roi_mask, [contour], -1, 255, -1)
        roi_pixels = combined_mask[roi_mask == 255]
        if len(roi_pixels) > 0:
            density = np.sum(roi_pixels > 0) / len(roi_pixels)
            if density < 0.3:  # Too sparse
                continue
        
        filtered_contours.append(contour)
    
    # 8. Intelligent contour merging with clustering
    if len(filtered_contours) > 1:
        # Use spatial clustering to group nearby changes
        centers = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            centers.append([x + w/2, y + h/2])
        
        if len(centers) > 1:
            centers = np.array(centers)
            # DBSCAN clustering to group nearby changes
            clustering = DBSCAN(eps=min(img_width, img_height) * 0.05, min_samples=1)
            labels = clustering.fit_predict(centers)
            
            # Group contours by cluster
            clustered_contours = {}
            for i, label in enumerate(labels):
                if label not in clustered_contours:
                    clustered_contours[label] = []
                clustered_contours[label].append(filtered_contours[i])
            
            # Merge contours within each cluster
            merged_contours = []
            for cluster_contours in clustered_contours.values():
                if len(cluster_contours) == 1:
                    merged_contours.append(cluster_contours[0])
                else:
                    # Merge multiple contours by creating bounding box
                    all_points = []
                    for contour in cluster_contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        all_points.extend([[x, y], [x+w, y+h]])
                    
                    all_points = np.array(all_points)
                    x_min, y_min = all_points.min(axis=0)
                    x_max, y_max = all_points.max(axis=0)
                    
                    # Create merged contour
                    merged_contour = np.array([
                        [[x_min, y_min]], [[x_max, y_min]], 
                        [[x_max, y_max]], [[x_min, y_max]]
                    ])
                    merged_contours.append(merged_contour)
        else:
            merged_contours = filtered_contours
    else:
        merged_contours = filtered_contours
    
    # 9. Final ranking and selection
    if merged_contours:
        # Score contours based on multiple factors
        scored_contours = []
        for contour in merged_contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calculate score based on size, position, and change intensity
            size_score = min(1.0, area / (img_area * 0.01))  # Larger is better, capped at 1%
            
            # Central position gets slight bonus
            center_x, center_y = x + w/2, y + h/2
            center_distance = np.sqrt((center_x - img_width/2)**2 + (center_y - img_height/2)**2)
            max_distance = np.sqrt((img_width/2)**2 + (img_height/2)**2)
            position_score = 1.0 - (center_distance / max_distance) * 0.2
            
            # Intensity score based on actual pixel differences in the region
            roi_diff = abs_diff[y:y+h, x:x+w]
            intensity_score = min(1.0, np.mean(roi_diff) / 100.0)
            
            total_score = size_score * 0.5 + position_score * 0.2 + intensity_score * 0.3
            scored_contours.append((total_score, contour))
        
        # Sort by score and limit number
        scored_contours.sort(key=lambda x: x[0], reverse=True)
        
        # Adaptive limit based on image characteristics
        base_limit = max(3, min(12, int(np.sqrt(img_area) / 150)))
        sensitivity_bonus = int((1.0 - threshold) * 5)
        max_changes = min(base_limit + sensitivity_bonus, 15)
        
        merged_contours = [contour for _, contour in scored_contours[:max_changes]]
    
    # 10. Fallback detection for edge cases
    if len(merged_contours) == 0 and threshold < 0.9:
        # Very sensitive fallback using simple absolute difference
        fallback_diff = cv2.absdiff(img1_cv, img2_cv)
        fallback_thresh = cv2.threshold(fallback_diff, 8, 255, cv2.THRESH_BINARY)[1]
        
        # Minimal morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fallback_thresh = cv2.morphologyEx(fallback_thresh, cv2.MORPH_CLOSE, kernel)
        
        fb_contours = cv2.findContours(fallback_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fb_contours = fb_contours[0] if len(fb_contours) == 2 else fb_contours[1]
        
        # Very relaxed filtering
        for contour in fb_contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if (w > 5 and h > 5 and area > 25 and 
                x > 2 and y > 2 and x + w < img_width - 2 and y + h < img_height - 2):
                merged_contours.append(contour)
                if len(merged_contours) >= 8:  # Limit fallback detections
                    break
    
    # 11. Generate enhanced annotations
    for contour in merged_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Enhanced change type detection
        change_type = _determine_change_type_enhanced(img1, img2, x, y, w, h)
        
        # Smart color selection based on change type
        if change_type == "removal":
            color = (255, 0, 0)  # Red
            draw_dashed_rectangle(draw, x, y, x + w, y + h, color, width=3)
        elif change_type == "addition":
            color = (0, 255, 0)  # Green
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        else:  # modification
            color = (255, 165, 0)  # Orange
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        # Adaptive circle placement
        circle_radius = max(12, min(20, int(min(w, h) * 0.15)))
        
        # Smart circle positioning to avoid overlap
        circle_positions = [
            (x + w + circle_radius + 3, y - 3),  # Top-right
            (x - circle_radius - 3, y - 3),      # Top-left  
            (x + w + circle_radius + 3, y + h + circle_radius + 3),  # Bottom-right
            (x - circle_radius - 3, y + h + circle_radius + 3)       # Bottom-left
        ]
        
        # Choose best position within image bounds
        circle_x, circle_y = circle_positions[0]  # Default
        for pos_x, pos_y in circle_positions:
            if (circle_radius <= pos_x <= img_width - circle_radius and
                circle_radius <= pos_y <= img_height - circle_radius):
                circle_x, circle_y = pos_x, pos_y
                break
        
        # Ensure circle is within bounds
        circle_x = max(circle_radius, min(circle_x, img_width - circle_radius))
        circle_y = max(circle_radius, min(circle_y, img_height - circle_radius))
        
        # Draw circle with improved styling
        draw.ellipse([circle_x - circle_radius, circle_y - circle_radius,
                     circle_x + circle_radius, circle_y + circle_radius], 
                    fill=color, outline=(255, 255, 255), width=2)
        
        # Enhanced text rendering
        number_text = str(change_number)
        try:
            text_bbox = draw.textbbox((0, 0), number_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            text_width, text_height = len(number_text) * 8, 12  # Fallback
        
        text_x = circle_x - text_width // 2
        text_y = circle_y - text_height // 2
        
        # Add text shadow for better visibility
        draw.text((text_x + 1, text_y + 1), number_text, fill=(0, 0, 0), font=font)
        draw.text((text_x, text_y), number_text, fill=(255, 255, 255), font=font)
        
        # Store change information
        changes.append({
            'number': change_number,
            'type': change_type,
            'bbox': (x, y, w, h),
            'description': get_change_description(change_type, change_number),
            'confidence': _calculate_change_confidence(img1, img2, x, y, w, h),
            'area': int(cv2.contourArea(contour))
        })
        
        change_number += 1
    
    # Return results with optional mask
    if return_mask:
        return img_annotated, changes, score, (combined_mask > 0).astype(np.uint8)
    return img_annotated, changes, score

def _determine_change_type_enhanced(img1, img2, x, y, w, h):
    """
    Enhanced change type determination using multiple analysis methods.
    """
    # Extract regions
    region1 = img1.crop((x, y, x + w, y + h))
    region2 = img2.crop((x, y, x + w, y + h))
    
    # Convert to arrays for analysis
    arr1 = np.array(region1.convert('RGB'))
    arr2 = np.array(region2.convert('RGB'))
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    
    # Multiple analysis metrics
    metrics = {}
    
    # 1. Brightness analysis
    metrics['mean1'] = np.mean(gray1)
    metrics['mean2'] = np.mean(gray2)
    metrics['brightness_diff'] = abs(metrics['mean2'] - metrics['mean1'])
    
    # 2. Texture analysis
    metrics['var1'] = np.var(gray1)
    metrics['var2'] = np.var(gray2)
    metrics['texture_diff'] = abs(metrics['var2'] - metrics['var1'])
    
    # 3. Edge analysis with multiple scales
    edges1_fine = cv2.Canny(gray1, 30, 100)
    edges2_fine = cv2.Canny(gray2, 30, 100)
    edges1_coarse = cv2.Canny(gray1, 80, 200)
    edges2_coarse = cv2.Canny(gray2, 80, 200)
    
    metrics['edge_density1'] = (np.sum(edges1_fine > 0) + np.sum(edges1_coarse > 0)) / (2 * w * h)
    metrics['edge_density2'] = (np.sum(edges2_fine > 0) + np.sum(edges2_coarse > 0)) / (2 * w * h)
    metrics['edge_diff'] = abs(metrics['edge_density2'] - metrics['edge_density1'])
    
    # 4. Color histogram analysis
    hist1 = cv2.calcHist([arr1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([arr2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    metrics['hist_correlation'] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 5. Gradient analysis
    grad_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
    
    grad_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
    
    metrics['gradient_diff'] = abs(np.mean(grad_mag2) - np.mean(grad_mag1))
    
    # Enhanced decision logic
    # Strong evidence for addition
    if (metrics['edge_density2'] > metrics['edge_density1'] + 0.03 and 
        metrics['mean2'] > metrics['mean1'] + 8 and
        metrics['var2'] > metrics['var1'] + 50):
        return "addition"
    
    # Strong evidence for removal
    elif (metrics['edge_density1'] > metrics['edge_density2'] + 0.03 and 
          metrics['mean1'] > metrics['mean2'] + 8 and
          metrics['var1'] > metrics['var2'] + 50):
        return "removal"
    
    # Brightness-based classification
    elif metrics['brightness_diff'] > 20:
        if metrics['hist_correlation'] < 0.7:  # Significant color change
            return "modification"
        elif metrics['mean2'] > metrics['mean1']:
            return "addition"
        else:
            return "removal"
    
    # Texture/structure changes
    elif (metrics['texture_diff'] > 200 or 
          metrics['edge_diff'] > 0.02 or 
          metrics['gradient_diff'] > 15):
        return "modification"
    
    # Default to modification for ambiguous cases
    return "modification"

def _calculate_change_confidence(img1, img2, x, y, w, h):
    """
    Calculate confidence score for detected change (0.0 to 1.0).
    """
    try:
        # Extract regions
        region1 = img1.crop((x, y, x + w, y + h))
        region2 = img2.crop((x, y, x + w, y + h))
        
        # Convert to arrays
        arr1 = np.array(region1.convert('RGB'))
        arr2 = np.array(region2.convert('RGB'))
        
        # Calculate multiple confidence factors
        factors = []
        
        # 1. Absolute difference magnitude
        abs_diff = np.mean(np.abs(arr1.astype(float) - arr2.astype(float)))
        factors.append(min(1.0, abs_diff / 100.0))
        
        # 2. SSIM difference (inverted)
        gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
        if gray1.shape == gray2.shape and min(gray1.shape) > 6:
            ssim_score = ssim(gray1, gray2, win_size=min(7, min(gray1.shape)//2*2-1), data_range=255)
            factors.append(1.0 - ssim_score)
        
        # 3. Edge difference significance
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        edge_diff_ratio = np.sum(np.abs(edges1.astype(float) - edges2.astype(float))) / (w * h * 255)
        factors.append(min(1.0, edge_diff_ratio * 5))
        
        # 4. Size-based confidence (larger changes are typically more reliable)
        area = w * h
        size_factor = min(1.0, area / 10000)  # Normalize by 100x100 pixel area
        factors.append(size_factor)
        
        # Combine factors with weights
        confidence = (factors[0] * 0.3 + 
                     factors[1] * 0.3 if len(factors) > 1 else factors[0] * 0.6) + \
                    (factors[2] * 0.2 if len(factors) > 2 else 0) + \
                    (factors[3] * 0.2 if len(factors) > 3 else 0)
        
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
        
    except Exception:
        return 0.5  # Default confidence for errors

def determine_change_type(img1, img2, x, y, w, h):
    """
    Legacy function - kept for backward compatibility.
    """
    return _determine_change_type_enhanced(img1, img2, x, y, w, h)

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

def detect_large_content_blocks(img1: Image.Image, img2: Image.Image, min_block_size=50000, return_mask=False):
    """
    Specialized detection for large content blocks like diagrams, screenshots, tables, etc.
    This is optimized for detecting significant visual additions or removals.
    Returns separate annotated images for original (showing removals) and updated (showing additions).
    """
    # Ensure same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    arr1 = np.array(img1.convert('RGB'))
    arr2 = np.array(img2.convert('RGB'))
    
    img_height, img_width = arr1.shape[:2]
    img_area = img_height * img_width
    
    # === LARGE CONTENT BLOCK DETECTION ===
    
    # 1. High-level structural difference using segmentation
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    
    # Apply strong gaussian blur to identify large regions
    blur_kernel = max(15, int(min(img_width, img_height) / 50))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
        
    blurred1 = cv2.GaussianBlur(gray1, (blur_kernel, blur_kernel), 0)
    blurred2 = cv2.GaussianBlur(gray2, (blur_kernel, blur_kernel), 0)
    
    # Find large-scale differences
    large_scale_diff = cv2.absdiff(blurred1, blurred2)
    
    # Use adaptive threshold for large content detection
    threshold_value = max(20, int(np.mean(large_scale_diff) + 0.5 * np.std(large_scale_diff)))
    _, block_mask = cv2.threshold(large_scale_diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological operations to consolidate regions
    # Large kernel to merge nearby changes into blocks
    consolidate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_kernel, blur_kernel))
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_CLOSE, consolidate_kernel)
    block_mask = cv2.morphologyEx(block_mask, cv2.MORPH_DILATE, consolidate_kernel)
    
    # 3. Find large contiguous regions
    contours = cv2.findContours(block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Filter for large blocks only
    large_blocks = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Must be a significant portion of the page
        area_ratio = area / img_area
        size_threshold = min_block_size / img_area
        
        # Accept if area is significant OR dimensions suggest a large block
        if (area_ratio > size_threshold or 
            (w > img_width * 0.3 and h > img_height * 0.2) or  # Wide horizontal blocks
            (w > img_width * 0.2 and h > img_height * 0.3)):   # Tall vertical blocks
            large_blocks.append(contour)
    
    # 4. Create annotated image showing large blocks
    img_annotated = Image.fromarray(arr2)
    draw = ImageDraw.Draw(img_annotated)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    changes = []
    change_number = 1
    
    for contour in large_blocks:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Analyze the content in this region to determine change type
        region1 = gray1[y:y+h, x:x+w]
        region2 = gray2[y:y+h, x:x+w]
        
        # Calculate content density (how much "stuff" is in each region)
        edges1 = cv2.Canny(region1, 30, 100)
        edges2 = cv2.Canny(region2, 30, 100)
        
        density1 = np.sum(edges1 > 0) / (w * h)
        density2 = np.sum(edges2 > 0) / (w * h)
        
        mean1 = np.mean(region1)
        mean2 = np.mean(region2)
        
        # Determine change type based on content analysis
        if density2 > density1 + 0.01 or mean2 < mean1 - 10:  # More content in region2
            change_type = "large_addition"
            color = (0, 200, 0)  # Bright green for large additions
            description = f"Large content block added (diagram/screenshot/table)"
        elif density1 > density2 + 0.01 or mean1 < mean2 - 10:  # More content in region1
            change_type = "large_removal"
            color = (200, 0, 0)  # Bright red for large removals
            description = f"Large content block removed (diagram/screenshot/table)"
        else:
            change_type = "large_modification"
            color = (200, 100, 0)  # Orange for large modifications
            description = f"Large content block modified (diagram/screenshot/table)"
        
        change_info = {
            'number': change_number,
            'type': change_type,
            'bbox': (x, y, w, h),
            'description': description,
            'area': int(area),
            'area_ratio': area / img_area,
            'content_density_1': float(density1),
            'content_density_2': float(density2),
            'color': color
        }
        
        changes.append(change_info)
        change_number += 1
    
    # Now create separate annotated images for original and updated documents
    img_original_annotated = Image.fromarray(arr1)  # Original image base
    img_updated_annotated = Image.fromarray(arr2)    # Updated image base
    
    draw_original = ImageDraw.Draw(img_original_annotated)
    draw_updated = ImageDraw.Draw(img_updated_annotated)
    
    # Draw highlights on appropriate images
    for change in changes:
        x, y, w, h = change['bbox']
        color = change['color']
        change_type = change['type']
        number = change['number']
        
        # Calculate circle position
        circle_radius = min(25, max(15, int(min(w, h) * 0.08)))
        circle_x = x + w + circle_radius + 5
        circle_y = y + circle_radius + 5
        circle_x = min(circle_x, img_width - circle_radius - 5)
        circle_y = min(circle_y, img_height - circle_radius - 5)
        
        # Draw on appropriate image based on change type
        if change_type == "large_removal":
            # Draw on ORIGINAL image to show what was removed
            draw_original.rectangle([x, y, x + w, y + h], outline=color, width=5)
            draw_original.ellipse([circle_x - circle_radius, circle_y - circle_radius,
                                 circle_x + circle_radius, circle_y + circle_radius], 
                                fill=color, outline=(255, 255, 255), width=3)
            
            # Draw number
            number_text = str(number)
            try:
                text_bbox = draw_original.textbbox((0, 0), number_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = len(number_text) * 12, 16
            
            text_x = circle_x - text_width // 2
            text_y = circle_y - text_height // 2
            draw_original.text((text_x + 2, text_y + 2), number_text, fill=(0, 0, 0), font=font)
            draw_original.text((text_x, text_y), number_text, fill=(255, 255, 255), font=font)
            
        elif change_type == "large_addition":
            # Draw on UPDATED image to show what was added
            draw_updated.rectangle([x, y, x + w, y + h], outline=color, width=5)
            draw_updated.ellipse([circle_x - circle_radius, circle_y - circle_radius,
                                circle_x + circle_radius, circle_y + circle_radius], 
                               fill=color, outline=(255, 255, 255), width=3)
            
            # Draw number
            number_text = str(number)
            try:
                text_bbox = draw_updated.textbbox((0, 0), number_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = len(number_text) * 12, 16
            
            text_x = circle_x - text_width // 2
            text_y = circle_y - text_height // 2
            draw_updated.text((text_x + 2, text_y + 2), number_text, fill=(0, 0, 0), font=font)
            draw_updated.text((text_x, text_y), number_text, fill=(255, 255, 255), font=font)
            
        else:  # modification - draw on both
            for draw_obj in [draw_original, draw_updated]:
                draw_obj.rectangle([x, y, x + w, y + h], outline=color, width=5)
                draw_obj.ellipse([circle_x - circle_radius, circle_y - circle_radius,
                                circle_x + circle_radius, circle_y + circle_radius], 
                               fill=color, outline=(255, 255, 255), width=3)
                
                number_text = str(number)
                try:
                    text_bbox = draw_obj.textbbox((0, 0), number_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except:
                    text_width, text_height = len(number_text) * 12, 16
                
                text_x = circle_x - text_width // 2
                text_y = circle_y - text_height // 2
                draw_obj.text((text_x + 2, text_y + 2), number_text, fill=(0, 0, 0), font=font)
                draw_obj.text((text_x, text_y), number_text, fill=(255, 255, 255), font=font)
    
    # Calculate similarity score
    total_changed_area = sum(cv2.contourArea(c) for c in large_blocks)
    similarity_score = 1.0 - (total_changed_area / img_area)
    
    if return_mask:
        return img_original_annotated, img_updated_annotated, changes, similarity_score, block_mask
    return img_original_annotated, img_updated_annotated, changes, similarity_score

def find_ultra_subtle_differences(img1: Image.Image, img2: Image.Image, return_mask: bool = False):
    """
    Ultra-sensitive detection for very subtle changes that standard algorithms miss.
    Designed for nearly identical images with minimal differences.
    """
    # Ensure same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to high-precision arrays
    arr1 = np.array(img1.convert('RGB')).astype(np.float64)
    arr2 = np.array(img2.convert('RGB')).astype(np.float64)
    
    img_height, img_width = arr1.shape[:2]
    img_area = img_height * img_width
    
    # === ULTRA-SENSITIVE PIPELINE ===
    
    # 1. High-precision absolute difference
    abs_diff = np.abs(arr1 - arr2)
    max_channel_diff = np.max(abs_diff, axis=2)
    mean_channel_diff = np.mean(abs_diff, axis=2)
    
    # 2. Very low threshold detection (detect even 1-2 pixel intensity changes)
    ultra_sensitive_masks = []
    
    # Detect any pixel with >1 intensity difference in any channel
    mask_1 = max_channel_diff > 1
    ultra_sensitive_masks.append(mask_1.astype(np.uint8) * 255)
    
    # Detect average channel differences >0.5
    mask_05 = mean_channel_diff > 0.5
    ultra_sensitive_masks.append(mask_05.astype(np.uint8) * 255)
    
    # 3. Perceptual difference using LAB color space
    lab1 = cv2.cvtColor(arr1.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    lab2 = cv2.cvtColor(arr2.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    
    # Delta E calculation with very low threshold
    dL = lab1[:,:,0] - lab2[:,:,0]
    dA = lab1[:,:,1] - lab2[:,:,1]
    dB = lab1[:,:,2] - lab2[:,:,2]
    deltaE = np.sqrt(dL**2 + dA**2 + dB**2)
    
    # Even tiny perceptual differences
    perceptual_mask = (deltaE > 0.5).astype(np.uint8) * 255
    ultra_sensitive_masks.append(perceptual_mask)
    
    # 4. Gradient-based detection for texture changes
    gray1 = cv2.cvtColor(arr1.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Multi-scale gradient analysis
    gradient_masks = []
    for sigma in [0.5, 1.0, 1.5]:
        g1_smooth = cv2.GaussianBlur(gray1.astype(np.float32), (0, 0), sigma)
        g2_smooth = cv2.GaussianBlur(gray2.astype(np.float32), (0, 0), sigma)
        
        grad_x1 = cv2.Sobel(g1_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y1 = cv2.Sobel(g1_smooth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
        
        grad_x2 = cv2.Sobel(g2_smooth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y2 = cv2.Sobel(g2_smooth, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
        
        grad_diff = np.abs(grad_mag1 - grad_mag2)
        grad_mask = (grad_diff > 0.5).astype(np.uint8) * 255
        gradient_masks.append(grad_mask)
    
    if gradient_masks:
        combined_gradient = np.maximum.reduce(gradient_masks)
        ultra_sensitive_masks.append(combined_gradient)
    
    # 5. Statistical analysis for local changes
    # Use sliding window to detect local statistical changes
    window_size = 5
    stat_mask = np.zeros_like(gray1, dtype=np.uint8)
    
    for i in range(0, img_height - window_size, 2):
        for j in range(0, img_width - window_size, 2):
            # Extract windows
            win1 = gray1[i:i+window_size, j:j+window_size]
            win2 = gray2[i:i+window_size, j:j+window_size]
            
            # Compare statistical properties
            mean_diff = abs(np.mean(win1) - np.mean(win2))
            var_diff = abs(np.var(win1) - np.var(win2))
            
            if mean_diff > 0.5 or var_diff > 1.0:
                stat_mask[i:i+window_size, j:j+window_size] = 255
    
    ultra_sensitive_masks.append(stat_mask)
    
    # 6. Combine all ultra-sensitive detections
    if ultra_sensitive_masks:
        combined_mask = np.maximum.reduce(ultra_sensitive_masks)
    else:
        combined_mask = np.zeros_like(gray1, dtype=np.uint8)
    
    # 7. Minimal morphological processing (preserve small changes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # 8. Find contours with very relaxed filtering
    contours = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Ultra-relaxed filtering - detect even single pixels
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Accept any change larger than 1 pixel
        if area >= 1 and w >= 1 and h >= 1:
            filtered_contours.append(contour)
    
    # 9. Create annotated image
    img_annotated = Image.fromarray(arr2.astype(np.uint8))
    draw = ImageDraw.Draw(img_annotated)
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    changes = []
    change_number = 1
    
    # 10. Generate annotations for all detected changes
    for contour in filtered_contours[:50]:  # Limit to prevent overcrowding
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Calculate change intensity in this region
        roi_diff = max_channel_diff[y:y+h, x:x+w]
        avg_intensity = np.mean(roi_diff)
        max_intensity = np.max(roi_diff)
        
        # Determine change type based on intensity
        if max_intensity < 2:
            change_type = "micro_change"
            color = (255, 255, 0)  # Yellow for micro changes
        elif max_intensity < 10:
            change_type = "subtle_change"
            color = (255, 165, 0)  # Orange for subtle changes
        else:
            change_type = "visible_change"
            color = (255, 0, 0)  # Red for visible changes
        
        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        
        # Draw circle with number
        circle_radius = max(8, min(15, int(min(w, h) * 0.2)))
        circle_x = x + w + circle_radius + 2
        circle_y = y - 2
        
        # Keep circle in bounds
        circle_x = min(circle_x, img_width - circle_radius - 2)
        circle_y = max(circle_y, circle_radius + 2)
        
        draw.ellipse([circle_x - circle_radius, circle_y - circle_radius,
                     circle_x + circle_radius, circle_y + circle_radius], 
                    fill=color, outline=(255, 255, 255), width=1)
        
        # Draw number
        number_text = str(change_number)
        try:
            text_bbox = draw.textbbox((0, 0), number_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except:
            text_width, text_height = len(number_text) * 6, 10
        
        text_x = circle_x - text_width // 2
        text_y = circle_y - text_height // 2
        
        draw.text((text_x, text_y), number_text, fill=(255, 255, 255), font=font)
        
        # Store change information
        changes.append({
            'number': change_number,
            'type': change_type,
            'bbox': (x, y, w, h),
            'description': f'{change_type.replace("_", " ").title()} detected (Max intensity: {max_intensity:.1f})',
            'avg_intensity': float(avg_intensity),
            'max_intensity': float(max_intensity),
            'area': int(area)
        })
        
        change_number += 1
    
    # Calculate overall similarity score
    total_different_pixels = np.sum(combined_mask > 0)
    similarity_score = 1.0 - (total_different_pixels / img_area)
    
    if return_mask:
        return img_annotated, changes, similarity_score, combined_mask
    return img_annotated, changes, similarity_score
