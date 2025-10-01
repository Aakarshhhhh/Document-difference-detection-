import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from sklearn.cluster import DBSCAN

def detect_missing_images_enhanced(img1: Image.Image, img2: Image.Image, 
                                 min_missing_area=5000, 
                                 sensitivity=0.7,
                                 return_mask=False):
    """
    Enhanced detection specifically designed for missing images, diagrams, screenshots, etc.
    
    This function focuses on detecting large visual elements that are present in one image
    but completely absent in the other, which is common when pictures/diagrams are removed
    from documents.
    
    Args:
        img1: First image (original)
        img2: Second image (updated) 
        min_missing_area: Minimum area in pixels for a missing element to be detected
        sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)
        return_mask: Whether to return the detection mask
    
    Returns:
        original_annotated: Original image with missing elements highlighted in red
        updated_annotated: Updated image with added elements highlighted in green
        changes: List of detected changes with metadata
        similarity_score: Overall similarity score
    """
    # Ensure same size
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to numpy arrays
    arr1 = np.array(img1.convert('RGB'))
    arr2 = np.array(img2.convert('RGB'))
    
    img_height, img_width = arr1.shape[:2]
    img_area = img_height * img_width
    
    # === MISSING IMAGE DETECTION PIPELINE ===
    
    # 1. Convert to grayscale for structural analysis
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    
    # 2. Enhanced preprocessing to identify structured content
    # Apply adaptive histogram equalization to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced1 = clahe.apply(gray1)
    enhanced2 = clahe.apply(gray2)
    
    # 3. Multi-scale difference detection
    missing_masks = []
    
    # Scale 1: Full resolution for fine details
    diff_full = cv2.absdiff(enhanced1, enhanced2)
    
    # Adaptive threshold based on sensitivity
    base_threshold = int(15 + (1.0 - sensitivity) * 25)  # 15-40 range
    _, mask_full = cv2.threshold(diff_full, base_threshold, 255, cv2.THRESH_BINARY)
    missing_masks.append(mask_full)
    
    # Scale 2: Half resolution for medium structures
    h_half, w_half = img_height // 2, img_width // 2
    gray1_half = cv2.resize(enhanced1, (w_half, h_half))
    gray2_half = cv2.resize(enhanced2, (w_half, h_half))
    diff_half = cv2.absdiff(gray1_half, gray2_half)
    _, mask_half = cv2.threshold(diff_half, base_threshold, 255, cv2.THRESH_BINARY)
    mask_half_full = cv2.resize(mask_half, (img_width, img_height))
    missing_masks.append(mask_half_full)
    
    # Scale 3: Quarter resolution for large structures
    h_quarter, w_quarter = img_height // 4, img_width // 4
    gray1_quarter = cv2.resize(enhanced1, (w_quarter, h_quarter))
    gray2_quarter = cv2.resize(enhanced2, (w_quarter, h_quarter))
    diff_quarter = cv2.absdiff(gray1_quarter, gray2_quarter)
    _, mask_quarter = cv2.threshold(diff_quarter, base_threshold, 255, cv2.THRESH_BINARY)
    mask_quarter_full = cv2.resize(mask_quarter, (img_width, img_height))
    missing_masks.append(mask_quarter_full)
    
    # 4. Edge-based detection for structured content
    # Detect edges in both images
    edges1 = cv2.Canny(enhanced1, 30, 100)
    edges2 = cv2.Canny(enhanced2, 30, 100)
    
    # Find areas where edges disappeared or appeared
    edge_diff = cv2.absdiff(edges1, edges2)
    
    # Dilate to connect nearby edge changes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_diff_dilated = cv2.dilate(edge_diff, kernel, iterations=2)
    missing_masks.append(edge_diff_dilated)
    
    # 5. Texture-based detection
    # Calculate local standard deviation (texture measure)
    def calculate_local_std(img, window_size=15):
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        img_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
        img_sq_mean = cv2.filter2D((img.astype(np.float32))**2, -1, kernel)
        img_std = np.sqrt(np.maximum(0, img_sq_mean - img_mean**2))
        return img_std.astype(np.uint8)
    
    texture1 = calculate_local_std(enhanced1)
    texture2 = calculate_local_std(enhanced2)
    texture_diff = cv2.absdiff(texture1, texture2)
    
    # Threshold texture differences
    texture_threshold = int(10 + (1.0 - sensitivity) * 15)
    _, texture_mask = cv2.threshold(texture_diff, texture_threshold, 255, cv2.THRESH_BINARY)
    missing_masks.append(texture_mask)
    
    # 6. Gradient magnitude analysis
    grad_x1 = cv2.Sobel(enhanced1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(enhanced1, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag1 = np.sqrt(grad_x1**2 + grad_y1**2)
    
    grad_x2 = cv2.Sobel(enhanced2, cv2.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(enhanced2, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag2 = np.sqrt(grad_x2**2 + grad_y2**2)
    
    grad_diff = np.abs(grad_mag1 - grad_mag2)
    grad_threshold = 20 + (1.0 - sensitivity) * 20
    grad_mask = (grad_diff > grad_threshold).astype(np.uint8) * 255
    missing_masks.append(grad_mask)
    
    # 7. Combine all detection masks
    combined_mask = np.maximum.reduce(missing_masks)
    
    # 8. Morphological operations to consolidate regions
    # Use larger kernels for missing image detection since we want to find big blocks
    morph_kernel_size = max(7, int(np.sqrt(img_area) / 100))
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    
    # Close gaps within regions
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, morph_kernel)
    
    # Fill holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    
    # Remove small noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, 
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    # 9. Find contours and filter for missing images
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for missing image characteristics
    missing_image_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Must be larger than minimum area
        if area < min_missing_area:
            continue
        
        # Should have reasonable aspect ratio (not extremely thin lines)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > 10:  # Skip very thin regions
            continue
        
        # Should have decent solidity (not too fragmented)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.3:  # Skip very fragmented regions
            continue
        
        # Should occupy a reasonable portion of the image
        area_ratio = area / img_area
        if area_ratio > 0.8:  # Skip if it covers most of the image
            continue
        
        missing_image_contours.append(contour)
    
    # 10. Analyze content to determine if removal or addition
    changes = []
    change_number = 1
    
    for contour in missing_image_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract regions from both images
        region1 = gray1[y:y+h, x:x+w]
        region2 = gray2[y:y+h, x:x+w]
        
        # Calculate various metrics to determine change type
        mean1, std1 = cv2.meanStdDev(region1)
        mean2, std2 = cv2.meanStdDev(region2)
        
        mean1, std1 = mean1[0][0], std1[0][0]
        mean2, std2 = mean2[0][0], std2[0][0]
        
        # Calculate edge density (indicates structured content like images)
        edges1_region = cv2.Canny(region1, 30, 100)
        edges2_region = cv2.Canny(region2, 30, 100)
        edge_density1 = np.sum(edges1_region > 0) / (w * h)
        edge_density2 = np.sum(edges2_region > 0) / (w * h)
        
        # Determine change type based on content analysis
        if edge_density1 > edge_density2 + 0.01 or std1 > std2 + 5:
            # More content/structure in image 1 -> removal
            change_type = "missing_image"
            description = f"Missing visual element (image/diagram/chart)"
        elif edge_density2 > edge_density1 + 0.01 or std2 > std1 + 5:
            # More content/structure in image 2 -> addition
            change_type = "added_image"
            description = f"Added visual element (image/diagram/chart)"
        else:
            # Ambiguous - call it modification
            change_type = "modified_image"
            description = f"Modified visual element (image/diagram/chart)"
        
        # Calculate confidence based on difference magnitude
        mean_diff = abs(mean1 - mean2)
        std_diff = abs(std1 - std2)
        edge_diff = abs(edge_density1 - edge_density2)
        
        confidence = min(1.0, (mean_diff / 50.0 + std_diff / 20.0 + edge_diff * 100) / 3)
        confidence = max(0.3, confidence)  # Minimum confidence
        
        change_info = {
            'number': change_number,
            'type': change_type,
            'bbox': (x, y, w, h),
            'description': description,
            'area': int(cv2.contourArea(contour)),
            'confidence': confidence,
            'edge_density_1': float(edge_density1),
            'edge_density_2': float(edge_density2),
            'mean_intensity_1': float(mean1),
            'mean_intensity_2': float(mean2),
            'texture_std_1': float(std1),
            'texture_std_2': float(std2)
        }
        
        changes.append(change_info)
        change_number += 1
    
    # 11. Create annotated images
    img_original_annotated = Image.fromarray(arr1)
    img_updated_annotated = Image.fromarray(arr2)
    
    draw_original = ImageDraw.Draw(img_original_annotated)
    draw_updated = ImageDraw.Draw(img_updated_annotated)
    
    # Load font
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw annotations
    for change in changes:
        x, y, w, h = change['bbox']
        change_type = change['type']
        number = change['number']
        
        # Color coding
        if change_type == "missing_image":
            color = (255, 0, 0)  # Red for missing
            highlight_original = True
            highlight_updated = False
        elif change_type == "added_image":
            color = (0, 200, 0)  # Green for added
            highlight_original = False
            highlight_updated = True
        else:  # modified
            color = (255, 165, 0)  # Orange for modified
            highlight_original = True
            highlight_updated = True
        
        # Draw rectangles and labels
        for draw_obj, should_highlight in [(draw_original, highlight_original), 
                                         (draw_updated, highlight_updated)]:
            if should_highlight:
                # Draw rectangle
                draw_obj.rectangle([x, y, x + w, y + h], outline=color, width=4)
                
                # Draw circle with number
                circle_radius = min(20, max(12, int(min(w, h) * 0.1)))
                circle_x = x + w + circle_radius + 5
                circle_y = y + circle_radius + 5
                
                # Keep circle in bounds
                circle_x = min(circle_x, img_width - circle_radius - 5)
                circle_y = min(circle_y, img_height - circle_radius - 5)
                circle_x = max(circle_x, circle_radius + 5)
                circle_y = max(circle_y, circle_radius + 5)
                
                draw_obj.ellipse([circle_x - circle_radius, circle_y - circle_radius,
                                circle_x + circle_radius, circle_y + circle_radius], 
                               fill=color, outline=(255, 255, 255), width=2)
                
                # Draw number
                number_text = str(number)
                try:
                    text_bbox = draw_obj.textbbox((0, 0), number_text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except:
                    text_width, text_height = len(number_text) * 10, 14
                
                text_x = circle_x - text_width // 2
                text_y = circle_y - text_height // 2
                
                draw_obj.text((text_x + 1, text_y + 1), number_text, fill=(0, 0, 0), font=font)
                draw_obj.text((text_x, text_y), number_text, fill=(255, 255, 255), font=font)
    
    # Calculate similarity score
    total_changed_area = sum(change['area'] for change in changes)
    similarity_score = 1.0 - (total_changed_area / img_area)
    
    if return_mask:
        return img_original_annotated, img_updated_annotated, changes, similarity_score, combined_mask
    return img_original_annotated, img_updated_annotated, changes, similarity_score


def detect_missing_visual_content(img1: Image.Image, img2: Image.Image, **kwargs):
    """
    Wrapper function with a more descriptive name.
    Alias for detect_missing_images_enhanced.
    """
    return detect_missing_images_enhanced(img1, img2, **kwargs)


def analyze_visual_content_regions(img: Image.Image, min_region_size=2000):
    """
    Analyze an image to identify regions that likely contain visual content
    (images, diagrams, charts, etc.) vs text regions.
    
    This can be useful for understanding what types of content are in an image.
    """
    arr = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(gray)
    
    # Calculate local statistics
    def calculate_local_stats(img, window_size=15):
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        img_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
        img_sq_mean = cv2.filter2D((img.astype(np.float32))**2, -1, kernel)
        img_std = np.sqrt(np.maximum(0, img_sq_mean - img_mean**2))
        return img_mean, img_std
    
    local_mean, local_std = calculate_local_stats(enhanced)
    
    # Detect edges
    edges = cv2.Canny(enhanced, 30, 100)
    
    # Identify potential visual content regions (high texture variation)
    visual_content_mask = (local_std > 15).astype(np.uint8) * 255
    
    # Morphological operations to consolidate regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    visual_content_mask = cv2.morphologyEx(visual_content_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(visual_content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and analyze regions
    visual_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_region_size:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate region characteristics
            region = enhanced[y:y+h, x:x+w]
            region_mean = np.mean(region)
            region_std = np.std(region)
            
            # Edge density in region
            region_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(region_edges > 0) / (w * h)
            
            # Classify region type
            if edge_density > 0.05 and region_std > 20:
                region_type = "complex_visual"  # Likely image, diagram, chart
            elif edge_density > 0.02:
                region_type = "structured_content"  # Likely table, formatted text
            else:
                region_type = "simple_content"  # Likely plain text
            
            visual_regions.append({
                'bbox': (x, y, w, h),
                'area': int(area),
                'type': region_type,
                'edge_density': float(edge_density),
                'mean_intensity': float(region_mean),
                'texture_std': float(region_std)
            })
    
    return visual_regions