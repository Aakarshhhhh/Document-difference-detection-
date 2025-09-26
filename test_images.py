import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from utils.image_comparison import find_image_differences

def download_and_save_image(url, filename):
    """Download image from URL and save locally"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(filename)
            print(f"Saved {filename}")
            return img
        else:
            print(f"Failed to download image: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def analyze_image_differences(img1_path, img2_path):
    """Analyze differences between two images with various settings"""
    try:
        # Load images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        print(f"Image 1 size: {img1.size}")
        print(f"Image 2 size: {img2.size}")
        
        # Test with different sensitivity settings
        sensitivity_levels = [
            ("ultra", "Ultra Sensitive (New Algorithm)"),
            (0.5, "Ultra Sensitive (Standard)"),
            (0.7, "Sensitive"), 
            (0.8, "Balanced"),
            (0.9, "Conservative")
        ]
        
        for threshold, name in sensitivity_levels:
            print(f"\n=== Testing {name} (threshold={threshold}) ===")
            
            try:
                # Run detection
                if threshold == "ultra":
                    from utils.image_comparison import find_ultra_subtle_differences
                    result_img, changes, ssim_score = find_ultra_subtle_differences(img1, img2)
                else:
                    result_img, changes, ssim_score = find_image_differences(
                        img1, img2,
                        threshold=threshold,
                        min_change_size=5,
                        min_change_area=100,
                        enable_edge_detection=True,
                        enable_color_detection=True
                    )
                
                print(f"SSIM Score: {ssim_score:.4f}")
                print(f"Changes detected: {len(changes)}")
                
                for i, change in enumerate(changes):
                    print(f"  Change {i+1}: {change['type']} at {change['bbox']} - {change['description']}")
                    if 'confidence' in change:
                        print(f"    Confidence: {change['confidence']:.2f}")
                    if 'max_intensity' in change:
                        print(f"    Max Intensity: {change['max_intensity']:.1f}")
                    if 'area' in change:
                        print(f"    Area: {change['area']} pixels")
                
                # Save result
                result_filename = f"result_{name.lower().replace(' ', '_')}.png"
                result_img.save(result_filename)
                print(f"Saved result to {result_filename}")
                
            except Exception as e:
                print(f"Error in detection: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"Error loading images: {e}")

def manual_difference_check(img1_path, img2_path):
    """Manual pixel-by-pixel difference analysis"""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print("Could not load images for manual check")
            return
            
        # Ensure same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Calculate absolute difference
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Statistics
        total_pixels = gray_diff.shape[0] * gray_diff.shape[1]
        changed_pixels = np.sum(gray_diff > 5)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        print(f"\n=== Manual Analysis ===")
        print(f"Total pixels: {total_pixels}")
        print(f"Changed pixels (>5 intensity): {changed_pixels}")
        print(f"Change percentage: {change_percentage:.4f}%")
        print(f"Max difference: {np.max(gray_diff)}")
        print(f"Mean difference: {np.mean(gray_diff):.2f}")
        print(f"Std difference: {np.std(gray_diff):.2f}")
        
        # Save difference visualization
        cv2.imwrite("manual_diff.png", gray_diff)
        cv2.imwrite("manual_diff_enhanced.png", gray_diff * 10)  # Enhanced for visibility
        
        # Threshold analysis
        thresholds = [1, 5, 10, 20, 30]
        for thresh in thresholds:
            changed = np.sum(gray_diff > thresh)
            pct = (changed / total_pixels) * 100
            print(f"Pixels changed > {thresh}: {changed} ({pct:.4f}%)")
            
    except Exception as e:
        print(f"Error in manual analysis: {e}")

if __name__ == "__main__":
    # For testing, let's create some sample images or load from file
    # Since I can't directly access the images you uploaded, I'll create a test framework
    
    print("Image Difference Detection Test")
    print("=" * 50)
    
    # You would replace these with your actual image paths
    img1_path = "test_image1.png"  # Replace with your first image path
    img2_path = "test_image2.png"  # Replace with your second image path
    
    # Check if images exist
    import os
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        analyze_image_differences(img1_path, img2_path)
        manual_difference_check(img1_path, img2_path)
    else:
        print(f"Please save your test images as:")
        print(f"  {img1_path}")
        print(f"  {img2_path}")
        print("Then run this script again.")