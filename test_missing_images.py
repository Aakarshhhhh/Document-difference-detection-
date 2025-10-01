#!/usr/bin/env python3
"""
Diagnostic test script for missing image detection.
This script helps identify why missing pictures/images aren't being highlighted.
"""
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.image_comparison import (
    find_image_differences, 
    detect_large_content_blocks,
    find_ultra_subtle_differences
)

def create_test_images_with_missing_picture():
    """Create test images - one with a picture, one without."""
    print("Creating test images...")
    
    # Create base document image (white background with text-like content)
    width, height = 800, 1000
    
    # Image 1: Document WITH a picture
    img1 = Image.new('RGB', (width, height), 'white')
    draw1 = ImageDraw.Draw(img1)
    
    # Add some text-like horizontal lines
    for i in range(6):
        y = 50 + i * 30
        draw1.rectangle([50, y, 700, y + 15], fill='black')
    
    # Add a PICTURE/IMAGE in the middle (simulate screenshot/diagram)
    picture_x, picture_y = 100, 250
    picture_w, picture_h = 500, 300
    
    # Draw picture background (grayish to simulate an image)
    draw1.rectangle([picture_x, picture_y, picture_x + picture_w, picture_y + picture_h], 
                   fill=(240, 240, 245), outline=(150, 150, 150), width=2)
    
    # Add some visual content inside the picture (simulate graph/chart elements)
    # Add title area
    draw1.rectangle([picture_x + 20, picture_y + 20, picture_x + picture_w - 20, picture_y + 50], 
                   fill=(220, 220, 225))
    
    # Add some "chart bars" or visual elements
    for i in range(5):
        bar_x = picture_x + 50 + i * 80
        bar_height = 30 + i * 15
        draw1.rectangle([bar_x, picture_y + picture_h - 50 - bar_height, 
                        bar_x + 60, picture_y + picture_h - 50], 
                       fill=(100, 150, 200))
    
    # Add some lines (like graph axes)
    draw1.line([picture_x + 30, picture_y + 70, picture_x + 30, picture_y + picture_h - 30], 
               fill=(80, 80, 80), width=2)
    draw1.line([picture_x + 30, picture_y + picture_h - 30, picture_x + picture_w - 30, picture_y + picture_h - 30], 
               fill=(80, 80, 80), width=2)
    
    # Add more text after the picture
    for i in range(4):
        y = picture_y + picture_h + 50 + i * 30
        if y < height - 50:
            draw1.rectangle([50, y, 650, y + 15], fill='black')
    
    # Image 2: Document WITHOUT the picture (missing image)
    img2 = Image.new('RGB', (width, height), 'white')
    draw2 = ImageDraw.Draw(img2)
    
    # Add the same text-like horizontal lines
    for i in range(6):
        y = 50 + i * 30
        draw2.rectangle([50, y, 700, y + 15], fill='black')
    
    # NOTE: We deliberately DON'T add the picture here - it's missing!
    
    # Add the text after where the picture would be
    for i in range(4):
        y = picture_y + picture_h + 50 + i * 30
        if y < height - 50:
            draw2.rectangle([50, y, 650, y + 15], fill='black')
    
    return img1, img2, (picture_x, picture_y, picture_w, picture_h)

def run_diagnostic_tests(img1, img2, expected_missing_area):
    """Run various detection algorithms and analyze their performance."""
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST RESULTS")
    print("="*60)
    
    results = {}
    
    # Test 1: Regular image difference detection with various sensitivity levels
    print("\n1. Testing regular image difference detection:")
    print("-" * 50)
    
    sensitivity_levels = [
        (0.5, "Ultra Sensitive"),
        (0.7, "High Sensitivity"), 
        (0.8, "Balanced"),
        (0.9, "Conservative")
    ]
    
    for threshold, name in sensitivity_levels:
        try:
            diff_img, changes, ssim_score = find_image_differences(
                img1, img2,
                threshold=threshold,
                min_change_size=5,
                min_change_area=1000,  # Lower area for better detection
                enable_edge_detection=True,
                enable_color_detection=True
            )
            
            results[f'regular_{threshold}'] = {
                'changes': len(changes),
                'ssim': ssim_score,
                'method': name
            }
            
            print(f"  {name:20} (t={threshold}): {len(changes):2d} changes detected, SSIM: {ssim_score:.3f}")
            
            # Save result for inspection
            diff_img.save(f"diagnostic_regular_{name.lower().replace(' ', '_')}.png")
            
        except Exception as e:
            print(f"  {name:20} (t={threshold}): ERROR - {e}")
            results[f'regular_{threshold}'] = {'changes': 0, 'error': str(e)}
    
    # Test 2: Large content block detection (should be best for missing images)
    print("\n2. Testing large content block detection:")
    print("-" * 50)
    
    try:
        original_annotated, updated_annotated, lb_changes, lb_similarity = detect_large_content_blocks(
            img1, img2, min_block_size=20000  # Lower threshold
        )
        
        results['large_blocks'] = {
            'changes': len(lb_changes),
            'similarity': lb_similarity,
            'types': [c.get('type', 'unknown') for c in lb_changes]
        }
        
        print(f"  Large Block Detection: {len(lb_changes)} changes detected, Similarity: {lb_similarity:.3f}")
        
        for i, change in enumerate(lb_changes):
            change_type = change.get('type', 'unknown')
            bbox = change.get('bbox', 'unknown')
            area = change.get('area', 0)
            print(f"    Change {i+1}: {change_type} at {bbox}, area: {area} pixels")
        
        # Save results
        original_annotated.save("diagnostic_large_blocks_original.png")
        updated_annotated.save("diagnostic_large_blocks_updated.png")
        
    except Exception as e:
        print(f"  Large Block Detection: ERROR - {e}")
        results['large_blocks'] = {'changes': 0, 'error': str(e)}
    
    # Test 3: Ultra-sensitive detection
    print("\n3. Testing ultra-sensitive detection:")
    print("-" * 50)
    
    try:
        ultra_img, ultra_changes, ultra_similarity = find_ultra_subtle_differences(img1, img2)
        
        results['ultra_sensitive'] = {
            'changes': len(ultra_changes),
            'similarity': ultra_similarity
        }
        
        print(f"  Ultra Sensitive: {len(ultra_changes)} changes detected, Similarity: {ultra_similarity:.3f}")
        
        # Save result
        ultra_img.save("diagnostic_ultra_sensitive.png")
        
    except Exception as e:
        print(f"  Ultra Sensitive: ERROR - {e}")
        results['ultra_sensitive'] = {'changes': 0, 'error': str(e)}
    
    # Test 4: Manual pixel difference analysis
    print("\n4. Manual pixel difference analysis:")
    print("-" * 50)
    
    try:
        # Convert to numpy arrays for manual analysis
        arr1 = np.array(img1.convert('RGB'))
        arr2 = np.array(img2.convert('RGB'))
        
        # Calculate absolute difference
        abs_diff = np.abs(arr1.astype(int) - arr2.astype(int))
        gray_diff = np.mean(abs_diff, axis=2)  # Convert to grayscale difference
        
        # Statistics
        total_pixels = gray_diff.shape[0] * gray_diff.shape[1]
        changed_pixels = np.sum(gray_diff > 5)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Find the area with maximum difference (should be where the picture was)
        max_diff_value = np.max(gray_diff)
        max_diff_location = np.unravel_index(np.argmax(gray_diff), gray_diff.shape)
        
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Changed pixels (>5 intensity): {changed_pixels:,}")
        print(f"  Change percentage: {change_percentage:.3f}%")
        print(f"  Max difference value: {max_diff_value:.1f}")
        print(f"  Max difference location: row {max_diff_location[0]}, col {max_diff_location[1]}")
        
        # Save difference visualization
        diff_visual = (gray_diff * 3).astype(np.uint8)  # Enhance for visibility
        diff_image = Image.fromarray(diff_visual, mode='L')
        diff_image.save("diagnostic_manual_diff.png")
        
        results['manual'] = {
            'changed_pixels': changed_pixels,
            'change_percentage': change_percentage,
            'max_diff': max_diff_value,
            'max_location': max_diff_location
        }
        
    except Exception as e:
        print(f"  Manual analysis: ERROR - {e}")
        results['manual'] = {'error': str(e)}
    
    return results

def analyze_expected_vs_actual(results, expected_area):
    """Analyze why detection might be failing."""
    print("\n" + "="*60)
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    expected_x, expected_y, expected_w, expected_h = expected_area
    print(f"\nExpected missing image area: ({expected_x}, {expected_y}) size {expected_w}x{expected_h}")
    print(f"Expected area: {expected_w * expected_h:,} pixels")
    
    # Check which method worked best
    best_method = None
    max_changes = 0
    
    for method_name, result in results.items():
        if 'changes' in result and result['changes'] > max_changes:
            max_changes = result['changes']
            best_method = method_name
    
    print(f"\nBest performing method: {best_method} with {max_changes} changes detected")
    
    # Recommendations
    print("\nRECOMMENDations:")
    print("-" * 30)
    
    if results.get('large_blocks', {}).get('changes', 0) > 0:
        print("✅ Large block detection is working - this should detect missing images")
        print("   Enable 'Large Block Detection' in your app settings")
    elif results.get('manual', {}).get('changed_pixels', 0) > 10000:
        print("⚠️  Manual analysis shows significant differences but algorithms missed them")
        print("   Try lowering the sensitivity thresholds:")
        print("   - Set threshold to 0.5 or lower")
        print("   - Set min_block_size to 10000 or lower")
        print("   - Ensure 'Large Block Detection' is enabled")
    elif any(results.get(f'regular_{t}', {}).get('changes', 0) > 0 for t in [0.5, 0.7, 0.8, 0.9]):
        print("✅ Regular detection found changes - try using lower thresholds")
        print("   Set your app to 'Ultra Sensitive' or 'High Sensitivity' mode")
    else:
        print("❌ No method successfully detected the missing image")
        print("   This suggests a problem with the algorithms or test setup")
    
    # Specific parameter recommendations
    print("\nSuggested app settings for missing image detection:")
    print("- Detection Mode: Ultra Sensitive")
    print("- Enable Large Block Detection: YES")
    print("- Minimum Block Size: 10,000 - 20,000 pixels")
    print("- Threshold: 0.5 - 0.7")
    print("- Enable Edge Detection: YES")
    print("- Enable Color Detection: YES")

def main():
    print("MISSING IMAGE DETECTION DIAGNOSTIC TEST")
    print("="*60)
    print("This test creates two images: one with a picture, one without.")
    print("It then runs various detection algorithms to see which ones")
    print("successfully identify the missing picture.")
    
    # Create test images
    img1, img2, expected_area = create_test_images_with_missing_picture()
    
    # Save test images for inspection
    img1.save("diagnostic_test_with_picture.png")
    img2.save("diagnostic_test_without_picture.png")
    print(f"\nTest images saved:")
    print(f"  - diagnostic_test_with_picture.png (has the picture)")
    print(f"  - diagnostic_test_without_picture.png (missing picture)")
    
    # Run diagnostic tests
    results = run_diagnostic_tests(img1, img2, expected_area)
    
    # Analyze results and provide recommendations
    analyze_expected_vs_actual(results, expected_area)
    
    print(f"\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print("Check the generated PNG files to visually verify the results.")
    print("The files starting with 'diagnostic_' show the detection results.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()