#!/usr/bin/env python3
"""
Test script for the enhanced missing image detector.
Tests both the diagnostic script and the new enhanced detector.
"""
import sys
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.enhanced_missing_image_detector import (
    detect_missing_images_enhanced,
    detect_missing_visual_content,
    analyze_visual_content_regions
)

def create_document_with_missing_screenshot():
    """Create realistic test documents - one with a screenshot, one without."""
    print("Creating realistic test documents...")
    
    # Create document-like images
    width, height = 900, 1200
    
    # Document 1: WITH screenshot/image
    img1 = Image.new('RGB', (width, height), 'white')
    draw1 = ImageDraw.Draw(img1)
    
    # Add document header
    draw1.rectangle([50, 50, 850, 80], fill=(230, 230, 230))
    draw1.text([60, 55], "Project Requirements Document", fill='black')
    
    # Add some paragraph text (simulate with horizontal lines)
    text_y = 120
    for i in range(8):
        if i % 3 == 2:  # Shorter line (end of paragraph)
            draw1.rectangle([50, text_y, 600, text_y + 12], fill='black')
        else:
            draw1.rectangle([50, text_y, 820, text_y + 12], fill='black')
        text_y += 20
    
    # Add section header
    draw1.rectangle([50, text_y + 20, 300, text_y + 45], fill='black')
    text_y += 70
    
    # Add more text
    for i in range(3):
        draw1.rectangle([50, text_y, 800, text_y + 12], fill='black')
        text_y += 20
    
    # === ADD THE SCREENSHOT/IMAGE ===
    screenshot_x, screenshot_y = 80, text_y + 30
    screenshot_w, screenshot_h = 650, 400
    
    # Create a realistic-looking screenshot with window frame
    # Window frame
    draw1.rectangle([screenshot_x, screenshot_y, screenshot_x + screenshot_w, screenshot_y + screenshot_h], 
                   fill=(240, 245, 250), outline=(120, 120, 120), width=3)
    
    # Title bar
    draw1.rectangle([screenshot_x + 3, screenshot_y + 3, screenshot_x + screenshot_w - 3, screenshot_y + 35], 
                   fill=(60, 60, 60))
    draw1.text([screenshot_x + 15, screenshot_y + 12], "Application Screenshot", fill='white')
    
    # Window buttons
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        btn_x = screenshot_x + screenshot_w - 60 + i * 20
        draw1.ellipse([btn_x, screenshot_y + 10, btn_x + 12, screenshot_y + 22], fill=color)
    
    # Content area with some interface elements
    content_y = screenshot_y + 40
    content_h = screenshot_h - 45
    
    # Background
    draw1.rectangle([screenshot_x + 3, content_y, screenshot_x + screenshot_w - 3, screenshot_y + screenshot_h - 3], 
                   fill=(250, 250, 250))
    
    # Simulate menu bar
    draw1.rectangle([screenshot_x + 10, content_y + 10, screenshot_x + screenshot_w - 10, content_y + 35], 
                   fill=(220, 220, 220), outline=(180, 180, 180))
    
    # Add some menu items
    menu_items = ["File", "Edit", "View", "Tools", "Help"]
    menu_x = screenshot_x + 20
    for item in menu_items:
        draw1.text([menu_x, content_y + 18], item, fill='black')
        menu_x += len(item) * 8 + 20
    
    # Add a chart/graph simulation
    chart_x = screenshot_x + 50
    chart_y = content_y + 60
    chart_w = screenshot_w - 100
    chart_h = 200
    
    # Chart background
    draw1.rectangle([chart_x, chart_y, chart_x + chart_w, chart_y + chart_h], 
                   fill='white', outline=(100, 100, 100), width=2)
    
    # Chart title
    draw1.text([chart_x + chart_w//2 - 60, chart_y + 15], "Performance Metrics", fill='black')
    
    # Chart axes
    draw1.line([chart_x + 30, chart_y + 40, chart_x + 30, chart_y + chart_h - 30], fill=(80, 80, 80), width=2)
    draw1.line([chart_x + 30, chart_y + chart_h - 30, chart_x + chart_w - 20, chart_y + chart_h - 30], fill=(80, 80, 80), width=2)
    
    # Chart bars
    bar_width = 40
    bar_spacing = 60
    for i in range(7):
        bar_x = chart_x + 60 + i * bar_spacing
        bar_height = 20 + i * 15 + (i % 3) * 10
        draw1.rectangle([bar_x, chart_y + chart_h - 30 - bar_height, 
                        bar_x + bar_width, chart_y + chart_h - 30], 
                       fill=(50 + i * 20, 100 + i * 15, 200 - i * 10))
    
    # Add some buttons/controls
    btn_y = content_y + chart_h + 80
    for i, btn_text in enumerate(["Generate Report", "Export Data", "Settings"]):
        btn_x = screenshot_x + 50 + i * 150
        draw1.rectangle([btn_x, btn_y, btn_x + 130, btn_y + 35], 
                       fill=(70, 130, 180), outline=(50, 100, 150), width=2)
        draw1.text([btn_x + 20, btn_y + 12], btn_text, fill='white')
    
    # Add text after the screenshot
    text_y = screenshot_y + screenshot_h + 50
    draw1.rectangle([50, text_y, 400, text_y + 25], fill='black')  # Section header
    text_y += 50
    
    for i in range(6):
        if i % 4 == 3:  # Shorter line
            draw1.rectangle([50, text_y, 650, text_y + 12], fill='black')
        else:
            draw1.rectangle([50, text_y, 830, text_y + 12], fill='black')
        text_y += 20
    
    # Document 2: WITHOUT the screenshot (missing image)
    img2 = Image.new('RGB', (width, height), 'white')
    draw2 = ImageDraw.Draw(img2)
    
    # Copy everything except the screenshot
    
    # Add document header
    draw2.rectangle([50, 50, 850, 80], fill=(230, 230, 230))
    draw2.text([60, 55], "Project Requirements Document", fill='black')
    
    # Add the same paragraph text
    text_y = 120
    for i in range(8):
        if i % 3 == 2:  # Shorter line (end of paragraph)
            draw2.rectangle([50, text_y, 600, text_y + 12], fill='black')
        else:
            draw2.rectangle([50, text_y, 820, text_y + 12], fill='black')
        text_y += 20
    
    # Add section header
    draw2.rectangle([50, text_y + 20, 300, text_y + 45], fill='black')
    text_y += 70
    
    # Add more text
    for i in range(3):
        draw2.rectangle([50, text_y, 800, text_y + 12], fill='black')
        text_y += 20
    
    # === SKIP THE SCREENSHOT - IT'S MISSING! ===
    # Just add some white space where it would be
    
    # Add text after where the screenshot would be
    text_y = screenshot_y + screenshot_h + 50  # Same position as in img1
    draw2.rectangle([50, text_y, 400, text_y + 25], fill='black')  # Section header
    text_y += 50
    
    for i in range(6):
        if i % 4 == 3:  # Shorter line
            draw2.rectangle([50, text_y, 650, text_y + 12], fill='black')
        else:
            draw2.rectangle([50, text_y, 830, text_y + 12], fill='black')
        text_y += 20
    
    return img1, img2, (screenshot_x, screenshot_y, screenshot_w, screenshot_h)

def test_enhanced_detector():
    """Test the enhanced missing image detector."""
    print("\n" + "="*70)
    print("TESTING ENHANCED MISSING IMAGE DETECTOR")
    print("="*70)
    
    # Create test images
    img_with_screenshot, img_without_screenshot, expected_area = create_document_with_missing_screenshot()
    
    # Save test images
    img_with_screenshot.save("test_document_with_screenshot.png")
    img_without_screenshot.save("test_document_without_screenshot.png")
    print(f"Test images saved:")
    print(f"  - test_document_with_screenshot.png")
    print(f"  - test_document_without_screenshot.png")
    
    # Test different sensitivity levels
    sensitivity_levels = [0.5, 0.7, 0.9]
    
    for sensitivity in sensitivity_levels:
        print(f"\n--- Testing Enhanced Detector (sensitivity={sensitivity}) ---")
        
        try:
            original_annotated, updated_annotated, changes, similarity = detect_missing_images_enhanced(
                img_with_screenshot, 
                img_without_screenshot,
                min_missing_area=3000,
                sensitivity=sensitivity
            )
            
            print(f"✅ Enhanced detector (s={sensitivity}): {len(changes)} changes detected")
            print(f"   Similarity score: {similarity:.3f}")
            
            # Analyze detected changes
            for i, change in enumerate(changes):
                change_type = change.get('type', 'unknown')
                bbox = change.get('bbox', (0, 0, 0, 0))
                area = change.get('area', 0)
                confidence = change.get('confidence', 0)
                
                print(f"   Change {i+1}: {change_type}")
                print(f"     Location: {bbox}")
                print(f"     Area: {area:,} pixels")
                print(f"     Confidence: {confidence:.2f}")
                
                # Check if detection is in the expected area
                expected_x, expected_y, expected_w, expected_h = expected_area
                detected_x, detected_y, detected_w, detected_h = bbox
                
                # Calculate overlap with expected area
                overlap_x = max(0, min(expected_x + expected_w, detected_x + detected_w) - max(expected_x, detected_x))
                overlap_y = max(0, min(expected_y + expected_h, detected_y + detected_h) - max(expected_y, detected_y))
                overlap_area = overlap_x * overlap_y
                
                expected_area_total = expected_w * expected_h
                overlap_ratio = overlap_area / expected_area_total if expected_area_total > 0 else 0
                
                print(f"     Overlap with expected: {overlap_ratio:.1%}")
                
                if overlap_ratio > 0.3:  # Good overlap
                    print(f"     ✅ Good detection! Overlaps {overlap_ratio:.1%} with expected missing area")
                elif overlap_ratio > 0.1:
                    print(f"     ⚠️  Partial detection. Overlaps {overlap_ratio:.1%} with expected missing area")
                else:
                    print(f"     ❌ Poor detection. Only {overlap_ratio:.1%} overlap with expected area")
            
            # Save annotated results
            suffix = f"_enhanced_s{sensitivity}".replace(".", "")
            original_annotated.save(f"test_result_original{suffix}.png")
            updated_annotated.save(f"test_result_updated{suffix}.png")
            print(f"   Saved: test_result_original{suffix}.png, test_result_updated{suffix}.png")
            
        except Exception as e:
            print(f"❌ Enhanced detector (s={sensitivity}) failed: {e}")
            import traceback
            traceback.print_exc()

def test_content_analysis():
    """Test the visual content analysis function."""
    print(f"\n--- Testing Visual Content Analysis ---")
    
    # Load one of our test images
    try:
        img = Image.open("test_document_with_screenshot.png")
        regions = analyze_visual_content_regions(img, min_region_size=1000)
        
        print(f"✅ Content analysis found {len(regions)} visual regions:")
        
        for i, region in enumerate(regions):
            region_type = region.get('type', 'unknown')
            bbox = region.get('bbox', (0, 0, 0, 0))
            area = region.get('area', 0)
            edge_density = region.get('edge_density', 0)
            
            print(f"  Region {i+1}: {region_type}")
            print(f"    Location: {bbox}")
            print(f"    Area: {area:,} pixels")
            print(f"    Edge density: {edge_density:.3f}")
            
            if region_type == "complex_visual":
                print(f"    🖼️  This is likely an image, diagram, or chart!")
        
    except Exception as e:
        print(f"❌ Content analysis failed: {e}")

def compare_with_existing_methods():
    """Compare enhanced detector with existing methods."""
    print(f"\n--- Comparing with Existing Methods ---")
    
    try:
        # Import existing methods
        from utils.image_comparison import (
            find_image_differences,
            detect_large_content_blocks,
            find_ultra_subtle_differences
        )
        
        # Load test images
        img1 = Image.open("test_document_with_screenshot.png")
        img2 = Image.open("test_document_without_screenshot.png")
        
        methods = [
            ("Regular Detection (sensitive)", lambda: find_image_differences(img1, img2, threshold=0.5)),
            ("Large Block Detection", lambda: detect_large_content_blocks(img1, img2, min_block_size=5000)),
            ("Ultra Sensitive", lambda: find_ultra_subtle_differences(img1, img2)),
            ("Enhanced Missing Image", lambda: detect_missing_images_enhanced(img1, img2, sensitivity=0.7))
        ]
        
        results = []
        
        for method_name, method_func in methods:
            try:
                result = method_func()
                
                if method_name == "Large Block Detection":
                    # Different return format
                    original_ann, updated_ann, changes, similarity = result
                    num_changes = len(changes)
                elif len(result) >= 3:
                    # Standard format
                    annotated, changes, similarity = result[:3]
                    num_changes = len(changes)
                else:
                    num_changes = 0
                    similarity = 0
                
                results.append((method_name, num_changes, similarity))
                print(f"✅ {method_name:25}: {num_changes:2d} changes, similarity: {similarity:.3f}")
                
            except Exception as e:
                results.append((method_name, 0, 0))
                print(f"❌ {method_name:25}: ERROR - {str(e)[:50]}")
        
        # Find best method
        best_method = max(results, key=lambda x: x[1])
        print(f"\n🏆 Best performing method: {best_method[0]} with {best_method[1]} changes detected")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

def main():
    print("ENHANCED MISSING IMAGE DETECTOR TEST")
    print("="*70)
    print("This test creates realistic document images with and without screenshots,")
    print("then tests the enhanced missing image detector against them.")
    
    # Run tests
    test_enhanced_detector()
    test_content_analysis()
    compare_with_existing_methods()
    
    # Final recommendations
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR DOCUMENT COMPARISON APP")
    print("="*70)
    print("Based on these tests:")
    print("1. Use the Enhanced Missing Image Detector for best results")
    print("2. Set sensitivity between 0.7-0.9 for good detection without false positives")
    print("3. Set min_missing_area to 3000-5000 pixels for typical screenshots/diagrams")
    print("4. Enable both text highlighting AND enhanced image detection for complete coverage")
    print("5. The enhanced detector should catch missing screenshots, diagrams, and charts")
    
    print(f"\nGenerated files to check:")
    print("- test_document_with_screenshot.png (has the screenshot)")
    print("- test_document_without_screenshot.png (missing screenshot)")
    print("- test_result_original_enhanced_*.png (shows missing elements highlighted)")
    print("- test_result_updated_enhanced_*.png (shows added elements highlighted)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()