#!/usr/bin/env python3
"""
Test script for large content block detection functionality.
"""

from PIL import Image, ImageDraw
import numpy as np
from utils.image_comparison import detect_large_content_blocks

def create_test_images():
    """Create test images - one with a large block, one without."""
    # Create base image (white with some text-like content)
    img1 = Image.new('RGB', (800, 1000), 'white')
    draw1 = ImageDraw.Draw(img1)
    
    # Add some text-like lines at the top
    for i in range(5):
        y = 50 + i * 30
        draw1.rectangle([50, y, 600, y + 15], fill='black')
    
    # Create second image with the same content plus a large block
    img2 = img1.copy()
    draw2 = ImageDraw.Draw(img2)
    
    # Add a large dark block (simulating a screenshot/diagram)
    draw2.rectangle([100, 400, 650, 750], fill=(50, 20, 80), outline=(200, 150, 100), width=3)
    
    # Add some content inside the block (simulating text/code)
    for i in range(8):
        y = 450 + i * 25
        draw2.rectangle([120, y, 500, y + 12], fill=(150, 255, 150))
    
    return img1, img2

def test_large_block_detection():
    """Test the large block detection function."""
    print("Creating test images...")
    img1, img2 = create_test_images()
    
    # Save test images for inspection
    img1.save("test_original.png")
    img2.save("test_updated.png")
    print("Test images saved as test_original.png and test_updated.png")
    
    print("\nRunning large block detection...")
    original_annotated, updated_annotated, changes, similarity_score = detect_large_content_blocks(img1, img2)
    
    print(f"\nResults:")
    print(f"Similarity Score: {similarity_score:.3f}")
    print(f"Changes Detected: {len(changes)}")
    
    for i, change in enumerate(changes):
        print(f"\nChange {i+1}:")
        print(f"  Type: {change['type']}")
        print(f"  Description: {change['description']}")
        print(f"  Bounding Box: {change['bbox']}")
        print(f"  Area: {change['area']} pixels")
        print(f"  Area Ratio: {change['area_ratio']:.3%}")
        print(f"  Content Density Original: {change['content_density_1']:.3f}")
        print(f"  Content Density Updated: {change['content_density_2']:.3f}")
    
    # Save annotated results
    original_annotated.save("test_result_original_annotated.png")
    updated_annotated.save("test_result_updated_annotated.png")
    print(f"\nAnnotated results saved as:")
    print(f"  - test_result_original_annotated.png (shows missing elements)")
    print(f"  - test_result_updated_annotated.png (shows added elements)")
    
    return len(changes) > 0

if __name__ == "__main__":
    success = test_large_block_detection()
    if success:
        print("\n✅ Test PASSED: Large block detection working correctly!")
    else:
        print("\n❌ Test FAILED: No large blocks detected")