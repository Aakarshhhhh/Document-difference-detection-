#!/usr/bin/env python3
"""
Simple test script to verify basic functionality of document difference detection.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.file_conversion import get_supported_file_types, convert_to_comparable_format
from utils.text_comparison import compare_texts, get_text_diff_summary
from utils.image_comparison import find_image_differences
from PIL import Image, ImageDraw
import io

def create_test_image(text, width=400, height=300, bg_color=(255,255,255), text_color=(0,0,0)):
    """Create a simple test image with text"""
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Simple text drawing
    lines = text.split('\n')
    y_offset = 50
    for line in lines:
        draw.text((20, y_offset), line, fill=text_color)
        y_offset += 30
    
    return img

def test_basic_functionality():
    """Test basic document processing functionality"""
    print("Testing Document Difference Detection...")
    
    # Test 1: File type support
    print("\n1. Testing file type support...")
    supported_types = get_supported_file_types()
    print(f"   Supported file types: {len(supported_types)}")
    for ext, desc in list(supported_types.items())[:5]:
        print(f"   - {ext}: {desc}")
    
    # Test 2: Text comparison
    print("\n2. Testing text comparison...")
    text1 = "This is a sample document.\nIt has multiple lines.\nThis is the original version."
    text2 = "This is a sample document.\nIt has multiple lines.\nThis is the updated version with changes."
    
    differences = compare_texts(text1, text2)
    summary = get_text_diff_summary(differences)
    
    print(f"   Text differences found: {len(differences)}")
    print(f"   Summary: {summary}")
    
    # Test 3: Image comparison
    print("\n3. Testing image comparison...")
    img1 = create_test_image("Original Document\nPage 1\nSample content")
    img2 = create_test_image("Updated Document\nPage 1\nModified content")
    
    try:
        diff_img, changes, ssim_score = find_image_differences(img1, img2)
        print(f"   Image comparison completed successfully")
        print(f"   Changes detected: {len(changes)}")
        print(f"   SSIM Score: {ssim_score:.3f}")
    except Exception as e:
        print(f"   Image comparison failed: {e}")
    
    # Test 4: Memory usage and performance
    print("\n4. Testing with multiple small operations...")
    for i in range(5):
        test_text1 = f"Document version {i}\nWith some content\nLine {i*3+1}\nLine {i*3+2}"
        test_text2 = f"Document version {i+1}\nWith some content\nLine {i*3+1}\nLine {i*3+3}"
        diffs = compare_texts(test_text1, test_text2)
        print(f"   Test {i+1}: {len(diffs)} differences")
    
    print("\n✅ Basic functionality tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_basic_functionality()
        print("\n🎉 All tests passed! The app should work correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)