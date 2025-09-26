#!/usr/bin/env python3
"""
Test script for text-level highlighting functionality.
This script tests the new text highlighting features.
"""

import sys
import os
from PIL import Image
from utils.text_comparison import create_text_diff_image, extract_text_with_positions

def test_text_highlighting():
    """Test text highlighting functionality with sample PDFs"""
    
    print("Text Highlighting Test")
    print("=" * 50)
    
    # You can test with actual PDF files by placing them in the directory
    pdf1_path = "sample1.pdf"  # Replace with your PDF files
    pdf2_path = "sample2.pdf"
    
    if os.path.exists(pdf1_path) and os.path.exists(pdf2_path):
        print(f"Testing with {pdf1_path} and {pdf2_path}")
        
        try:
            # Read PDF files
            with open(pdf1_path, 'rb') as f1, open(pdf2_path, 'rb') as f2:
                pdf1_bytes = f1.read()
                pdf2_bytes = f2.read()
            
            # Test text extraction with positions
            print("\nTesting text extraction...")
            words1 = extract_text_with_positions(pdf1_bytes, 0)
            words2 = extract_text_with_positions(pdf2_bytes, 0)
            
            print(f"Extracted {len(words1)} words from first document")
            print(f"Extracted {len(words2)} words from second document")
            
            if words1:
                print(f"Sample words from first document: {[w['text'] for w in words1[:5]]}")
            if words2:
                print(f"Sample words from second document: {[w['text'] for w in words2[:5]]}")
            
            # Test text highlighting
            print("\nTesting text highlighting...")
            result_img, text_changes = create_text_diff_image(pdf1_bytes, pdf2_bytes, 0)
            
            if result_img:
                print(f"✅ Text highlighting successful!")
                print(f"Found {len(text_changes)} text changes:")
                
                for i, change in enumerate(text_changes):
                    print(f"  {i+1}. {change['type'].title()}: '{change['text']}'")
                
                # Save result
                result_img.save("text_highlighting_result.png")
                print("Saved result as 'text_highlighting_result.png'")
                
            else:
                print("❌ Text highlighting failed")
                
        except Exception as e:
            print(f"❌ Error during test: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("Please place two PDF files named 'sample1.pdf' and 'sample2.pdf' in this directory")
        print("Or modify the script to use your own PDF file paths")
        
        # Create a simple demonstration
        print("\nCreating a demonstration of the text highlighting concept:")
        print("🟢 Green highlighting = Text additions")
        print("🔴 Red highlighting = Text deletions") 
        print("🟠 Orange highlighting = Text modifications")
        print("\nThe system will:")
        print("1. Extract text with precise positions from both PDF documents")
        print("2. Compare the text word-by-word using sequence matching")
        print("3. Highlight differences directly on the document image")
        print("4. Provide a list of all changes with their types and locations")

def demo_text_extraction():
    """Demonstrate how text extraction with positions works"""
    
    print("\n" + "=" * 50)
    print("TEXT EXTRACTION DEMONSTRATION")
    print("=" * 50)
    
    print("The text highlighting system works by:")
    print("\n1. **Extract text with positions**: Each word gets precise coordinates")
    print("   Example: {'text': 'Cloud', 'bbox': (100, 200, 150, 220)}")
    
    print("\n2. **Compare word sequences**: Uses difflib.SequenceMatcher")
    print("   - Finds insertions, deletions, and replacements")
    print("   - Works at word level for precise highlighting")
    
    print("\n3. **Generate overlay**: Creates transparent colored rectangles")
    print("   - Green (0, 255, 0) with 80% transparency for additions")
    print("   - Red (255, 0, 0) with 80% transparency for deletions")
    
    print("\n4. **Composite result**: Overlays highlights on document image")
    print("   - High DPI rendering (2x zoom) for crisp text")
    print("   - Alpha blending for professional appearance")

if __name__ == "__main__":
    test_text_highlighting()
    demo_text_extraction()
    
    print("\n" + "=" * 50)
    print("To use text highlighting in the main application:")
    print("1. Upload two PDF documents")
    print("2. Enable 'Word-Level Text Highlighting' in Advanced Options")
    print("3. The system will show highlighted differences like in your reference image")
    print("=" * 50)