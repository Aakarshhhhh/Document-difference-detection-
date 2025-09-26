#!/usr/bin/env python3
"""
Test script to verify text highlighting functionality.
"""
import sys
import os
from PIL import Image, ImageDraw

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.text_comparison import compare_documents_with_text_overlay
import io

def create_simple_pdf_bytes(text_content, filename=None):
    """Create a simple PDF with given text content for testing"""
    try:
        import fitz  # PyMuPDF
        
        # Create a new PDF document
        doc = fitz.open()  # new PDF
        page = doc.new_page()  # new page
        
        # Add text to the page
        text_rect = fitz.Rect(50, 50, 500, 700)  # text area
        page.insert_textbox(text_rect, text_content, fontsize=12, fontname="helv")
        
        # Save to bytes
        pdf_bytes = doc.tobytes()
        doc.close()
        
        if filename:
            with open(filename, 'wb') as f:
                f.write(pdf_bytes)
            print(f"Test PDF saved as {filename}")
        
        return pdf_bytes
        
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return None

def test_text_highlighting():
    """Test text highlighting with simple PDF documents"""
    print("Testing Text Highlighting Functionality...")
    
    # Create test documents
    original_text = """This is a test document.
It contains several lines of text.
Some words will be deleted.
Other words will remain unchanged.
This line will be modified significantly."""
    
    updated_text = """This is a test document.
It contains several lines of text.
New words have been added here.
Other words will remain unchanged.
This line has been changed completely with different content."""
    
    print("\n1. Creating test PDF documents...")
    pdf1_bytes = create_simple_pdf_bytes(original_text, "test_original.pdf")
    pdf2_bytes = create_simple_pdf_bytes(updated_text, "test_updated.pdf")
    
    if not pdf1_bytes or not pdf2_bytes:
        print("❌ Failed to create test PDFs")
        return False
    
    print("✅ Test PDFs created successfully")
    
    # Test highlighting
    print("\n2. Testing text highlighting...")
    try:
        original_highlighted, updated_highlighted, changes = compare_documents_with_text_overlay(
            pdf1_bytes, pdf2_bytes, page1_num=0, page2_num=0
        )
        
        if original_highlighted is None or updated_highlighted is None:
            print("❌ Highlighting failed - returned None")
            return False
        
        print("✅ Highlighting completed successfully")
        print(f"   Original image size: {original_highlighted.size}")
        print(f"   Updated image size: {updated_highlighted.size}")
        print(f"   Changes detected: {len(changes)}")
        
        # Analyze changes
        deletions = sum(1 for c in changes if c['type'] == 'deletion')
        additions = sum(1 for c in changes if c['type'] == 'addition')
        
        print(f"   Deletions (red highlights): {deletions}")
        print(f"   Additions (green highlights): {additions}")
        
        # Show change details
        if changes:
            print("\n3. Change details:")
            for i, change in enumerate(changes[:5]):  # Show first 5 changes
                doc_type = "Original" if change['document'] == 'original' else "Updated"
                color = "🔴" if change['type'] == 'deletion' else "🟢"
                print(f"   {color} {change['type'].title()} in {doc_type}: '{change['text']}'")
        
        # Save highlighted images for verification
        try:
            original_highlighted.save("test_original_highlighted.png")
            updated_highlighted.save("test_updated_highlighted.png")
            print(f"\n✅ Highlighted images saved:")
            print(f"   - test_original_highlighted.png (with red deletions)")
            print(f"   - test_updated_highlighted.png (with green additions)")
        except Exception as e:
            print(f"⚠️ Could not save images: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during highlighting: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_highlight_visibility():
    """Test that highlights are actually visible"""
    print("\n4. Testing highlight visibility...")
    
    # Create a simple test image
    test_img = Image.new('RGB', (400, 300), 'white')
    test_overlay = Image.new('RGBA', (400, 300), (255, 255, 255, 0))
    draw = ImageDraw.Draw(test_overlay)
    
    # Draw test highlights
    deletion_color = (255, 0, 0, 160)  # Red
    addition_color = (0, 200, 0, 160)  # Green
    
    # Draw some rectangles
    draw.rectangle([50, 50, 150, 80], fill=deletion_color)  # Red highlight
    draw.rectangle([200, 50, 300, 80], fill=addition_color)  # Green highlight
    
    # Composite
    result = Image.alpha_composite(test_img.convert('RGBA'), test_overlay)
    result = result.convert('RGB')
    
    try:
        result.save("test_highlight_visibility.png")
        print("✅ Highlight visibility test image saved: test_highlight_visibility.png")
        
        # Check if highlights are visible by sampling pixels
        red_pixel = result.getpixel((100, 65))  # Should be red-ish
        green_pixel = result.getpixel((250, 65))  # Should be green-ish
        white_pixel = result.getpixel((350, 65))  # Should be white
        
        print(f"   Red highlight pixel: RGB{red_pixel}")
        print(f"   Green highlight pixel: RGB{green_pixel}")
        print(f"   White background pixel: RGB{white_pixel}")
        
        # Basic checks
        if red_pixel[0] > red_pixel[1] and red_pixel[0] > red_pixel[2]:
            print("   ✅ Red highlighting is visible")
        else:
            print("   ⚠️ Red highlighting may not be visible enough")
            
        if green_pixel[1] > green_pixel[0] and green_pixel[1] > green_pixel[2]:
            print("   ✅ Green highlighting is visible")
        else:
            print("   ⚠️ Green highlighting may not be visible enough")
        
    except Exception as e:
        print(f"   ⚠️ Could not test highlight visibility: {e}")

if __name__ == "__main__":
    try:
        success = test_text_highlighting()
        test_highlight_visibility()
        
        if success:
            print("\n🎉 Text highlighting tests completed successfully!")
            print("\n📋 Summary:")
            print("   • Red highlights show deletions on the original document")
            print("   • Green highlights show additions on the updated document") 
            print("   • Check the generated PNG files to visually verify the highlighting")
        else:
            print("\n❌ Text highlighting tests failed")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)