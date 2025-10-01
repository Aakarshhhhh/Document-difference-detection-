from PIL import Image, ImageDraw
import numpy as np
from utils.enhanced_missing_image_detector import detect_missing_images_enhanced

# Test with simple detection
print('Testing enhanced missing image detection...')

# Create simple test images
img1 = Image.new('RGB', (800, 600), 'white')
img2 = Image.new('RGB', (800, 600), 'white')

# Add a dark block to img2 (simulating the terminal screenshot)
draw2 = ImageDraw.Draw(img2)
draw2.rectangle([100, 200, 500, 450], fill=(50, 20, 80))  # Dark purple like terminal

try:
    original_ann, updated_ann, changes, similarity = detect_missing_images_enhanced(
        img1, img2,
        min_missing_area=2000,  # Lower threshold
        sensitivity=0.8
    )
    
    print(f'Detection successful!')
    print(f'Changes found: {len(changes)}')
    print(f'Similarity: {similarity:.3f}')
    
    for i, change in enumerate(changes):
        change_type = change.get('type', 'unknown')
        area = change.get('area', 0)
        print(f'  Change {i+1}: {change_type} - Area: {area} pixels')
        
    # Save test results
    original_ann.save('quick_test_original.png')
    updated_ann.save('quick_test_updated.png')
    print('Test images saved as quick_test_*.png')
        
except Exception as e:
    print(f'Detection failed: {e}')
    import traceback
    traceback.print_exc()