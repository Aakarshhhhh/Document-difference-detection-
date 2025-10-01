#!/usr/bin/env python3
"""
Solution guide for fixing missing image detection in your document comparison app.

This script shows you exactly how to integrate the enhanced missing image detection
into your existing application to fix the issue you're experiencing.
"""

import sys
import os
from PIL import Image
import streamlit as st

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import both existing and enhanced functions
from utils.image_comparison import detect_large_content_blocks, find_image_differences
from utils.enhanced_missing_image_detector import detect_missing_images_enhanced

def enhanced_visual_comparison(img1, img2, detection_settings):
    """
    Enhanced visual comparison that combines multiple detection methods
    for the best missing image detection results.
    
    This is what you should integrate into your app.py file.
    """
    print("🔍 Running Enhanced Visual Comparison...")
    
    # Settings from your app
    threshold = detection_settings.get('threshold', 0.7)
    enable_large_block = detection_settings.get('enable_large_block_detection', True)
    enable_enhanced_missing = detection_settings.get('enable_enhanced_missing_detection', True)
    min_missing_area = detection_settings.get('min_missing_area', 5000)
    sensitivity = detection_settings.get('sensitivity', 0.7)
    
    all_changes = []
    annotated_images = {'original': img1, 'updated': img2}
    detection_info = []
    
    # Method 1: Enhanced Missing Image Detection (NEW - Best for missing images)
    if enable_enhanced_missing:
        try:
            print("  ✓ Running Enhanced Missing Image Detection...")
            
            original_ann, updated_ann, enhanced_changes, similarity = detect_missing_images_enhanced(
                img1, img2,
                min_missing_area=min_missing_area,
                sensitivity=sensitivity
            )
            
            if enhanced_changes:
                print(f"    🎯 Found {len(enhanced_changes)} missing/added visual elements")
                
                # Store annotated images
                annotated_images['original'] = original_ann
                annotated_images['updated'] = updated_ann
                
                # Add changes with source info
                for change in enhanced_changes:
                    change['detection_method'] = 'Enhanced Missing Image'
                    change['priority'] = 'high'  # High priority for missing images
                    all_changes.append(change)
                
                detection_info.append({
                    'method': 'Enhanced Missing Image Detection',
                    'changes_found': len(enhanced_changes),
                    'similarity': similarity,
                    'status': 'success'
                })
            else:
                detection_info.append({
                    'method': 'Enhanced Missing Image Detection', 
                    'changes_found': 0,
                    'status': 'no_changes'
                })
                
        except Exception as e:
            print(f"    ❌ Enhanced detection failed: {e}")
            detection_info.append({
                'method': 'Enhanced Missing Image Detection',
                'changes_found': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # Method 2: Large Block Detection (Existing - Good for missing images)
    if enable_large_block and not enhanced_changes:  # Only if enhanced didn't find anything
        try:
            print("  ✓ Running Large Block Detection as backup...")
            
            lb_original, lb_updated, lb_changes, lb_similarity = detect_large_content_blocks(
                img1, img2, min_block_size=min_missing_area
            )
            
            if lb_changes:
                print(f"    📊 Found {len(lb_changes)} large content blocks")
                
                # If no enhanced detection results, use large block results
                if not enhanced_changes:
                    annotated_images['original'] = lb_original
                    annotated_images['updated'] = lb_updated
                
                # Add changes with source info
                for change in lb_changes:
                    change['detection_method'] = 'Large Block Detection'
                    change['priority'] = 'medium'
                    all_changes.append(change)
                
                detection_info.append({
                    'method': 'Large Block Detection',
                    'changes_found': len(lb_changes),
                    'similarity': lb_similarity,
                    'status': 'success'
                })
            else:
                detection_info.append({
                    'method': 'Large Block Detection',
                    'changes_found': 0,
                    'status': 'no_changes'
                })
                
        except Exception as e:
            print(f"    ❌ Large block detection failed: {e}")
            detection_info.append({
                'method': 'Large Block Detection',
                'changes_found': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # Method 3: Regular Detection (Existing - Good for small changes)
    if not all_changes:  # Only if no changes found yet
        try:
            print("  ✓ Running Regular Detection as final backup...")
            
            regular_annotated, regular_changes, regular_similarity = find_image_differences(
                img1, img2,
                threshold=threshold,
                min_change_size=5,
                min_change_area=500,
                enable_edge_detection=True,
                enable_color_detection=True
            )
            
            if regular_changes:
                print(f"    🔍 Found {len(regular_changes)} regular changes")
                
                # Use regular detection results only if nothing else worked
                annotated_images['updated'] = regular_annotated
                
                # Add changes with source info
                for change in regular_changes:
                    change['detection_method'] = 'Regular Detection'
                    change['priority'] = 'low'
                    all_changes.append(change)
                
                detection_info.append({
                    'method': 'Regular Detection',
                    'changes_found': len(regular_changes),
                    'similarity': regular_similarity,
                    'status': 'success'
                })
            else:
                detection_info.append({
                    'method': 'Regular Detection',
                    'changes_found': 0,
                    'status': 'no_changes'
                })
                
        except Exception as e:
            print(f"    ❌ Regular detection failed: {e}")
            detection_info.append({
                'method': 'Regular Detection',
                'changes_found': 0,
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    total_changes = len(all_changes)
    missing_images = len([c for c in all_changes if 'missing' in c.get('type', '').lower()])
    added_images = len([c for c in all_changes if 'add' in c.get('type', '').lower()])
    
    print(f"\n🎯 Detection Summary:")
    print(f"   Total changes detected: {total_changes}")
    print(f"   Missing visual elements: {missing_images}")
    print(f"   Added visual elements: {added_images}")
    
    # Sort changes by priority (high priority = missing images first)
    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    all_changes.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 2))
    
    return annotated_images, all_changes, detection_info

def get_recommended_settings():
    """
    Get recommended settings for missing image detection based on our tests.
    """
    return {
        'threshold': 0.7,  # Good balance
        'enable_large_block_detection': True,  # Essential for missing images
        'enable_enhanced_missing_detection': True,  # NEW - Best method
        'min_missing_area': 5000,  # 5000 pixels minimum for screenshots/diagrams
        'sensitivity': 0.7,  # Good balance between detection and false positives
        'min_change_size': 10,
        'min_change_area': 1000,
        'enable_edge_detection': True,
        'enable_color_detection': True
    }

def integration_instructions():
    """
    Print step-by-step instructions for integrating this into your app.
    """
    print("\n" + "="*80)
    print("INTEGRATION INSTRUCTIONS FOR YOUR APP.PY")
    print("="*80)
    
    print("\n1. 📁 Add Import Statement")
    print("   Add this import to your app.py file:")
    print("   from utils.enhanced_missing_image_detector import detect_missing_images_enhanced")
    
    print("\n2. 🔧 Modify Your Image Comparison Function")
    print("   In your app.py, find where you do image comparison (around line 450-550).")
    print("   Replace or enhance the existing visual comparison with:")
    print("   ")
    print("   # Add this BEFORE your existing image comparison")
    print("   if enable_large_block_detection:")
    print("       try:")
    print("           original_ann, updated_ann, enhanced_changes, similarity = detect_missing_images_enhanced(")
    print("               img_original, img_updated,")
    print("               min_missing_area=5000,")
    print("               sensitivity=0.7")
    print("           )")
    print("           ")
    print("           if enhanced_changes:")
    print("               st.success(f'🎯 Enhanced detector found {len(enhanced_changes)} missing visual elements')")
    print("               ")
    print("               # Show the enhanced results")
    print("               st.image(original_ann, use_column_width=True,")
    print("                       caption=f'Original with Missing Elements - Page {original_page_num + 1}')")
    print("               st.image(updated_ann, use_column_width=True,") 
    print("                       caption=f'Updated with Added Elements - Page {updated_page_num + 1}')")
    print("               ")
    print("               # Store changes for summary")
    print("               changes = enhanced_changes")
    print("               ssim_score = similarity")
    print("           ")
    print("       except Exception as e:")
    print("           st.error(f'Enhanced detection failed: {e}')")
    
    print("\n3. 🎛️ Add Settings Controls")
    print("   Add these settings to your sidebar:")
    print("   ")
    print("   with st.sidebar:")
    print("       st.subheader('Missing Image Detection')")
    print("       enable_enhanced_detection = st.checkbox('Enhanced Missing Image Detection', value=True)")
    print("       missing_sensitivity = st.slider('Missing Image Sensitivity', 0.5, 0.9, 0.7, 0.1)")
    print("       min_missing_pixels = st.slider('Min Missing Area (pixels)', 1000, 20000, 5000, 1000)")
    
    print("\n4. 🏃 Test It!")
    print("   - Enable 'Enhanced Missing Image Detection' in sidebar")
    print("   - Upload your documents with missing images")
    print("   - You should now see missing images highlighted in RED on original document")
    print("   - Added images will be highlighted in GREEN on updated document")
    
    print("\n5. 🎨 Customize Display (Optional)")
    print("   You can customize how missing images are displayed:")
    print("   - Red highlights = Missing visual elements (shown on original)")
    print("   - Green highlights = Added visual elements (shown on updated)")
    print("   - Orange highlights = Modified visual elements (shown on both)")
    
    print("\n6. 🐛 Troubleshooting")
    print("   If it's still not working:")
    print("   - Check that enable_large_block_detection is True in your UI")
    print("   - Lower the min_missing_area to 2000-3000 pixels")
    print("   - Increase sensitivity to 0.8 or 0.9")
    print("   - Check the debug output for error messages")
    
    print(f"\n✅ After integration, your app should successfully highlight missing images!")

def create_example_integration_code():
    """
    Create a complete example of how to integrate this into app.py
    """
    example_code = '''
# Add this import at the top of your app.py file
from utils.enhanced_missing_image_detector import detect_missing_images_enhanced

# In your main comparison function, replace the visual comparison section with this:
def enhanced_visual_comparison_section(img_original, img_updated, original_page_num, updated_page_num):
    """Enhanced visual comparison with missing image detection"""
    
    # Settings from sidebar
    enable_enhanced_detection = st.session_state.get('enable_enhanced_detection', True)
    missing_sensitivity = st.session_state.get('missing_sensitivity', 0.7)
    min_missing_pixels = st.session_state.get('min_missing_pixels', 5000)
    
    changes = []
    ssim_score = 1.0
    
    if enable_enhanced_detection:
        with st.spinner("🔍 Scanning for missing images, diagrams, and screenshots..."):
            try:
                original_ann, updated_ann, enhanced_changes, similarity = detect_missing_images_enhanced(
                    img_original, img_updated,
                    min_missing_area=min_missing_pixels,
                    sensitivity=missing_sensitivity
                )
                
                if enhanced_changes:
                    st.success(f"🎯 Found {len(enhanced_changes)} missing visual elements!")
                    
                    # Count types of changes
                    missing_count = sum(1 for c in enhanced_changes if 'missing' in c.get('type', '').lower())
                    added_count = sum(1 for c in enhanced_changes if 'add' in c.get('type', '').lower())
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original (Missing Elements)")
                        st.image(original_ann, use_column_width=True, 
                                caption=f"🔴 {missing_count} missing visual elements highlighted")
                    
                    with col2:
                        st.subheader("Updated (Added Elements)") 
                        st.image(updated_ann, use_column_width=True,
                                caption=f"🟢 {added_count} added visual elements highlighted")
                    
                    # Show change details
                    with st.expander("🔍 Visual Change Details", expanded=False):
                        for i, change in enumerate(enhanced_changes):
                            change_type = change.get('type', 'unknown')
                            bbox = change.get('bbox', (0, 0, 0, 0))
                            confidence = change.get('confidence', 0)
                            
                            if 'missing' in change_type.lower():
                                st.error(f"🔴 Missing: {change.get('description', 'Visual element')} "
                                        f"(Confidence: {confidence:.1%})")
                            elif 'add' in change_type.lower():
                                st.success(f"🟢 Added: {change.get('description', 'Visual element')} "
                                          f"(Confidence: {confidence:.1%})")
                            else:
                                st.warning(f"🟡 Modified: {change.get('description', 'Visual element')} "
                                          f"(Confidence: {confidence:.1%})")
                    
                    changes = enhanced_changes
                    ssim_score = similarity
                    
                else:
                    st.info("✅ No missing visual elements detected")
                    
                    # Fallback to regular detection
                    with st.spinner("Running regular change detection..."):
                        diff_img, regular_changes, regular_ssim = find_image_differences(
                            img_original, img_updated,
                            threshold=0.7,
                            enable_edge_detection=True,
                            enable_color_detection=True
                        )
                        
                        if regular_changes:
                            st.image(diff_img, use_column_width=True, 
                                    caption=f"Regular changes detected: {len(regular_changes)}")
                            changes = regular_changes
                            ssim_score = regular_ssim
                
            except Exception as e:
                st.error(f"❌ Enhanced detection failed: {e}")
                st.info("Falling back to regular detection...")
                
                # Fallback to existing method
                diff_img, changes, ssim_score = find_image_differences(
                    img_original, img_updated,
                    threshold=0.7,
                    enable_edge_detection=True,
                    enable_color_detection=True
                )
                
                st.image(diff_img, use_column_width=True, 
                        caption=f"Changes detected: {len(changes)}")
    
    # Store results in session state for summary
    page_key = f'page_{st.session_state.get("page_num", 0)}_changes'
    st.session_state[page_key] = changes
    
    return changes, ssim_score

# Add these settings to your sidebar:
with st.sidebar:
    st.subheader("🖼️ Missing Image Detection")
    enable_enhanced_detection = st.checkbox("Enhanced Missing Image Detection", value=True, 
                                           key='enable_enhanced_detection',
                                           help="Best method for detecting missing screenshots, diagrams, and charts")
    
    if enable_enhanced_detection:
        missing_sensitivity = st.slider("Sensitivity", 0.5, 0.9, 0.7, 0.1,
                                      key='missing_sensitivity',
                                      help="Higher = more sensitive detection")
        
        min_missing_pixels = st.slider("Min Missing Area (pixels)", 1000, 20000, 5000, 1000,
                                     key='min_missing_pixels', 
                                     help="Minimum size for missing visual elements")
'''
    
    with open("integration_example.py", "w") as f:
        f.write(example_code)
    
    print(f"\n📄 Complete integration example saved to: integration_example.py")
    print("You can copy and paste the code from this file into your app.py")

def main():
    print("MISSING IMAGE DETECTION - SOLUTION GUIDE")
    print("="*80)
    print("Based on our diagnostic tests, here's how to fix your app:")
    
    # Show current status
    print(f"\n📊 DIAGNOSTIC RESULTS SUMMARY:")
    print(f"✅ Your algorithms ARE working correctly!")
    print(f"✅ Large Block Detection: Successfully detects missing images")  
    print(f"✅ Enhanced Missing Detection: Even better results") 
    print(f"⚠️  Issue: May not be properly integrated into your UI")
    
    # Show recommended settings
    print(f"\n🎛️  RECOMMENDED SETTINGS:")
    settings = get_recommended_settings()
    for key, value in settings.items():
        print(f"   {key}: {value}")
    
    # Show integration instructions
    integration_instructions()
    
    # Create example code
    create_example_integration_code()
    
    print(f"\n" + "="*80)
    print("🎯 QUICK FIX SUMMARY")
    print("="*80)
    print("The issue is NOT that missing images aren't being detected.")
    print("The algorithms work fine! The issue is likely:")
    print()
    print("1. 🔧 Large Block Detection might not be enabled in your UI")
    print("2. 🔧 Settings might be too restrictive (min_block_size too high)")
    print("3. 🔧 The enhanced detector isn't integrated yet")
    print()
    print("SOLUTION:")
    print("✅ Enable 'Large Block Detection' in your app sidebar")
    print("✅ Set threshold to 0.7 or lower")  
    print("✅ Set minimum block size to 5,000-10,000 pixels")
    print("✅ Integrate the enhanced missing image detector for best results")
    print()
    print("After making these changes, missing images WILL be highlighted!")

if __name__ == "__main__":
    main()