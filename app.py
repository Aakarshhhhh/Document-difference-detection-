import streamlit as st
from PIL import Image, ImageDraw
import io
import fitz
import streamlit.components.v1 as components

from utils.file_conversion import convert_to_comparable_format, get_file_type
from utils.image_comparison import find_image_differences, compare_images_pixel_wise, find_ultra_subtle_differences, detect_large_content_blocks
from utils.enhanced_missing_image_detector import detect_missing_images_enhanced
from utils.text_comparison import compare_texts, highlight_text_differences_html, get_text_diff_summary, find_word_level_changes, create_text_diff_image
import json
import datetime


def calibrate_visual_params(img_original, img_updated, gt_mask_file=None):
    """Sweep detection params to find a setting that captures clear changes with low noise."""
    candidate_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    candidate_min_sizes = [5, 10, 15]
    candidate_min_areas = [100, 200, 400, 800]

    best = None
    best_score = float('-inf')

    gt_mask_arr = None
    if gt_mask_file is not None:
        try:
            gt = Image.open(gt_mask_file).convert('RGB')
            import numpy as np
            gta = np.array(gt)
            # Consider red-ish pixels as positives
            gt_mask_arr = ((gta[:,:,0] > 180) & (gta[:,:,1] < 100) & (gta[:,:,2] < 100)).astype(np.uint8)
        except Exception:
            gt_mask_arr = None

    for thr in candidate_thresholds:
        for msz in candidate_min_sizes:
            for mar in candidate_min_areas:
                try:
                    _, changes, ssim_score, diff_mask = find_image_differences(
                        img_original, img_updated,
                        threshold=thr,
                        min_change_size=msz,
                        min_change_area=mar,
                        enable_edge_detection=True,
                        enable_color_detection=True,
                        return_mask=True
                    )
                except Exception:
                    continue

                num_changes = len(changes)
                if gt_mask_arr is not None and diff_mask is not None and gt_mask_arr.shape == diff_mask.shape:
                    import numpy as np
                    pred = diff_mask.astype(bool)
                    gt = gt_mask_arr.astype(bool)
                    inter = np.logical_and(pred, gt).sum()
                    union = np.logical_or(pred, gt).sum()
                    iou = (inter / union) if union > 0 else 0.0
                    score = iou * 100.0
                else:
                    # Heuristic scoring when no ground truth
                    if num_changes == 0:
                        score = -5
                    else:
                        target = 6
                        score = 10 - abs(num_changes - target)
                    score -= thr * 1.0
                    score += (msz / 10.0) + (mar / 400.0)

                if score > best_score:
                    best_score = score
                    best = {
                        'threshold': thr,
                        'min_change_size': msz,
                        'min_change_area': mar,
                        'num_changes': num_changes,
                        'ssim': ssim_score
                    }

    return best

def main():
    st.set_page_config(layout="wide", page_title="Review Document Changes")
    st.title("Review Document Changes")
    st.markdown("Compare two versions of a document and identify changes")

    # Sidebar for file uploads
    with st.sidebar:
        st.header("Upload Documents")
        
        # Get supported file types
        from utils.file_conversion import get_supported_file_types
        supported_types = get_supported_file_types()
        file_extensions = list(supported_types.keys())
        
        uploaded_file1 = st.file_uploader("Upload Original Document", type=file_extensions)
        uploaded_file2 = st.file_uploader("Upload Updated Document", type=file_extensions)
        
        if uploaded_file1 and uploaded_file2:
            st.success("Files uploaded successfully!")
            
            # Show file info
            st.info(f"Original: {uploaded_file1.name} ({supported_types.get(uploaded_file1.name.split('.')[-1].lower(), 'Unknown')})")
            st.info(f"Updated: {uploaded_file2.name} ({supported_types.get(uploaded_file2.name.split('.')[-1].lower(), 'Unknown')})")
            
            # Analysis options
            st.header("Analysis Options")
            show_pixel_comparison = st.checkbox("Enable Pixel-wise Comparison", value=False)
            pixel_threshold = st.slider("Pixel Threshold (RGB delta)", 5, 100, 30)
            
            # Enhanced threshold options
            st.subheader("Detection Sensitivity")
            
            # Override with quick settings if applied
            if st.session_state.get('quick_setup_applied', False):
                default_mode_index = 0  # Ultra Sensitive
            else:
                default_mode_index = 3  # Balanced
                
            detection_mode = st.selectbox(
                "Detection Mode",
                ["Ultra Sensitive", "Very Sensitive", "Sensitive", "Balanced", "Conservative"],
                index=default_mode_index
            )
            
            # Map detection modes to thresholds
            threshold_map = {
                "Ultra Sensitive": 0.5,
                "Very Sensitive": 0.6,
                "Sensitive": 0.7,
                "Balanced": 0.8,
                "Conservative": 0.9
            }
            threshold = threshold_map[detection_mode]
            
            # Noise reduction options
            st.subheader("Noise Reduction")
            enable_noise_reduction = st.checkbox("Enable Advanced Noise Reduction", value=True)
            min_change_area = st.slider("Minimum Change Area (pixels²)", 100, 2000, 400)
            
            # Advanced options
            with st.expander("Advanced Detection Options"):
                enable_edge_detection = st.checkbox("Enable Edge-based Detection", value=True)
                enable_color_detection = st.checkbox("Enable Color Change Detection", value=True)
                enable_template_matching = st.checkbox("Enable Template Matching", value=False)
                enable_text_highlighting = st.checkbox("Enable Word-Level Text Highlighting", value=True, help="Highlight specific text changes like additions, deletions, and modifications directly on the document")
                enable_large_block_detection = st.checkbox("Enable Large Visual Element Detection", value=True, help="Detect large added/removed visual elements like diagrams, screenshots, tables, and code blocks")
                # Quick setup overrides
                quick_setup = st.session_state.get('quick_setup_applied', False)
                
                enable_enhanced_missing_detection = st.checkbox("Enhanced Missing Image Detection", value=True if quick_setup else True, help="Best method for detecting missing screenshots, diagrams, and charts")
                missing_image_sensitivity = st.slider("Missing Image Sensitivity", 0.5, 0.9, 0.7 if quick_setup else 0.7, 0.1, help="Higher = more sensitive to missing visual elements")
                min_missing_area = st.slider("Min Missing Area (pixels)", 1000, 20000, 5000 if quick_setup else 5000, 1000, help="Minimum size for missing visual elements")
                min_change_size = st.slider("Minimum Change Size (pixels)", 5, 50, 10)
            
            # Quick Settings for Missing Image Detection
            st.header("🎯 Quick Setup for Missing Images")
            
            col_quick1, col_quick2 = st.columns(2)
            with col_quick1:
                if st.button("🚀 Optimize for Missing Images", help="Click to automatically set best settings for detecting missing screenshots, diagrams, and charts"):
                    st.session_state['quick_setup_applied'] = True
                    st.success("✅ Settings optimized!")
            
            with col_quick2:
                if st.button("⚡ Force Aggressive Detection", help="Force ultra-aggressive detection on current page"):
                    st.session_state['force_aggressive'] = True
                    st.warning("⚡ Aggressive detection will be applied!")
            
            # Show current status
            if st.session_state.get('quick_setup_applied', False):
                st.info("🎯 Optimization: Enhanced Detection ON, Ultra Sensitive mode")
            
            if st.session_state.get('force_aggressive', False):
                st.warning("⚡ Aggressive mode: Min Area 1000px, Sensitivity 0.9")
            
            # Additional options
            st.header("Display Options")
            gt_mask = st.file_uploader("Ground-truth mask (optional, PNG with red marks)", type=["png"])
            show_text_comparison = st.checkbox("Show Text Comparison", value=True)
            show_visual_annotations = st.checkbox("Show Visual Annotations", value=True)
            show_debug_info = st.checkbox("Show Debug Information", value=False)
            apply_calibrated = st.checkbox("Apply Calibrated Settings (if available)", value=False)

            # ROI controls
            with st.expander("Region of Interest (focus detection)"):
                roi_enable = st.checkbox("Enable ROI", value=False)
                col_roi1, col_roi2 = st.columns(2)
                with col_roi1:
                    roi_x = st.number_input("ROI X", min_value=0, value=0)
                    roi_w = st.number_input("ROI Width", min_value=0, value=0)
                with col_roi2:
                    roi_y = st.number_input("ROI Y", min_value=0, value=0)
                    roi_h = st.number_input("ROI Height", min_value=0, value=0)

    if uploaded_file1 and uploaded_file2:
        # Convert files to comparable formats with error handling
        try:
            with st.spinner("Converting original document..."):
                images1, text1 = convert_to_comparable_format(uploaded_file1)
                if not images1:
                    st.error("Failed to process original document. Please check the file format.")
                    return
        except Exception as e:
            st.error(f"Error processing original document: {str(e)}")
            return
            
        try:
            with st.spinner("Converting updated document..."):
                images2, text2 = convert_to_comparable_format(uploaded_file2)
                if not images2:
                    st.error("Failed to process updated document. Please check the file format.")
                    return
        except Exception as e:
            st.error(f"Error processing updated document: {str(e)}")
            return
        
        # Calculate overall statistics
        num_pages1 = len(images1)
        num_pages2 = len(images2)
        page_additions = max(0, num_pages2 - num_pages1)
        page_removals = max(0, num_pages1 - num_pages2)
        
        # Analyze text changes with progress indication
        with st.spinner("Analyzing text differences..."):
            text_differences = compare_texts(text1, text2)
            text_summary = get_text_diff_summary(text_differences)
        
        # Create intelligent page mapping with optimization for large documents
        with st.spinner("Creating page mappings (this may take a moment for large documents)..."):
            from utils.text_comparison import create_page_mapping, get_page_mapping_info, get_page_mapping_summary
            
            # Split text into pages for mapping
            try:
                # Split text1 and text2 into page-wise lists
                text1_pages = text1.split('\n\n') if len(text1.split('\n\n')) == num_pages1 else [text1[i:i+1000] for i in range(0, len(text1), 1000)][:num_pages1]
                text2_pages = text2.split('\n\n') if len(text2.split('\n\n')) == num_pages2 else [text2[i:i+1000] for i in range(0, len(text2), 1000)][:num_pages2]
                
                # Ensure we have the right number of pages
                if len(text1_pages) != num_pages1:
                    text1_pages = [text1] * num_pages1  # Fallback
                if len(text2_pages) != num_pages2:
                    text2_pages = [text2] * num_pages2  # Fallback
                
                # For large documents (>20 pages), use simplified mapping
                if num_pages1 > 20 or num_pages2 > 20:
                    st.info(f"Large document detected ({max(num_pages1, num_pages2)} pages). Using simplified page mapping for better performance.")
                    page_mapping = {i: i for i in range(min(num_pages1, num_pages2))}
                    similarity_matrix = None
                else:
                    page_mapping, similarity_matrix = create_page_mapping(text1_pages, text2_pages, similarity_threshold=0.2)
                
                mapping_summary = get_page_mapping_summary(page_mapping, num_pages1, num_pages2)
                
            except Exception as e:
                st.warning(f"Page mapping failed: {e}. Using simple sequential mapping.")
                page_mapping = {i: i for i in range(min(num_pages1, num_pages2))}
                similarity_matrix = None
                mapping_summary = {'total_logical_pages': max(num_pages1, num_pages2), 'mapped_pages': min(num_pages1, num_pages2)}
        
        # Store page mapping in session state
        if 'page_mapping' not in st.session_state:
            st.session_state['page_mapping'] = page_mapping
            st.session_state['mapping_summary'] = mapping_summary
        
        # Overall summary metrics
        st.subheader("Document Comparison Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Pages", num_pages1)
        with col2:
            st.metric("Updated Pages", num_pages2)
        with col3:
            st.metric("Page Additions", page_additions, delta=page_additions)
        with col4:
            st.metric("Page Removals", page_removals, delta=-page_removals)
        
        st.markdown("---")

        if not images1 or not images2:
            st.error("Could not process both documents for visual comparison. Check file types or content.")
            return

        # Initialize session state for page navigation
        if 'page_num' not in st.session_state:
            st.session_state.page_num = 0

        max_pages = max(num_pages1, num_pages2)
        
        # Page navigation with mapping info
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        
        # Get the current page mapping for display
        current_logical_page = st.session_state.page_num
        
        # Ensure current_logical_page is within bounds
        if current_logical_page >= max_pages:
            st.session_state.page_num = max_pages - 1
            current_logical_page = st.session_state.page_num
        elif current_logical_page < 0:
            st.session_state.page_num = 0
            current_logical_page = st.session_state.page_num
        
        original_page_num, updated_page_num = get_page_mapping_info(
            current_logical_page, 
            st.session_state['page_mapping'], 
            num_pages1, 
            num_pages2
        )
        
        with col_nav1:
            # Use unique keys to prevent conflicts
            prev_key = f"prev_page_{current_logical_page}"
            if st.button("⬅ Previous Page", disabled=st.session_state.page_num == 0, key=prev_key):
                st.session_state.page_num = max(0, st.session_state.page_num - 1)
                # Clear page-specific cached data
                st.session_state.pop(f'original_highlighted_{current_logical_page}', None)
                st.session_state.pop(f'updated_highlighted_{current_logical_page}', None)
                st.rerun()
        
        with col_nav2:
            # Show logical page number and physical page mapping
            logical_page_display = st.session_state.page_num + 1
            original_display = f"Orig: {original_page_num + 1}" if original_page_num is not None else "Orig: N/A"
            updated_display = f"Upd: {updated_page_num + 1}" if updated_page_num is not None else "Upd: N/A"
            st.markdown(f"<h3 style='text-align: center;'>Logical Page {logical_page_display} of {max_pages}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.9em;'>({original_display}, {updated_display})</p>", unsafe_allow_html=True)
        
        with col_nav3:
            # Use unique keys to prevent conflicts
            next_key = f"next_page_{current_logical_page}"
            if st.button("Next Page ➡", disabled=st.session_state.page_num >= max_pages - 1, key=next_key):
                st.session_state.page_num = min(max_pages - 1, st.session_state.page_num + 1)
                # Clear page-specific cached data
                st.session_state.pop(f'original_highlighted_{current_logical_page}', None)
                st.session_state.pop(f'updated_highlighted_{current_logical_page}', None)
                st.rerun()
        
        st.markdown("---")

        # Pre-process highlighting if needed (before displaying images)
        original_highlighted_img = None
        updated_highlighted_img = None
        text_changes = []
        
        if (original_page_num is not None and updated_page_num is not None and
            enable_text_highlighting and 
            get_file_type(uploaded_file1.name) == 'pdf' and 
            get_file_type(uploaded_file2.name) == 'pdf'):
            
            # Check if we already have highlighted images for this page
            original_key = f'original_highlighted_{current_logical_page}'
            updated_key = f'updated_highlighted_{current_logical_page}'
            
            if original_key in st.session_state and updated_key in st.session_state:
                # Use cached highlighted images
                original_highlighted_img = st.session_state[original_key]
                updated_highlighted_img = st.session_state[updated_key]
                text_changes = st.session_state.get(f'text_changes_{current_logical_page}', [])
            else:
                # Generate new highlighted images
                with st.spinner("Analyzing text-level changes..."):
                    # Get PDF bytes for both documents
                    uploaded_file1.seek(0)
                    uploaded_file2.seek(0)
                    pdf1_bytes = uploaded_file1.read()
                    pdf2_bytes = uploaded_file2.read()
                    
                    # Create text-highlighted images for both documents
                    original_highlighted_img, updated_highlighted_img, text_changes = create_text_diff_image(
                        pdf1_bytes, pdf2_bytes, original_page_num, updated_page_num
                    )
                    
                    if original_highlighted_img is not None and updated_highlighted_img is not None:
                        # Store the highlighted images for display (page-specific)
                        st.session_state[original_key] = original_highlighted_img
                        st.session_state[updated_key] = updated_highlighted_img
                        st.session_state[f'text_changes_{current_logical_page}'] = text_changes
        
        # Main comparison view
        col1, col2, col3 = st.columns([4, 4, 2])

        with col1:
            st.subheader("Original Version")
            if original_page_num is not None:
                try:
                    # Check if we have large-block annotations for this page
                    lb_original_key = f'lb_original_{current_logical_page}'
                    lb_original_img = st.session_state.get(lb_original_key)
                    
                    # Display in priority order: text+visual highlights > text highlights > visual highlights > original
                    if original_highlighted_img is not None and lb_original_img is not None:
                        # Show text highlights with note about visual changes
                        st.image(original_highlighted_img, use_column_width=True, 
                               caption=f"🔴 Original with Text Deletions - Page {original_page_num + 1}")
                        deletions = sum(1 for c in text_changes if c['type'] == 'deletion')
                        st.success(f"✨ Showing {deletions} red text deletion highlights")
                        
                        # Also show large-block highlights for missing visual content
                        st.image(lb_original_img, use_column_width=True,
                               caption=f"🖼️ Original with Missing Visual Elements - Page {original_page_num + 1}")
                        removals = sum(1 for c in st.session_state.get(f'page_{current_logical_page}_changes', []) if 'large_removal' in str(c.get('type', '')))
                        if removals > 0:
                            st.info(f"🔴 Showing {removals} missing visual element(s) highlighted in red")
                    
                    elif original_highlighted_img is not None:
                        st.image(original_highlighted_img, use_column_width=True, 
                               caption=f"🔴 Original with Deletions - Page {original_page_num + 1}")
                        deletions = sum(1 for c in text_changes if c['type'] == 'deletion')
                        st.success(f"✨ Showing {deletions} red deletion highlights")
                    
                    elif lb_original_img is not None:
                        st.image(lb_original_img, use_column_width=True,
                               caption=f"🖼️ Original with Missing Visual Elements - Page {original_page_num + 1}")
                        removals = sum(1 for c in st.session_state.get(f'page_{current_logical_page}_changes', []) if 'large_removal' in str(c.get('type', '')) or c.get('type') == 'removal')
                        if removals > 0:
                            st.info(f"🔴 Showing {removals} missing visual element(s) highlighted in red")
                    
                    elif 0 <= original_page_num < len(images1):
                        st.image(images1[original_page_num], use_column_width=True, caption=f"Original - Page {original_page_num + 1}")
                    else:
                        st.error(f"Original page {original_page_num + 1} is out of bounds (max: {len(images1)})")
                except Exception as e:
                    st.error(f"Error displaying original page: {e}")
            else:
                st.info("Page not present in Original Document.")
            
            # Display original text for the page
            original_page_text = ""
            if get_file_type(uploaded_file1.name) == 'pdf' and original_page_num is not None:
                uploaded_file1.seek(0)
                doc = fitz.open(stream=uploaded_file1.read(), filetype="pdf")
                if original_page_num < doc.page_count:
                    original_page_text = doc.load_page(original_page_num).get_text()
                doc.close()

        with col2:
            st.subheader("Updated Version")
            if updated_page_num is not None:
                try:
                    if 0 <= updated_page_num < len(images2):
                        img_updated = images2[updated_page_num]
                    else:
                        st.error(f"Updated page {updated_page_num + 1} is out of bounds (max: {len(images2)})")
                        st.session_state[f'page_{current_logical_page}_changes'] = []
                        return
                except Exception as e:
                    st.error(f"Error accessing updated page: {e}")
                    st.session_state[f'page_{current_logical_page}_changes'] = []
                    return
                
                # Run enhanced visual detection first (before any display logic)
                if enable_enhanced_missing_detection and original_page_num is not None:
                    lb_updated_key = f'lb_updated_{current_logical_page}'
                    
                    # Only run if we haven't already processed this page
                    if lb_updated_key not in st.session_state:
                        with st.spinner("🔍 Scanning for missing images, diagrams, and screenshots..."):
                            try:
                                img_original = images1[original_page_num]
                                
                                # Use more aggressive settings for better detection
                                if st.session_state.get('force_aggressive', False):
                                    # Ultra-aggressive forced settings
                                    effective_sensitivity = 0.9
                                    effective_min_area = 1000
                                    st.session_state['force_aggressive'] = False  # Reset after use
                                else:
                                    # Standard aggressive settings
                                    effective_sensitivity = max(0.8, missing_image_sensitivity)  # At least 0.8
                                    effective_min_area = min(3000, min_missing_area)  # At most 3000 pixels
                                
                                enhanced_original_img, enhanced_updated_img, enhanced_changes, enhanced_similarity = detect_missing_images_enhanced(
                                    img_original, img_updated,
                                    min_missing_area=effective_min_area,
                                    sensitivity=effective_sensitivity
                                )
                                
                                if enhanced_changes:
                                    st.success(f"🎯 Found {len(enhanced_changes)} visual changes!")
                                    
                                    # Count types of changes
                                    missing_count = sum(1 for c in enhanced_changes if 'missing' in c.get('type', '').lower())
                                    added_count = sum(1 for c in enhanced_changes if 'add' in c.get('type', '').lower())
                                    
                                    # Store enhanced results for display
                                    if missing_count > 0:
                                        st.session_state[f'lb_original_{current_logical_page}'] = enhanced_original_img
                                        st.info(f"🔴 {missing_count} missing visual elements highlighted in red")
                                    
                                    if added_count > 0:
                                        st.session_state[f'lb_updated_{current_logical_page}'] = enhanced_updated_img
                                        st.info(f"🟢 {added_count} added visual elements highlighted in green")
                                    
                                    # Convert enhanced changes to standard format and store
                                    enhanced_changes_formatted = [{
                                        'type': 'addition' if 'add' in c.get('type', '').lower() else ('removal' if 'missing' in c.get('type', '').lower() else 'modification'),
                                        'description': c.get('description', 'Visual element'),
                                        'bbox': c.get('bbox', (0, 0, 0, 0)),
                                        'number': i + 1,
                                        'confidence': c.get('confidence', 0),
                                        'area': c.get('area', 0)
                                    } for i, c in enumerate(enhanced_changes)]
                                    
                                    # Store changes for summary
                                    st.session_state[f'page_{current_logical_page}_changes'] = enhanced_changes_formatted
                                else:
                                    # No changes found, ensure we don't have stale data
                                    st.session_state.pop(f'lb_original_{current_logical_page}', None)
                                    st.session_state.pop(f'lb_updated_{current_logical_page}', None)
                                    st.session_state[f'page_{current_logical_page}_changes'] = []
                                    
                            except Exception as e:
                                if show_debug_info:
                                    st.error(f"❌ Enhanced detection failed: {e}")
                                # Clean up on error
                                st.session_state.pop(f'lb_original_{current_logical_page}', None)
                                st.session_state.pop(f'lb_updated_{current_logical_page}', None)
                                st.session_state[f'page_{current_logical_page}_changes'] = []
                
                # Check if we have large-block annotations for this page
                lb_updated_key = f'lb_updated_{current_logical_page}'
                lb_updated_img = st.session_state.get(lb_updated_key)
                
                # Display highlighted version - priority order: visual highlights > text highlights > original
                # Check for enhanced detection results first
                if lb_updated_img is not None:
                    st.image(lb_updated_img, use_column_width=True,
                           caption=f"🖼️ Updated with Added Visual Elements - Page {updated_page_num + 1}")
                    additions_visual = sum(1 for c in st.session_state.get(f'page_{current_logical_page}_changes', []) if 'large_addition' in str(c.get('type', '')) or c.get('type') == 'addition')
                    if additions_visual > 0:
                        st.success(f"🟢 Showing {additions_visual} added visual element(s) highlighted in green")
                    
                    # Also show text highlights if available
                    if updated_highlighted_img is not None:
                        st.image(updated_highlighted_img, use_column_width=True, 
                                caption=f"🟢 Updated with Text Additions - Page {updated_page_num + 1}")
                        
                        # Count and display change information
                        deletions = sum(1 for c in text_changes if c['type'] == 'deletion')
                        additions = sum(1 for c in text_changes if c['type'] == 'addition')
                        st.info(f"📚 Found {len(text_changes)} word-level changes ({deletions} deletions, {additions} additions)")
                        
                elif updated_highlighted_img is not None:
                    st.image(updated_highlighted_img, use_column_width=True, 
                            caption=f"🟢 Updated with Text Additions - Page {updated_page_num + 1}")
                    
                    # Count and display change information
                    deletions = sum(1 for c in text_changes if c['type'] == 'deletion')
                    additions = sum(1 for c in text_changes if c['type'] == 'addition')
                    st.info(f"📚 Found {len(text_changes)} word-level changes ({deletions} deletions, {additions} additions)")
                    
                    if show_debug_info:
                        st.success(f"✨ Text highlighting active: Original page has {deletions} red highlights, Updated page has {additions} green highlights")
                    
                    # Show text change details
                    if text_changes and show_debug_info:
                        with st.expander("Text Changes Details", expanded=False):
                            for i, change in enumerate(text_changes):
                                color = "🟢" if change['type'] == 'addition' else "🔴"
                                doc_indicator = f"({change.get('document', 'unknown')})"
                                st.write(f"{color} **{change['type'].title()}** {doc_indicator}: '{change['text']}'")
                    
                    # Hybrid: also run large-block detector to catch missing/added diagrams
                    if enable_large_block_detection:
                        with st.spinner("Scanning for large visual changes (diagrams/images)..."):
                            lb_original_img, lb_updated_img, lb_changes, lb_ssim = detect_large_content_blocks(images1[original_page_num], images2[updated_page_num])
                            if lb_changes:
                                # Store the large-block annotated images for later use
                                st.session_state[f'lb_original_{current_logical_page}'] = lb_original_img
                                st.session_state[f'lb_updated_{current_logical_page}'] = lb_updated_img
                                if show_debug_info:
                                    st.info(f"🖼️ Large-block detector found {len(lb_changes)} region(s)")
                    else:
                        lb_changes = []
                    
                    # Store text changes and large-block changes together for summary
                    changes = text_changes + [{
                        'type': 'addition' if c['type'] == 'large_addition' else ('removal' if c['type'] == 'large_removal' else 'modification'),
                        'description': c['description'],
                        'bbox': c['bbox']
                    } for c in (lb_changes or [])]
                    ssim_score = 1.0  # Not meaningful in hybrid view
                    
                    # Store changes in session state for summary
                    st.session_state[f'page_{current_logical_page}_changes'] = changes
                    
                elif original_page_num is not None:
                    # Perform image comparison if both pages exist but no text highlighting
                    with st.spinner("Analyzing visual differences..."):
                        img_original = images1[original_page_num]

                        # Calibration controls
                        with st.expander("Calibration (optimize detection on this page)"):
                            if st.button("Calibrate on this page"):
                                with st.spinner("Calibrating parameters..."):
                                    rec = calibrate_visual_params(img_original, img_updated, gt_mask_file=gt_mask)
                                    st.session_state['calibrated_params'] = rec
                            rec = st.session_state.get('calibrated_params')
                            if rec:
                                st.success(f"Recommended: threshold={rec['threshold']}, min_size={rec['min_change_size']}, min_area={rec['min_change_area']} (changes={rec['num_changes']}, SSIM={rec['ssim']:.2f})")
                        
                        if show_visual_annotations:
                            # Initialize variables to avoid scope issues
                            diff_img = None
                            changes = []
                            ssim_score = 1.0
                            
                            if not enable_text_highlighting:
                                # Use calibrated parameters if requested and available
                                rec = st.session_state.get('calibrated_params') if apply_calibrated else None
                                use_thr = rec['threshold'] if rec else threshold
                                use_msz = rec['min_change_size'] if rec else min_change_size
                                use_mar = rec['min_change_area'] if rec else min_change_area
                                
                                # Use ultra-sensitive detection for "Ultra Sensitive" mode
                                if detection_mode == "Ultra Sensitive":
                                    with st.spinner("Running ultra-sensitive analysis..."):
                                        diff_img, changes, ssim_score = find_ultra_subtle_differences(
                                            img_original, img_updated
                                        )
                                        st.info(f"🔍 Ultra-sensitive mode detected {len(changes)} micro-changes")
                                else:
                                    diff_img, changes, ssim_score = find_image_differences(
                                        img_original,
                                        img_updated,
                                        threshold=use_thr,
                                        min_change_size=use_msz,
                                        min_change_area=use_mar,
                                        enable_edge_detection=enable_edge_detection,
                                        enable_color_detection=enable_color_detection,
                                        roi=(roi_x, roi_y, roi_w, roi_h) if roi_enable and roi_w > 0 and roi_h > 0 else None
                                    )
                                
                                # Note: Enhanced missing image detection now runs earlier in the process
                                
                                # Also run large-block detector if enabled (as backup)
                                if enable_large_block_detection:
                                    with st.spinner("Scanning for large visual elements..."):
                                        lb_original_img, lb_updated_img, lb_changes, _ = detect_large_content_blocks(img_original, img_updated)
                                        if lb_changes:
                                            # Store for use in the original document display
                                            st.session_state[f'lb_original_{current_logical_page}'] = lb_original_img
                                            st.session_state[f'lb_updated_{current_logical_page}'] = lb_updated_img
                                            
                                            st.image(lb_updated_img, use_column_width=True, caption=f"Large visual changes - Page {updated_page_num + 1}")
                                            st.info(f"🖼️ Detected {len(lb_changes)} large visual change(s) (diagrams/images/tables)")
                                            
                                            # Merge changes with existing ones
                                            lb_changes_formatted = [{
                                                'type': 'addition' if c['type'] == 'large_addition' else ('removal' if c['type'] == 'large_removal' else 'modification'),
                                                'description': c['description'],
                                                'bbox': c['bbox'],
                                                'number': len(changes) + i + 1
                                            } for i, c in enumerate(lb_changes)]
                                            changes.extend(lb_changes_formatted)
                                        elif diff_img is not None:
                                            st.image(diff_img, use_column_width=True, caption=f"Updated - Page {updated_page_num + 1} (SSIM: {ssim_score:.2f})")
                                else:
                                    if diff_img is not None:
                                        st.image(diff_img, use_column_width=True, caption=f"Updated - Page {updated_page_num + 1} (SSIM: {ssim_score:.2f})")
                            
                            
                            # Enhanced debug information
                            if show_debug_info:
                                with st.expander("Debug Information", expanded=True):
                                    st.write(f"**Detection Mode:** {detection_mode}")
                                    st.write(f"**SSIM Score:** {ssim_score:.4f}")
                                    st.write(f"**Threshold:** {threshold}")
                                    st.write(f"**Changes detected:** {len(changes)}")
                                    st.write(f"**Edge Detection:** {'Enabled' if enable_edge_detection else 'Disabled'}")
                                    st.write(f"**Color Detection:** {'Enabled' if enable_color_detection else 'Disabled'}")
                                    st.write(f"**Template Matching:** {'Enabled' if enable_template_matching else 'Disabled'}")
                                    st.write(f"**Min Change Size:** {min_change_size} pixels")
                                    
                                    if changes:
                                        st.write("**Detailed Changes:**")
                                        for i, change in enumerate(changes):
                                            st.write(f"• Change {i+1}: {change['type']} - {change['description']}")
                                            if 'bbox' in change:
                                                bbox = change['bbox']
                                                st.write(f"  Location: ({bbox[0]}, {bbox[1]}) Size: {bbox[2]}x{bbox[3]}")
                                            # Show additional ultra-sensitive details
                                            if detection_mode == "Ultra Sensitive" and 'max_intensity' in change:
                                                st.write(f"  Max Intensity: {change['max_intensity']:.1f}")
                                                st.write(f"  Avg Intensity: {change.get('avg_intensity', 0):.1f}")
                                            if 'confidence' in change:
                                                st.write(f"  Confidence: {change['confidence']:.2f}")
                        else:
                            st.image(img_updated, use_column_width=True, caption=f"Updated - Page {updated_page_num + 1}")
                            # Still perform analysis for summary
                            rec = st.session_state.get('calibrated_params') if apply_calibrated else None
                            use_thr = rec['threshold'] if rec else threshold
                            use_msz = rec['min_change_size'] if rec else min_change_size
                            use_mar = rec['min_change_area'] if rec else min_change_area
                            
                            # Use appropriate detection method
                            if detection_mode == "Ultra Sensitive":
                                _, changes, ssim_score = find_ultra_subtle_differences(img_original, img_updated)
                            else:
                                _, changes, ssim_score = find_image_differences(
                                    img_original,
                                    img_updated,
                                    threshold=use_thr,
                                    min_change_size=use_msz,
                                    min_change_area=use_mar,
                                    enable_edge_detection=enable_edge_detection,
                                    enable_color_detection=enable_color_detection,
                                    roi=(roi_x, roi_y, roi_w, roi_h) if roi_enable and roi_w > 0 and roi_h > 0 else None
                                )
                            
                            # Also run large-block detector if enabled
                            if enable_large_block_detection:
                                lb_original_img, lb_updated_img, lb_changes, _ = detect_large_content_blocks(img_original, img_updated)
                                if lb_changes:
                                    # Store for use in the original document display
                                    st.session_state[f'lb_original_{current_logical_page}'] = lb_original_img
                                    st.session_state[f'lb_updated_{current_logical_page}'] = lb_updated_img
                                    
                                    lb_changes_formatted = [{
                                        'type': 'addition' if c['type'] == 'large_addition' else ('removal' if c['type'] == 'large_removal' else 'modification'),
                                        'description': c['description'],
                                        'bbox': c['bbox'],
                                        'number': len(changes) + i + 1
                                    } for i, c in enumerate(lb_changes)]
                                    changes.extend(lb_changes_formatted)
                        
                        # Pixel-wise comparison if enabled
                        if show_pixel_comparison:
                            with st.expander("Pixel-wise Comparison Details"):
                                pixel_mask = compare_images_pixel_wise(img_original, img_updated, threshold=pixel_threshold)
                                st.write(f"Pixel differences detected: {pixel_mask.sum()} pixels")
                                
                                # Create a visualization of pixel differences
                                import numpy as np
                                diff_visualization = np.zeros_like(np.array(img_original))
                                diff_visualization[pixel_mask] = [255, 0, 0]  # Red for differences
                                
                                # Overlay on original image
                                original_array = np.array(img_original)
                                overlay = np.where(pixel_mask[..., np.newaxis], 
                                                 np.array([255, 0, 0]), 
                                                 original_array)
                                overlay_img = Image.fromarray(overlay.astype(np.uint8))
                                st.image(overlay_img, caption="Pixel-level differences (Red)", use_column_width=True)
                        
                        # Store changes in session state for summary
                        st.session_state[f'page_{current_logical_page}_changes'] = changes
                else:
                    # Display regular updated image when no highlighting is available
                    st.image(img_updated, use_column_width=True, caption=f"Updated - Page {updated_page_num + 1}")
                    # If enhanced detection already ran but found no changes, preserve that
                    if f'page_{current_logical_page}_changes' not in st.session_state:
                        st.session_state[f'page_{current_logical_page}_changes'] = []
            else:
                st.info("Page not present in Updated Document.")
                st.session_state[f'page_{current_logical_page}_changes'] = []

            # Display updated text for the page
            updated_page_text = ""
            if get_file_type(uploaded_file2.name) == 'pdf' and updated_page_num is not None:
                uploaded_file2.seek(0)
                doc = fitz.open(stream=uploaded_file2.read(), filetype="pdf")
                if updated_page_num < doc.page_count:
                    updated_page_text = doc.load_page(updated_page_num).get_text()
                doc.close()

        with col3:
            st.subheader("Changes Summary")
            
            # Count visual changes from all pages
            visual_additions = 0
            visual_removals = 0
            visual_modifications = 0
            
            for key, value in st.session_state.items():
                if key.startswith('page_') and key.endswith('_changes'):
                    for change in value:
                        if change['type'] == 'addition':
                            visual_additions += 1
                        elif change['type'] == 'removal':
                            visual_removals += 1
                        elif change['type'] == 'modification':
                            visual_modifications += 1
            
            # Overall changes summary
            total_additions = text_summary['additions'] + page_additions + visual_additions
            total_removals = text_summary['removals'] + page_removals + visual_removals
            total_modifications = text_summary['modifications'] + visual_modifications
            
            # Use Streamlit metrics for better visibility
            col_add, col_mod, col_rem = st.columns(3)
            
            with col_add:
                st.metric("Additions", total_additions, delta=total_additions if total_additions > 0 else None)
            with col_mod:
                st.metric("Modifications", total_modifications, delta=total_modifications if total_modifications > 0 else None)
            with col_rem:
                st.metric("Removals", total_removals, delta=-total_removals if total_removals > 0 else None)
            
            # Current page changes
            st.subheader("Changes on this Page")
            current_changes = st.session_state.get(f'page_{current_logical_page}_changes', [])
            
            if current_changes:
                for i, change in enumerate(current_changes):
                    change_type = change['type']
                    # Handle both old format (with 'number') and new format (without 'number')
                    change_num = change.get('number', i + 1)
                    description = change.get('description', f"{change_type}: {change.get('text', 'Visual change')}")
                    
                    if change_type == 'addition':
                        color = '#e6ffe6'
                        icon = '➕'
                    elif change_type == 'removal':
                        color = '#ffe6e6'
                        icon = '➖'
                    else:  # modification
                        color = '#fff3e0'
                        icon = '✏️'
                    
                    # Use Streamlit components for better visibility
                    with st.container():
                        st.markdown(f"**{change_num}.** {icon} {description}")
                        # Show document indicator for text changes
                        if 'document' in change:
                            doc_indicator = f"📄 {change['document']} document"
                            st.caption(doc_indicator)
                        st.markdown("---")
            else:
                st.info("No changes detected on this page.")
            
            # Text changes for current page
            if show_text_comparison and original_page_text and updated_page_text:
                st.subheader("Text Changes")
                text_diff_html = highlight_text_differences_html(original_page_text, updated_page_text)
                st.components.v1.html(text_diff_html, height=300, scrolling=True)
        
        # Action buttons
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            if st.button("💾 Save Draft", use_container_width=True):
                # Save current state to session
                draft_data = {
                    'file1_name': uploaded_file1.name,
                    'file2_name': uploaded_file2.name,
                    'current_page': st.session_state.page_num,
                    'total_pages': max_pages,
                    'changes_summary': {
                        'additions': total_additions,
                        'removals': total_removals,
                        'modifications': total_modifications
                    },
                    'timestamp': st.session_state.get('timestamp', 'Unknown')
                }
                st.session_state['draft_data'] = draft_data
                st.success("Draft saved successfully!")
        
        with col_btn2:
            if st.button("📤 Submit Review", use_container_width=True):
                # Generate review summary
                review_summary = generate_review_summary(
                    uploaded_file1.name, uploaded_file2.name,
                    total_additions, total_removals, total_modifications,
                    st.session_state
                )
                
                # Store review data
                st.session_state['review_data'] = review_summary
                st.success("Review submitted successfully!")
                
                # Show review summary
                with st.expander("Review Summary", expanded=True):
                    st.json(review_summary)
        
        with col_btn3:
            if st.button("📊 Generate Report", use_container_width=True):
                generate_comparison_report(
                    uploaded_file1.name, uploaded_file2.name,
                    total_additions, total_removals, total_modifications,
                    st.session_state
                )

        # Reset file pointers
        uploaded_file1.seek(0)
        uploaded_file2.seek(0)

    else:
        st.info("Please upload two documents to compare using the sidebar.")
        
        # Show example of what the interface looks like
        st.markdown("### Example Interface")
        st.markdown("""
        This tool will help you:
        - **Compare documents** side-by-side with visual annotations
        - **Detect changes** including additions, modifications, and removals
        - **Highlight differences** with numbered annotations and color coding
        - **Navigate pages** to review changes systematically
        - **Generate summaries** of all detected changes
        """)

def generate_review_summary(file1_name, file2_name, additions, removals, modifications, session_state):
    """Generates a comprehensive review summary."""
    import datetime
    
    # Collect all page changes
    all_page_changes = []
    for key, value in session_state.items():
        if key.startswith('page_') and key.endswith('_changes'):
            page_num = int(key.split('_')[1])
            all_page_changes.extend([(page_num, change) for change in value])
    
    # Sort by page number
    all_page_changes.sort(key=lambda x: x[0])
    
    summary = {
        'review_metadata': {
            'original_file': file1_name,
            'updated_file': file2_name,
            'review_timestamp': datetime.datetime.now().isoformat(),
            'reviewer': 'Document Comparison Tool'
        },
        'overall_statistics': {
            'total_additions': additions,
            'total_removals': removals,
            'total_modifications': modifications,
            'total_changes': additions + removals + modifications
        },
        'page_by_page_changes': {},
        'change_details': []
    }
    
    # Group changes by page
    for page_num, change in all_page_changes:
        if page_num not in summary['page_by_page_changes']:
            summary['page_by_page_changes'][page_num] = []
        summary['page_by_page_changes'][page_num].append(change)
        summary['change_details'].append({
            'page': page_num + 1,  # Convert to 1-based indexing
            'change_number': change.get('number', 'N/A'),
            'type': change['type'],
            'description': change.get('description', f"{change['type']}: {change.get('text', 'Visual change')}")
        })
    
    return summary

def generate_comparison_report(file1_name, file2_name, additions, removals, modifications, session_state):
    """Generates a detailed comparison report."""
    import datetime
    st.info("Generating comparison report...")
    
    # Create report content
    report_content = f"""
# Document Comparison Report

## Files Compared
- **Original Document:** {file1_name}
- **Updated Document:** {file2_name}
- **Comparison Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Additions:** {additions}
- **Total Removals:** {removals}
- **Total Modifications:** {modifications}
- **Total Changes:** {additions + removals + modifications}

## Detailed Changes by Page
"""
    
    # Add page-by-page details
    for key, value in session_state.items():
        if key.startswith('page_') and key.endswith('_changes') and value:
            page_num = int(key.split('_')[1]) + 1  # Convert to 1-based
            report_content += f"\n### Page {page_num}\n"
            for change in value:
                description = change.get('description', f"{change['type']}: {change.get('text', 'Visual change')}")
                report_content += f"- **{change['type'].title()}:** {description}\n"
    
    # Display the report
    st.markdown(report_content)
    
    # Offer download
    if st.button("📥 Download Report"):
        st.download_button(
            label="Download as Markdown",
            data=report_content,
            file_name=f"comparison_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()