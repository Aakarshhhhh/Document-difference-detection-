import streamlit as st
from PIL import Image, ImageDraw
import io
import fitz
import streamlit.components.v1 as components

from utils.file_conversion import convert_to_comparable_format, get_file_type
from utils.image_comparison import find_image_differences, compare_images_pixel_wise
from utils.text_comparison import compare_texts, highlight_text_differences_html, get_text_diff_summary, find_word_level_changes
import json
import datetime


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
            
            # Enhanced threshold options
            st.subheader("Detection Sensitivity")
            detection_mode = st.selectbox(
                "Detection Mode",
                ["Ultra Sensitive", "Very Sensitive", "Sensitive", "Balanced", "Conservative"],
                index=3  # Default to "Balanced" for better noise reduction
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
                min_change_size = st.slider("Minimum Change Size (pixels)", 5, 50, 10)
            
            # Additional options
            st.header("Display Options")
            show_text_comparison = st.checkbox("Show Text Comparison", value=True)
            show_visual_annotations = st.checkbox("Show Visual Annotations", value=True)
            show_debug_info = st.checkbox("Show Debug Information", value=False)

    if uploaded_file1 and uploaded_file2:
        # Convert files to comparable formats
        with st.spinner("Converting original document..."):
            images1, text1 = convert_to_comparable_format(uploaded_file1)
        with st.spinner("Converting updated document..."):
            images2, text2 = convert_to_comparable_format(uploaded_file2)
        
        # Calculate overall statistics
        num_pages1 = len(images1)
        num_pages2 = len(images2)
        page_additions = max(0, num_pages2 - num_pages1)
        page_removals = max(0, num_pages1 - num_pages2)
        
        # Analyze text changes
        text_differences = compare_texts(text1, text2)
        text_summary = get_text_diff_summary(text_differences)
        
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
        
        # Page navigation
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        with col_nav1:
            if st.button("⬅ Previous Page", disabled=st.session_state.page_num == 0):
                if st.session_state.page_num > 0:
                    st.session_state.page_num -= 1
        with col_nav2:
            st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state.page_num + 1} of {max_pages}</h3>", unsafe_allow_html=True)
        with col_nav3:
            if st.button("Next Page ➡", disabled=st.session_state.page_num >= max_pages - 1):
                if st.session_state.page_num < max_pages - 1:
                    st.session_state.page_num += 1
        
        st.markdown("---")

        # Main comparison view
        col1, col2, col3 = st.columns([4, 4, 2])
        
        current_page_num = st.session_state.page_num

        with col1:
            st.subheader("Original Version")
            if current_page_num < num_pages1:
                st.image(images1[current_page_num], use_column_width=True, caption=f"Original - Page {current_page_num + 1}")
            else:
                st.info("Page not present in Original Document.")
            
            # Display original text for the page
            original_page_text = ""
            if get_file_type(uploaded_file1.name) == 'pdf':
                uploaded_file1.seek(0)
                doc = fitz.open(stream=uploaded_file1.read(), filetype="pdf")
                if current_page_num < doc.page_count:
                    original_page_text = doc.load_page(current_page_num).get_text()
                doc.close()

        with col2:
            st.subheader("Updated Version")
            if current_page_num < num_pages2:
                img_updated = images2[current_page_num]
                
                # Perform image comparison if both pages exist
                if current_page_num < num_pages1 and current_page_num < num_pages2:
                    with st.spinner("Analyzing visual differences..."):
                        img_original = images1[current_page_num]
                        
                        if show_visual_annotations:
                            diff_img, changes, ssim_score = find_image_differences(img_original, img_updated, threshold)
                            st.image(diff_img, use_column_width=True, caption=f"Updated - Page {current_page_num + 1} (SSIM: {ssim_score:.2f})")
                            
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
                        else:
                            st.image(img_updated, use_column_width=True, caption=f"Updated - Page {current_page_num + 1}")
                            # Still perform analysis for summary
                            _, changes, ssim_score = find_image_differences(img_original, img_updated, threshold)
                        
                        # Pixel-wise comparison if enabled
                        if show_pixel_comparison:
                            with st.expander("Pixel-wise Comparison Details"):
                                pixel_mask = compare_images_pixel_wise(img_original, img_updated)
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
                        st.session_state[f'page_{current_page_num}_changes'] = changes
                else:
                    st.image(img_updated, use_column_width=True, caption=f"Updated - Page {current_page_num + 1}")
                    st.session_state[f'page_{current_page_num}_changes'] = []
            else:
                st.info("Page not present in Updated Document (likely an addition).")
                st.session_state[f'page_{current_page_num}_changes'] = []

            # Display updated text for the page
            updated_page_text = ""
            if get_file_type(uploaded_file2.name) == 'pdf':
                uploaded_file2.seek(0)
                doc = fitz.open(stream=uploaded_file2.read(), filetype="pdf")
                if current_page_num < doc.page_count:
                    updated_page_text = doc.load_page(current_page_num).get_text()
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
            current_changes = st.session_state.get(f'page_{current_page_num}_changes', [])
            
            if current_changes:
                for change in current_changes:
                    change_type = change['type']
                    change_num = change['number']
                    description = change['description']
                    
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
            'change_number': change['number'],
            'type': change['type'],
            'description': change['description']
        })
    
    return summary

def generate_comparison_report(file1_name, file2_name, additions, removals, modifications, session_state):
    """Generates a detailed comparison report."""
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
                report_content += f"- **{change['type'].title()}:** {change['description']}\n"
    
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