import difflib
import re
from typing import List, Dict, Tuple
import fitz  # PyMuPDF for text positioning
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def compare_texts(text1: str, text2: str):
    """
    Compares two text strings and returns a list of differences.
    Each difference is a tuple: (type, line_number, content)
    Type can be 'addition', 'removal', 'modification', 'equal'.
    Enhanced with word-level analysis but filtered to reduce noise.
    """
    # Normalize whitespace for better comparison
    text1_normalized = ' '.join(text1.split())
    text2_normalized = ' '.join(text2.split())
    
    # Use line-by-line comparison as primary method (less noisy)
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    # Use unified_diff for line-level changes
    line_differ = difflib.unified_diff(lines1, lines2, fromfile='Original', tofile='Updated', lineterm='')
    
    differences = []
    for line in line_differ:
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        elif line.startswith('-'):
            content = line[1:].strip()
            if content and len(content) > 2:  # Filter out very short changes
                differences.append(('removal', 0, content))
        elif line.startswith('+'):
            content = line[1:].strip()
            if content and len(content) > 2:  # Filter out very short changes
                differences.append(('addition', 0, content))
    
    # For word-level changes, only check if there are significant differences
    if len(differences) == 0:
        # Use SequenceMatcher for word-level analysis only if no line changes
        words1 = text1_normalized.split()
        words2 = text2_normalized.split()
        
        if len(words1) > 10 or len(words2) > 10:  # Only for substantial texts
            matcher = difflib.SequenceMatcher(None, words1, words2)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'delete' and (i2 - i1) > 2:  # Only significant deletions
                    removed_text = ' '.join(words1[i1:i2])
                    differences.append(('removal', 0, removed_text))
                elif tag == 'insert' and (j2 - j1) > 2:  # Only significant additions
                    added_text = ' '.join(words2[j1:j2])
                    differences.append(('addition', 0, added_text))
                elif tag == 'replace' and (i2 - i1) > 1 and (j2 - j1) > 1:  # Only significant modifications
                    old_text = ' '.join(words1[i1:i2])
                    new_text = ' '.join(words2[j1:j2])
                    differences.append(('modification', 0, f"'{old_text}' → '{new_text}'"))
    
    # Filter out very short or whitespace-only changes
    filtered_differences = []
    for diff in differences:
        content = diff[2]
        if content and len(content.strip()) > 3:  # Minimum 3 characters
            filtered_differences.append(diff)
    
    return filtered_differences

def highlight_text_differences_html(text1: str, text2: str):
    """
    Generates enhanced HTML to display text differences with color coding and numbered annotations.
    """
    # Use SequenceMatcher for more detailed word-level comparison
    matcher = difflib.SequenceMatcher(None, text1, text2)
    
    # Create HTML with enhanced styling
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .comparison-container { display: flex; gap: 20px; }
            .text-panel { flex: 1; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .original { background-color: #f9f9f9; }
            .updated { background-color: #f0f8ff; }
            .changes-panel { flex: 0 0 300px; }
            .change-item { 
                margin: 10px 0; padding: 10px; border-radius: 5px; 
                border-left: 4px solid;
            }
            .addition { background-color: #e6ffe6; border-color: #4CAF50; }
            .removal { background-color: #ffe6e6; border-color: #f44336; }
            .modification { background-color: #fff3e0; border-color: #ff9800; }
            .highlight-added { background-color: #a5d6a7; padding: 2px 4px; border-radius: 3px; }
            .highlight-removed { background-color: #ef9a9a; padding: 2px 4px; border-radius: 3px; text-decoration: line-through; }
            .highlight-modified { background-color: #ffcc80; padding: 2px 4px; border-radius: 3px; }
            .change-number { 
                display: inline-block; width: 20px; height: 20px; 
                background-color: #2196F3; color: white; 
                border-radius: 50%; text-align: center; 
                line-height: 20px; font-size: 12px; margin-right: 8px;
            }
        </style>
    </head>
    <body>
        <h2>Document Text Comparison</h2>
        <div class="comparison-container">
            <div class="text-panel original">
                <h3>Original Version</h3>
                <div id="original-text"></div>
            </div>
            <div class="text-panel updated">
                <h3>Updated Version</h3>
                <div id="updated-text"></div>
            </div>
            <div class="changes-panel">
                <h3>Changes Summary</h3>
                <div id="changes-list"></div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate detailed comparison
    changes = []
    change_counter = 1
    
    # Process line-by-line differences
    differ = difflib.unified_diff(text1.splitlines(), text2.splitlines(), 
                                 fromfile='Original', tofile='Updated', lineterm='')
    
    original_html = ""
    updated_html = ""
    
    # Process each line
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    # Use SequenceMatcher for more sophisticated comparison
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    
    i = 0
    j = 0
    change_num = 1
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for line in lines1[i1:i2]:
                original_html += f"<div>{escape_html(line)}</div>"
            for line in lines2[j1:j2]:
                updated_html += f"<div>{escape_html(line)}</div>"
        elif tag == 'delete':
            for line in lines1[i1:i2]:
                original_html += f'<div class="highlight-removed"><span class="change-number">{change_num}</span>{escape_html(line)}</div>'
                changes.append({
                    'number': change_num,
                    'type': 'removal',
                    'description': f'Line removed: "{line[:50]}{"..." if len(line) > 50 else ""}"'
                })
                change_num += 1
        elif tag == 'insert':
            for line in lines2[j1:j2]:
                updated_html += f'<div class="highlight-added"><span class="change-number">{change_num}</span>{escape_html(line)}</div>'
                changes.append({
                    'number': change_num,
                    'type': 'addition',
                    'description': f'Line added: "{line[:50]}{"..." if len(line) > 50 else ""}"'
                })
                change_num += 1
        elif tag == 'replace':
            # Handle replacements as both deletion and addition
            for line in lines1[i1:i2]:
                original_html += f'<div class="highlight-removed"><span class="change-number">{change_num}</span>{escape_html(line)}</div>'
            for line in lines2[j1:j2]:
                updated_html += f'<div class="highlight-added"><span class="change-number">{change_num}</span>{escape_html(line)}</div>'
            changes.append({
                'number': change_num,
                'type': 'modification',
                'description': f'Lines replaced: {i2-i1} lines changed'
            })
            change_num += 1
    
    # Generate changes list HTML
    changes_html = ""
    for change in changes:
        change_class = change['type']
        changes_html += f'''
        <div class="change-item {change_class}">
            <div><span class="change-number">{change['number']}</span>{change['description']}</div>
        </div>
        '''
    
    # Replace placeholders in HTML
    html = html.replace('<div id="original-text"></div>', original_html)
    html = html.replace('<div id="updated-text"></div>', updated_html)
    html = html.replace('<div id="changes-list"></div>', changes_html)
    
    return html

def escape_html(text):
    """Escape HTML special characters."""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))

def get_text_diff_summary(text_differences):
    """Summarizes text differences with detailed counts."""
    additions = sum(1 for d_type, _, _ in text_differences if d_type == 'addition')
    removals = sum(1 for d_type, _, _ in text_differences if d_type == 'removal')
    modifications = sum(1 for d_type, _, _ in text_differences if d_type == 'modification')
    
    return {
        "additions": additions,
        "removals": removals,
        "modifications": modifications,
        "total_changes": additions + removals + modifications
    }

def find_word_level_changes(text1: str, text2: str):
    """
    Finds word-level changes between two texts.
    Returns a list of changes with detailed information.
    """
    words1 = text1.split()
    words2 = text2.split()
    
    matcher = difflib.SequenceMatcher(None, words1, words2)
    changes = []
    change_num = 1
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            changes.append({
                'number': change_num,
                'type': 'removal',
                'content': ' '.join(words1[i1:i2]),
                'position': i1
            })
            change_num += 1
        elif tag == 'insert':
            changes.append({
                'number': change_num,
                'type': 'addition',
                'content': ' '.join(words2[j1:j2]),
                'position': j1
            })
            change_num += 1
        elif tag == 'replace':
            changes.append({
                'number': change_num,
                'type': 'modification',
                'old_content': ' '.join(words1[i1:i2]),
                'new_content': ' '.join(words2[j1:j2]),
                'position': i1
            })
            change_num += 1
    
    return changes

def extract_text_with_positions(pdf_bytes, page_num=0):
    """
    Extract text from PDF with precise positioning information for each word.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_num >= doc.page_count:
            return []
        
        page = doc.load_page(page_num)
        
        # Get text with detailed positioning
        text_dict = page.get_text("dict")
        
        words_with_positions = []
        
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Extract individual words from the span
                        text = span["text"]
                        bbox = span["bbox"]  # (x0, y0, x1, y1)
                        font_info = {
                            "font": span.get("font", ""),
                            "size": span.get("size", 12),
                            "flags": span.get("flags", 0)
                        }
                        
                        # Split into words and estimate positions
                        words = text.split()
                        if words:
                            word_width = (bbox[2] - bbox[0]) / len(text) if len(text) > 0 else 0
                            
                            char_pos = 0
                            for word in words:
                                word_x0 = bbox[0] + char_pos * word_width
                                word_x1 = word_x0 + len(word) * word_width
                                word_bbox = (word_x0, bbox[1], word_x1, bbox[3])
                                
                                words_with_positions.append({
                                    'text': word,
                                    'bbox': word_bbox,
                                    'font_info': font_info
                                })
                                
                                char_pos += len(word) + 1  # +1 for space
        
        doc.close()
        return words_with_positions
        
    except Exception as e:
        print(f"Error extracting text with positions: {e}")
        return []

def compare_documents_with_text_overlay(pdf1_bytes, pdf2_bytes, page1_num=0, page2_num=None):
    """
    Compare two PDF documents and return separate highlighted images for original (with deletions) 
    and updated (with additions) documents, just like in the reference image.
    """
    try:
        # Use the same page number for both if only one is provided
        if page2_num is None:
            page2_num = page1_num
        
        # Extract text with positions from both documents
        words1 = extract_text_with_positions(pdf1_bytes, page1_num)
        words2 = extract_text_with_positions(pdf2_bytes, page2_num)
        
        # Get word-level differences
        words1_list = [w['text'] for w in words1]
        words2_list = [w['text'] for w in words2]
        
        # Use SequenceMatcher for word-level comparison
        matcher = difflib.SequenceMatcher(None, words1_list, words2_list)
        
        # Render both document pages at high DPI
        zoom = 2.0  # 144 DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # Get original document image
        doc1 = fitz.open(stream=pdf1_bytes, filetype="pdf")
        page1 = doc1.load_page(page1_num)
        pix1 = page1.get_pixmap(matrix=mat, alpha=False)
        img1_bytes = pix1.tobytes("png")
        from io import BytesIO
        original_image = Image.open(BytesIO(img1_bytes)).convert('RGB')
        doc1.close()
        
        # Get updated document image
        doc2 = fitz.open(stream=pdf2_bytes, filetype="pdf")
        page2 = doc2.load_page(page2_num)
        pix2 = page2.get_pixmap(matrix=mat, alpha=False)
        img2_bytes = pix2.tobytes("png")
        updated_image = Image.open(BytesIO(img2_bytes)).convert('RGB')
        doc2.close()
        
        # Create overlays for both images
        original_overlay = Image.new('RGBA', original_image.size, (255, 255, 255, 0))
        updated_overlay = Image.new('RGBA', updated_image.size, (255, 255, 255, 0))
        
        draw_original = ImageDraw.Draw(original_overlay)
        draw_updated = ImageDraw.Draw(updated_overlay)
        
        # Colors for highlighting - more visible colors
        deletion_color = (255, 0, 0, 160)    # Brighter red with more opacity for deletions
        addition_color = (0, 200, 0, 160)    # Brighter green with more opacity for additions
        
        # Process differences
        changes_detected = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':  # Text removed - highlight on original document
                for idx in range(i1, i2):
                    if idx < len(words1):
                        word_info = words1[idx]
                        bbox = word_info['bbox']
                        scaled_bbox = (
                            bbox[0] * zoom, bbox[1] * zoom,
                            bbox[2] * zoom, bbox[3] * zoom
                        )
                        
                        # Highlight deletion in red on original document
                        draw_original.rectangle(scaled_bbox, fill=deletion_color)
                        changes_detected.append({
                            'type': 'deletion',
                            'text': word_info['text'],
                            'bbox': scaled_bbox,
                            'document': 'original'
                        })
            
            elif tag == 'insert':  # Text added - highlight on updated document
                for idx in range(j1, j2):
                    if idx < len(words2):
                        word_info = words2[idx]
                        bbox = word_info['bbox']
                        scaled_bbox = (
                            bbox[0] * zoom, bbox[1] * zoom,
                            bbox[2] * zoom, bbox[3] * zoom
                        )
                        
                        # Highlight addition in green on updated document
                        draw_updated.rectangle(scaled_bbox, fill=addition_color)
                        changes_detected.append({
                            'type': 'addition', 
                            'text': word_info['text'],
                            'bbox': scaled_bbox,
                            'document': 'updated'
                        })
            
            elif tag == 'replace':  # Text modified
                # Highlight old text (deletion) in red on original document
                for idx in range(i1, i2):
                    if idx < len(words1):
                        word_info = words1[idx]
                        bbox = word_info['bbox']
                        scaled_bbox = (
                            bbox[0] * zoom, bbox[1] * zoom,
                            bbox[2] * zoom, bbox[3] * zoom
                        )
                        
                        draw_original.rectangle(scaled_bbox, fill=deletion_color)
                        changes_detected.append({
                            'type': 'deletion',
                            'text': word_info['text'],
                            'bbox': scaled_bbox,
                            'document': 'original'
                        })
                
                # Highlight new text (addition) in green on updated document
                for idx in range(j1, j2):
                    if idx < len(words2):
                        word_info = words2[idx]
                        bbox = word_info['bbox']
                        scaled_bbox = (
                            bbox[0] * zoom, bbox[1] * zoom,
                            bbox[2] * zoom, bbox[3] * zoom
                        )
                        
                        draw_updated.rectangle(scaled_bbox, fill=addition_color)
                        changes_detected.append({
                            'type': 'addition',
                            'text': word_info['text'], 
                            'bbox': scaled_bbox,
                            'document': 'updated'
                        })
        
        # Composite the overlays onto the base images
        original_result = Image.alpha_composite(original_image.convert('RGBA'), original_overlay)
        updated_result = Image.alpha_composite(updated_image.convert('RGBA'), updated_overlay)
        
        original_result = original_result.convert('RGB')
        updated_result = updated_result.convert('RGB')
        
        return original_result, updated_result, changes_detected
        
    except Exception as e:
        print(f"Error in text overlay comparison: {e}")
        return None, None, []

def create_text_diff_image(file1_bytes, file2_bytes, page1_num=0, page2_num=None):
    """
    Create highlighted images for both original and updated documents showing text-level changes.
    Returns (original_with_deletions, updated_with_additions, changes_list)
    """
    try:
        # Use the same page number for both if only one is provided
        if page2_num is None:
            page2_num = page1_num
        
        # First try PDF text extraction with positioning
        original_img, updated_img, changes = compare_documents_with_text_overlay(
            file1_bytes, file2_bytes, page1_num, page2_num
        )
        
        if original_img is not None and updated_img is not None:
            return original_img, updated_img, changes
        
        # Fallback to basic rendering if highlighting fails
        doc1 = fitz.open(stream=file1_bytes, filetype="pdf")
        doc2 = fitz.open(stream=file2_bytes, filetype="pdf")
        
        if page1_num < doc1.page_count and page2_num < doc2.page_count:
            # Render both pages without highlighting
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            
            # Original page
            page1 = doc1.load_page(page1_num)
            pix1 = page1.get_pixmap(matrix=mat, alpha=False)
            img1_bytes = pix1.tobytes("png")
            from io import BytesIO
            original_image = Image.open(BytesIO(img1_bytes)).convert('RGB')
            
            # Updated page  
            page2 = doc2.load_page(page2_num)
            pix2 = page2.get_pixmap(matrix=mat, alpha=False)
            img2_bytes = pix2.tobytes("png")
            updated_image = Image.open(BytesIO(img2_bytes)).convert('RGB')
            
            doc1.close()
            doc2.close()
            
            return original_image, updated_image, []
        
        doc1.close()
        doc2.close()
        return None, None, []
        
    except Exception as e:
        print(f"Error creating text diff image: {e}")
        return None, None, []

def create_page_mapping(text1_list, text2_list, similarity_threshold=0.3):
    """
    Create a mapping between pages of two documents based on text similarity.
    text1_list and text2_list are lists of strings, one per page.
    Returns a mapping dict and similarity matrix.
    Optimized for performance with large documents.
    """
    try:
        import numpy as np
        
        num_pages1 = len(text1_list)
        num_pages2 = len(text2_list)
        
        if num_pages1 == 0 or num_pages2 == 0:
            return {}, np.array([])
            
        # For very large documents, use simple sequential mapping
        if num_pages1 > 50 or num_pages2 > 50:
            simple_mapping = {i: i for i in range(min(num_pages1, num_pages2))}
            return simple_mapping, None
        
        # Calculate similarity matrix with optimization
        similarity_matrix = np.zeros((num_pages1, num_pages2))
        
        for i in range(num_pages1):
            for j in range(num_pages2):
                # Calculate text similarity using difflib
                text1_clean = text1_list[i].strip().lower()[:1000]  # Limit to first 1000 chars for performance
                text2_clean = text2_list[j].strip().lower()[:1000]  # Limit to first 1000 chars for performance
                
                # Skip very short texts
                if len(text1_clean) < 10 and len(text2_clean) < 10:
                    similarity_matrix[i, j] = 1.0 if i == j else 0.0
                    continue
                
                # Use SequenceMatcher for similarity with autojunk=False for speed
                similarity = difflib.SequenceMatcher(None, text1_clean, text2_clean, autojunk=False).ratio()
                similarity_matrix[i, j] = similarity
        
        # Create mapping using greedy approach (highest similarity first)
        page_mapping = {}
        used_pages2 = set()
        
        # Sort page pairs by similarity (highest first)
        page_pairs = []
        for i in range(num_pages1):
            for j in range(num_pages2):
                if similarity_matrix[i, j] >= similarity_threshold:
                    page_pairs.append((i, j, similarity_matrix[i, j]))
        
        page_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Assign mappings (greedy: highest similarity first)
        for page1, page2, similarity in page_pairs:
            if page1 not in page_mapping and page2 not in used_pages2:
                page_mapping[page1] = page2
                used_pages2.add(page2)
        
        return page_mapping, similarity_matrix
    
    except ImportError:
        # If numpy is not available, fall back to simple mapping
        print("NumPy not available, using simple page mapping")
        simple_mapping = {}
        for i in range(min(len(text1_list), len(text2_list))):
            simple_mapping[i] = i
        return simple_mapping, None
    except Exception as e:
        print(f"Error creating page mapping: {e}")
        return {}, None

def get_page_mapping_info(logical_page, page_mapping, num_pages1, num_pages2):
    """
    Get the physical page numbers for a given logical page index.
    Returns (original_page_num, updated_page_num) or (None, None) if page doesn't exist.
    """
    # Try to find the best mapping for this logical page
    original_page = None
    updated_page = None
    
    # Check if we have a direct mapping
    if logical_page in page_mapping:
        original_page = logical_page
        updated_page = page_mapping[logical_page]
    elif logical_page < num_pages1 and logical_page < num_pages2:
        # Both documents have this page number, assume direct mapping
        original_page = logical_page
        updated_page = logical_page
    elif logical_page < num_pages1:
        # Only original has this page
        original_page = logical_page
        updated_page = None
    elif logical_page < num_pages2:
        # Only updated has this page (look for reverse mapping)
        # Find if any original page maps to this logical page
        for orig_page, upd_page in page_mapping.items():
            if upd_page == logical_page:
                original_page = orig_page
                updated_page = logical_page
                break
        if original_page is None:
            # No reverse mapping found, treat as new page
            original_page = None
            updated_page = logical_page
    
    return original_page, updated_page

def get_page_mapping_summary(page_mapping, num_pages1, num_pages2):
    """
    Generate a human-readable summary of page mappings.
    """
    total_logical_pages = max(num_pages1, num_pages2)
    mapped_pages = len(page_mapping)
    unmapped_original = num_pages1 - mapped_pages
    unmapped_updated = num_pages2 - mapped_pages
    
    summary = {
        'total_logical_pages': total_logical_pages,
        'mapped_pages': mapped_pages,
        'unmapped_original': max(0, unmapped_original),
        'unmapped_updated': max(0, unmapped_updated),
        'mapping_details': []
    }
    
    # Add details for each logical page
    for logical_page in range(total_logical_pages):
        orig_page, upd_page = get_page_mapping_info(logical_page, page_mapping, num_pages1, num_pages2)
        
        if orig_page is not None and upd_page is not None:
            status = "Mapped"
        elif orig_page is not None:
            status = "Only in original"
        elif upd_page is not None:
            status = "Only in updated"
        else:
            status = "Missing"
        
        summary['mapping_details'].append({
            'logical_page': logical_page + 1,  # 1-based for display
            'original_page': (orig_page + 1) if orig_page is not None else None,
            'updated_page': (upd_page + 1) if upd_page is not None else None,
            'status': status
        })
    
    return summary
