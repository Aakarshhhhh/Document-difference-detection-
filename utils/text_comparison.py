import difflib
import re
from typing import List, Dict, Tuple

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