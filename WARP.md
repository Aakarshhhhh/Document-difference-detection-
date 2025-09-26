# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

A Streamlit-based document comparison system that detects and visualizes differences between document versions. The application supports multiple file formats (PDF, DOCX, XLSX, images, text files) and provides both visual and textual difference detection with numbered annotations and color-coded changes.

## Development Commands

### Running the Application
```powershell
# Start the Streamlit app
streamlit run app.py

# Run with custom port
streamlit run app.py --server.port 8502

# Run with debug logging
streamlit run app.py --logger.level debug
```

### Environment Setup
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install optional OCR support (Tesseract)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Testing & Development
```powershell
# Run Python modules directly for testing
python -m utils.image_comparison
python -m utils.text_comparison
python -m utils.file_conversion

# Test specific functions interactively
python -c "from utils.image_comparison import find_image_differences; help(find_image_differences)"

# Check Streamlit configuration
streamlit config show
```

## Architecture & Code Structure

### Core Components

**Main Application (`app.py`)**
- Streamlit web interface with sidebar for uploads and options
- Page-by-page document navigation system
- Real-time change statistics and summary panels
- Session state management for drafts and calibration
- Three-column layout: Original | Updated | Changes Summary

**Image Processing Pipeline (`utils/image_comparison.py`)**
- Multi-method change detection: SSIM analysis, absolute difference, edge detection, color change detection
- Intelligent image alignment using ECC (Enhanced Correlation Coefficient) with ORB fallback
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
- Contour filtering with area, solidity, and perimeter thresholds
- Smart contour merging to reduce noise while preserving distinct changes
- Adaptive parameter tuning based on detection sensitivity modes

**Document Conversion System (`utils/file_conversion.py`)**
- Unified interface for multiple document formats
- High-DPI PDF rendering (220-240 DPI) for precise visual comparison
- Text-to-image conversion with automatic pagination
- Font fallback system (Arial → DejaVu Sans → default)
- OCR integration for image-based documents using pytesseract

**Text Analysis Engine (`utils/text_comparison.py`)**
- Word-level and line-level difference detection using difflib
- HTML generation for visual text comparison with color coding
- Noise reduction through minimum change length filtering
- Enhanced sequence matching with opcodes processing

### Key Algorithms

**Change Detection Flow:**
1. Document normalization and alignment
2. Multi-method difference detection (SSIM + absolute + edge + color)
3. Contour extraction and intelligent filtering
4. Change classification (addition/removal/modification)
5. Visual annotation with numbered circles and colored rectangles

**Calibration System:**
- Parameter sweeping across threshold/size/area combinations
- Ground truth mask support for IoU-based evaluation
- Heuristic scoring when no ground truth available
- Session-based parameter storage and recommendation

### Critical Implementation Details

**Session State Management:**
- `page_num`: Current page navigation state
- `calibrated_params`: Optimized detection parameters
- `page_{n}_changes`: Per-page change storage for summary generation
- `draft_data` & `review_data`: Save/submit functionality

**Performance Optimizations:**
- Lazy image loading and caching
- Adaptive change limits based on image size
- Multi-level morphological operations for noise reduction
- Memory-efficient PDF processing with stream handling

**Change Classification Logic:**
- Edge density analysis for structure detection
- Brightness and variance comparisons
- Region-based feature extraction
- Heuristic scoring for addition/removal/modification categorization

## Development Guidelines

### Adding New File Format Support
1. Create conversion function in `utils/file_conversion.py`
2. Add text extraction logic for the format
3. Update `get_supported_file_types()` dictionary
4. Add file extension to Streamlit uploader configuration
5. Test with various file sizes and content types

### Modifying Detection Algorithms
- Image comparison parameters are in `find_image_differences()`
- Threshold mappings are in the detection mode dictionary
- Contour filtering logic is in the main detection loop
- Change classification is in `determine_change_type()`

### UI/UX Customization
- Color schemes are defined in HTML templates within text comparison
- Column layouts use Streamlit's column system (typically 3 columns: 4,4,2 ratio)
- Analysis options are in the sidebar with expandable sections
- Progress indicators use Streamlit spinners with descriptive messages

### Debugging & Calibration
- Enable "Show Debug Information" for detailed algorithm insights
- Use ground truth masks (PNG with red markers) for calibration validation
- Parameter calibration runs multiple combinations automatically
- SSIM scores provide quantitative change assessment

## Common Issues & Solutions

**Memory Issues with Large Documents:**
- Reduce DPI in `convert_pdf_to_images()` (default: 220)
- Implement page-wise processing instead of loading all pages
- Use image downsampling for preview modes

**False Positive Detection:**
- Increase minimum change area threshold
- Enable advanced noise reduction options
- Use conservative detection mode
- Apply ROI (Region of Interest) focusing

**Performance Optimization:**
- Cache converted images using Streamlit's session state
- Implement lazy loading for multi-page documents
- Use lower resolution for real-time preview, higher for final analysis

**OCR Integration:**
- Ensure Tesseract PATH is configured correctly
- Handle OCR failures gracefully with fallback messages
- Consider preprocessing images for better OCR accuracy

This system is designed for comprehensive document comparison with both automated detection and manual review capabilities. The modular architecture allows for easy extension and customization of detection algorithms and supported formats.