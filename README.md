# Document Difference Detection System

A comprehensive Python-based document comparison tool built with Streamlit that detects and visualizes changes between two versions of the same document. The system supports multiple file formats and provides detailed visual annotations similar to the reference interface shown in your image.

## Features

### 🎯 Core Functionality
- **Multi-format Support**: PDF, DOCX, XLSX, TXT, RTF, CSV, and various image formats (JPG, PNG, BMP, TIFF, GIF)
- **Visual Change Detection**: Numbered annotations with color-coded changes (additions, modifications, removals)
- **Text Comparison**: Word-level and line-level text difference highlighting
- **Pixel-wise Analysis**: Precise pixel-level comparison for detailed visual analysis
- **Page Navigation**: Easy navigation through multi-page documents
- **Comprehensive Reporting**: Generate detailed comparison reports

### 🎨 Visual Interface
- **Side-by-side Comparison**: Original vs Updated document views
- **Numbered Annotations**: Changes are numbered and color-coded for easy reference
- **Changes Summary Panel**: Real-time statistics and categorized change counts
- **Interactive Navigation**: Previous/Next page buttons with page indicators
- **Responsive Design**: Optimized for different screen sizes

### 📊 Analysis Capabilities
- **Change Categorization**: Automatically categorizes changes as additions, modifications, or removals
- **SSIM Scoring**: Structural Similarity Index for quantitative change assessment
- **Text Extraction**: OCR support for image-based documents
- **Multi-page Support**: Handles documents with multiple pages
- **Session Management**: Save drafts and submit reviews

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone or download the project files
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Tesseract OCR for image text extraction:
   - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

### Basic Workflow
1. **Upload Documents**: Use the sidebar to upload original and updated documents
2. **Configure Options**: Set analysis options like threshold and comparison type
3. **Review Changes**: Navigate through pages to review detected changes
4. **Save/Submit**: Save drafts or submit reviews as needed

### Supported File Types
- **PDF Documents**: Full text extraction and visual comparison
- **Word Documents (DOCX)**: Text extraction and visual rendering
- **Excel Spreadsheets (XLSX)**: Data extraction and tabular comparison
- **Text Files (TXT)**: Direct text comparison
- **Rich Text Format (RTF)**: Basic text extraction
- **CSV Files**: Tabular data comparison
- **Image Files**: OCR-based text extraction and visual comparison

### Analysis Options
- **Change Detection Threshold**: Adjust sensitivity (0.1 to 1.0)
- **Pixel-wise Comparison**: Enable for detailed visual analysis
- **Visual Annotations**: Toggle numbered change annotations
- **Text Comparison**: Enable/disable text difference highlighting

## File Structure

```
project/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This file
└── utils/
    ├── file_conversion.py         # File format conversion utilities
    ├── image_comparison.py        # Visual difference detection
    └── text_comparison.py         # Text analysis and highlighting
```

## Key Components

### Image Comparison (`utils/image_comparison.py`)
- **SSIM Analysis**: Structural similarity index for change detection
- **Contour Detection**: Identifies changed regions using OpenCV
- **Visual Annotations**: Draws numbered circles and colored rectangles
- **Change Classification**: Determines if changes are additions, removals, or modifications

### Text Comparison (`utils/text_comparison.py`)
- **Word-level Analysis**: Detailed text difference detection
- **HTML Rendering**: Color-coded text highlighting
- **Change Summarization**: Categorizes and counts text changes
- **Line-by-line Comparison**: Comprehensive text analysis

### File Conversion (`utils/file_conversion.py`)
- **Multi-format Support**: Handles various document types
- **Image Rendering**: Converts documents to comparable image formats
- **Text Extraction**: Extracts text content for analysis
- **Pagination**: Handles multi-page documents

## Advanced Features

### Pixel-wise Comparison
When enabled, provides detailed pixel-level analysis:
- Shows exact pixel differences
- Overlays differences on original image
- Provides pixel count statistics

### Change Detection Algorithm
1. **Preprocessing**: Convert documents to comparable formats
2. **Visual Analysis**: Use SSIM and contour detection
3. **Text Analysis**: Compare extracted text content
4. **Classification**: Categorize changes by type
5. **Annotation**: Add numbered visual indicators

### Report Generation
- **Markdown Reports**: Detailed comparison summaries
- **JSON Export**: Machine-readable change data
- **Download Support**: Save reports locally

## Customization

### Adding New File Types
1. Add conversion function in `utils/file_conversion.py`
2. Update `get_supported_file_types()` function
3. Add file extension to the uploader in `app.py`

### Modifying Change Detection
1. Adjust threshold values in `utils/image_comparison.py`
2. Modify contour detection parameters
3. Update change classification logic

### UI Customization
1. Modify color schemes in the HTML templates
2. Adjust column layouts in `app.py`
3. Add new analysis options in the sidebar

## Troubleshooting

### Common Issues
1. **Font Errors**: Install system fonts or update font paths
2. **OCR Issues**: Ensure Tesseract is properly installed
3. **Memory Issues**: Reduce image resolution or file size
4. **File Format Errors**: Check file integrity and format support

### Performance Optimization
- Use smaller file sizes for faster processing
- Adjust image resolution in conversion settings
- Enable/disable features based on needs

## Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue with detailed information

---

**Note**: This system is designed for document comparison and change detection. For production use, consider additional security measures and performance optimizations based on your specific requirements.
