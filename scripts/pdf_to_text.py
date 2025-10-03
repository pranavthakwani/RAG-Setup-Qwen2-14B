#!/usr/bin/env python3
"""
PDF to Text Converter for Kore ChatBot
Converts all PDF files in data/pdfs/ directory to text files in data/txts/ directory
"""

import os
import sys
from pathlib import Path
import PyPDF2
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                logger.warning(f"PDF {pdf_path} is encrypted. Skipping.")
                return None

            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"

            return text.strip()

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return None

def convert_pdfs_to_text(pdf_dir, txt_dir):
    """Convert all PDF files in pdf_dir to text files in txt_dir."""
    pdf_path = Path(pdf_dir)
    txt_path = Path(txt_dir)

    # Create txt directory if it doesn't exist
    txt_path.mkdir(exist_ok=True)

    # Get all PDF files
    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF files to convert")

    success_count = 0
    error_count = 0

    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        try:
            # Extract text
            text = extract_text_from_pdf(pdf_file)

            if text:
                # Create output filename (replace .pdf with .txt)
                txt_filename = pdf_file.stem + ".txt"
                txt_filepath = txt_path / txt_filename

                # Write text to file
                with open(txt_filepath, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)

                success_count += 1
                logger.debug(f"Converted: {pdf_file.name} -> {txt_filename}")
            else:
                error_count += 1
                logger.warning(f"No text extracted from: {pdf_file.name}")

        except Exception as e:
            error_count += 1
            logger.error(f"Failed to convert {pdf_file.name}: {str(e)}")

    return success_count, error_count

def main():
    """Main function to run the PDF conversion."""
    # Define paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    pdf_dir = project_root / "data" / "pdfs"
    txt_dir = project_root / "data" / "txts"

    logger.info("Starting PDF to Text conversion...")
    logger.info(f"PDF directory: {pdf_dir}")
    logger.info(f"Output directory: {txt_dir}")

    # Check if PDF directory exists
    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_dir}")
        sys.exit(1)

    # Convert PDFs
    success_count, error_count = convert_pdfs_to_text(pdf_dir, txt_dir)

    # Report results
    logger.info("Conversion completed!")
    logger.info(f"Successfully converted: {success_count} files")
    logger.info(f"Errors/Skipped: {error_count} files")

    if error_count > 0:
        logger.warning(f"Some files had errors. Check the logs above for details.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
