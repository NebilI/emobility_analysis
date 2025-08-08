import fitz  # PyMuPDF
import os
import argparse

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extracts all images from a PDF and saves them to a directory.

    :param pdf_path: Path to the PDF file.
    :param output_dir: Directory to save the extracted images.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        print(f"[*] Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error: Failed to open PDF file '{pdf_path}'. Reason: {e}")
        return

    image_count = 0
    print(f"[*] Processing {doc.page_count} pages in '{pdf_path}'...")

    # Iterate through each page of the PDF
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        
        # The get_images() method returns a list of image metadata
        image_list = page.get_images(full=True)

        if image_list:
            print(f"[*] Found {len(image_list)} images on page {page_index + 1}")
        else:
            continue # No images on this page

        # Iterate through each image on the current page
        for image_index, img in enumerate(image_list, start=1):
            # The first item in the img tuple is the XREF of the image
            xref = img[0]

            # Extract the raw image bytes and its extension
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Generate a unique filename for the image
            image_filename = f"image_p{page_index + 1}_{image_index}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            # Save the image bytes to a file
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            image_count += 1

    doc.close()
    if image_count > 0:
        print(f"\n[+] Success! Extracted {image_count} images to '{output_dir}'.")
    else:
        print("\n[-] No images were found in the document.")

def main():
    parser = argparse.ArgumentParser(description="A script to extract all images from a PDF file.")
    parser.add_argument("pdf_file", help="Path to the PDF file.")
    parser.add_argument("-o", "--output", default="extracted_images",
                        help="Directory to save the extracted images (default: 'extracted_images').")
    
    args = parser.parse_args()

    if not os.path.isfile(args.pdf_file):
        print(f"Error: The file '{args.pdf_file}' was not found or is not a file.")
        return

    extract_images_from_pdf(args.pdf_file, args.output)

if __name__ == "__main__":
    main()