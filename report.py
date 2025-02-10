import os
import argparse
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def count_images_in_folder(root_folder, extensions={'.jpg', '.jpeg', '.png'}):
    """
    Recursively counts files in root_folder that have one of the given extensions.
    """
    count = 0
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                count += 1
    return count

def count_files_in_subfolder(folder, subfolder_name, extension_filter):
    """
    Counts files in a given subfolder (non-recursive) that pass the extension_filter.
    """
    subfolder_path = os.path.join(folder, subfolder_name)
    if not os.path.exists(subfolder_path):
        return 0
    count = 0
    for fname in os.listdir(subfolder_path):
        if extension_filter(fname):
            count += 1
    return count

def generate_pdf_report(samples_dir, output_pdf):
    """
    Generates a PDF report that contains, for each data folder:
      - A summary of counts (total images, ref images, ref masks, target image existence, inferred mask existence)
      - A table of reference images (thumbnails) and their corresponding masks.
      - A table showing the target image and the inferred mask.
    """
    styles = getSampleStyleSheet()
    story = []

    # Title page
    title = Paragraph("Data Folders Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 24))
    
    # List the data folders (subdirectories) inside the samples directory.
    data_folders = sorted(
        [os.path.join(samples_dir, d) for d in os.listdir(samples_dir)
         if os.path.isdir(os.path.join(samples_dir, d))]
    )
    
    for folder in data_folders:
        folder_name = os.path.basename(folder)
        story.append(Paragraph(f"Folder: {folder_name}", styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Get summary information
        total_images = count_images_in_folder(folder)
        num_ref_images = count_files_in_subfolder(folder, "refs", lambda f: os.path.splitext(f)[1].lower() == '.jpg')
        num_ref_masks = count_files_in_subfolder(folder, "masks", lambda f: f.lower().endswith("_mask.png"))
        target_exists = os.path.exists(os.path.join(folder, "target.jpg"))
        output_exists = os.path.exists(os.path.join(folder, "output.png"))
        
        summary_data = [
            ["Total Images", total_images],
            ["Reference Images", num_ref_images],
            ["Reference Masks", num_ref_masks],
            ["Target Image", "Yes" if target_exists else "No"],
            ["Inferred Mask", "Yes" if output_exists else "No"]
        ]
        summary_table = Table(summary_data, colWidths=[150, 100])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 12))
        
        # Section: Reference images and their masks
        refs_folder = os.path.join(folder, "refs")
        masks_folder = os.path.join(folder, "masks")
        if os.path.exists(refs_folder):
            ref_files = sorted([f for f in os.listdir(refs_folder) if f.lower().endswith('.jpg')])
            if ref_files:
                story.append(Paragraph("Reference Images and Masks", styles['Heading2']))
                story.append(Spacer(1, 6))
                
                # Table header: two columns ("Ref Image" and "Ref Mask")
                ref_table_data = [["Ref Image", "Ref Mask"]]
                for ref in ref_files:
                    ref_img_path = os.path.join(refs_folder, ref)
                    # Assume corresponding mask follows naming convention: {base}_mask.png
                    base = os.path.splitext(ref)[0]
                    mask_filename = f"{base}_mask.png"
                    mask_img_path = os.path.join(masks_folder, mask_filename)
                    
                    # Create thumbnail objects (scale them to 150x150 points)
                    try:
                        ref_thumb = RLImage(ref_img_path, width=150, height=150)
                    except Exception:
                        ref_thumb = Paragraph("Error loading image", styles['Normal'])
                    if os.path.exists(mask_img_path):
                        try:
                            mask_thumb = RLImage(mask_img_path, width=150, height=150)
                        except Exception:
                            mask_thumb = Paragraph("Error loading mask", styles['Normal'])
                    else:
                        mask_thumb = Paragraph("Mask not found", styles['Normal'])
                    
                    ref_table_data.append([ref_thumb, mask_thumb])
                
                ref_table = Table(ref_table_data, colWidths=[200, 200])
                ref_table.setStyle(TableStyle([
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
                ]))
                story.append(ref_table)
                story.append(Spacer(1, 12))
        
        # Section: Target image and inferred mask
        story.append(Paragraph("Target Image and Inferred Mask", styles['Heading2']))
        story.append(Spacer(1, 6))
        target_img_path = os.path.join(folder, "target.jpg")
        output_img_path = os.path.join(folder, "output.png")
        if os.path.exists(target_img_path):
            try:
                target_img = RLImage(target_img_path, width=200, height=200)
            except Exception:
                target_img = Paragraph("Error loading target image", styles['Normal'])
        else:
            target_img = Paragraph("Target image not found", styles['Normal'])
        if os.path.exists(output_img_path):
            try:
                output_img = RLImage(output_img_path, width=200, height=200)
            except Exception:
                output_img = Paragraph("Error loading output mask", styles['Normal'])
        else:
            output_img = Paragraph("Inferred mask not found", styles['Normal'])
        
        target_table = Table([[target_img, output_img]], colWidths=[250, 250])
        target_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        story.append(target_table)
        story.append(Spacer(1, 24))
        
        # Page break after each folder section
        story.append(PageBreak())
    
    # Build and save the PDF document.
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    doc.build(story)
    print(f"PDF report generated: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a PDF report for data folders including image counts and visible images."
    )
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples",
        help="Path to the root folder containing data subfolders."
    )
    parser.add_argument(
        "--output_pdf",
        type=str,
        default="report.pdf",
        help="Filename for the output PDF report."
    )
    args = parser.parse_args()
    generate_pdf_report(args.samples_dir, args.output_pdf)

if __name__ == "__main__":
    main()
