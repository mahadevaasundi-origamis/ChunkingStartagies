import base64
import io
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TableItem, TextItem, PictureItem, SectionHeaderItem


class PDFParser:
    """
    A comprehensive PDF parser that extracts structured content including text, tables, and images,
    and merges consecutive blocks across pages.
    """
    
    def __init__(self, min_width=200, min_height=200, min_area=1000):
        """
        Initialize the PDFParser with image filtering parameters.
        
        Args:
            min_width: Minimum image width in pixels
            min_height: Minimum image height in pixels
            min_area: Minimum image area in pixels
        """
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
    
    @staticmethod
    def _image_to_base64(pil_image):
        """Helper to convert PIL Image to Base64 string for JSON output"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def extract_structured_json(self, file_path: str):
        """
        Extract structured content from PDF including text, tables, and images.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing extracted content blocks
        """
        # 1. Configuration
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.generate_picture_images = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            print(f"Processing {file_path}...")
            doc = converter.convert(file_path).document
            
        except Exception as e:
            print(f"Conversion failed: {str(e)}")
            return []

        output_data = []
        
        # State variables for grouping
        current_page = -1
        current_type = None
        current_content_buffer = []

        # 2. Helper to flush the buffer
        def flush_buffer():
            if not current_content_buffer:
                return
            
            # Join content based on type
            if current_type == "text":
                final_content = "\n".join(current_content_buffer)
            elif current_type == "table":
                final_content = "\n".join(current_content_buffer)
            elif current_type == "image":
                final_content = current_content_buffer if len(current_content_buffer) > 1 else current_content_buffer[0]
            else:
                final_content = ""

            # Only add non-empty blocks
            if current_type == "text" and not final_content.strip():
                return
            if current_type == "image" and not current_content_buffer:
                return

            output_data.append({
                "page_no": current_page,
                "content_type": current_type,
                "page_content": final_content
            })

        # 3. Iterate through all items
        try:
            for item, level in doc.iterate_items():
                # Determine Item Type
                item_type = "unknown"
                item_page = -1
                
                # Get Page Number safely
                if hasattr(item, "prov") and item.prov:
                    item_page = item.prov[0].page_no
                
                if isinstance(item, TableItem):
                    item_type = "table"
                elif isinstance(item, PictureItem):
                    item_type = "image"
                elif isinstance(item, (TextItem, SectionHeaderItem)):
                    # Ignore empty whitespace items
                    if not item.text.strip():
                        continue
                    item_type = "text"
                else:
                    continue  # Skip unknown types

                # Trigger new block if page or type changes
                if item_page != current_page or item_type != current_type:
                    flush_buffer()
                    
                    # Reset state
                    current_page = item_page
                    current_type = item_type
                    current_content_buffer = []

                # Process Content by Type
                if item_type == "table":
                    try:
                        df = item.export_to_dataframe()
                        if df is not None and not df.empty:
                            html_table = df.to_html(index=False, border=1)
                            current_content_buffer.append(html_table)
                        else:
                            print(f"Empty table on page {item_page}, skipping")
                    except Exception as e:
                        print(f"Table processing error on page {item_page}: {str(e)}")
                        
                elif item_type == "image":
                    try:
                        img_obj = item.get_image(doc)
                        if img_obj:
                            width, height = img_obj.size
                            
                            # Filter by size
                            if width >= self.min_width and height >= self.min_height and (width * height) >= self.min_area:
                                
                                # Create base64 string
                                b64_str = self._image_to_base64(img_obj)
                                
                                img_data = {
                                    "base64": f"data:image/png;base64,{b64_str}",
                                    "width": width,
                                    "height": height,
                                    "page": item_page
                                }
                                current_content_buffer.append(img_data)
                                
                                print(f"Extracted image: {width}x{height}px on page {item_page}")
                            else:
                                print(f"Skipping small image: {width}x{height}px on page {item_page}")
                    except Exception as e:
                        print(f"Image processing error on page {item_page}: {str(e)}")
                        
                elif item_type == "text":
                    current_content_buffer.append(item.text)

            # Final flush
            flush_buffer()

            # Sort by page number for consistent output
            output_data.sort(key=lambda x: x['page_no'])

        except Exception as e:
            print(f"Error during document iteration: {str(e)}")
            
        return output_data
    
    def merge_cross_page_blocks(self, data):
        """
        Merges consecutive blocks across page boundaries if they have the same content_type.
        
        Args:
            data: List of content blocks from extract_structured_json
            
        Returns:
            List of merged content blocks
        """
        if not data:
            print("Empty data")
            return []
        
        merged_data = []
        i = 0
        
        while i < len(data):
            current_block = data[i].copy()
            start_page = current_block['page_no']
            last_page = start_page
            
            # Look ahead to see if next block has same content_type
            while i + 1 < len(data):
                next_block = data[i + 1]
                
                # Check if content types match
                if current_block['content_type'] == next_block['content_type']:
                    # Track the last page number
                    last_page = next_block['page_no']
                    
                    # Merge content based on type
                    if current_block['content_type'] == 'text':
                        current_block['page_content'] += "\n\n" + next_block['page_content']
                    elif current_block['content_type'] == 'table':
                        current_block['page_content'] += "\n" + next_block['page_content']
                    elif current_block['content_type'] == 'image':
                        # For images, ensure we're working with lists of image objects
                        if not isinstance(current_block['page_content'], list):
                            current_block['page_content'] = [current_block['page_content']]
                        
                        if isinstance(next_block['page_content'], list):
                            current_block['page_content'].extend(next_block['page_content'])
                        else:
                            current_block['page_content'].append(next_block['page_content'])
                    
                    i += 1  # Skip the merged block
                else:
                    break  # Content types don't match, stop merging
            
            # Set the final page_no range
            if start_page == last_page:
                current_block['page_no'] = start_page
            else:
                current_block['page_no'] = f"{start_page}-{last_page}"
            
            merged_data.append(current_block)
            i += 1
        
        print(f"Merged {len(data)} blocks into {len(merged_data)} blocks")
        return merged_data
    
    def parse_pdf(self, file_path: str, output_file: str = None, merge_blocks: bool = True):
        """
        Complete PDF parsing pipeline: extract and optionally merge blocks.
        
        Args:
            file_path: Path to the PDF file
            output_file: Optional path to save JSON output
            merge_blocks: Whether to merge consecutive blocks of same type
            
        Returns:
            List of content blocks (merged or unmerged based on merge_blocks parameter)
        """
        # Extract content
        extracted_data = self.extract_structured_json(file_path)
        
        # Optionally merge blocks
        if merge_blocks:
            final_data = self.merge_cross_page_blocks(extracted_data)
        else:
            final_data = extracted_data
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            print(f"Output saved to: {output_file}")
        
        print(f"Extraction Complete. Processed {len(final_data)} content blocks.")
        return final_data

if __name__ == "__main__":

    parser = PDFParser()
    
    pdf_path = r"files/ES Mod1@AzDOCUMENTS2.pdf"
    parser.parse_pdf(pdf_path, output_file="merged_output.json")
