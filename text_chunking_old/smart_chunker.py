import os
import shutil
import re
from fastapi.responses import JSONResponse
import openai
import json
import time
import base64
import html
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from constants import *
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Dict, Tuple, Union, Optional

from models import GetDocumentsModel
from routers.upload import getFileList
from services.KannadaConverter import KannadaConverter
from services.form_recognizer import FormRecognizer
from utility.credit_helper import update_user_transaction
from utility.load_model import generate_embeddings
from routers.logging_config import queue_logger
from utility.helper import get_Update_ConfigItem, get_user_container, get_upload_container, get_transaction_container
from PIL import Image
import fitz  # PyMuPDF
from websocket_manager import websocket_manager
from services.tesseract_ocr import OCRProcessorTesseract
from dotenv import load_dotenv
from utility.azkeyvault_manager import KeyVaultManager
from langdetect import detect
import json

load_dotenv('acs.env')

upload_container = get_upload_container()
user_container = get_user_container()
tran_container = get_transaction_container()

#---------------------------------Azure Key Vault--------------------------------------------------------------------------------------------
KeyVaultManager = KeyVaultManager()
DOCUAGENT_KEYVAULT_NAME = os.getenv("DOCUAGENT_KEYVAULT_NAME")

AZURE_OPENAI_API_REGION = KeyVaultManager.fetch_tag_value_from_key_and_tag(DOCUAGENT_KEYVAULT_NAME, 'AZURE_OPENAI_API_REGION')
OPENAI_API_TYPE = KeyVaultManager.fetch_tag_value_from_key_and_tag(DOCUAGENT_KEYVAULT_NAME, 'OPENAI_API_TYPE')
AZURE_OPENAI_API_BASE = KeyVaultManager.fetch_tag_value_from_key_and_tag(DOCUAGENT_KEYVAULT_NAME, 'AZURE_OPENAI_API_BASE')
AZURE_OPENAI_API_KEY = KeyVaultManager.fetch_tag_value_from_key_and_tag(DOCUAGENT_KEYVAULT_NAME, 'AZURE_OPENAI_API_KEY')
EMBEDDING_MODEL_DEPLOYMENT_NAME = KeyVaultManager.fetch_tag_value_from_key_and_tag(DOCUAGENT_KEYVAULT_NAME, 'EMBEDDING_MODEL_DEPLOYMENT_NAME_1')
AZURE_OPENAI_CHATGPT_DEPLOYMENT = KeyVaultManager.fetch_tag_value_from_key_and_tag(DOCUAGENT_KEYVAULT_NAME, 'GPT3_LLM_MODEL_DEPLOYMENT_NAME')
AZURE_OPENAI_CHATGPT_MODEL = KeyVaultManager.fetch_tag_value_from_key_and_tag(DOCUAGENT_KEYVAULT_NAME, 'GPT3_LLM_MODEL_NAME')

class DocumentAnalyzer:
    """Analyzes documents to determine the most appropriate chunking method."""
    
    def __init__(self):
        self.table_patterns = [
            r'\|\s*[^\n]+\s*\|',  # Markdown tables
            r'\+[-+]+\+',         # ASCII tables
            r'<table',            # HTML tables
        ]
        self.header_patterns = [
            r'^#+\s',            # Markdown headers
            r'^\d+\.\s',         # Numbered sections
            r'^[IVX]+\.',        # Roman numerals
            r'^\s*[A-Z][A-Z\s]+:', # Section headers
        ]
        self.invoice_patterns = [
            r'invoice',
            r'bill to',
            r'amount due',
            r'total',
            r'payment terms',
            r'date:',
            r'invoice number',
        ]
        self.contract_patterns = [
            r'agreement',
            r'contract',
            r'terms and conditions',
            r'party',
            r'effective date',
            r'termination',
            r'clause',
        ]
        self.research_patterns = [
            r'abstract',
            r'introduction',
            r'methodology',
            r'results',
            r'discussion',
            r'conclusion',
            r'references',
        ]

    def analyze_document(self, text: str, file_extension: str) -> str:
        """
        Analyzes document content and returns the recommended chunking method.
        Returns: 'hybrid', 'table', 'semantic', 'header', or 'recursive'
        """
        # Check file extension first
        if file_extension.lower() in ['.xls', '.xlsx', '.csv']:
            return 'table'
            
        # Convert text to lowercase for pattern matching
        text_lower = text.lower()
        
        # Count patterns
        table_count = sum(len(re.findall(pattern, text)) for pattern in self.table_patterns)
        header_count = sum(len(re.findall(pattern, text)) for pattern in self.header_patterns)
        invoice_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.invoice_patterns)
        contract_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.contract_patterns)
        research_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.research_patterns)
        
        # Determine document type and chunking method
        if invoice_count > 2 or table_count > 3:
            return 'table'
        elif contract_count > 2 or header_count > 3:
            return 'header'
        elif research_count > 2:
            return 'semantic'
        elif table_count > 0 or header_count > 0:
            return 'hybrid'
        else:
            return 'recursive'

def create_sections(
    category_id: str,
    blob_name: str,
    page_map: List[Tuple[int, int, str]],
    mode: str,
    language: str,
    blob_Connection_String: str,
    blob_container_name: str,
    base_threshold: int = 10000,
    buffer_percent: int = 20,
    overlap_sent_count: int = 3
) -> List[Dict[str, Union[str, bool]]]:
    """
    Smart chunking function that selects the appropriate chunking method based on document analysis.
    """
    chunk_id_prefix = blob_name.replace(" ", "_").replace(".", "_")
    input_data = []

    # Handle image files
    if blob_name.lower().endswith((".jpg", ".png", ".jpeg")):
        try:
            image_descriptions = get_image_description(blob_name, mode, blob_Connection_String, blob_container_name)
            for idx, content in enumerate(image_descriptions):
                input_data.append({
                    'id': f"{chunk_id_prefix}_{idx}",
                    'title': blob_name,
                    'category': category_id,
                    'sourcepage': blob_name_from_file_page(blob_name),
                    'content': content
                })
        except Exception as e:
            queue_logger.error(f"Image error for '{blob_name}': {e}")
        return input_data

    # Combine text from first few pages for analysis
    analysis_text = ""
    for page_num, _, text in page_map[:3]:  # Analyze first 3 pages
        analysis_text += text + "\n"

    # Get file extension
    _, file_extension = os.path.splitext(blob_name)

    # Analyze document and select chunking method
    analyzer = DocumentAnalyzer()
    chunking_method = analyzer.analyze_document(analysis_text, file_extension)
    queue_logger.info(f"Selected chunking method for {blob_name}: {chunking_method}")

    # Apply selected chunking method
    if chunking_method == 'table':
        # Use table-aware chunking
        return create_table_aware_sections(
            category_id, blob_name, page_map,
            base_threshold, buffer_percent, overlap_sent_count
        )
    elif chunking_method == 'header':
        # Use header-based chunking
        return create_sections_Markdown(
            category_id, blob_name, page_map, mode, language,
            blob_Connection_String, blob_container_name,
            base_threshold, buffer_percent, overlap_sent_count
        )
    elif chunking_method == 'semantic':
        # Use semantic chunking
        return create_semantic_sections(
            category_id, blob_name, page_map,
            base_threshold, buffer_percent, overlap_sent_count
        )
    elif chunking_method == 'recursive':
        # Use recursive chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=base_threshold,
            chunk_overlap=int(base_threshold * buffer_percent / 100),
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Combine all text
        all_text = ""
        for _, _, text in page_map:
            all_text += text + "\n"
            
        # Split text
        chunks = text_splitter.split_text(all_text)
        
        # Create input data
        for idx, chunk in enumerate(chunks):
            input_data.append({
                'id': f"{chunk_id_prefix}_{idx}",
                'title': blob_name,
                'category': category_id,
                'sourcepage': f"{blob_name}::1-{len(page_map)}",
                'content': chunk.strip()
            })
    else:  # hybrid
        # Use hybrid chunking (combines multiple methods)
        return create_hybrid_sections(
            category_id, blob_name, page_map,
            base_threshold, buffer_percent, overlap_sent_count
        )

    return input_data

# Import the chunking methods from other files
from embed_hybrid import create_sections as create_hybrid_sections
from embed_Recursive import create_sections as create_recursive_sections
from embed_MarkdownHeader import create_sections_Markdown
from embed_table_aware import create_table_aware_sections
from embed_Semantic_Chunker import create_sections as  create_semantic_sections
# Rest of the code remains the same...

def modify_string(input_string):
    # Remove characters that don't match the allowed set: letters, digits, underscore, dash, equal sign, for Index key(id)
    modified_string = re.sub(r'[^a-zA-Z0-9_=\-]', '_', input_string)
    return modified_string

def table_to_html(table):
    table_html = "<table>"
    rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
    for row_cells in rows:
        table_html += "<tr>"
        for cell in row_cells:
            tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
            cell_spans = ""
            if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
            if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
            table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
        table_html +="</tr>"
    table_html += "</table>"
    return table_html

def fetch_file_from_azure_blob(blob_name,blob_encoded_content, mode, blob_Connection_String, blob_container_name):
    """
    Read blob pdf and convert it into text
    """
    try:
        page_map = []

        fr = FormRecognizer(blob_name,mode,blob_Connection_String,blob_container_name)
        pdf_content = blob_encoded_content

        pdf_text = ""
        offset = 0

        extracted_result = fr.extract(pdf_content)

        for page_num, page in enumerate(extracted_result.pages):
            queue_logger.info(f"page_num - {str(page_num)}")
            tables_on_page = [table for table in extracted_result.tables if table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1]*page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >=0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing characters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += extracted_result.content[page_offset + idx]
                elif table_id not in added_tables:
                    page_text += table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num+1, offset, page_text))
            offset += len(page_text)

        return page_map

    except Exception as e:
        queue_logger.error(f"An error occurred in fetch_file_from_azure_blob: {e}", exc_info=True)
        page_map = []
        queue_logger.error(f"Error processing file: {blob_name}")
        queue_logger.error(f"File size: {len(pdf_content)} bytes")
        queue_logger.error(f"File content: {pdf_content[:100]}... (truncated)")

def blob_name_from_file_page(filename, page=0):
    base_name, extension = os.path.splitext(filename)
    if extension.lower() in EXTENSIONS:
        return f"{base_name}-{page}{extension}"
    else:
        return os.path.basename(filename)

def split_text_into_batches(text, batch_size=MAX_SECTION_LENGTH):
    batches = []
    current_position = 0

    while current_position < len(text):
        end_position = min(current_position + batch_size, len(text))

        if end_position == len(text):
            batches.append(text[current_position:end_position])
            break

        for i in range(end_position, current_position, -1):
            if text[i] in (SENTENCE_ENDINGS + WORDS_BREAKS):
                end_position = i + 1
                break

        batches.append(text[current_position:end_position].strip())
        current_position = end_position

    return batches

def get_image_description(blob_name,mode,blob_Connection_String,blob_container_name):
    # Get a reference to the blob
    fr = FormRecognizer(blob_name,mode,blob_Connection_String,blob_container_name)
    content = fr.download()

    # Encode the downloaded image in base64
    encoded_image = base64.b64encode(content).decode('utf-8')

    try:
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            model=AZURE_OPENAI_CHATGPT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze the image and give detailed description of every item and person in the image"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        content = response.choices[0].message.content
        return split_text_into_batches(content)
    except Exception as e:
        queue_logger.error(f"Error occurred while generating image description: {e}")

def get_file_content(blob_name,language,mode,blob_Connection_String,blob_container_name):
    """
    Process a file stored as a blob and extract its text content.
    Uses FormRecognizer for file processing and applies OCR on image blocks.
    """
    fr = FormRecognizer(blob_name, mode,blob_Connection_String,blob_container_name)
    blob_encoded_content, ext = fr.download_and_process()

    queue_logger.info(f"---Processing file: {blob_name}")

    if ext in ('.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.txt','.jpg', '.jpeg', '.png'):
        converted_file_name = blob_name.replace(ext, '.pdf')
        queue_logger.info(f"---Updating file extension to PDF: {converted_file_name}")
    else:
        converted_file_name = blob_name

    if language not in ['en', 'hi', 'ta', 'mr', 'it']:
        doc = fitz.open(stream=blob_encoded_content, filetype="pdf")
        file_content_str = ""
        page_wise_content = []
        prev_text = ""

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_dict = page.get_text("dict")
            page_text = ""
            queue_logger.info(f"Processing page {page_num + 1}")

            for block in page_dict["blocks"]:
                if block.get("type") == 0:  # Text block
                    block_text = block.get("text", "").strip()

                    if block_text:
                        if language in ['en', 'hi', 'ta', 'te', 'mr', 'ml', 'bn', 'or', 'gu']:
                            page_text += block_text + "\n"
                        elif language == 'kn':
                            converter = KannadaConverter()
                            processed_text = converter.process_line(block_text)
                            page_text += processed_text + "\n"
                        else:
                            page_text += block_text + "\n"
                    else:
                        rect = fitz.Rect(block["bbox"])
                        pix = page.get_pixmap(clip=rect)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                        tesseract_map = {
                            'en': "eng",
                            'kn': "eng+kan", 'hi': "eng+hin", 'ta': "eng+tam",
                            'te': "eng+tel", 'mr': "eng+mar", 'ml': "eng+mal",
                            'bn': "eng+ben", 'or': "eng+ori", 'gu': "eng+guj"
                        }
                        tesseract_language = tesseract_map.get(language)

                        if tesseract_language:
                            ocr_processor = OCRProcessorTesseract(blob_name, tesseract_language, blob_Connection_String, blob_container_name)
                            image_text = ocr_processor.process_image_for_ocr(img)
                        else:
                            image_text = ""

                        page_text += image_text + "\n"

                elif block.get("type") == 1:  # Image block
                    rect = fitz.Rect(block["bbox"])
                    pix = page.get_pixmap(clip=rect)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    tesseract_map = {
                        'en': "eng",
                        'kn': "eng+kan", 'hi': "eng+hin", 'ta': "eng+tam",
                        'te': "eng+tel", 'mr': "eng+mar", 'ml': "eng+mal",
                        'bn': "eng+ben", 'or': "eng+ori", 'gu': "eng+guj"
                    }
                    tesseract_language = tesseract_map.get(language)

                    if tesseract_language:
                        ocr_processor = OCRProcessorTesseract(blob_name, tesseract_language,blob_Connection_String,blob_container_name)
                        image_text = ocr_processor.process_image_for_ocr(img)
                    else:
                        image_text = ""
                    page_text += image_text + "\n"

            file_content_str += page_text + "\n"
            page_wise_content.append((page_num + 1, len(prev_text), page_text))
            prev_text = page_text

        if not file_content_str.strip():
            file_content_str = "No text extracted from the document."

        queue_logger.info("Completed file content extraction.")
        return {
            "blob_encoded_content": blob_encoded_content,
            "file_content_str": file_content_str,
            "page_wise_content": page_wise_content,
            "converted_file_name": converted_file_name
        }
  
    else:
        page_wise_content = fetch_file_from_azure_blob(blob_name, blob_encoded_content, mode, blob_Connection_String, blob_container_name)
        file_content_str = page_wise_content[0][2] if page_wise_content else ""
        
        return {
            "blob_encoded_content": blob_encoded_content,
            "file_content_str": file_content_str,
            "page_wise_content": page_wise_content,
            "converted_file_name": converted_file_name
        }

async def chunking_file(category_id, blob_name, language, mode="search",comp_id=None):
    try:
        blob_config = await get_Update_ConfigItem(comp_id, "blob_connection")
        blob_Connection_String = blob_config['Connection_String']
        blob_container_name = blob_config['Container_Name']
    
        print(f"Blob Connection String: {blob_Connection_String}")

        contentObj = get_file_content(blob_name,language,mode,blob_Connection_String,blob_container_name)

        blob_encoded_content = contentObj['blob_encoded_content']
        file_content_str = contentObj['file_content_str']
        page_wise_content = contentObj['page_wise_content']
        converted_file_name = contentObj['converted_file_name']

        lang = detect(file_content_str)
        queue_logger.info(f"---Language detected: {lang}")

        page_map = page_wise_content
   
        if mode == "search":
            return create_sections(category_id, converted_file_name, page_map,mode, language,blob_Connection_String,blob_container_name,base_threshold=10000, buffer_percent=20, overlap_sent_count = 3), converted_file_name	
        else:
            return page_map
            
    except Exception as e:
        queue_logger.error(f"An error occurred in chunking_file: {e}", exc_info=True)
        return []

async def startEmbedding(file_name, email,comp_id,role):
    queue_logger.info(f"***Generating embeddings for file: {file_name}***")
    credit_used = 0
    
    try:
        query = "SELECT * FROM gi_uploads r WHERE r.file_name = @blob_name"
        query_params = [{"name": "@blob_name", "value": file_name}]
        query_result = upload_container.query_items(query=query, parameters=query_params, enable_cross_partition_query=True)

        file_item = next(query_result, None)
        if not file_item:
            queue_logger.error("File item not found. Check your query or data.")
            return

        language = file_item.get('language', 'en')
        queue_logger.info(f"--Fetched file item from Cosmos DB: {file_item}")
    except Exception as e:
        queue_logger.error(f"XXXAn error occurred in Step 1:\n {e}", exc_info=True)

    try:
        chunks_list = []
        chunk_id_prefix = file_name.replace(" ","_").replace(".","_")
        chunks_list, converted_file_name = await chunking_file(file_item['category_id'], file_name, language,mode="search", comp_id=comp_id)
        queue_logger.info(f"File Checked into {len(chunks_list)} chunks")
    except Exception as e:
        queue_logger.error(f"XXXAn error occurred in Step 2:\n {e}", exc_info=True)

    if chunks_list:
        chunk_ids=[]
        total_tokens = 0
        
        try:
            for item in chunks_list:
                chunk_ids.append(item['id'])
                content = item['content']
                content_embeddings, token_used = generate_embeddings(content)
                total_tokens = total_tokens + token_used
                item['contentVector'] = content_embeddings
                item['comp_id'] = file_item['comp_id']
                item['uploaded_by'] = file_item['uploaded_by']
        except Exception as e:
            queue_logger.error(f"XXXAn error occurred in Step 3:\n {e}", exc_info=True)

        try:
            vector_config = await get_Update_ConfigItem(comp_id, "vector_db")
            azure_search_credential = AzureKeyCredential(vector_config["Admin_Key"])
            search_client = SearchClient(
                endpoint= vector_config["Service_Endpoint"],
                index_name= vector_config["Search_Index_Name"],
                credential= azure_search_credential)
   
            batch_size = 1000
            for i in range(0, len(chunks_list), batch_size):
                batch = chunks_list[i:i + batch_size]
                search_client.upload_documents(batch)
                queue_logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(chunks_list) - 1) // batch_size + 1}")
        except Exception as e:
            queue_logger.error(f"XXXAn error occurred in Step 4:\n {i // batch_size + 1}: {e}")
            
        try:
            end_time = time.time()
            ex_time = end_time - file_item['ex_time']
            config_item = await get_Update_ConfigItem(file_item['comp_id'], "token_per_credit")
            token_per_credit = config_item['Token_Per_Credit']
            credit_used = round(total_tokens/token_per_credit,1)
            file_item['chunk_ids'] = str(chunk_ids)
            file_item['token_used'] = total_tokens
            file_item['credit_used'] = credit_used
            file_item['ex_time'] = ex_time
            file_item['status'] = 1
            file_item['converted_file_name'] = converted_file_name
            upload_container.replace_item(item=file_item, body=file_item)
            queue_logger.info(f"Updated file item in Cosmos DB: {file_item}")
        except Exception as e:
            queue_logger.error(f"XXXAn error occurred in Step 5:\n {e}")
            
        try:
            user_query = f"SELECT * FROM c WHERE LOWER(c.email) = LOWER('{email}')"
            user_list = list(user_container.query_items(query=user_query,enable_cross_partition_query=True))
            existing_user = user_list[0]
            existing_user["credit_used"] += credit_used
            existing_user["credit_balance"] -= credit_used
            user_container.upsert_item(existing_user)
   
            tran_data = {
                "email": email,
                "comp_id": file_item['comp_id'],
                "amount": credit_used,
                "balance" : existing_user["credit_balance"],
                "service_type": "Upload",
                "transaction_type": 2
            }
            res = await update_user_transaction(tran_data)
            queue_logger.info(f"Updated User Transaction in Cosmos DB: {res}")
        except Exception as e:
            queue_logger.error(f"XXXAn error occurred in Step 6:\n {e}")

    else:
        file_item['status'] = -1
        upload_container.replace_item(item=file_item, body=file_item)
        queue_logger.error("No chunks found. Exiting the function.")
        return

    try:
        for directory in ['tempfiles', 'temp_folder']:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                queue_logger.info(f"Deleted directory: {directory}")
    except Exception as e:
        queue_logger.error(f"XXXAn error occurred in Step 7 while deleting temp directories:\n {e}")
    
    queue_logger.info(f"***Embedding generation completed for file: {file_name}***")

    fileListPayload = GetDocumentsModel(
        comp_id=comp_id,
        email=email,
        role=role
    )
    file_list_response = await getFileList(fileListPayload)
    
    if file_list_response is None:
        DocumentList = []
    else:
        if isinstance(file_list_response, JSONResponse):
            try:
                file_list_content = json.loads(file_list_response.body.decode("utf-8"))
                DocumentList = file_list_content.get("DocumentList", [])
            except Exception as e:
                DocumentList = []
                error_msg = f"Error parsing file list response: {str(e)}"
                queue_logger.error(error_msg)
        else:
            DocumentList = []

    final_response = {
        "websocket_name":"uploads",
        "DocumentList": DocumentList
    }
    try:
        await websocket_manager.broadcast(json.dumps(final_response))	
    except Exception as e:
        queue_logger.error(str(e))
