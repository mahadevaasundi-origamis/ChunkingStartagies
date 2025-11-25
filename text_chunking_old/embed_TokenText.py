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
from langchain.text_splitter import TokenTextSplitter

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
from services.tesseract_ocr import OCRProcessorTesseract  # Ensure this is the correct module and class name
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

#-------------------------------------------------------------------------------------------------------------------------------------------------


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
		# queue_logger.info("Form recognizer results:" + str(extracted_result.pages))

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
			#queue_logger.info(f"Page text length: {len(page_text)}")

			# build page text by replacing characters in table spans with table html
			page_text = ""
			added_tables = set()
			for idx, table_id in enumerate(table_chars):
				if table_id == -1:
					page_text += extracted_result.content[page_offset + idx]
					#queue_logger.info("page text1: %s", page_text)
				elif table_id not in added_tables:
					page_text += table_to_html(tables_on_page[table_id])
					#queue_logger.info("page text2:%s",page_text)
					added_tables.add(table_id)

			page_text += " "
			page_map.append((page_num+1, offset, page_text))
			offset += len(page_text)
			# queue_logger.info("page map:"+str(page_map))
		# with open("page_map.txt", "w") as f:
		#     f.write(str(page_map))
		return page_map

	except Exception as e:
		queue_logger.error(f"An error occurred in fetch_file_from_azure_blob: {e}", exc_info=True)
		# Initialize page_map even if an exception occurs
		page_map = []

		# Log additional details about the file
		queue_logger.error(f"Error processing file: {blob_name}")
		queue_logger.error(f"File size: {len(pdf_content)} bytes")
		queue_logger.error(f"File content: {pdf_content[:100]}... (truncated)")


# def split_text(page_map):
# 	def find_page(offset):
# 		num_pages = len(page_map)
# 		for i in range(num_pages - 1):
# 			if offset >= page_map[i][1] and offset < page_map[i + 1][1]:
# 				return i
# 		return num_pages - 1

# 	all_text = "".join(p[2] for p in page_map)
# 	length = len(all_text)
# 	start = 0
# 	end = length
# 	while (start + SECTION_OVERLAP) < length:
# 		last_word = -1
# 		end = start + MAX_SECTION_LENGTH

# 		if end > length:
# 			end = length
# 		else:
# 			# Try to find the end of the sentence
# 			while end < length and (end - start - MAX_SECTION_LENGTH) < SENTENCE_SEARCH_LIMIT and all_text[end] not in SENTENCE_ENDINGS:
# 				if all_text[end] in WORDS_BREAKS:
# 					last_word = end
# 				end += 1
# 			if end < length and all_text[end] not in SENTENCE_ENDINGS and last_word > 0:
# 				end = last_word # Fall back to at least keeping a whole word
# 		if end < length:
# 			end += 1

# 		# Try to find the start of the sentence or at least a whole word boundary
# 		last_word = -1
# 		while start > 0 and start > end - MAX_SECTION_LENGTH - 2 * SENTENCE_SEARCH_LIMIT and all_text[start] not in SENTENCE_ENDINGS:
# 			if all_text[start] in WORDS_BREAKS:
# 				last_word = start
# 			start -= 1
# 		if all_text[start] not in SENTENCE_ENDINGS and last_word > 0:
# 			start = last_word
# 		if start > 0:
# 			start += 1

# 		section_text = all_text[start:end]
# 		yield (section_text, find_page(start))

# 		last_table_start = section_text.rfind("<table")
# 		if (last_table_start > 2 * SENTENCE_SEARCH_LIMIT and last_table_start > section_text.rfind("</table")):
# 			start = min(end - SECTION_OVERLAP, start + last_table_start)
# 		else:
# 			start = end - SECTION_OVERLAP

# 	if (start + SECTION_OVERLAP) < end:
# 		yield (all_text[start:end], find_page(start))

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
		# Determine the end of the batch
		end_position = min(current_position + batch_size, len(text))

		# If we are at the end of the text, take the remaining text
		if end_position == len(text):
			batches.append(text[current_position:end_position])
			break

		# Look for the nearest sentence ending or word break
		for i in range(end_position, current_position, -1):
			if text[i] in (SENTENCE_ENDINGS + WORDS_BREAKS):
				end_position = i + 1
				break

		# Append the batch to the list
		batches.append(text[current_position:end_position].strip())

		# Move to the next position
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
		# Extract and return the description
		return split_text_into_batches(content)
	except Exception as e:
		queue_logger.error(f"Error occured while generating image description: {e}")

# def create_sections(category_id, blob_name, page_map,mode, language,blob_Connection_String,blob_container_name):
# 	# file_id = filename_to_id(filename)
# 	chunk_id_prefix = blob_name.replace(" ","_").replace(".","_")
# 	input_data = []
# 	if blob_name.lower().endswith((".jpg", ".png", ".jpeg")):
# 		text = get_image_description(blob_name,mode,blob_Connection_String,blob_container_name)
# 		item = []
# 		for index, content in enumerate(text):
# 			item.append({
# 				'id': f"{chunk_id_prefix}_{index}",
# 				'title': blob_name,
# 				'category': category_id,
# 				'sourcepage': blob_name_from_file_page(blob_name),
# 				'content': content
# 			})
# 		return item
# 	else:
# 		if language in ['kn', 'ca', 'ml', 'bn', 'or', 'gu', 'te']: 		# Only Add the Tessract Languages
# 			text = []
# 			#Page map is an array of tuples (page_num, offset, text) Need to make text same as output of split_text
# 			for i, (page_num, offset, content) in enumerate(page_map):
# 				text.append((content, page_num))
# 		else:
# 			text = split_text(page_map) #split_text returns a generator of tuples (text, page_num)


# 		for i, (content, pagenum) in enumerate(text):
# 			item = {
# 				'id': f"{chunk_id_prefix}_{i+1}",
# 				'title': blob_name,
# 				'category': category_id,
# 				"sourcepage": blob_name_from_file_page(blob_name, pagenum),
# 				# 'blob_name': blob_name,
# 				'content': content
# 			}
# 			input_data.append(item)

# 		return input_data


def split_into_sentences(text):
	sentences = re.split(r'(?<=[.!?])\s+', text.strip())
	return [s.strip() for s in sentences if s.strip()]

def chunk_sentences_by_char_limit(sentences, limit):
	chunks = []
	current_chunk = ""
	for sentence in sentences:
		if len(current_chunk) + len(sentence) + 1 <= limit:
			current_chunk += " " + sentence if current_chunk else sentence
		else:
			chunks.append(current_chunk)
			current_chunk = sentence
	if current_chunk:
		chunks.append(current_chunk)
	return chunks

def create_sections(
	category_id, blob_name, page_map, mode, language,
	blob_Connection_String, blob_container_name,
	base_threshold, buffer_percent, overlap_sent_count
):
	chunk_id_prefix = blob_name.replace(" ", "_").replace(".", "_")
	input_data = []

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
	else:
		# Combine all text from page_map into a single string
		all_text = ""
		page_ranges = []
		current_page = None
		start_page = None
		
		for page_num, _, text in page_map:
			if current_page is None:
				current_page = page_num
				start_page = page_num
			elif page_num != current_page + 1:
				# If there's a gap in page numbers, add the current range
				page_ranges.append((start_page, current_page))
				start_page = page_num
			current_page = page_num
			all_text += text + "\n"
		
		# Add the final page range
		if start_page is not None:
			page_ranges.append((start_page, current_page))

		# Initialize TokenTextSplitter
		text_splitter = TokenTextSplitter(
			chunk_size=base_threshold,
			chunk_overlap=int(base_threshold * buffer_percent / 100),
			encoding_name="cl100k_base"  # This is the encoding used by GPT models
		)

		# Split the text into chunks
		chunks = text_splitter.split_text(all_text)

		# Create input data entries for each chunk
		for idx, chunk in enumerate(chunks):
			# Find which page range this chunk belongs to
			chunk_start = all_text.find(chunk)
			chunk_end = chunk_start + len(chunk)
			
			# Find the corresponding page range
			current_pos = 0
			chunk_page_range = None
			for start_page, end_page in page_ranges:
				range_text = ""
				for page_num, _, text in page_map:
					if start_page <= page_num <= end_page:
						range_text += text + "\n"
						current_pos += len(text) + 1
						if current_pos >= chunk_start:
							chunk_page_range = (start_page, end_page)
							break
				if chunk_page_range:
					break

			if chunk_page_range:
				start_page, end_page = chunk_page_range
				page_range = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)
			else:
				page_range = "1"  # Default to page 1 if we can't determine the range

			input_data.append({
				'id': f"{chunk_id_prefix}_{page_range}_{idx}",
				'title': blob_name,
				'category': category_id,
				'sourcepage': f"{blob_name}::{page_range}",
				'content': chunk.strip()
			})

	return input_data

def get_file_content(blob_name,language,mode,blob_Connection_String,blob_container_name):
	"""
	Process a file stored as a blob and extract its text content.
	Uses FormRecognizer for file processing and applies OCR on image blocks.
	Leverages OCRProcessor (from computer_vision.py) for Azure OCR and
	OCRProcessorTesseract (from tesseract_ocr.py) for Tesseract OCR based on language.
	"""

	# Initialize FormRecognizer (logic defined in embedService.py)
	fr = FormRecognizer(blob_name, mode,blob_Connection_String,blob_container_name)
	blob_encoded_content, ext = fr.download_and_process()

	queue_logger.info(f"---Processing file: {blob_name}")

	# Update file extension if needed (e.g., convert .doc, .xls, etc. to PDF)
	if ext in ('.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.txt','.jpg', '.jpeg', '.png'):
		converted_file_name = blob_name.replace(ext, '.pdf')
		queue_logger.info(f"---Updating file extension to PDF: {converted_file_name}")
	else:
		converted_file_name = blob_name

	if language not in ['en', 'hi', 'ta', 'mr', 'it']:
		# Open the PDF from the byte stream using PyMuPDF (fitz)
		doc = fitz.open(stream=blob_encoded_content, filetype="pdf")
		file_content_str = ""
		page_wise_content = []
		prev_text = ""

		# Process each page of the document
		for page_num in range(len(doc)):
			page = doc[page_num]
			page_dict = page.get_text("dict")
			page_text = ""
			queue_logger.info(f"Processing page {page_num + 1}")

			# Process each block in the page
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
		# Use Form Recognizer for text extraction
		page_wise_content  = fetch_file_from_azure_blob(blob_name, blob_encoded_content, mode, blob_Connection_String, blob_container_name)
		file_content_str = page_wise_content[0][2] if page_wise_content else ""
		
		return {
			"blob_encoded_content": blob_encoded_content,
			"file_content_str": file_content_str,
			"page_wise_content": page_wise_content,
			"converted_file_name": converted_file_name
		}

async def chunking_file(category_id, blob_name, language, mode="search",comp_id=None):
	try:
	 
		#Getting Blob Details from config container 
		blob_config = await get_Update_ConfigItem(comp_id, "blob_connection")
		blob_Connection_String = blob_config['Connection_String']
		blob_container_name = blob_config['Container_Name']
	
		print(f"Blob Connection String: {blob_Connection_String}")


		# GET File content
		contentObj = get_file_content(blob_name,language,mode,blob_Connection_String,blob_container_name)

		blob_encoded_content = contentObj['blob_encoded_content']
		file_content_str = contentObj['file_content_str']
		page_wise_content = contentObj['page_wise_content']
		converted_file_name = contentObj['converted_file_name']

		# Detect Language
		lang = detect(file_content_str)
  
		queue_logger.info(f"---Language detected: {lang}")

		# if(language in ['kn', 'ca', 'ml', 'bn', 'or', 'gu', 'te']):
		# 	# page_map  = fetch_file_from_azure_blob_ocr(blob_name, blob_encoded_content)
		# 	queue_logger.info(f"User Language is picked --------> {language} and page map is --------> Get File Content")
		# 	page_map = page_wise_content
		# elif language in ['en', 'hi', 'ta', 'mr']:
		# 	queue_logger.info(f"User Language is picked --------> {language} and page map is --------> Form Recognizer")
		# 	page_map  = fetch_file_from_azure_blob(blob_name, blob_encoded_content, mode)
		# else:
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
	# Step 1: Get file item from Cosmos DB
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

	# Step 2: Chunk the file
	try:
		chunks_list = []
		chunk_id_prefix = file_name.replace(" ","_").replace(".","_")
		chunks_list, converted_file_name = await chunking_file(file_item['category_id'], file_name, language,mode="search", comp_id=comp_id)
		queue_logger.info(f"File Checked into {len(chunks_list)} chunks")
		# print(chunks_list)
		
	except Exception as e:
		queue_logger.error(f"XXXAn error occurred in Step 2:\n {e}", exc_info=True)

	if chunks_list:
		chunk_ids=[]
		total_tokens = 0
		
		# Step 3: Generate embeddings for each chunk
		try:
			for item in chunks_list:
				chunk_ids.append(item['id'])
				content = item['content']
				content_embeddings, token_used = generate_embeddings(content)
				total_tokens = total_tokens + token_used
				item['contentVector'] = content_embeddings
				item['comp_id'] = file_item['comp_id']
				item['uploaded_by'] = file_item['uploaded_by']
				# queue_logger.info(f"Generated embeddings for chunk: {file_name}")
		except Exception as e:
			queue_logger.error(f"XXXAn error occurred in Step 3:\n {e}", exc_info=True)


		#Step 4: Saving the embeddings in Azure AI Search
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
			
		#Step 5: Updating File info in Cosmos DB
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
			
		# Step 6: Update Transactions Table
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

	# Step 7: Delete The File in Tempfiles folder
	try:
		for directory in ['tempfiles', 'temp_folder']:
			if os.path.exists(directory):
				shutil.rmtree(directory)
				queue_logger.info(f"Deleted directory: {directory}")
	except Exception as e:
		queue_logger.error(f"XXXAn error occurred in Step 7 while deleting temp directories:\n {e}")
	
	queue_logger.info(f"***Embedding generation completed for file: {file_name}***")


	# Step 8 - Get the latest file list and send to frontend
	fileListPayload = GetDocumentsModel(
		comp_id=comp_id,
		email=email,
		role=role
	)
	file_list_response = await getFileList(fileListPayload)
	
	# Check if file_list_response exists and handle if it's None
	if file_list_response is None:
		DocumentList = []
	else:
		# Check if it's a JSONResponse and extract the data if necessary
		if isinstance(file_list_response, JSONResponse):
			try:
				file_list_content = json.loads(file_list_response.body.decode("utf-8"))  # Decode JSONResponse body
				DocumentList = file_list_content.get("DocumentList", [])  # Extract the 'DocumentList' key
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
