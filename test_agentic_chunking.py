import logging
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()


# 1. Import necessary libraries for file loading
from langchain_community.document_loaders import PyMuPDFLoader

# 2. Import your AgenticChunking class
# Assuming your class is saved in 'agentic_chunking.py'
from agentic_chunking import AgenticChunking


# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgenticRunner")




# ModelName = "gpt-oss:120b-cloud"

ModelName = os.getenv('GPT4_1_NANO_LLM_MODEL_DEPLOYMENT_NAME')

logger.info(f"Using Model: {ModelName}")

def extract_content_from_pdf(file_path):
    """
    Loads a PDF and converts it into the page_map format expected by the Chunker.
    Format: [(page_number, bbox_data, text_content), ...]
    """
    logger.info(f"Loading file: {file_path}")
    
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    
    page_map = []
    offset = 0
    
    for i, page in enumerate(pages):
        # page_number: usually 0-indexed in loader, we make it 1-indexed for readability
        page_num = page.metadata.get("page", i) + 1
        
        # text: content of the page
        text = page.page_content
        
        # bbox: We don't need real bbox data for chunking text, so we pass None
        # The structure requires a tuple of length 3
        page_map.append((page_num, offset, text))
        offset = len(text) # Update offset

        
    logger.info(f"Extracted {len(page_map)} pages.")
    return page_map

def run_test(file_path, model_name=ModelName): # for local model use "llama3"
    
    # 1. Extract Text
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    page_map = extract_content_from_pdf(file_path)

    # 2. Initialize Chunker
    # Ensure 'chonkie' is installed and 'ollama serve' is running
    print(f"\n--- Initializing AgenticChunking with model '{model_name}' ---")
    try:
        chunker = AgenticChunking(
            filename=os.path.basename(file_path),
            max_tokens=2000, # Smaller window for faster testing, increase for prod
            model_name=model_name
        )
    except ImportError as e:
        print(f"Dependency Error: {e}")
        return
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # 3. Run Chunking
    print("--- Starting Chunking Process (This calls the LLM, it may take time) ---")
    try:
        chunks = chunker.chunk(
            page_map=page_map,
            index_id_field="id",           # The key for the ID in the output dict
            index_content_field="content", # The key for the text in the output dict
            index_sourcepage_field="source", # The key for source info
            logger=logger
        )
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Display Results
    print(f"\n--- Processing Complete. Generated {len(chunks)} chunks ---")
    
    # Save to JSON for inspection
    import json
    output_filename = f"{os.path.basename(file_path)}_chunks.json"
    
    # Helper to serialize datetime objects in metadata
    def json_serial(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError (f"Type {type(obj)} not serializable")

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, default=json_serial)
        
    print(f"Full results saved to: {output_filename}")
    
    # Print a preview of the first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i+1}]")
        print(f"ID: {chunk.get('id')}")
        print(f"Tokens: {chunk.get('metadata', {}).get('token_count')}")
        print(f"Content Preview: {chunk.get('content', '')[:150]}...")

if __name__ == "__main__":
    # CHANGE THIS to your actual PDF file path
    target_file = r"files/ES Mod1@AzDOCUMENTS2.pdf" 



    
    # Create a dummy file if it doesn't exist just to test the script logic
    # if not os.path.exists(target_file):
    #     print(f"'{target_file}' not found. Creating a dummy PDF for testing...")
    #     from reportlab.pdfgen import canvas
    #     c = canvas.Canvas(target_file)
    #     c.drawString(100, 750, "This is page 1. Introduction to Agentic Chunking.")
    #     c.drawString(100, 730, "The concept involves using LLMs to split text.")
    #     c.showPage()
    #     c.drawString(100, 750, "This is page 2. Implementation details.")
    #     c.drawString(100, 730, "We use LangChain and Ollama.")
    #     c.save()
    #     print("Dummy PDF created.")

    # Run the test
    # Ensure you have the model pulled: `ollama pull llama3`
    run_test(target_file, model_name=ModelName)