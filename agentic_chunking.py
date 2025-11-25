from typing import List, Optional, Literal, Dict, Any
import hashlib
import uuid
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()



# --- Pydantic V1 Imports ---
# Crucial: LangChain's OutputParser is strictly tied to Pydantic V1 logic in many versions.
try:
    from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
except ImportError:
    try:
        from pydantic.v1 import BaseModel, Field, ValidationError
    except ImportError:
        from pydantic import BaseModel, Field, ValidationError

# --- LangChain Imports ---
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Base Class Import ---
# Assumes 'providers.services' exists in your project structure
# from providers.services import Chunker

# --- Chonkie Import ---
try:
    from chonkie import TokenChunker
except ImportError:
    TokenChunker = None


# GPT4_1_LLM_MODEL_DEPLOYMENT_NAME=gpt-41
# GPT4_1_MODEL_NAME=gpt-4.1
# GPT4_1_MINI_LLM_MODEL_DEPLOYMENT_NAME=gpt-41-mini
# GPT4_1_MINI_MODEL_NAME=gpt-4.1-mini
# GPT4_1_NANO_LLM_MODEL_DEPLOYMENT_NAME=gpt-41-nano
# GPT4_1_NANO_MODEL_NAME=gpt-4.1-nano
GPT4_1_AZURE_OPENAI_API_KEY=os.getenv('GPT4_1_AZURE_OPENAI_API_KEY')
GPT4_1_AZURE_OPENAI_API_BASE=os.getenv('GPT4_1_AZURE_OPENAI_API_BASE')
GPT4_1_OPENAI_API_TYPE=os.getenv('GPT4_1_OPENAI_API_TYPE')
GPT4_1_AZURE_OPENAI_API_VERSION=os.getenv('GPT4_1_AZURE_OPENAI_API_VERSION')

print(f"GPT4_1_AZURE_OPENAI_API_KEY: {GPT4_1_AZURE_OPENAI_API_KEY}")



# ==========================================
# 1. System Prompt Definition
# ==========================================
def document_chunker_prompt():
    return """
    You are an expert in document segmentation and chunking. Your task is to split the given page content into semantically meaningful chunks while strictly preserving structure, layout, and hierarchy — as they would appear in a Word document or formal report — to support accurate retrieval in downstream RAG systems.

    ⚠️ Follow the rules below with absolute precision. Do not improvise or ignore instructions.

    ---

    📏 Chunking Rules:

    1. **Chunk Size Limit:**
    - Each chunk must NOT exceed 1000 tokens.
    - Tables and bullet lists may exceed this limit but must remain whole — do not split rows or list items.
    - If a chunk exceeds the limit, split it into multiple semantically coherent chunks while preserving headings and meaning.

    2. **Table Format:**
    - For any `chunk_type: "table"`, return the table content in **valid HTML table format** (`<table>`, `<tr>`, `<td>`, etc.).
    - Preserve headers, row order, and alignment as closely as possible to the original.
    - If a section contains both text and a table, treat the entire chunk as a `"table"` and return both — but wrap only the table portion in HTML.

    3. **Exclude Figure Content:**
    - Strictly ignore `<figure>` tags, logos, scanned images, and decorative graphics unless they contain real explanatory text.
    - Do **not** generate any `chunk_type: "figure"` entries unless a meaningful caption or description is included.

    4. **Minimum Content Rule:**
    - Do not create a chunk if it contains fewer than **10 words**, unless it is a bullet list item or table row.
    - Merge short content with nearby chunks when possible, ensuring it remains coherent.

    5. **Preserve Structural Integrity:**
    - Never split:
        - Bullet lists — keep the full list in a single chunk.
        - Tables — keep the full table in one chunk.
        - Headings from their related paragraphs — always keep them together.

    6. **Chunk Types:**
    Use one of the following values for `chunk_type`:
    - `"text"`: For normal prose or paragraph-style content.
    - `"bullet_list"`: For grouped bullet or numbered lists.
    - `"table"`: For structured data (rows/columns or key-value pairs). Format the table as HTML.
    - `"figure"`: Only if a meaningful caption or explanation exists (otherwise exclude).

    7. **Metadata for Each Chunk:**
    For each chunk, include the following fields:
    - `chunk_type`: One of `"text"`, `"bullet_list"`, or `"table"` (avoid `"figure"` unless content is substantial).
    - `section_title`: A clear, meaningful title (inferred from headings if not explicitly given).
    - `chunk_summary`: A 1-line summary of the content.
    - `content`: The raw chunk content — plain text for all types except HTML for `"table"`.

    ---

    ✅ **Output Format:**
    Return a valid **JSON array** only. Each object in the array must have exactly:
    - `"chunk_type"` (string)
    - `"section_title"` (string)
    - `"chunk_summary"` (string)
    - `"content"` (string — plain text or HTML depending on chunk_type)

    Do **not** include any markdown, headers, or explanatory text outside the JSON.
    """

# ==========================================
# 2. Pydantic Models for LLM Interaction
# ==========================================

class LLMChunkResult(BaseModel):
    """Represents a single chunk returned by the LLM."""
    chunk_type: str = Field(description="One of 'text', 'bullet_list', 'table', 'figure'")
    section_title: str = Field(description="A clear, meaningful title for the section")
    chunk_summary: str = Field(description="A 1-line summary of the content")
    content: str = Field(description="The content text or HTML table")

class SemanticSplitResult(BaseModel):
    """The root object expected from the LLM."""
    chunks: List[LLMChunkResult] = Field(description="List of semantic chunks")

# ==========================================
# 3. Pydantic Models for Final Metadata
# ==========================================

class ChunkMetadata(BaseModel):
    doc_id: str
    chunk_id: str
    source_path: str
    mime_type: str
    page_number: Optional[int] = None
    bbox: Optional[List[float]] = None 
    section_title: Optional[str] = "General"
    chunk_summary: Optional[str] = None 
    chunk_type: str = Field(description="The type of chunk determined by LLM (text, table, etc.)")
    content_type: Literal["narrative", "tabular", "code", "markdown"]
    token_count: int
    hash_sha256: str
    vector_metric: str = "cosine"
    embedding_model: str
    table_schema: Optional[str] = None 
    neighbors: List[str] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class Chunk(BaseModel):
    text: str
    tokens: List[int]
    start_char: int
    end_char: int
    metadata: ChunkMetadata

# ==========================================
# 4. Agentic Chunking Class
# ==========================================

class AgenticChunking():
    """
    Splits text using an LLM that returns a strict Pydantic structure based on specific prompting rules.
    """

    def __init__(self, filename: str, max_tokens: int = 4000, model_name: str = "deepseek-v3.1:671b-cloud", embedding_model_name: str = "text-embedding-3-small"):
        self.filename = filename
        self.max_tokens = int(max_tokens) if max_tokens is not None else 4000
        self.embedding_model_name = embedding_model_name
        
        # Initialize Chonkie for managing the context window size safely
        if TokenChunker:
            # We target 85% of max_tokens to leave room for the longer system prompt
            self.pre_chunker = TokenChunker(chunk_size=int(self.max_tokens * 0.85))
        else:
            raise ImportError("The 'chonkie' library is required. Please install it.")

        # Initialize ChatOllama with JSON format enforcement
        # self.llm = ChatOllama(
        #     model=model_name,
        #     temperature=0,
        #     format="json", 
        # )
        self.llm = AzureChatOpenAI(
            model_name=model_name,
            temperature=0,
            max_tokens=self.max_tokens,
            openai_api_key=GPT4_1_AZURE_OPENAI_API_KEY,
            openai_api_base=GPT4_1_AZURE_OPENAI_API_BASE,
            openai_api_type=GPT4_1_OPENAI_API_TYPE,
            openai_api_version=GPT4_1_AZURE_OPENAI_API_VERSION
        )
        
        # Setup the Parser
        self.parser = PydanticOutputParser(pydantic_object=SemanticSplitResult)
        
        # Define the Prompt Template
        # IMPORTANT: Using tuple syntax ("system", ...) and ("human", ...) enables strict variable replacement
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", document_chunker_prompt()),
            ("human", "{format_instructions}\n\n📄 Here is the page content:\n{text}")
        ])

        print(f"AgenticChunking initialized. Model: {model_name}, Context: {self.max_tokens}")

    def chunk(self, page_map: List[tuple], index_id_field: str, index_content_field: str, index_sourcepage_field: str, logger: logging.Logger) -> List[dict]:
        """
        Main execution method following sentence_chunking signature.
        """
        logger.debug(f"Chunking {self.filename} with {len(page_map)} pages")
        
        # Normalize input
        normalized_map = []
        for item in page_map:
            if len(item) == 3:
                page_num, _, text = item
                normalized_map.append((page_num, text))
            else:
                raise ValueError(f"Unexpected page_map format: {item}")

        final_output_dicts: List[dict] = []
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, self.filename))
        
        current_buffer_text = ""
        current_page_nums = []
        
        for page_num, page_text in normalized_map:
            # Tentatively add next page
            test_text = (current_buffer_text + "\n" + page_text) if current_buffer_text else page_text
            
            # Check size using Chonkie
            pre_check = list(self.pre_chunker.chunk(test_text))
            
            if len(pre_check) > 1 and current_buffer_text:
                # Buffer is full. Process it.
                logger.info(f"Context window full. Processing pages {current_page_nums[0]}-{current_page_nums[-1]}...")
                
                batch_chunks = self._process_batch(
                    current_buffer_text, doc_id, current_page_nums, logger, 
                    index_id_field, index_sourcepage_field, index_content_field
                )
                final_output_dicts.extend(batch_chunks)
                
                # Reset buffer with current page
                current_buffer_text = page_text
                current_page_nums = [page_num]
            else:
                current_buffer_text = test_text
                current_page_nums.append(page_num)

        # Process any remaining text
        if current_buffer_text:
            logger.info(f"Processing final batch pages {current_page_nums[0]}-{current_page_nums[-1]}...")
            
            batch_chunks = self._process_batch(
                current_buffer_text, doc_id, current_page_nums, logger, 
                index_id_field, index_sourcepage_field, index_content_field
            )
            final_output_dicts.extend(batch_chunks)

        # Post-Processing: Link Neighbors
        for i in range(len(final_output_dicts)):
            curr = final_output_dicts[i]
            neighbors = []
            if i > 0:
                neighbors.append(final_output_dicts[i-1][index_id_field])
            if i < len(final_output_dicts) - 1:
                neighbors.append(final_output_dicts[i+1][index_id_field])
            
            curr["metadata"]["neighbors"] = neighbors

        return final_output_dicts

    def _process_batch(self, text: str, doc_id: str, page_nums: List[int], logger: logging.Logger, id_field: str, source_field: str, content_field: str) -> List[dict]:
        """
        Sends text to LLM, parses rich object response, and formats into Chunk objects.
        """
        # 1. Get Semantic Splits from LLM (Returns List[LLMChunkResult])
        llm_chunks = self._agentic_split(text, logger)
        
        batch_results = []
        page_range_str = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])
        current_char_idx = 0
        
        for idx, llm_chunk in enumerate(llm_chunks):
            # Extract data from LLM object
            clean_text = llm_chunk.content.strip()
            chunk_type_llm = llm_chunk.chunk_type.lower()
            section_title = llm_chunk.section_title
            chunk_summary = llm_chunk.chunk_summary

            if not clean_text:
                continue

            # Map LLM chunk_type to Metadata content_type
            content_type_mapped = "narrative" # Default
            if chunk_type_llm == "table":
                content_type_mapped = "tabular"
            elif chunk_type_llm == "bullet_list":
                content_type_mapped = "markdown"
            elif chunk_type_llm == "code":
                content_type_mapped = "code"

            # 2. Calculate Metadata
            chunk_unique_id = f"{self.filename.replace(' ', '_')}_{page_range_str}_{idx}"
            
            # Use Chonkie to count tokens
            token_info = list(self.pre_chunker.chunk(clean_text))
            token_count = sum(c.token_count for c in token_info) if token_info else len(clean_text) // 4
            
            meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_id=chunk_unique_id,
                source_path=self.filename,
                mime_type="application/pdf",
                page_number=page_nums[0],
                section_title=section_title, # From LLM
                chunk_summary=chunk_summary, # From LLM
                chunk_type=chunk_type_llm,   # Raw LLM Type stored here
                content_type=content_type_mapped, # Mapped from LLM
                token_count=token_count,
                hash_sha256=hashlib.sha256(clean_text.encode()).hexdigest(),
                embedding_model=self.embedding_model_name,
                neighbors=[] 
            )

            # 3. Format as Dictionary for the Indexer
            output_dict = {
                id_field: chunk_unique_id,
                source_field: f"{self.filename}::{page_range_str}",
                content_field: clean_text,
                "metadata": meta.dict(), # Use .dict() for Pydantic V1
                "page_number": page_nums[0],
                "token_count": token_count
            }
            
            batch_results.append(output_dict)
            current_char_idx += len(clean_text) + 1
            
        return batch_results

    def _agentic_split(self, text: str, logger: logging.Logger) -> List[LLMChunkResult]:
        """
        Invokes the LLM to get a Pydantic-structured response (List of objects).
        """
        try:
            # Create the chain: Prompt -> LLM -> Parser
            chain = self.prompt_template | self.llm | self.parser
            
            # Invoke
            result: SemanticSplitResult = chain.invoke({
                "text": text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            return result.chunks

        except Exception as e:
            logger.error(f"Structured output parsing failed: {e}")
            logger.warning("Falling back to raw splitting due to LLM error.")
            
            # Fallback: Simple text split if strict parsing fails
            fallback_chunks = text.split("\n\n")
            return [
                LLMChunkResult(
                    chunk_type="text",
                    section_title="General",
                    chunk_summary="Fallback chunk",
                    content=c
                ) for c in fallback_chunks if c.strip()
            ]