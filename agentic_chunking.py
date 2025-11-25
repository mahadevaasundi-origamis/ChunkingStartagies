from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime
import hashlib
import uuid
import logging

# LangChain Imports
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_core.agents import AgentExecutor, AgentOutputParser
from langchain_core.



# Base class import (assumed from your environment)
# from providers.services import Chunker

# Chonkie Import

from chonkie import TokenChunker
# --- 1. Pydantic Models for Final Output (Metadata) ---

class ChunkMetadata(BaseModel):
    doc_id: str
    chunk_id: str
    source_path: str
    mime_type: str
    page_number: Optional[int] = None
    bbox: Optional[List[float]] = None 
    section_title: Optional[str] = "General"
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

# --- 2. Pydantic Model for LLM Structured Response ---

class SemanticSplitResult(BaseModel):
    """
    The structured response expected from the LLM.
    """
    chunks: List[str] = Field(
        description="A list of semantically distinct text segments extracted from the input text."
    )

# --- 3. Agentic Chunking Class ---

class AgenticChunking():
    """
    Splits text using an LLM that returns a strict Pydantic structure.
    Uses Chonkie for context window management and token counting.
    """

    def __init__(self, filename: str, max_tokens: int = 4000, model_name: str = "deepseek-v3.1:671b-cloud", embedding_model_name: str = "text-embedding-3-small"):
        self.filename = filename
        self.max_tokens = int(max_tokens) if max_tokens is not None else 4000
        self.embedding_model_name = embedding_model_name
        
        # Initialize Chonkie for managing the context window size safely
        if TokenChunker:
            # We target 90% of max_tokens to leave room for the prompt instructions
            self.pre_chunker = TokenChunker(chunk_size=int(self.max_tokens * 0.90))
        else:
            raise ImportError("The 'chonkie' library is required. Please install it.")

        # Initialize ChatOllama (Better for structured output than base Ollama)
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            format="json", # Force JSON mode for reliability
        )
        
        # Setup the Parser
        self.parser = PydanticOutputParser(pydantic_object=SemanticSplitResult)
        
        # Define the Prompt Template
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert content analyzer. Your job is to split text into semantically meaningful sections."),
            HumanMessage(content="{format_instructions}\n\nAnalyze and split the following text into a list of semantic chunks:\n\nTEXT:\n{text}")
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
        
        # -- Logic: Buffer pages until we hit the context limit, then send to LLM --
        current_buffer_text = ""
        current_page_nums = []
        
        for page_num, page_text in normalized_map:
            # Tentatively add next page
            test_text = (current_buffer_text + "\n" + page_text) if current_buffer_text else page_text
            
            # Check size using Chonkie
            # chunk() returns a list of chunk objects. If > 1, it implies we exceeded the limit.
            pre_check = list(self.pre_chunker.chunk(test_text))
            
            if len(pre_check) > 1 and current_buffer_text:
                # Buffer is full. Process it.
                logger.info(f"Context window full. Processing pages {current_page_nums[0]}-{current_page_nums[-1]}...")
                batch_chunks = self._process_batch(current_buffer_text, doc_id, current_page_nums, logger, index_id_field, index_sourcepage_field, index_content_field)
                final_output_dicts.extend(batch_chunks)
                
                # Reset buffer with current page
                current_buffer_text = page_text
                current_page_nums = [page_num]
            else:
                # Fits in window, keep accumulating
                current_buffer_text = test_text
                current_page_nums.append(page_num)

        # Process any remaining text
        if current_buffer_text:
            logger.info(f"Processing final batch pages {current_page_nums[0]}-{current_page_nums[-1]}...")
            batch_chunks = self._process_batch(current_buffer_text, doc_id, current_page_nums, logger, index_id_field, index_sourcepage_field, index_content_field)
            final_output_dicts.extend(batch_chunks)

        # Post-Processing: Link Neighbors
        # (This modifies the 'metadata' dict inside the output list)
        for i in range(len(final_output_dicts)):
            curr = final_output_dicts[i]
            
            # Identify neighbors
            neighbors = []
            if i > 0:
                neighbors.append(final_output_dicts[i-1][index_id_field])
            if i < len(final_output_dicts) - 1:
                neighbors.append(final_output_dicts[i+1][index_id_field])
                
            # Inject into the metadata dict (and the root if necessary)
            curr["metadata"]["neighbors"] = neighbors

        return final_output_dicts

    def _process_batch(self, text: str, doc_id: str, page_nums: List[int], logger: logging.Logger, id_field: str, source_field: str, index_content_field: str) -> List[dict]:
        """
        Sends text to LLM, parses Pydantic response, and formats into Chunk objects.
        """
        # 1. Get Semantic Splits from LLM
        split_texts = self._agentic_split(text, logger)
        
        batch_results = []
        page_range_str = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])
        
        # Calculate char offsets roughly relative to this batch
        # (Note: absolute document offset requires global tracking, simplifying to batch-relative here)
        current_char_idx = 0
        
        for idx, segment_text in enumerate(split_texts):
            clean_text = segment_text.strip()
            if not clean_text:
                continue

            # 2. Calculate Metadata
            chunk_unique_id = f"{self.filename.replace(' ', '_')}_{page_range_str}_{idx}"
            
            # Use Chonkie to count tokens of the result
            # TokenChunker().chunk(text) returns a list of chunks, we just take the first/sum
            token_info = list(self.pre_chunker.chunk(clean_text))
            token_count = sum(c.token_count for c in token_info) if token_info else len(clean_text) // 4
            
            # Create Pydantic Metadata
            meta = ChunkMetadata(
                doc_id=doc_id,
                chunk_id=chunk_unique_id,
                source_path=self.filename,
                mime_type="application/pdf",
                page_number=page_nums[0], # Attribution to first page of batch
                section_title="General",
                content_type="narrative",
                token_count=token_count,
                hash_sha256=hashlib.sha256(clean_text.encode()).hexdigest(),
                embedding_model=self.embedding_model_name,
                neighbors=[] # Populated in post-processing
            )

            # Create Pydantic Chunk Wrapper
            chunk_obj = Chunk(
                text=clean_text,
                tokens=[], # Chonkie might expose this, or leave empty if not strictly needed
                start_char=current_char_idx,
                end_char=current_char_idx + len(clean_text),
                metadata=meta
            )
            
            # Increment offset
            current_char_idx += len(clean_text) + 1 # +1 for newline/space

            # 3. Format as Dictionary for the Indexer
            # We return the flat structure usually expected by indexers, 
            # but include the rich metadata object inside.
            output_dict = {
                id_field: chunk_unique_id,
                source_field: f"{self.filename}::{page_range_str}",
                index_content_field: clean_text,
                "metadata": meta.model_dump(), # The strict Pydantic metadata
                # Flattened fields often required by legacy indexers
                "page_number": page_nums[0],
                "token_count": token_count
            }
            
            batch_results.append(output_dict)
            
        return batch_results

    def _agentic_split(self, text: str, logger: logging.Logger) -> List[str]:
        """
        Invokes the LLM to get a Pydantic-structured response.
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
            
            # Fallback: Manual splitting if the structured output fails completely
            return text.split("\n\n")