# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
# from llama_index.core.node_parser import SentenceSplitter

# # Configure chunking globally (default is 1024/20)
# Settings.chunk_size = 512
# Settings.chunk_overlap = 50

# # Or use a specific parser explicitly
# # node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# documents = SimpleDirectoryReader("./data").load_data()
# index = VectorStoreIndex.from_documents(documents)


from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings
import os
from llama_index.core.node_parser import SentenceSplitter


# documents = SimpleDirectoryReader("./data").load_data()
# Load Bandhan.pdf from current directory
# Construct absolute path to the data file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Bandhan.pdf")

documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
# Configure chunking globally (default is 1024/20)

Settings.chunk_size = 512
Settings.chunk_overlap = 50

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Use a local embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Explicitly parse documents into nodes

parser = SentenceSplitter.from_defaults(
    chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap
)
nodes = parser.get_nodes_from_documents(documents, show_progress=True)

for i, node in enumerate(nodes):
    print(f"--- Chunk {i+1}/{len(nodes)} ---\n{node.get_content()}\n")

index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine(similarity_top_k=4)
