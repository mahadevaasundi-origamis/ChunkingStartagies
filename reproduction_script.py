from langchain_community.document_loaders import UnstructuredPDFLoader
import os

file = r"files\Bandhan.pdf"
if not os.path.exists(file):
    print(f"File {file} does not exist. Please check the path.")
    # Create a dummy PDF for testing if the file doesn't exist
    if not os.path.exists("files"):
        os.makedirs("files")
    with open(file, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/Resources <<\n/Font <<\n/F1 4 0 R\n>>\n>>\n/MediaBox [0 0 612 792]\n/Contents 5 0 R\n>>\nendobj\n4 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\n5 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 24 Tf\n100 100 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000117 00000 n\n0000000236 00000 n\n0000000324 00000 n\ntrailer\n<<\n/Size 6\n/Root 1 0 R\n>>\nstartxref\n418\n%%EOF")
    print(f"Created dummy file {file} for testing.")

print(f"File {file} found. Proceeding to load.")

try:
    loader = UnstructuredPDFLoader(
        file_path=file,
        mode="elements"
    )

    print(f"loader contents: {loader}")

    pages = loader.load()

    print(f"Loaded {len(pages)} pages from the PDF document.")
    if pages:
        print(f"First page content:\n{pages[0].page_content[:500]}")
    
    print("SUCCESS: PDF loaded successfully.")

except Exception as e:
    print(f"ERROR: Failed to load PDF. {e}")
    import traceback
    traceback.print_exc()
