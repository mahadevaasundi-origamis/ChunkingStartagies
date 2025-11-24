import sys
import importlib.metadata

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

packages = ["unstructured", "unstructured_inference", "pdfminer.six", "pdfminer", "langchain-community", "langchain"]
for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"{package}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: Not Found")
