from pathlib import Path
import pytesseract
from PyPDF2 import PdfReader
from PIL import Image
from docx import Document
from pptx import Presentation
import pandas as pd
from project import settings
from pdfminer.high_level import extract_text as pdfminer_extract_text
import cv2
import pandas as pd
import spacy
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, MatchAny, MatchValue
import time
import time
from sentence_transformers import SentenceTransformer
from langchain.schema import Document as LangChainDocument  # <-- Rename it
import json
import re
import difflib
import pandas as pd
from io import StringIO
import xml.etree.ElementTree as ET

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Optional

from langchain.chains.llm import LLMChain
from .models import Document as DocumentModel
# model = SentenceTransformer('all-MiniLM-L6-v2')
import os
import subprocess
import platform
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from moviepy.editor import VideoFileClip


nlp = spacy.load("en_core_web_sm")  # For NER

# Initialize Embeddings and Chat Model
embedding_model = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)
llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY,temperature=0)

# Connect to Qdrant (cloud)
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    timeout=600.0  # seconds: increase to handle large bulk uploads without timing out
)
collection_name = settings.QDRANT_COLLECTION_NAME




# Check if collection exists and create if it doesn't
def create_qdrant_collection_if_not_exists():
    try:
        # Check if the collection exists
        qdrant_client.get_collection(collection_name=collection_name)
        print(f"[+] Collection '{collection_name}' already exists.")
    except Exception:
        # Collection doesn't exist, create it
        print(f"[+] Collection '{collection_name}' not found. Creating it...")
        vector_params = VectorParams(size=1536, distance=Distance.COSINE)  # Adjust vector size and distance metric as needed
        # vector_params = VectorParams(size=384, distance=Distance.COSINE)  # 384 for MiniLM
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_params,
        )
        print(f"[+] Collection '{collection_name}' created successfully.")
        
# qdrant_client.delete_collection(collection_name="xamplify_docs")

create_qdrant_collection_if_not_exists()

# from langchain_community.embeddings import HuggingFaceEmbeddings

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# Qdrant
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedding_model,
)

# ----------- File Extraction Logic -----------

# def extract_text_from_file(file_path: str, original_file_name: str) -> str:

#     ext = Path(original_file_name).suffix.lower()
#     if ext == ".pdf":
#         return extract_text_from_pdf(file_path)
#     elif ext == ".docx":
#         return extract_text_from_docx(file_path)
#     elif ext in [".png", ".jpg", ".jpeg"]:
#         return extract_text_from_image(file_path)
#     elif ext == ".pptx":
#         return extract_text_from_pptx(file_path)
#     elif ext in [".xls", ".xlsx", ".csv"]:
#         return extract_structured_from_excel_or_csv(file_path)
#         # return extract_text_from_excel_or_csv(file_path)
#     elif ext == ".txt":
#         return extract_text_from_txt(file_path)
#     elif ext == ".json":
#         return extract_structured_from_json(file_path)
#         # return extract_text_from_json(file_path)
#     elif ext == ".xml":
#         return extract_text_from_xml(file_path)
#     elif ext == ".html":
#         return extract_text_from_html(file_path)
#     elif ext == ".md":  # Adding support for Markdown files
#         return extract_text_from_md(file_path)
#     elif ext == ".yaml" or ext == ".yml":  # Adding support for YAML files
#         return extract_text_from_yaml(file_path)
#     elif ext in [".ini", ".cfg"]:  # Adding support for configuration files
#         return extract_text_from_ini(file_path)
#     else:
#         raise ValueError("Unsupported file type!")

def extract_text_from_file(file_path: str, original_file_name: str) -> str:
    ext = Path(original_file_name).suffix.lower()
    extracted_data = None  # Initialize a variable to hold the extracted text
    
    print(f"[+] Starting text extraction for {original_file_name} with extension {ext}...")
    
    if ext == ".pdf":
        extracted_data = extract_text_from_pdf(file_path)
    
    elif ext == ".docx":
        extracted_data = extract_text_from_docx(file_path)
    
    elif ext in [".png", ".jpg", ".jpeg"]:
        extracted_data = extract_text_from_image(file_path)
    
    elif ext == ".pptx":
        extracted_data = extract_text_from_pptx(file_path)
    
    elif ext in [".xls", ".xlsx", ".csv"]:
        # Read tabular data as a formatted table string
        extracted_data = extract_text_from_excel_or_csv(file_path)
    
    elif ext == ".txt":
        extracted_data = extract_text_from_txt(file_path)
    
    elif ext == ".json":
        extracted_data = extract_structured_from_json(file_path)
        if isinstance(extracted_data, dict):
            extracted_data = json.dumps(extracted_data) # Serialize to JSON string
    
    elif ext == ".xml":
        extracted_data = extract_text_from_xml(file_path)
    
    elif ext == ".html":
        extracted_data = extract_text_from_html(file_path)
    
    elif ext == ".md":  # Adding support for Markdown files
        extracted_data = extract_text_from_md(file_path)
    
    elif ext in [".yaml", ".yml"]:  # Adding support for YAML files
        extracted_data = extract_text_from_yaml(file_path)
    
    elif ext in [".ini", ".cfg"]:  # Adding support for configuration files
        extracted_data = extract_text_from_ini(file_path)

    elif ext == ".ppt":  # Legacy PowerPoint support
        extracted_data = extract_text_from_ppt(file_path)

    elif ext in [".mp4", ".avi", ".mov"]:  # Video support
        extracted_data = extract_text_from_video(file_path)

    elif ext in [".mp3", ".wav", ".m4a"]:  # Audio support
        extracted_data = extract_text_from_audio(file_path)
    
    else:
        raise ValueError("Unsupported file type!")

    if isinstance(extracted_data, (str, list)):
        print(f"[+] Successfully extracted data from {original_file_name}: {type(extracted_data)} with length {len(extracted_data)}")
    elif isinstance(extracted_data, dict):
        print(f"[+] Successfully extracted data from {original_file_name}: {type(extracted_data)} with {len(extracted_data.keys())} keys")
    else:
        print(f"[!] Error: Extracted data from {original_file_name} is of an unexpected type: {type(extracted_data)}")
    print(extracted_data, '\n\n Extracted Data \n\n')
    return extracted_data

def extract_text_from_video(file_path: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            video = VideoFileClip(file_path)
            audio = video.audio
            audio.write_audiofile(tmp_audio.name, logger=None)

            return extract_text_from_audio(tmp_audio.name)
    except Exception as e:
        print(f"[!] Error processing video: {e}")
        return ""
    # finally:
    #     if os.path.exists(tmp_audio.name):
    #         os.unlink(tmp_audio.name)

import whisper
def extract_text_from_audio(file_path: str) -> str:
    try:
        model = whisper.load_model("base")  # or "small", "medium", "large"
        result = model.transcribe(file_path, language="en")  # or let it auto-detect
        return result["text"]
    except Exception as e:
        print(f"[!] Whisper failed: {e}")
        return ""

# ----------- Dynamic Metadata Extraction Logic -----------

# def extract_metadata(text: str, file_ext: str) -> dict:
#     metadata = {}

#     if file_ext in [".csv", ".xls", ".xlsx"]:
#         try:
#             # Try extracting header columns
#             lines = text.strip().split("\n")
#             header = lines[0].split()
#             for col in header:
#                 metadata[f"column_{col.lower()}"] = True
#         except Exception as e:
#             print(f"[!] CSV/Excel metadata extraction failed: {e}")
    
#     elif file_ext in [".pdf", ".docx", ".pptx", ".txt"]:
#         doc = nlp(text[:3000])  # Analyze only first 3000 chars for speed
#         entities = {"ORG": [], "GPE": [], "DATE": []}
#         for ent in doc.ents:
#             if ent.label_ in entities:
#                 entities[ent.label_].append(ent.text)

#         if entities["ORG"]:
#             metadata["organizations"] = list(set(entities["ORG"]))
#         if entities["GPE"]:
#             metadata["locations"] = list(set(entities["GPE"]))
#         if entities["DATE"]:
#             metadata["dates"] = list(set(entities["DATE"]))

#     elif file_ext in [".png", ".jpg", ".jpeg"]:
#         metadata["detected_from"] = "image"
    
#     else:
#         metadata["type"] = "unknown"

#     return metadata

def extract_metadata(data, file_ext: str) -> dict:
    metadata = {}

    if file_ext in [".csv", ".xls", ".xlsx", ".json"]:
        try:
            if isinstance(data, (dict, list)):
                # Structured - infer columns
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    header = data[0].keys()
                elif isinstance(data, dict):
                    header = data.keys()
                else:
                    header = []
                for col in header:
                    metadata[f"column_{col.lower()}"] = True
        except Exception as e:
            print(f"[!] Structured metadata extraction failed: {e}")

    elif file_ext in [".pdf", ".docx", ".pptx", ".txt"]:
        if isinstance(data, str):
            doc = nlp(data[:3000])  # Only if it's text
            entities = {"ORG": [], "GPE": [], "DATE": []}
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)

            if entities["ORG"]:
                metadata["organizations"] = list(set(entities["ORG"]))
            if entities["GPE"]:
                metadata["locations"] = list(set(entities["GPE"]))
            if entities["DATE"]:
                metadata["dates"] = list(set(entities["DATE"]))

    elif file_ext in [".png", ".jpg", ".jpeg"]:
        metadata["detected_from"] = "image"
    
    else:
        metadata["type"] = "unknown"

    return metadata

# ----------- Specific Extractors -----------

# 1. PDF
# def extract_text_from_pdf(path):
#     try:
#         text = pdfminer_extract_text(path)

#         if not text.strip():
#             # If text is empty even after extraction, maybe password protected
#             raise ValueError("The PDF appears to be empty or password-protected.")

#     except Exception as e:
#         print(f"[!] PDF extraction failed: {e}")
#         raise ValueError("Failed to extract text from PDF. It might be password-protected or corrupted.") from e

#     return text.strip()
def extract_text_from_pdf(path):
    try:
        # Check encryption first
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("The uploaded PDF is password-protected. Please upload an unlocked PDF.")

        # If not encrypted, continue extraction
        text = pdfminer_extract_text(path)

        if not text.strip():
            raise ValueError("The PDF appears to be empty or unreadable after extraction.")

    except ValueError as ve:
        print(f"[!] PDF Extraction error: {ve}")
        raise ve

    except Exception as e:
        print(f"[!] Unexpected error during PDF extraction: {e}")
        raise ValueError("Failed to extract text from PDF. It might be password-protected, empty, or corrupted.") from e

    return text.strip()

# 2. DOCX
def extract_text_from_docx(path):
    doc = Document(path)
    fullText = []

    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            fullText.append(para.text)

    # Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            fullText.append('\t'.join(row_data))

    return '\n'.join(fullText).strip()

# 3. PPTX
def extract_text_from_pptx(path):
    prs = Presentation(path)
    text = ""

    for slide in prs.slides:
        # Slide shapes
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                text += shape.text + "\n"
        # Slide notes
        if slide.has_notes_slide:
            notes_slide = slide.notes_slide
            if notes_slide and notes_slide.notes_text_frame:
                text += notes_slide.notes_text_frame.text + "\n"

    return text.strip()

def extract_text_from_ppt(file_path: str) -> str:
    pptx_path = file_path + "x"  # Temporary file path for converted pptx
    try:
        print("[+] Converting PPT to PPTX...",platform.system())
        if platform.system() == "Windows":
            import comtypes.client
            powerpoint = comtypes.client.CreateObject("Powerpoint.Application")
            powerpoint.Visible = 1
            ppt = powerpoint.Presentations.Open(file_path, WithWindow=False)
            ppt.SaveAs(pptx_path, 24)  # 24 = pptx
            ppt.Close()
            powerpoint.Quit()
        else:
            # For Linux: Convert using unoconv or libreoffice
            subprocess.run(["libreoffice", "--headless", "--convert-to", "pptx", file_path, "--outdir", os.path.dirname(file_path)], check=True)

        # Now extract text from .pptx
        return extract_text_from_pptx(pptx_path)

    except Exception as e:
        print(f"[!] Error converting or reading PPT: {e}")
        return ""

    finally:
        if os.path.exists(pptx_path):
            os.remove(pptx_path)
# 4. Excel/CSV
def extract_text_from_excel_or_csv(path: str) -> str:
    """
    Read an Excel (.xls/.xlsx) or CSV file and return a human-readable table
    (markdown format if possible, otherwise plain text).
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        # Read CSV into DataFrame and return as CSV text
        try:
            df = read_csv_with_encoding_fallback(path)
        except NameError:
            df = pd.read_csv(path)
        return df.to_csv(index=False)
    elif ext in [".xlsx", ".xls"]:
        # Read first sheet of Excel and return as CSV text
        try:
            df = pd.read_excel(path, sheet_name=0)
        except Exception:
            # Fallback to specific engine
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        return df.to_csv(index=False)
    else:
        raise ValueError("Unsupported Excel/CSV file extension for table extraction.")

# 4. Structured Excel/CSV (NEW)
def read_csv_with_encoding_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")  # fallback

# def extract_structured_from_excel_or_csv(path):
#     ext = Path(path).suffix.lower()

#     if ext == ".csv":
#         df = read_csv_with_encoding_fallback(path)
#         return df.to_dict(orient="records")

#     elif ext in [".xlsx", ".xls"]:
#         if ext == ".xlsx":
#             print("[+] Using openpyxl for .xlsx file")
#             try:
#                 xls = pd.ExcelFile(path, engine="openpyxl")
#             except Exception as e:
#                 print(f"[!] openpyxl failed: {e}. Trying xlrd...")
#                 xls = pd.ExcelFile(path, engine="xlrd")
#         else:
#             xls = pd.ExcelFile(path, engine="xlrd")

#         dfs = {}
#         print("[+] Reading Excel file...", xls.sheet_names)
#         for sheet_name in xls.sheet_names:
#             sheet_df = xls.parse(sheet_name)
#             dfs[sheet_name] = sheet_df.to_dict(orient="records")
#         print(dfs)
#         return dfs

#     else:
#         raise ValueError("Unsupported Excel/CSV file extension!")

def extract_structured_from_excel_or_csv(path):
    ext = Path(path).suffix.lower()

    if ext == ".csv":
        df = read_csv_with_encoding_fallback(path)
        return df.to_dict(orient="records")

    elif ext in [".xlsx", ".xls"]:
        if ext == ".xlsx":
            print("[+] Using openpyxl for .xlsx file")
            try:
                xls = pd.ExcelFile(path, engine="openpyxl")
            except Exception as e:
                print(f"[!] openpyxl failed: {e}. Trying xlrd...")
                xls = pd.ExcelFile(path, engine="xlrd")
        else:
            xls = pd.ExcelFile(path, engine="xlrd")

        dfs = {}
        print("[+] Reading Excel file...", xls.sheet_names)
        for sheet_name in xls.sheet_names:
            sheet_df = xls.parse(sheet_name)

            # Checking for any potential issues in data types
            print(f"Processing sheet: {sheet_name}")
            print(sheet_df.dtypes)  # Display column data types
            
            if not sheet_df.empty:
                # Convert any potential Timestamp to string
                for col in sheet_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(sheet_df[col]):
                        sheet_df[col] = sheet_df[col].astype(str)
                
                # Ensure to_dict can process the data safely
                try:
                    dfs[sheet_name] = sheet_df.to_dict(orient="records")
                except Exception as e:
                    print(f"[!] Error converting sheet '{sheet_name}' to dict: {e}")
            else:
                print(f"[!] Warning: Sheet '{sheet_name}' is empty.")
                
        # Output the resulting structure clearly
        print("[+] Resulting structured data:")
        for k, v in dfs.items():
            print(f"Sheet: {k}, Records: {len(v)}")
            print(v)  # Print the content for visibility
        return dfs

    else:
        raise ValueError("Unsupported Excel/CSV file extension!")

# 5. Image
def preprocess_image(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1,1), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def extract_text_from_image(path):
    preprocessed_img = preprocess_image(path)
    text = pytesseract.image_to_string(preprocessed_img)
    return text.strip()

# 6. TXT
def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 7. JSON
# def extract_text_from_json(file_path: str) -> str:
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return json.dumps(data, indent=4)  # pretty prints JSON

def extract_structured_from_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# 8. XML
def extract_text_from_xml(file_path: str) -> str:
    tree = ET.parse(file_path)
    root = tree.getroot()
    return ET.tostring(root, encoding='utf-8').decode('utf-8')

# 9. HTML
def extract_text_from_html(file_path: str) -> str:
    from bs4 import BeautifulSoup
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()

# 10. Markdown
def extract_text_from_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 11. YAML
def extract_text_from_yaml(file_path: str) -> str:
    import yaml
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return yaml.dump(data)

# 12. INI
def extract_text_from_ini(file_path: str) -> str:
    import configparser
    config = configparser.ConfigParser()
    config.read(file_path)
    output = []
    for section in config.sections():
        output.append(f"[{section}]")
        for key, value in config.items(section):
            output.append(f"{key} = {value}")
    return "\n".join(output)

# ----------- VectorStore Insert Logic -----------

def insert_document_to_vectorstore(text: str, source_type: str, file_ext: str,vector_id: str):
    
    docs = smart_split_text(text, file_ext)
    print(f"[+] Splitting text into {len(docs)} chunks...")

    # Extract global metadata once for the whole file
    # Extract metadata (including file_ext) and persist vector_id
    global_metadata = extract_metadata(text, file_ext)
    global_metadata["vector_id"] = vector_id
    global_metadata["file_ext"] = file_ext


    # Merge the global metadata into each document's metadata
    for doc in docs:
        if not hasattr(doc, "metadata") or not isinstance(doc.metadata, dict):
            doc.metadata = {}
        doc.metadata.update(global_metadata)


    if source_type not in ("file", "integration"):
        raise ValueError(f"Invalid source_type '{source_type}'")
    # Batch upload to Qdrant with progress logs
    batch_size = 64
    total_docs = len(docs)
    batches = [docs[i:i+batch_size] for i in range(0, total_docs, batch_size)]
    total_batches = len(batches)
    print(f"[+] Inserting {total_docs} chunks into Qdrant in {total_batches} batches...")
    for idx, batch in enumerate(batches):
        try:
            qdrant.add_documents(batch, batch_size=len(batch))
        except Exception as e:
            print(f"[!] Failed to insert batch {idx+1}/{total_batches}: {e}")
            raise
        print(f"[+] Uploaded batch {idx+1}/{total_batches}")

    

def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.create_documents([text])



# Split CSV/Excel table text into one row per document, preserving headers
def split_csv_rows(text: str) -> list:
    # Split into non-empty lines
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        print("[!] No lines found in CSV text.")
        return []
    # Detect header row: prefer markdown-style '|col|' or simple first line
    header = None
    data_start = 1
    for idx, l in enumerate(lines):
        if l.strip().startswith("|") and l.count("|") >= 2:
            header = l.strip()
            data_start = idx + 1
            break
    if not header:
        header = lines[0].strip()
        data_start = 1
    # Exclude any sheet headers or separators (like lines starting with '#')
    data_lines = [l for l in lines[data_start:] if not l.strip().startswith('#')]
    if not data_lines:
        print("[!] No data lines found after header.")
        return []
    docs = []
    for i, row in enumerate(data_lines):
        row = row.strip()
        chunk_text = f"{header}\n{row}"
        md = {"chunk": i + 1}
        docs.append(LangChainDocument(page_content=chunk_text, metadata=md))
    print(f"[+] Split CSV/Excel into {len(docs)} row-level documents.")
    return docs

def split_unstructured_text(text: str, chunk_size=1000, chunk_overlap=200):
    if len(text.strip()) == 0:
        print("[!] No text provided for unstructured splitting.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = splitter.create_documents([text])
    if len(docs) == 0:
        print("[!] No documents created from unstructured text.")
    
    for i, doc in enumerate(docs):
        doc.metadata["chunk"] = i + 1
        print(f"[+] Created chunk {i + 1} with content of length {len(doc.page_content)}.")
    
    return docs

def split_json_rows(json_text):
    from langchain.schema import Document
    try:
        data = json.loads(json_text)
        documents = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                documents.append(Document(page_content=str(item), metadata={"source": "JSON_item", "chunk": i + 1}))
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        documents.append(Document(page_content=str(item), metadata={"source": key, "chunk": i + 1, "type": type(item).__name__}))
                elif isinstance(value, dict):
                    documents.append(Document(page_content=str(value), metadata={"source": key, "type": type(value).__name__}))
                elif value is not None:
                    documents.append(Document(page_content=str(value), metadata={"source": key, "type": type(value).__name__}))
        return documents
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []
    

def smart_split_text(text: str, file_ext: str):
    if file_ext == ".json" or (text.startswith('{') and text.endswith('}')):
        print(f"[+] Detected JSON format. Preparing to parse...")
        return split_json_rows(text)
    
    elif file_ext in [".csv", ".xls", ".xlsx"]:
        print(f"[+] Splitting CSV/Excel text into rows...")
        docs = split_csv_rows(text)
        print(f"[+] Split into {len(docs)} documents.")
        return docs
    
    else:
        print(f"[+] Splitting unstructured text into chunks...")
        docs = split_unstructured_text(text)
        print(f"[+] Split into {len(docs)} documents.")
        return docs

def handle_table_query(csv_text: str, question: str):
    """
    Perform NLP-driven queries on tabular CSV/Excel data.
    Supports patterns like:
      - how many <target> from/in <value>
      - give me <target> from/in <value>
    Uses fuzzy matching for target column names and value matching for filtering.
    Returns a string answer, or None if parsing/matching fails.
    """
    try:
        df = pd.read_csv(StringIO(csv_text))
        ql = question.strip().lower()
        # Numeric filter support: e.g. 'companies with more than 3000 employees'
        # Support patterns like 'X [from Y]? with more than N Z'
        num_pattern = r"(?P<target>[\w\s]+?)(?:\s+from\s+[\w\s]+?)?\s+with\s+(?:more|greater|over)\s+than\s+(?P<number>\d+)\s+(?P<num_field>[\w\s]+)"
        mnum = re.search(num_pattern, ql)
        if mnum:
            target_raw = mnum.group('target').strip().lower()
            num_value = int(mnum.group('number'))
            num_field_raw = mnum.group('num_field').strip().lower()
            # Detect location if present
            location_raw = None
            # Extract location up to 'with'
            mloc = re.search(r"(?:from|in)\s+(?P<loc>[\w\s]+?)(?=\s+with|$)", ql)
            if mloc:
                location_raw = mloc.group('loc').strip().lower()
            # Handle demonyms like 'srilankan'
            elif 'sri lanka' in ql or 'srilankan' in ql:
                location_raw = 'sri lanka'
            # Fuzzy match target column
            cols = df.columns.tolist()
            cols_low = [c.lower() for c in cols]
            # Map generic queries for company/organization names to a likely name column
            synonyms = ['company', 'companies', 'organization', 'organizations', 'org', 'orgs', 'firm', 'firms', 'name']
            if any(syn in target_raw for syn in synonyms):
                # pick the first column containing name-like text
                name_cols = [c for c in cols if any(k in c.lower() for k in ['name','company','org','firm'])]
                tgt_col = name_cols[0] if name_cols else None
            else:
                # fuzzy match target column
                fm_tgt = difflib.get_close_matches(target_raw, cols_low, n=1, cutoff=0.3)
                if fm_tgt:
                    tgt_col = cols[cols_low.index(fm_tgt[0])]
                else:
                    tgt_col = next((cols[i] for i,c in enumerate(cols_low)
                                    if target_raw in c or c in target_raw), None)
            # Fuzzy match numeric field column
            fm_num = difflib.get_close_matches(num_field_raw, cols_low, n=1, cutoff=0.3)
            if fm_num:
                num_col = cols[cols_low.index(fm_num[0])]
            else:
                num_col = next((cols[i] for i,c in enumerate(cols_low)
                                if num_field_raw in c or c in num_field_raw), None)
            # Find location column
            filt_col = None
            if location_raw:
                # try common names first
                filt_col = next((c for c in cols if 'country' in c.lower()), None)
                if not filt_col:
                    # fuzzy match location field
                    fm_loc = difflib.get_close_matches(location_raw, cols_low, n=1, cutoff=0.3)
                    if fm_loc:
                        filt_col = cols[cols_low.index(fm_loc[0])]
            # Build mask
            mask = pd.Series([True] * len(df))
            if location_raw and filt_col:
                mask &= df[filt_col].astype(str).str.lower().str.contains(location_raw, regex=False)
            # Numeric comparison
            if num_col:
                try:
                    num_series = pd.to_numeric(df[num_col], errors='coerce')
                    mask &= num_series > num_value
                except Exception:
                    pass
            if not tgt_col:
                return None
            # Extract full rows for matching mask
            records = df.loc[mask].dropna(how='all').to_dict(orient='records')
            if ql.startswith('how many'):
                return str(len(records))
            return records
        # end numeric filter
        # --- continue standard pattern parsing ---
        # Parse intent: action, target column name, filter value
        m = re.search(
            r"(?:how many|give me|list|show me|what are)\s+"     # action
            r"(?P<target>[\w\s]+?)\s+"                         # target field
            r"(?:are\s+)?(?:from|in)\s+"                       # preposition
            r"(?P<value>[\w\s]+)",                            # filter value
            ql
        )
        if not m:
            return None
        target_raw = m.group('target').strip().lower()
        # Normalize 'names of X' to 'X'
        for prefix in ['names of ', 'name of ', 'names for ', 'name for ']:
            if target_raw.startswith(prefix):
                target_raw = target_raw[len(prefix):].strip()
                break
        value_raw = m.group('value').strip().lower()
        # Prepare columns for matching
        cols = df.columns.tolist()
        cols_low = [c.lower() for c in cols]
        # Map generic terms to name column for organization/company queries
        synonyms = ['company', 'companies', 'organization', 'organizations', 'org', 'orgs', 'firm', 'firms']
        if target_raw in synonyms:
            # pick the first text column likely holding names
            name_cols = [c for c in cols if any(k in c.lower() for k in ['name', 'company', 'org', 'firm'])]
            if name_cols:
                tgt_col = name_cols[0]
                # Identify filter column for location
                filt_col = next((c for c in cols if any(k in c.lower() for k in ['country', 'location', 'region'])), None)
                if not filt_col:
                    for c in cols:
                        try:
                            if df[c].astype(str).str.lower().str.contains(value_raw, regex=False).any():
                                filt_col = c; break
                        except Exception:
                            continue
                if not filt_col:
                    return None
                mask = df[filt_col].astype(str).str.lower().str.contains(value_raw, regex=False)
                results = df.loc[mask, tgt_col].dropna().unique().tolist()
                if ql.startswith('how many'):
                    return str(len(results))
                return ', '.join(map(str, results))
        # Match target column via fuzzy match + substring
        cols = df.columns.tolist()
        cols_low = [c.lower() for c in cols]
        # Fuzzy match
        # Fuzzy match target column, relax cutoff and allow singular/plural match
        fm = difflib.get_close_matches(target_raw, cols_low, n=1, cutoff=0.3)
        if fm:
            tgt_col = cols[cols_low.index(fm[0])]
        else:
            # Fallback to substring match in either direction
            tgt_col = next((cols[i] for i,c in enumerate(cols_low)
                            if target_raw in c or c in target_raw), None)
        if not tgt_col:
            return None
        # Identify filter column: country/location
        filt_col = next((c for c in cols if any(k in c.lower() for k in ['country', 'location', 'region'])), None)
        if not filt_col:
            for c in cols:
                try:
                    if df[c].astype(str).str.lower().str.contains(value_raw, regex=False).any():
                        filt_col = c; break
                except Exception:
                    continue
        if not filt_col:
            return None
        # Filter rows
        mask = df[filt_col].astype(str).str.lower().str.contains(value_raw, regex=False)
        # Return full rows for matching
        records = df.loc[mask].dropna(how='all').to_dict(orient='records')
        if ql.startswith('how many'):
            return str(len(records))
        return records
    except Exception:
        return None



# ----------- Query Logic -----------

# def ask_question(query: str, source_type: str):
#     retriever = None
#     start = time.time()

#     if source_type == "file":
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#         # retriever = qdrant.as_retriever()
#     elif source_type == "integration":
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#     end = time.time()
#     print("Retrieved in: ", end - start)

    # start_time = time.time()
    # qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # result = qa.run(query)
    # end_time = time.time()
    # print("Question answered in: ", end_time - start_time)
    # return result

# def ask_question(query: str, source_type: str, documents=None):
#     from langchain.chains.retrieval_qa.base import RetrievalQA
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     from langchain.schema import Document as LangChainDocument
    
#     retriever = None
#     start = time.time()

#     if documents:
#         # Use provided documents as context
#         text = "\n".join(doc["content"] for doc in documents)
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         docs = splitter.create_documents([text])
#         for i, doc in enumerate(docs):
#             doc.metadata["chunk"] = i + 1
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#     else:
#         # Default behavior: retrieve from entire collection
#         retriever = qdrant.as_retriever(search_kwargs={"k": 5})

#     end = time.time()
#     print("Retrieved in: ", end - start)

#     start_time = time.time()
#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#     result = qa.run(query)
#     end_time = time.time()
#     print("Question answered in: ", end_time - start_time)
#     return result

# ========================Previous Working Code==========================
# def ask_question(query: str, source_type: str, documents=None):
#     try:
#         start = time.time()

#         # Define the default system prompt
#         system_prompt = """
#         You are a helpful document analyst. Your role is to provide accurate, concise, and detailed answers based on the provided document context. 
#         Use the information in the documents to answer the question directly and avoid including irrelevant details. 
#         If the documents do not contain enough information to answer the question, state so clearly.
#         Format your response clearly and professionally, using bullet points or paragraphs as appropriate.
#         """

#         # Create a prompt template for the QA chain
#         prompt_template = PromptTemplate(
#             template="""{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:""",
#             input_variables=["system_prompt", "context", "question"]
#         )

#         if documents:
#             print("[+] Using provided documents as context")
#             # Create a filter for the vector_id
#             vector_id = documents[0]["metadata"]["vector_id"]
#             filter = Filter(
#                 must=[
#                     FieldCondition(
#                         key="metadata.vector_id",
#                         match=MatchValue(value=vector_id)
#                     )
#                 ]
#             )
#             # Use the existing Qdrant collection with a filter
#             print(len(documents))
#             print("[+] Filter for vector_id:", vector_id)
#             print("[+] Filter:", filter)
#             retriever = qdrant.as_retriever(search_kwargs={"k": len(documents), "filter": filter})
#             print(f"[+] Created retriever with filter for vector_id: {vector_id}")
#         else:
#             # Use the default Qdrant collection
#             retriever = qdrant.as_retriever(search_kwargs={"k": 5})
#             print("[+] Using default Qdrant collection retriever")

#         end = time.time()
#         print(f"[+] Retrieved in: {end - start:.2f} seconds")

#         start_time = time.time()
#         # Configure the RetrievalQA chain with the custom prompt
#         qa = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type_kwargs={
#                 "prompt": prompt_template.partial(system_prompt=system_prompt)
#             }
#         )
#         # Run the query with increased max_tokens
#         result = qa.invoke({"query": query})["result"]  # Use invoke for modern LangChain versions
#         end_time = time.time()
#         print(f"[+] Question answered in: {end_time - start_time:.2f} seconds")

#         return result

#     except Exception as e:
#         print(f"[!] Error in ask_question: {e}")
#         raise



def estimate_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in a text (assuming ~4 characters per token)."""
    return len(text) // 4

def summarize_batch(batch_docs: List[LangChainDocument]) -> str:
    """Summarize a batch of documents."""
    combined_context = "\n".join([doc.page_content for doc in batch_docs])
    summary = summarize_context(combined_context)
    return summary if isinstance(summary, str) else str(summary)

def summarize_context(context_text: str) -> str:
    """Summarize a large text into a small focused version."""
    prompt = f"Summarize the following content concisely and accurately:\n\n{context_text}\n\nSummary:"
    response = llm.invoke(prompt)
    
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, dict) and "text" in response:
        return response["text"]
    return str(response)

def batch_documents(documents: List[LangChainDocument], batch_size: int = 5):
    """Yield successive batches of documents."""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def ask_question_for_single_document(query: str, source_type: str, documents: Optional[List[dict]] = None, chat_history: Optional[List[dict]] = None) -> str:
    try:
        start = time.time()

        system_prompt = """
        You are a helpful document analyst. Your role is to provide accurate, concise, and detailed answers based on the provided document context. 
        Use the information in the documents to answer the question directly and avoid including irrelevant details. 
        If the documents do not contain enough information to answer the question, state so clearly.
        """

        final_context = ""
        max_token_threshold = 4000  # Adjust based on your LLM's token limit
        batch_size = 5  # Number of documents per batch for large datasets

        memory_context = ""

        if chat_history:
            memory_pairs = []
            for turn in chat_history:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    memory_pairs.append(f"User: {content}")
                elif role == "assistant":
                    memory_pairs.append(f"Assistant: {content}")
            memory_context = "\n".join(memory_pairs)
            print(f"[+] Built memory context with {len(memory_pairs)} turns")

        if documents:
            # If this document is a table (Excel/CSV), use a Pandas DataFrame agent for QA
            file_ext = documents[0].get("metadata", {}).get("file_ext", "")
            # If this document is a table (CSV/Excel), try simple table query
            if file_ext in [".csv", ".xls", ".xlsx"]:
                # Try structured table query first
                from .utils import handle_table_query, llm
                # Reconstruct full CSV text
                rows = []
                header = None
                for d in documents:
                    parts = d.get("content", "").split("\n", 1)
                    if header is None and parts:
                        header = parts[0]
                    if len(parts) > 1:
                        rows.append(parts[1])
                csv_text = header + "\n" + "\n".join(rows)
                table_res = handle_table_query(csv_text, query)
                if table_res is not None:
                    return table_res
                # Fallback: use DataFrame agent for large tables
                try:
                    from io import StringIO
                    import pandas as pd
                    from langchain_experimental.agents import create_pandas_dataframe_agent
                    df = pd.read_csv(StringIO(csv_text))
                    agent = create_pandas_dataframe_agent(llm, df, verbose=False)
                    return agent.run(query)
                except Exception as e:
                    print(f"[!] DataFrame agent failed for table fallback: {e}")
            # Continue with vector retrieval for non-table or fallback
            print("[+] Using provided documents as context")
            vector_id = documents[0]["metadata"]["vector_id"]
            filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.vector_id",
                        match=MatchValue(value=vector_id)
                    )
                ]
            )
            retriever = qdrant.as_retriever(search_kwargs={"k": len(documents), "filter": filter})
            retrieved_docs = retriever.get_relevant_documents(query)
            print(f"[+] Retrieved {len(retrieved_docs)} documents")

            # Estimate total token count of retrieved documents
            combined_content = "\n".join([doc.page_content for doc in retrieved_docs])
            estimated_tokens = estimate_tokens(combined_content)
            print(f"[+] Estimated tokens in retrieved documents: {estimated_tokens}")

            if estimated_tokens <= max_token_threshold and len(retrieved_docs) <= 10:
                # For small datasets, use raw document content directly
                print("[+] Using raw document content (small dataset)")
                final_context = combined_content
            else:
                # For large datasets, batch and summarize
                print("[+] Batching and summarizing (large dataset)")
                batches = list(batch_documents(retrieved_docs, batch_size=batch_size))
                summaries = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(summarize_batch, batch) for batch in batches]
                    for future in futures:
                        summaries.append(future.result())

                # Combine and summarize all batch summaries
                combined_summary = "\n".join(summaries)
                final_context = summarize_context(combined_summary)
                print(f"[+] Summarized {len(batches)} batches and reduced to final context")

        else:
            print("[+] Using default Qdrant collection retriever")
            retriever = qdrant.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.get_relevant_documents(query)
            final_context = "\n".join([doc.page_content for doc in retrieved_docs])

        end = time.time()
        print(f"[+] Retrieved and processed context in: {end - start:.2f} seconds")

        # Final QA
        prompt_template = PromptTemplate(
            template="""
                {system_prompt}

                Context:
                {context}

                Conversation History:
                {memory}

                Current Question:
                {question}

                Answer:
                """,
                input_variables=["system_prompt", "context", "memory", "question"]
        )

        qa_chain = LLMChain(
            llm=llm,
            prompt=prompt_template.partial(
                system_prompt=system_prompt,
                context=final_context,
                memory=memory_context
            )
        )

        start_time = time.time()
        result = qa_chain.invoke({"question": query})["text"]
        end_time = time.time()
        print(f"[+] Question answered in: {end_time - start_time:.2f} seconds")

        return result

    except Exception as e:
        print(f"[!] Error in ask_question: {e}")
        raise

def ask_question(query: str, source_type: str, documents: Optional[List[dict]] = None, chat_history: Optional[List[dict]] = None) -> str:
    try:
        start = time.time()

        system_prompt = """
        You are a helpful document analyst. Your role is to provide accurate, concise, and detailed answers based on the provided document context. 
        Use the information in the documents to answer the question directly and avoid including irrelevant details. 
        If the documents do not contain enough information to answer the question, state so clearly.
        """

        final_context = ""
        max_token_threshold = 4000  # Approximate LLM context limit
        batch_size = 5

        memory_context = ""

        if chat_history:
            memory_pairs = []
            for turn in chat_history:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role == "user":
                    memory_pairs.append(f"User: {content}")
                elif role == "assistant":
                    memory_pairs.append(f"Assistant: {content}")
            memory_context = "\n".join(memory_pairs)

        if documents:
            # If this is a table (CSV/Excel), use handle_table_query for full table QA
            if all(d.get("metadata", {}).get("file_ext") in [".csv", ".xls", ".xlsx"] for d in documents):
                from .utils import handle_table_query
                # Reconstruct CSV text: header + all rows
                rows = []
                header = None
                for d in documents:
                    parts = d.get("content", "").split("\n", 1)
                    if header is None and parts:
                        header = parts[0]
                    if len(parts) > 1:
                        rows.append(parts[1])
                csv_text = header + "\n" + "\n".join(rows)
                table_ans = handle_table_query(csv_text, query)
                if table_ans is not None:
                    return table_ans
            print("[+] Using provided documents as context")
            # Group documents by vector_id (i.e., per file)
            grouped_docs = {}
            for doc in documents:
                vector_id = doc.get("metadata", {}).get("vector_id", "unknown")
                grouped_docs.setdefault(vector_id, []).append(doc)

            print(f"[+] Grouped into {len(grouped_docs)} files for summarization")

            mini_summaries = []

            # Summarize each file separately
            for vector_id, docs in grouped_docs.items():
                combined_text = "\n".join([d["content"] for d in docs])
                est_tokens = estimate_tokens(combined_text)

                if est_tokens <= max_token_threshold:
                    # Directly use combined content if small enough
                    mini_summary = summarize_context(combined_text)
                else:
                    # For huge files, split and batch summarize
                    batches = list(batch_documents(
                        [LangChainDocument(page_content=d["content"]) for d in docs],
                        batch_size=batch_size
                    ))
                    batch_summaries = []
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(summarize_batch, batch) for batch in batches]
                        for future in futures:
                            batch_summaries.append(future.result())

                    combined_summary = "\n".join(batch_summaries)
                    mini_summary = summarize_context(combined_summary)

                mini_summaries.append(mini_summary)

            # Combine all file summaries
            combined_context = "\n\n".join(mini_summaries)
            final_context = combined_context

            print(f"[+] Combined {len(mini_summaries)} file summaries into final context")

        else:
            print("[+] Using default retriever context (no specific documents)")
            retriever = qdrant.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.get_relevant_documents(query)
            final_context = "\n".join([doc.page_content for doc in retrieved_docs])

        end = time.time()
        print(f"[+] Retrieved and processed context in: {end - start:.2f} seconds")

        # Build full prompt
        prompt_template = PromptTemplate(
            template="""
                {system_prompt}

                Context:
                {context}

                Conversation History:
                {memory}

                Current Question:
                {question}

                Answer:
                """,
                input_variables=["system_prompt", "context", "memory", "question"]
        )

        qa_chain = LLMChain(
            llm=llm,
            prompt=prompt_template.partial(
                system_prompt=system_prompt,
                context=final_context,
                memory=memory_context
            )
        )

        start_time = time.time()
        result = qa_chain.invoke({"question": query})["text"]
        end_time = time.time()
        print(f"[+] Question answered in: {end_time - start_time:.2f} seconds")

        return result

    except Exception as e:
        print(f"[!] Error in ask_question: {e}")
        raise

# -----------  Deep Search + Reasoning ----------- 

# def ask_question(query: str, source_type: str):
#     retriever = None
#     if source_type == "file":
#         retriever = MultiQueryRetriever.from_llm(
#             retriever=qdrant.as_retriever(search_kwargs={"k": 20}),
#             llm=llm
#         )
#     elif source_type == "integration":
#         retriever = MultiQueryRetriever.from_llm(
#             retriever=qdrant.as_retriever(search_kwargs={"k": 20}),
#             llm=llm
#         )

#     reasoning_prompt = """
#     Use the following context to answer the question. Think step-by-step before concluding.
#     If you don't know the answer, say "I don't know" instead of making up facts.

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer (with reasoning):
#     """

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type_kwargs={"prompt": PromptTemplate.from_template(reasoning_prompt)},
#         chain_type="stuff"  # or "map_reduce" for large inputs
#     )

#     result = qa.run(query)
#     return result

# def ask_question(query: str, source_type: str):
#     retriever = None
#     if source_type == "file":
#         retriever = qdrant.as_retriever()
#     elif source_type == "integration":
#         retriever = qdrant.as_retriever()

#     # 1. Define a smarter prompt
#     prompt_template = """You are a helpful assistant. 
#         Use the following context to answer the question.
#         If unsure, reason carefully and answer based on the information provided.

#         Context:
#         {context}

#         Question:
#         {question}

#         Answer (step-by-step reasoning first, then final answer):
#         """

#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=prompt_template,
#     )

#     # 2. Create a better QA Chain with reasoning prompt
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",  # you can try "map_reduce" or "refine" for even deeper reasoning
#         chain_type_kwargs={"prompt": prompt},
#     )

#     # 3. Run the query
#     result = qa_chain.run(query)
#     return result



def retrieve_documents_by_vector_id(vector_id: str) -> list:
    try:
        # Create a filter for the vector_id
        filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.vector_id",
                    match=MatchValue(value=vector_id)
                )
            ]
        )

        print(f"[+] Searching for documents with vector_id: {vector_id}")

        # Query Qdrant for documents matching the vector_id
        search_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filter,
            limit=1000,  # Increase to support large tables with many rows
            with_vectors=False,
            with_payload=True
        )

        print(f"[+] Search result: {search_result}")
        if not search_result[0]:  # Check if any records were found
            print(f"[!] No documents found for vector_id: {vector_id}. Attempting re-ingestion...")
            
            # Fetch document from the database
            document = DocumentModel.objects.filter(vector_id=vector_id).first()
            if not document:
                print(f"[!] No document found in the database for vector_id: {vector_id}")
                return []

            # Re-ingest the document into Qdrant
            print(f"[+] Document found in database. Re-ingesting document into Qdrant for vector_id: {vector_id}")
            file_ext = Path(document.title).suffix.lower()
            source_type = "file"  # This can be dynamic if required

            # Check if the content is in JSON format (optional handling)
            try:
                # Try to load the document content as JSON
                json_content = json.loads(document.content)
                print("[+] Detected JSON format. Preparing to parse...")
                document_content = json_content  # Assuming it's a list of chunks or other data
            except json.JSONDecodeError as e:
                print(f"[!] JSON decode error: {e}. Treating content as plain text.")
                document_content = document.content  # Treat as plain text or process differently

            # Manually chunk the text content into smaller pieces for ingestion
            chunk_size = 1000  # Set a size based on your use case
            text_chunks = [document_content[i:i+chunk_size] for i in range(0, len(document_content), chunk_size)]

            print(f"[+] Splitting text into {len(text_chunks)} chunks...")
            
            # Insert chunks into Qdrant
            if text_chunks:
                for chunk in text_chunks:
                    insert_document_to_vectorstore(chunk, source_type, file_ext, vector_id)
                    print(f"[+] Inserted chunk into Qdrant.")

            print(f"[+] Re-ingestion complete. Retrying retrieval...")

            # Retry the search after re-ingestion
            search_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=filter,
                limit=100,  # Adjust based on expected number of chunks
                with_vectors=False,
                with_payload=True
            )

            if not search_result[0]:
                print(f"[!] Re-ingestion completed, but still no documents found for vector_id: {vector_id}")
                return []

        # Extract documents from the search result
        documents = []
        for record in search_result[0]:  # search_result[0] contains the records
            doc_content = record.payload.get("page_content", "")
            doc_metadata = record.payload.get("metadata", {})
            documents.append({
                "content": doc_content,
                "metadata": doc_metadata
            })

        return documents

    except Exception as e:
        print(f"[!] Error retrieving documents by vector_id: {e}")
        return []
    
def delete_documents_by_vector_id(vector_id: str) -> list:
    try:
        # Delete from Qdrant
        print(f"[+] Deleting documents from Qdrant with vector_id: {vector_id}")
        filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.vector_id",  # Adjust based on Qdrant payload structure
                    match=MatchValue(value=vector_id)
                )
            ]
        )
        qdrant_client.delete(
            collection_name=collection_name,
            points_selector=filter
        )
        print(f"[+] Successfully deleted Qdrant documents with vector_id: {vector_id}")
        return True
    except Exception as e:
        print(f"[!] Error deleting documents from Qdrant: {e}")
        return False
    

class EnrichedIngestMixin:
    """
    Call this from your existing IngestAPIView after extracting file text.
    Adds auto-summary and metadata enrichment to Document
    """
    def enrich_document(self, document_obj, file_text, file_ext):
        try:
            summary = summarize_context(file_text[:3000])
            metadata = extract_metadata(file_text, file_ext)

            document_obj.summary = summary
            document_obj.keywords = metadata
            document_obj.save()

            print(f"[+] Document enriched with summary and keywords")
        except Exception as e:
            print(f"[!] Failed to enrich document: {e}")

# In utils.py (add this to utils file)
def retrieve_documents_by_vector_ids(vector_ids: List[str]) -> list:
    from qdrant_client.http.models import Filter, FieldCondition
    from .utils import qdrant_client, collection_name

    filter = Filter(
        must=[
            FieldCondition(
                key="metadata.vector_id",
                match=MatchAny(any=vector_ids)
            )
        ]
    )

    print("[+] Searching documents for multiple vector_ids:", vector_ids)

    try:
        search_result = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=filter,
            limit=1000,  # Increase to support large tables with many rows
            with_vectors=False,
            with_payload=True
        )

        if not search_result[0]:
            return []

        documents = []
        for record in search_result[0]:
            doc_content = record.payload.get("page_content", "")
            doc_metadata = record.payload.get("metadata", {})
            documents.append({"content": doc_content, "metadata": doc_metadata})

        return documents

    except Exception as e:
        print("[!] Multi-vector retrieval error:", str(e))
        return []
