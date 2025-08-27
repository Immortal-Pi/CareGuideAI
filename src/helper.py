from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document 
# from langchain.embeddings import HuggingFaceEmbeddings
from typing import List 
# from langchain.schema import Document 


# Extract data from the PDF file 
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob='*.pdf',
                           loader_cls=PyPDFLoader)
    
    document=loader.load()
    return document 

def filter_to_minimal_docs(docs:List[Document])-> List[Document]:
    """
    Given a list of Document object, return a new list of document object 
    containing only 'source' in metadata and the original page_content
    """
    minimal_docs: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs 

# split the data into chunks 
def text_split(extraced_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunk=text_splitter.split_documents(extraced_data)
    return text_chunk


# Download the embeddings from huggingface 
# def download_hugging_face_embeddings():
#     embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     return embeddings 

# load Azure embedding model 
 