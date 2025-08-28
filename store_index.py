from dotenv import load_dotenv
import os 
from src.helper import  filter_to_minimal_docs, text_split, load_pdf_files 
from pinecone import Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore 
from src.model_loader import ModelLoader, ConfigLoader
load_dotenv()
from src.logging import logger 

logger.info('loading pdf files from dir data')
extracted_data=load_pdf_files(data='data/')
logger.info('filtering the extracted data')
filter_data=filter_to_minimal_docs(extracted_data)
logger.info('converting the data into chunks')
text_chunks=text_split(filter_data)


# embeddings=download_hugging_face_embeddings()

# pinecone 
config=ConfigLoader()
pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name=config['pinecone']['index_name']

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=config['pinecone']['dimension'],
        metric=config['pinecone']['metric'],
        spec=ServerlessSpec(cloud=config['pinecone']['cloud'],region=config['pinecone']['region'])
    )

logger.info('loading config')
index=pc.Index(index_name)
llm=ModelLoader(model_provider='openai')
emb=ModelLoader(model_provider='openai_embeddings').load_llm()
logger.info('storing data into ')
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=emb,
    batch_size=config['pinecone']['batch_size'],
    embeddings_chunk_size=config['pinecone']['embeddings_chunk_size'],
)