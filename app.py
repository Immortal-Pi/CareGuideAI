from flask import Flask, render_template,jsonify,request 
from langchain_pinecone import PineconeVectorStore 
from langchain_openai import ChatOpenAI 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv 
from src.prompt import system_prompt 
import os
from src.model_loader import ModelLoader, ConfigLoader

load_dotenv()

app=Flask(__name__)

llm=ModelLoader(model_provider='openai').load_llm()
emb=ModelLoader(model_provider='openai_embeddings').load_llm()
config=ConfigLoader()

index_name=config['pinecone']['index_name']

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=emb
)

retriever=docsearch.as_retriever(search_type='similarity',search_kwargs={'k':3})

# chatModel=ChatOpenAI(model=config['llm']['openai']['model_name'])
prompt=ChatPromptTemplate.from_messages(
    [
        ('system',system_prompt),
        ('human',"{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["get","post"])
def chat():
    msg=request.form["msg"]
    input=msg 
    print(input)
    response=rag_chain.invoke({'input':msg})
    print(f"Response:{response['answer']}")
    return str(response['answer'])


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)


