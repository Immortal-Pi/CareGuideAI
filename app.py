from flask import Flask, render_template,jsonify,request,session 
from langchain_pinecone import PineconeVectorStore 
from langchain_openai import ChatOpenAI 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv 
from src.prompt import system_prompt 
import os
from src.model_loader import ModelLoader, ConfigLoader
from langchain_community.chat_message_histories import ChatMessageHistory 
from langchain_core.runnables.history import RunnableWithMessageHistory 
from uuid import uuid4
from secrets import token_urlsafe
load_dotenv()

app=Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", token_urlsafe(32))

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
        MessagesPlaceholder(variable_name='chat_history'),
        ('human',"{input}"),
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

# in memory message store 
_session_store ={} 

def get_history(session_id:str)-> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id]=ChatMessageHistory()
    return _session_store[session_id]

rag_chain_with_history=RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer',
)

@app.route("/")
def index():
    # esnsure we have session 
    if "session_id" not in session:
        session['session_id']=str(uuid4())
    return render_template('chat.html')

@app.route("/get",methods=["get","post"])
def chat():
    if "session_id" not in session:
        session['session_id']=str(uuid4())

    msg=request.form["msg"]
    input=msg 
    print(input)
    response=rag_chain_with_history.invoke(
        {"input":msg, "system_prompt":system_prompt},
        config={'configurable':{"session_id":session["session_id"]}}
    )
    print(f"Response:{response['answer']}")
    return str(response['answer'])


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)


