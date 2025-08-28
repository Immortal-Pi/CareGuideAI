import os 
from dotenv import load_dotenv 
from typing import Literal, Optional, Any 
from pydantic import BaseModel, Field 
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from src.config_loader import load_config 
from pydantic import BaseModel, ConfigDict 
from typing import Any 
load_dotenv()

class ConfigLoader():
    def __init__(self):
        print(f'Loading config .....')
        self.config=load_config()
        

    def __getitem__(self,key):
        return self.config[key]
    

class ModelLoader(BaseModel):
    model_provider: Literal['groq','openai','openai_embeddings']='groq'
    config: Optional[ConfigLoader]=Field(default=None, exclude=True)

    def model_post_init(self,__context:Any)->None:
        self.config=ConfigLoader() 

    class Config:
        arbitrary_types_allowed=True 
    def load_llm(self):
        """ 
        Load and return LLM model 
        """ 
        print('LLM loading....')
        print(f'Loading model from provider: {self.model_provider}')

        if self.model_provider=='groq':
            print('LLM loading...')
            print(f'Loading model from provider: {self.model_provider}')
            groq_api_key=os.getenv('GROQ_API_KEY')
            model_name=self.config['llm']['groq']['model_name']
            llm=ChatGroq(model=model_name, api_key=groq_api_key)

        elif self.model_provider=='openai':
            print('Loading LLM from Azure-OpenAI.....')
            deployment_name=self.config['llm']['openai']['model_name']
            end_point=self.config['llm']['openai']['end_point']
            api_version=self.config['llm']['openai']['api_version']
            api_key=os.getenv('AZURE_OPENAI_KEY') 
            llm=AzureChatOpenAI(azure_deployment=deployment_name,api_key=api_key,api_version=api_version, azure_endpoint=end_point)

        elif self.model_provider=='openai_embeddings':
            # Azure open ai embeddings
            print('Loading LLM from Azure-OpenAI.....')
            model_name=self.config['llm']['openai_embeddings']['model_name']
            azure_deployment=self.config['llm']['openai_embeddings']['model_name']
            end_point=self.config['llm']['openai_embeddings']['end_point']
            api_version=self.config['llm']['openai_embeddings']['api_version']
            api_key=os.getenv('AZURE_OPENAI_KEY') 
            llm=AzureOpenAIEmbeddings(model=model_name, api_key=api_key,api_version=api_version) 
        return llm 

           

       

        