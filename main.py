# Imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline




# STORE your PDF's in folder named `data`
loader = DirectoryLoader('/data', show_progress=True, use_multithreading=True, loader_cls=PyPDFLoader)
docs = loader.load()

# Creating Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=45)  #3294
docs = text_splitter.split_documents(docs)

#Embedding model
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Creating Vector db
db = Chroma.from_documents(docs,embed_model,persist_directory="/chroma_db")

retriever = db.as_retriever(search_type= "mmr",k=10)

# Model id
checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  

#Downloading Model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,device_map='auto',torch_dtype=torch.float32)

# Hf pipeline
pipe = pipeline('text2text-generation',model = base_model,tokenizer = tokenizer,max_length = 256,do_sample = True,temperature = 0.7,top_p= 0.95)
local_llm = HuggingFacePipeline(pipeline=pipe)

#Prompt Template
template = """ Using only the information provided in the following context, answer the question:

{context}

question:{question}

Please provide a concise and accurate response based on the given context. If the answer cannot be determined from the context, explicitly state "Unable to determine from the provided context."
"""
prompt = ChatPromptTemplate.from_template(template)

#load model
model = local_llm

# Creating chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

query = input('Enter query:')


print('Response:',chain.invoke(query))


