## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
To design and implement a question-answering chatbot that can process a PDF document and provide relevant answers to user queries. The system uses LangChain for document loading, text chunking, embeddings, and conversational retrieval. Its effectiveness is evaluated by testing responses to queries from the PDF content.

### DESIGN STEPS:

#### STEP 1:
Load the PDF using PyPDFLoader.
#### STEP 2:
Split text into chunks with RecursiveCharacterTextSplitter.
#### STEP 3:
Generate embeddings using OpenAIEmbeddings.
#### STEP 4:
Store vectors in DocArrayInMemorySearch.
#### STEP 5:
Create a retriever to fetch relevant chunks.
#### STEP 6:
Integrate ChatOpenAI with ConversationalRetrievalChain.
#### STEP 7:
Add ConversationBufferMemory for chat history.
#### STEP 8:
Build a simple chatbot loop for user interaction.
#### STEP 9:
Test with diverse queries and evaluate accuracy.

### PROGRAM:
```
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  
openai_api_key = os.environ['OPENAI_API_KEY']

def load_pdf_to_db(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever
def create_conversational_chain(retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        memory=memory
    )
    return conversational_chain
if __name__ == "__main__":
    pdf_file_path = "exp3/blackarch-guide-en.pdf"  
    retriever = load_pdf_to_db(pdf_file_path)
    
    chatbot = create_conversational_chain(retriever)
    
    print("Welcome to the PDF Question-Answering Chatbot!")
    print("Type 'exit' to quit.")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Thanks for chatting! Goodbye!")
            break
        
        result = chatbot({"question": user_query})
        print("Chatbot:", result["answer"])
```

### OUTPUT:
![alt text](<Screenshot 2025-10-03 110920.png>)

### RESULT:
The chatbot was successfully implemented. It accurately retrieved and answered queries from the PDF document, demonstrating the effectiveness of LangChain in building document-based conversational AI.
