import streamlit as st
import numpy as np
import os

from langchain.chains import RetrievalQA, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS  # Replace with another vector store if needed
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.schema import Document

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
# from langchain.chains import ToolUseChain

from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ.get('GROQ_API_KEY')
groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name='llama3-8b-8192'
)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return vector store for RAG."""
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            with open(uploaded_file.name, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()
            documents.extend(docs)
            os.remove(uploaded_file.name)
        elif uploaded_file.name.endswith(".docx"):
            with open(uploaded_file.name, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            loader = Docx2txtLoader(uploaded_file.name)
            docs = loader.load()
            documents.extend(docs)
            os.remove(uploaded_file.name)
        elif uploaded_file.name.endswith(".txt"):
            with open(uploaded_file.name, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with open(uploaded_file.name, "r", encoding="utf-8") as text_file:
                content = text_file.read()
                docs = [Document(page_content=content)]
            documents.extend(docs)
            os.remove(uploaded_file.name)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
    if documents:
        texts = [doc.page_content for doc in documents]
        # embeddings = embedding_model.encode(texts)
        vector_store = FAISS.from_texts(texts, embedding_model)
        return vector_store
    return None


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # The title and greeting message of the Streamlit application
    with open('greeting_message.txt', 'r') as file:
        greet = file.read()

    st.title("Chat with Proteus420!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')

    uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    vector_store = process_uploaded_files(uploaded_files)
    if uploaded_files and vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    with open('script.txt', 'r') as file2:
        script = file2.read()
    system_prompt = st.sidebar.text_area("System prompt:", value=script, height=200)
    if st.sidebar.button("Submit"):
        with open("script.txt", "w") as file:
            file.write(system_prompt)
        st.sidebar.success("System prompt updated!")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{'human': '', 'AI': greet}]
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=15, memory_key="chat_history", return_messages=True
        )

    if st.sidebar.button("New Chat"):
        st.session_state.chat_history = [{'human': '', 'AI': greet}]
        st.session_state.memory = ConversationBufferWindowMemory(
            k=15, memory_key="chat_history", return_messages=True
        )
        st.success("New chat started!")
        
    for message in st.session_state.chat_history:
        
        if message['human']:
            st.markdown(f"<div style='text-align: right; padding: 10px; background-color: #e1ffc7; border-radius: 8px; margin-bottom: 5px;'>User: {message['human']}</div>", unsafe_allow_html=True)
        if message['AI']:
            st.markdown(f"<div style='text-align: left; padding: 10px; background-color: #d0e9ff; border-radius: 8px; margin-bottom: 5px;'>AI: {message['AI']}</div>", unsafe_allow_html=True)

    user_question = st.text_input("")  

    # If the user has asked a question,
    if st.button("Send"):
        if user_question and user_question.strip(): 
            # Construct a chat prompt template using various components

            # def retrieve_documents(query):
            #     retriever = vector_store.as_retriever(search_kwargs={"k": 2})
            #     docs = retriever.get_relevant_documents(query)
            #     return "\n".join([doc.page_content for doc in docs])
            # retriever_tool = Tool(
            #     name="RetrieveCompanyDocs",
            #     func=retrieve_documents,
            #     description="Use this tool to answer questions about the company or specific documents."
            # )

            # # Define the list of tools
            # tools = [retriever_tool]
            if vector_store:
                custom_prompt = PromptTemplate(
                template="""You are Alex from Proteus420 and your task is to let people know about the company.
                Answer users queries in a professional and human like tone.
                Context: {context}.""",
                input_variables = ['context']
                )
            
                llm_chain = LLMChain(llm=groq_chat, prompt=custom_prompt, memory=st.session_state.memory)
                doc_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
                qa_chain = RetrievalQA(
                    retriever=retriever,
                    combine_documents_chain=doc_chain,
                    return_source_documents=True,
                )
                result = qa_chain({'query':user_question})
                response = result['result']
            else:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(content=system_prompt),  # This is the persistent system prompt that is always included at the start of the chat.
                        MessagesPlaceholder(variable_name="chat_history"),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.
                        HumanMessagePromptTemplate.from_template("{human_input}"),  # This template is where the user's current input will be injected into the prompt.
                    ]
                )

                conversation = LLMChain(
                    llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
                    prompt=prompt,  # The constructed prompt template.
                    verbose=True,   # Enables verbose output, which can be useful for debugging.
                    memory=st.session_state.memory,  # The conversational memory object that stores and manages the conversation history.
                )
                response = conversation.predict(human_input=user_question)

            st.session_state.memory.save_context(
                {"input": user_question}, {"output": response}
            )
            st.session_state.chat_history.append(
                {"human": user_question, "AI": response}
            )
            st.rerun()

        else:
            st.warning("Please enter a message before sending.")

if __name__ == "__main__":
    main()
