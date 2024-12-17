import streamlit as st
import numpy as np
import os
import json

from langchain.chains import RetrievalQA, StuffDocumentsChain, ConversationalRetrievalChain
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

groq_api_key = st.secrets['GROQ_API_KEY']
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
    # st.sidebar.title('Customization')

    # uploaded_files = st.sidebar.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

    # with open('script.txt', 'r', encoding='utf-8', errors='replace') as file2:
    #     script = file2.read()
    # system_prompt = st.sidebar.text_area("System prompt:", value=script, height=200)
    # if st.sidebar.button("Submit"):
    #     with open("script.txt", "w") as file:
    #         file.write(system_prompt)
    #     st.sidebar.success("System prompt updated!")

    with open("prompt.json", "r") as file:
        prompt_data = json.load(file)
        def json_to_prompt(json_data):
            # Extracting the instructions (if they exist in the provided JSON)
            instructions = "\n".join(f"- {inst}" for inst in json_data.get("instructions", []))
            
            # Constructing the call script sections
            script_parts = [
                f"**Greeting**: {json_data['call_script']['greeting']}",
                f"**Qualify Interest**: \n  - If yes: {json_data['call_script']['qualify_interest']['if_yes']}\n  - If no: {json_data['call_script']['qualify_interest']['if_no']}",
                f"**Follow-Up No Challenge**: \n  - If yes: {json_data['call_script']['qualify_interest']['follow_up_no_challenge']['if_yes']}\n  - If no: {json_data['call_script']['qualify_interest']['follow_up_no_challenge']['if_no']}",
                f"**Offer a Demo**: {json_data['call_script']['offer_demo']['if_challenge']}",
                f"**Capture Details**: \n  - If single or satellite: {json_data['call_script']['offer_demo']['capture_details']['if_single_or_satellite']['ask_budget']}\n    - {json_data['call_script']['offer_demo']['capture_details']['if_single_or_satellite']['budget_options']['under_50k']}\n    - {json_data['call_script']['offer_demo']['capture_details']['if_single_or_satellite']['budget_options']['between_100k_500k']}",
                f"**Capture Details**: \n  - If multinational: {json_data['call_script']['offer_demo']['capture_details']['if_multinational']['ask_budget']}\n    - {json_data['call_script']['offer_demo']['capture_details']['if_multinational']['budget_options']['between_100k_500k']}\n    - {json_data['call_script']['offer_demo']['capture_details']['if_multinational']['budget_options']['over_500k']}",
                f"**Final Call to Action**: {json_data['call_script']['final_call_to_action']}"
            ]
            
            # Combine the script parts into one large string
            script_text = "\n\n".join(script_parts)
            
            # Return the formatted output
            return f"Assistant Name: {json_data['assistant_name']} at {json_data['company']}\n\nInstructions:\n{instructions}\n\nCall Script:\n{script_text}"
        prompt = json_to_prompt(prompt_data)
        
    # vector_store = process_uploaded_files(uploaded_files)
    # if "rag_chain" not in st.session_state:
    #     if uploaded_files and vector_store:
    #         retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    #         custom_prompt = PromptTemplate(
    #             input_variables=["context", "question", "chat_history"],
    #             template=f"""
    #             {prompt} 

    #             Additional Context from Retrieved Documents:
    #             {{context}}

    #             Chat History (up to now):
    #             {{chat_history}}

    #             Guidelines:
    #             1. Stick to the provided call script unless the user asks about unrelated topics.
    #             2. Lead the conversation and proactively guide the user toward setting an appointment.
    #             3. Use the retrieved context only for questions outside the scope of the script.
    #             4. Avoid repeating the same responses unless explicitly requested.
    #             5. Acknowledge the user's input and keep your responses concise.

    #             User Query:
    #             {{question}}

    #             Your Response:"""
    #         )
    #         rag_chain = ConversationalRetrievalChain.from_llm(
    #                     llm=groq_chat,
    #                     retriever=retriever,
    #                     return_source_documents=True,
    #                     combine_docs_chain_kwargs={"prompt": custom_prompt}
    #                 )
    #         st.session_state.rag_chain = rag_chain
    
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
            vector_store = False
            if vector_store:

                # Integrate the custom prompt into the RAG pipeline
                history = []
                for chat in st.session_state.chat_history:
                    history.append((chat['human'],chat['AI']))

                rag_chain = st.session_state.rag_chain
                result = rag_chain({"question": user_question, "chat_history": history})
                response = result['answer']
                source = result['source_documents']
              
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
            print(st.session_state.chat_history)
            print("*"*20)
            st.rerun()
            
        else:
            st.warning("Please enter a message before sending.")

if __name__ == "__main__":
    main()
