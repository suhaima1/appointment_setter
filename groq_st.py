import streamlit as st
import os
from groq import Groq
import random

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# model_name = "meta-llama/Llama-2-7b"
# # access_token = os.environ.get('ACCESS_TOKEN_HF')
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Create a Hugging Face pipeline
# hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)

# # Integrate the Hugging Face pipeline into LangChain
# huggingface_llm = HuggingFacePipeline(pipeline=hf_pipeline)


def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key = os.environ.get('GROQ_API_KEY')

    # Display the Groq logo
    # spacer, col = st.columns([5, 1])  
    # with col:  
    #     st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    with open('greeting_message.txt', 'r') as file:
        greet = file.read()

    st.title("Chat with Proteus420!")
    # st.write(greet)

    # Add customization options to the sidebar
    st.sidebar.title('Customization')

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
    # if "user_input" not in st.session_state:
    #     st.session_state.user_input = ""
    # memory = ConversationBufferWindowMemory(k=15, memory_key="chat_history", return_messages=True)
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
        if user_question.strip(): 
            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=system_prompt
                    ),  # This is the persistent system prompt that is always included at the start of the chat.

                    MessagesPlaceholder(
                        variable_name="chat_history"
                    ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                    HumanMessagePromptTemplate.from_template(
                        "{human_input}"
                    ),  # This template is where the user's current input will be injected into the prompt.
                ]
            )

            groq_chat = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name='llama3-8b-8192'
            )
            

            # Create a conversation chain using the LangChain LLM (Language Learning Model)
            conversation = LLMChain(
                llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
                prompt=prompt,  # The constructed prompt template.
                verbose=True,   # Enables verbose output, which can be useful for debugging.
                memory=st.session_state.memory,  # The conversational memory object that stores and manages the conversation history.
            )
            # The chatbot's answer is generated by sending the full prompt to the Groq API.
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