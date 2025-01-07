import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
import streamlit as st

client = Cerebras(
    # This is the default and can be omitted
    api_key=st.secrets["CEREBRAS_API_KEY"]
)


with open('greeting_message.txt', 'r') as file:
    greet = file.read()
    
with open('script.txt', 'r', encoding='utf-8', errors='replace') as file2:
    script = file2.read()
    
messages=[
    {
        "role": "system",
        "content": script
    },
    {
        "role": "assistant",
        "content": greet
    }
]

def chatbot(input):
    messages.append({"role": "user", "content": input})
    stream = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b",
        stream=True,
        max_completion_tokens=2048,
        temperature=0.8,
        top_p=1
    )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
    return response

def main():
    """
    This function is the main entry point of the application. It sets up the client, the Streamlit interface, and handles the chat interaction.
    """

    st.title("Chat with Proteus420!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{'human': '', 'AI': greet}]

    
    system_prompt = st.sidebar.text_area("System prompt:", value=script, height=200)
    if st.sidebar.button("Submit"):
        with open("script.txt", "w") as file:
            file.write(system_prompt)
        st.sidebar.success("System prompt updated!")

    if st.sidebar.button("New Chat"):
        st.session_state.chat_history = [{'human': '', 'AI': greet}]
        messages = [
            {
                "role": "system",
                "content": script
            },
            {
                "role": "assistant",
                "content": greet
            }
        ]
        st.success("New chat started!")

    for message in st.session_state.chat_history:
        if message['human']:
            st.markdown(f"<div style='text-align: right; padding: 10px; background-color: #e1ffc7; border-radius: 8px; margin-bottom: 5px;'>User: {message['human']}</div>", unsafe_allow_html=True)
        if message['AI']:
            st.markdown(f"<div style='text-align: left; padding: 10px; background-color: #d0e9ff; border-radius: 8px; margin-bottom: 5px;'>AI: {message['AI']}</div>", unsafe_allow_html=True)

    user_question = st.text_input("")  

    if st.button("Send"):
        if user_question and user_question.strip(): 
            response = chatbot(user_question)
            print(response)
            st.session_state.chat_history.append(
                {"human": user_question, "AI": response}
            )
        
            st.rerun()
            
        else:
            st.warning("Please enter a message before sending.")

if __name__ == "__main__":
    main()
