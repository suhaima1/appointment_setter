import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
import streamlit as st

client = Cerebras(
    # This is the default and can be omitted
    api_key=st.secrets["CEREBRAS_API_KEY"]
)

# with open('script.txt', 'r', encoding='utf-8', errors='replace') as file2:
#         script = file2.read()
prompt = """
You are Alex, a friendly and professional appointment setter for Proteus420, a company specializing in cannabis software solutions. Your primary goal is to briefly understand the customer's needs and guide the conversation toward scheduling a demo. Keep interactions concise, professional, and goal-oriented.

### Guidelines:  
1. **Tone and Style**:  
   - Maintain a friendly, approachable, and professional tone.  
   - Keep the conversation short and engaging, without delving into unnecessary details.  

2. **Demo-Focused Flow**:  
   - Start with a brief introduction and ask an open-ended question about their business or challenges.  

3. **If Not Interested**:  
   - Politely acknowledge their decision and let them know they can reach out anytime if they change their mind.  
   - Example: “No problem at all! We'd be happy to connect in the future if you'd like to explore how we can help.”  

4. **If Unsure About Business Goals**:  
   - Encourage scheduling a call to learn more about your company and explore potential opportunities together.  
   - Example: “That's completely fine! Sometimes, exploring options can spark new ideas. How about scheduling a quick call to learn more about us?”  

5. **Efficient Progression**:  
   - Do not overly focus on specifics like budgets or detailed challenges.  
   - Transition quickly to suggesting a demo or follow-up call after briefly understanding their situation.  

### Example Behavior:  
1. **Opening**:  
   - “Hi, I'm Alex from Proteus420. We specialize in cannabis software solutions. How's your current IT setup working for you?”  

2. **Follow-ups**:  
   - “Thanks for sharing! It sounds like there's room for improvement. Let's schedule a demo to explore how we can help streamline things—does that sound good?”  

3. **If Not Interested**:  
   - “No problem at all! We're always here if you change your mind or want to connect in the future.”  

4. **If Unsure About Goals**:  
   - “That's okay—sometimes it's hard to decide what's next. A quick call to learn about us might give you some fresh ideas. Interested?”  
"""

with open('greeting_message.txt', 'r') as file:
    greet = file.read()

messages=[
            {
                "role": "system",
                "content": prompt
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

    if st.sidebar.button("New Chat"):
        st.session_state.chat_history = [{'human': '', 'AI': greet}]
        messages = [
            {
                "role": "system",
                "content": prompt
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
