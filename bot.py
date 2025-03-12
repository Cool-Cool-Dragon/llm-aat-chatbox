import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("行测小精灵", page_icon=":100:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是聚焦行测领域的机器人，储备了丰富知识和解题技巧。若您想在行测中突破瓶颈、获取高分，不妨把我当作引路人，现在就可以告诉我具体问题。"},
    ]

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
