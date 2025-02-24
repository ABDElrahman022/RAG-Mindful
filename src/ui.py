# # This module contains the Streamlit UI for the chatbot.

# import streamlit as st
# import uuid
# from chatbot import generate_response
# from database import save_chat, get_chat_history

# # Generate a unique session ID for each user
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())

# st.set_page_config(page_title="Mindful Chatbot")

# # Sidebar with chatbot introduction
# with st.sidebar:
#     st.title('Hi there! I am a mental health chatbot!')

# # Retrieve past chat history from MongoDB
# st.session_state.messages = get_chat_history(st.session_state.session_id)

# # Display previous chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

# # Handle user input
# if input := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": input})
#     save_chat(st.session_state.session_id, "user", input)

#     with st.chat_message("user"):
#         st.write(input)

#     with st.chat_message("assistant"):
#         with st.spinner("Generating..."):
#             response = generate_response(input, st.session_state.messages)
#             st.write(response)

#     # Save chatbot response to MongoDB
#     save_chat(st.session_state.session_id, "assistant", response)
#     st.session_state.messages.append({"role": "assistant", "content": response})

# This module contains the Streamlit UI for the chatbot.

import streamlit as st
import uuid
from chatbot import generate_response
from database import save_chat, get_chat_history

# Ensure session_id is properly initialized
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.set_page_config(page_title="Mindful Chatbot")

# Sidebar with chatbot introduction
with st.sidebar:
    st.title('Hi there! I am a mental health chatbot!')

# Retrieve past chat history from MongoDB, ensuring it's always a list
chat_history = get_chat_history(st.session_state.session_id)
st.session_state.messages = chat_history if chat_history else [] 

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle user input
if input := st.chat_input():
    user_message = {"role": "user", "content": input}
    st.session_state.messages.append(user_message)
    save_chat(st.session_state.session_id, "user", input)

    with st.chat_message("user"):
        st.write(input)

    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            response = generate_response(input, st.session_state.messages)
            st.write(response)

    # Save chatbot response to MongoDB
    assistant_message = {"role": "assistant", "content": response}
    save_chat(st.session_state.session_id, "assistant", response)
    st.session_state.messages.append(assistant_message)
