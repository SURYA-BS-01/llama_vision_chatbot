import streamlit as st
from groq import Groq
import base64
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load API Key from environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ API Key. Please set it in the environment variables.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Function to encode image as base64
def encode_image(image):
    return base64.b64encode(image.read()).decode("utf-8")

# Streamlit App
st.set_page_config(page_title="AI OCR Chat", page_icon="📄", layout="wide")

# Sidebar
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/OCR-A_characters.svg/800px-OCR-A_characters.svg.png",
    width=200,
)
st.sidebar.title("AI OCR Chatbot")
st.sidebar.markdown("Extract text from images and chat with it.")

# File uploader
uploaded_file = st.file_uploader("Upload an image 📷", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    session_id = st.text_input("Session ID", value="default_session")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    # Convert image to base64
    encoded_image = encode_image(uploaded_file)

    model = "llama-3.2-11b-vision-preview"

    # Perform OCR using Groq Vision
    with st.spinner("Extracting text... 🔄"):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract and return only the text present in the given image as markdown format. Do not provide any additional details, explanations, or formatting beyond what is present in the image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                        ],
                    }
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            extracted_text = completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error during OCR extraction: {e}")
            extracted_text = ""

    # Display extracted text
    st.subheader("Extracted Text 📜")
    st.text_area("Output", extracted_text, height=200)

    # Initialize text embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store extracted text in ChromaDB
    db = Chroma.from_texts([extracted_text], embeddings, persist_directory="./chroma_db")
    retriever = db.as_retriever(search_kwargs={"k": 5})  # Ensure correct retrieval behavior

    # Contextualized question retrieval
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question, which may reference context from the chat history, 
        first formulate a standalone question that can be understood without the chat history.
        Then, answer the question strictly based on the provided content and chat history. 
        The response must be derived from the given content whenever possible. 
        Only if the provided content does not contain enough information to answer the question, 
        think beyond the given context and generate the best possible answer.
    """
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Question-answering system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, then only think outside the given context. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Function to retrieve chat session history
    def get_session_history() -> BaseChatMessageHistory:
        return st.session_state.chat_history

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda _: get_session_history(),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # **Display Chat History**
    st.subheader("Chat History 🗂️")

    # Create a container for the chat messages
    chat_container = st.container()

    # Iterate through chat history and display messages
    with chat_container:
        messages = get_session_history().messages
        displayed_messages = set()  # Track displayed messages to prevent duplicates

        for msg in messages:
            if msg.content not in displayed_messages:  # Avoid re-displaying messages
                if msg.type == "human":
                    st.markdown(f"**You:** {msg.content}")
                else:
                    st.markdown(f"**Assistant:** {msg.content}")
                displayed_messages.add(msg.content)  # Store displayed messages


    # User input for question-answering
    user_input = st.text_input("Your question:")

    if user_input:
        session_history = get_session_history()
        session_history.add_user_message(user_input)  # Store user query

        try:
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            assistant_response = response["answer"]
            session_history.add_ai_message(assistant_response)  # Store assistant response

            # Display assistant response
            st.subheader("Assistant's Response 💬")
            st.write(assistant_response)

        except Exception as e:
            st.error(f"Error processing your question: {e}")

    # Download button
    st.download_button("📥 Download Extracted Text", extracted_text, file_name="extracted_text.txt", mime="text/plain")

# Footer
st.markdown(
    """
---
👨‍💻 Built with ❤️ using **Streamlit**, **LangChain**, and **Groq API**
""",
    unsafe_allow_html=True,
)

# According to the text, planning in essay writing involves organizing
#  your thoughts 
# and presenting your arguments clearly in paragraphs, and 
