import os
import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from miltidocloader import load_doc
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# Create your pipeline
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Wrap it for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)


# Load documents
documents = load_doc()
chat_history = []

# Initialize the Hugging Face pipeline for QA
#qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
hf_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Wrap it for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)



# Define text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

# Split documents
splitter = text_splitter.split_documents(documents)

# Create vector store with Chroma
embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-nli-stsb-mean-tokens")
vector_db = Chroma.from_documents(
    documents=splitter,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_db.persist()

# Initialize question generator template (you can refine this part if needed)
question_generator_prompt = """
    Given the user's input, please create a clear and specific question that is designed to retrieve relevant information from the document database. The question should directly relate to the content of interviews with famous persons present in the uploaded documents.

    The system will then search the uploaded documents to find the best answer to the user's question. Ensure that the question is framed in a way that allows the system to provide an accurate and context-specific response based on the interviews.

    Please ensure the following:
    1. The question should be directly linked to the interviews with famous persons in the documents.
    2. It should be specific and unambiguous to avoid irrelevant results.
    3. Avoid general or vague questions like "Tell me about X". Instead, ask specific queries like "What does the interview say about X?" or "What is the main perspective shared by Y in the interview?".
"""

question_generator_template = PromptTemplate(input_variables=["input"], template=question_generator_prompt)

# Now it’s valid inside LLMChain
question_generator = LLMChain(
    llm=llm,
    prompt=question_generator_template
)

# Streamlit app
st.title("📚 Multi-Document Question Answering")
st.header("Ask questions about your uploaded documents!")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_query():
    input_text = st.chat_input("Enter your question here")
    return input_text

user_input = get_query()

if user_input:
    # Store user input in session state
    st.session_state.past.append(user_input)

    # Generate answer using QA pipeline
    response = hf_pipeline({
        'question': user_input,
        'context': " ".join([doc.page_content for doc in splitter])
    })

    # Store the model’s answer in session state
    st.session_state.generated.append(response['answer'])

# Display chat history (user question + model answer)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        user_msg = st.session_state['past'][i]
        bot_msg = st.session_state['generated'][i]

        # Display user message first
        message(user_msg, is_user=True, key=f"{i}_user")

        # Then display model's response
        message(bot_msg, key=f"{i}_bot")
