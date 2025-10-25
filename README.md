# **Multi-Document Question Answering Application**
![Multi-Document Question Answering Application](images/screenshot.png)

This project is a **Multi-Document Question Answering** web application built using **Streamlit**, **LangChain**, and various Natural Language Processing (NLP) models. The app allows users to upload documents (PDF, DOCX, or TXT) and ask questions related to the content of those documents. The system retrieves the most relevant information based on the user's query, providing answers derived from the text of the uploaded documents.

## **Features**

* **Upload Multiple Document Formats**: Users can upload PDF, DOCX, and TXT documents.
* **Question Answering**: Users can ask specific questions, and the system will return answers based on the uploaded documents.
* **Contextual Search**: The app generates specific and contextually accurate questions based on the document's content.
* **Streamlit Interface**: An interactive and user-friendly interface to upload documents, ask questions, and view answers.
* **Multi-Document Handling**: The app can process and analyze multiple documents simultaneously.

## **Technologies Used**

* **Streamlit**: A framework for building interactive web applications.
* **LangChain**: A framework for building language model chains, enabling document retrieval and question answering.
* **Hugging Face Transformers**: Used for various NLP tasks like **text2text generation** and **question answering**.
* **Chroma**: Vector store used for efficient retrieval of documents.
* **FAISS**: Facebook AI Similarity Search library, used for high-performance similarity search.
* **PyPDFLoader**: A document loader to handle PDF files.
* **Docx2txtLoader**: A document loader for DOCX files.
* **TextLoader**: A document loader for plain text files.

## **How It Works**

1. **Upload Documents**: Users can upload documents (PDF, DOCX, TXT) to the application. These documents are processed and stored for querying.
2. **Split Documents**: The documents are split into smaller chunks using `CharacterTextSplitter` for efficient searching and retrieval.
3. **Generate Questions**: A question generator is used to generate clear and specific questions from the user input.
4. **Retrieve Answers**: The uploaded documents are searched to find the most relevant answers to the user's query.
5. **Provide Feedback**: The system returns the most contextually relevant answers and displays them on the screen in an easy-to-follow chat format.

## **Setup Instructions**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/multi-document-qa-app.git
   cd multi-document-qa-app
   ```

2. **Set Up Environment Variables:**

   Create a `.env` file in the project root directory and add your HuggingFace API token:

   ```bash
   HUGGINGFACE_API_TOKEN=your-huggingface-api-token
   ```

3. **Run the Streamlit App:**

   Once the dependencies are installed and the environment variables are set up, run the app:

   ```bash
   streamlit run app.py
   ```

   The app will be accessible at `http://localhost:8501`.

## **File Structure**

```
.
├── app.py                # Main Streamlit application
├── helper.py             # Helper functions for document processing, QA pipeline, etc.
├── .env                  # Environment variables (HuggingFace API Token)
├── docs/                 # Folder to store documents for processing
└── README.md             # This README file
```

## **Dependencies**

The required dependencies are listed in the `requirements.txt` file:

```txt
streamlit
langchain
requests
transformers
faiss-cpu
python-dotenv
chroma
streamlit-chat
```

## **Environment Variables**

This project requires the following environment variable:

* **HUGGINGFACE_API_TOKEN**: Your HuggingFace API token for accessing the QA model.


## **Key Features**

* **Multi-Document Support**: Upload multiple documents and query across all of them.
* **Streamlit Chat Interface**: User-friendly chat interface that displays both the user's questions and the model's answers.
* **Question Generation**: The system generates specific and clear questions based on user input.
* **Efficient Document Retrieval**: Using FAISS and Chroma for high-speed and accurate document retrieval.

