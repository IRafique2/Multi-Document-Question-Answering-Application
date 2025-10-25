from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os
import streamlit as st

@st.cache_data()
def load_doc():
    documents = []
    for file in os.listdir('docs'):
        path = os.path.join('docs', file)
        try:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(path)
            elif file.endswith('.txt'):
                # 👇 specify encoding to handle Windows text files safely
                loader = TextLoader(path, encoding='utf-8')
            else:
                continue

            documents.extend(loader.load())

        except UnicodeDecodeError:
            # fallback for non-UTF8 encoded files
            loader = TextLoader(path, encoding='latin-1')
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"⚠️ Error loading {file}: {e}")
            continue

    return documents
