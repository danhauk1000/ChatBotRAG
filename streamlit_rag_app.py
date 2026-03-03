import streamlit as st
import os
from openai import OpenAI
import numpy as np
import faiss
from PyPDF2 import PdfReader
import io

# Configuração da página
st.set_page_config(page_title="NovaFarma - OpenAI RAG", page_icon="💊", layout="wide")

# Inicialização do cliente OpenAI
# Nota: A chave API deve ser configurada no ambiente como OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(text_list):
    """Gera embeddings usando o modelo text-embedding-3-small."""
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    return np.array([data.embedding for data in response.data]).astype('float32')

def extract_text_from_pdf(pdf_file):
    """Extrai texto de um arquivo PDF."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Divide o texto em pedaços menores."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Interface do Usuário
st.title("🚀 NovaFarma - RAG com OpenAI")
st.markdown("Utilizando `text-embedding-3-small` e o modelo `gpt-5-nano` (conforme solicitado).")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.chunks = []

with st.sidebar:
    st.header("📁 Documentos")
    uploaded_file = st.file_uploader("Upload de PDF para o RAG", type="pdf")
    
    if uploaded_file and st.button("Processar Documento"):
        with st.spinner("Extraindo texto e gerando embeddings..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)
            st.session_state.chunks = chunks
            
            # Gerar embeddings
            embeddings = get_embeddings(chunks)
            
            # Criar índice FAISS
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            st.session_state.vector_store = index
            st.success(f"Documento processado! {len(chunks)} fragmentos gerados.")

# Chat
st.header("💬 Chat com a Clara (OpenAI Edition)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Como posso ajudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context = ""
        if st.session_state.vector_store is not None:
            # Busca semântica
            query_embedding = get_embeddings([prompt])
            D, I = st.session_state.vector_store.search(query_embedding, k=3)
            
            retrieved_chunks = [st.session_state.chunks[i] for i in I[0]]
            context = "\n\nCONTEXTO DOS DOCUMENTOS:\n" + "\n---\n".join(retrieved_chunks)

        # Chamada ao modelo gpt-5-nano
        try:
            response = client.chat.completions.create(
                model="gpt-5-nano", # Nome do modelo solicitado pelo usuário
                messages=[
                    {"role": "system", "content": f"Você é a Clara, assistente da NovaFarma. Use o contexto abaixo se disponível para responder.{context}"},
                    {"role": "user", "content": prompt}
                ]
            )
            full_response = response.choices[0].message.content
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Erro ao chamar o modelo gpt-5-nano: {e}")
            st.info("Nota: O modelo 'gpt-5-nano' pode não estar disponível publicamente ainda. Verifique o nome do modelo.")

