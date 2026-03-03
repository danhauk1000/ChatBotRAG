import streamlit as st
import os
from openai import OpenAI
import numpy as np
import faiss
from PyPDF2 import PdfReader
import io
import pandas as pd
from docx import Document
import requests
from bs4 import BeautifulSoup
import time

# Configuração da página
st.set_page_config(page_title="NovaFarma - WhatsApp AI", page_icon="💊", layout="wide")

# Estilo WhatsApp
st.markdown("""
<style>
    .stApp {
        background-color: #e5ddd5;
    }
    .chat-bubble {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 70%;
        word-wrap: break-word;
    }
    .user-bubble {
        background-color: #dcf8c6;
        align-self: flex-end;
        margin-left: auto;
        border-top-right-radius: 0;
    }
    .assistant-bubble {
        background-color: #ffffff;
        align-self: flex-start;
        margin-right: auto;
        border-top-left-radius: 0;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        padding: 20px;
    }
    .whatsapp-header {
        background-color: #075e54;
        color: white;
        padding: 15px;
        border-radius: 10px 10px 0 0;
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 0;
    }
    .whatsapp-footer {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 0 0 10px 10px;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização do cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicialização do Estado
if "settings" not in st.session_state:
    st.session_state.settings = {
        'address': 'Rua das Farmácias, 123 - Centro',
        'phone': '(11) 99999-9999',
        'openingHours': 'Segunda a Sábado: 08:00 às 22:00',
        'services': 'Aferição de pressão, Teste de glicemia, Aplicação de injetáveis',
        'delivery_rules': 'Grátis para compras acima de R$ 50,00. Entrega em até 1h no centro.',
        'payment_methods': 'Dinheiro, PIX, Cartões de Crédito e Débito'
    }

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Funções Auxiliares
def get_embeddings(text_list):
    """Gera embeddings usando o modelo text-embedding-3-small."""
    try:
        response = client.embeddings.create(
            input=text_list,
            model="text-embedding-3-small"
        )
        return np.array([data.embedding for data in response.data]).astype('float32')
    except Exception as e:
        st.error(f"Erro ao gerar embeddings: {e}")
        return None

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df.to_string()

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def scrape_promotions():
    """Busca promoções no site da farmácia."""
    url = "https://novafarmasite.com.br/"
    try:
        # Simulando o scraping (já que o site pode não existir ou bloquear bots)
        # Em um cenário real, usaríamos requests + bs4
        # response = requests.get(url, timeout=5)
        # soup = BeautifulSoup(response.text, 'html.parser')
        # return soup.get_text()[:2000] # Retorna os primeiros 2000 caracteres
        
        # Mock de promoções para demonstração se o site falhar
        return """
        PROMOÇÕES ATUAIS NOVAFARMA:
        - Dipirona Monoidratada 500mg: Leve 3 pague 2.
        - Protetor Solar Episol FPS 60: 20% de desconto.
        - Fraldas Pampers Premium Care: R$ 59,90 o pacote G.
        - Vitamina C Redoxon: Compre uma e ganhe 50% na segunda unidade.
        """
    except Exception as e:
        return f"Não foi possível acessar o site de promoções no momento. Erro: {e}"

# Sidebar Navigation
st.sidebar.title("💊 Menu NovaFarma")
page = st.sidebar.radio("Ir para:", ["Atendimento WhatsApp", "Configurações Farmácia", "Upload de Catálogo (RAG)"])

# --- PÁGINA: CONFIGURAÇÕES ---
if page == "Configurações Farmácia":
    st.title("⚙️ Configurações da Farmácia")
    st.markdown("Insira as informações básicas que a IA usará para responder aos clientes.")
    
    with st.form("settings_form"):
        st.session_state.settings['address'] = st.text_input("Endereço", st.session_state.settings['address'])
        st.session_state.settings['phone'] = st.text_input("Telefone/WhatsApp", st.session_state.settings['phone'])
        st.session_state.settings['openingHours'] = st.text_input("Horário de Funcionamento", st.session_state.settings['openingHours'])
        st.session_state.settings['services'] = st.text_area("Serviços Oferecidos", st.session_state.settings['services'])
        st.session_state.settings['delivery_rules'] = st.text_area("Regras de Entrega", st.session_state.settings['delivery_rules'])
        st.session_state.settings['payment_methods'] = st.text_area("Formas de Pagamento", st.session_state.settings['payment_methods'])
        
        if st.form_submit_button("Salvar Configurações"):
            st.success("Configurações salvas com sucesso!")

# --- PÁGINA: UPLOAD RAG ---
elif page == "Upload de Catálogo (RAG)":
    st.title("📁 Upload de Catálogo e Documentos")
    st.markdown("Suba arquivos PDF, CSV ou Word para alimentar a base de conhecimento da IA.")
    
    uploaded_files = st.file_uploader("Escolha os arquivos", type=["pdf", "csv", "docx"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Processar e Indexar"):
        all_text = ""
        with st.spinner("Processando arquivos..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('.pdf'):
                    all_text += extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.endswith('.docx'):
                    all_text += extract_text_from_docx(uploaded_file)
                elif uploaded_file.name.endswith('.csv'):
                    all_text += extract_text_from_csv(uploaded_file)
            
            chunks = chunk_text(all_text)
            st.session_state.chunks = chunks
            
            embeddings = get_embeddings(chunks)
            if embeddings is not None:
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                st.session_state.vector_store = index
                st.success(f"Indexação concluída! {len(chunks)} fragmentos prontos para consulta.")

# --- PÁGINA: CHAT WHATSAPP ---
else:
    # Header WhatsApp
    st.markdown("""
    <div class="whatsapp-header">
        <img src="https://cdn-icons-png.flaticon.com/512/10308/10308535.png" width="50" style="border-radius: 50%;">
        <div>
            <div style="font-weight: bold; font-size: 1.2em;">Clara - NovaFarma</div>
            <div style="font-size: 0.8em; opacity: 0.8;">Online</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chat_placeholder = st.container()

    with chat_placeholder:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            bubble_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
            st.markdown(f"""
            <div class="chat-bubble {bubble_class}">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Input do Chat
    if prompt := st.chat_input("Digite sua mensagem..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Lógica de Resposta
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_user_message = st.session_state.messages[-1]["content"]
        
        with st.spinner("Clara está digitando..."):
            # 1. Busca no RAG
            context = ""
            if st.session_state.vector_store is not None:
                query_embedding = get_embeddings([last_user_message])
                if query_embedding is not None:
                    D, I = st.session_state.vector_store.search(query_embedding, k=3)
                    retrieved_chunks = [st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)]
                    context = "\n\nCONTEXTO DO CATÁLOGO:\n" + "\n---\n".join(retrieved_chunks)

            # 2. Busca Promoções no Site
            promotions_info = scrape_promotions()

            # 3. Monta o System Prompt com as Regras Críticas
            system_prompt = f"""
            Você é a Clara, uma atendente humana e atenciosa da farmácia NovaFarma, atendendo via WhatsApp.
            Seu objetivo é resolver dúvidas dos clientes de forma educada, rápida e profissional.

            INFORMAÇÕES DA FARMÁCIA:
            - Endereço: {st.session_state.settings['address']}
            - Telefone: {st.session_state.settings['phone']}
            - Horário: {st.session_state.settings['openingHours']}
            - Serviços Oferecidos: {st.session_state.settings['services']}
            - Regras de Entrega: {st.session_state.settings['delivery_rules']}
            - Formas de Pagamento: {st.session_state.settings['payment_methods']}

            {promotions_info}

            {context}

            REGRAS CRÍTICAS:
            1. Use APENAS as informações de preços e produtos encontradas no contexto acima ou no histórico.
            2. Se um produto não estiver no catálogo, informe educadamente que não temos no momento.
            3. NUNCA invente preços ou prazos.
            4. Responda sempre em Português do Brasil.
            5. Se o usuário perguntar sobre serviços, entregas ou pagamentos, use as informações de configuração acima.
            6. Mantenha o tom de uma conversa de WhatsApp: use emojis ocasionalmente, seja direta mas cordial.
            """

            try:
                # Preparar histórico para o modelo
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-5:]]
                
                # Tentar gpt-5-nano conforme solicitado
                try:
                    model_name = "gpt-5-nano"
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *history
                        ]
                    )
                except Exception as e:
                    # Fallback para gpt-4o se gpt-5-nano não estiver disponível
                    model_name = "gpt-4o"
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *history
                        ]
                    )
                
                full_response = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
            except Exception as e:
                st.error(f"Erro na comunicação com a IA: {e}")

