import streamlit as st
import google.generativeai as genai
import os
import json
from PIL import Image
import io
import PyPDF2
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Clara RAG - Assistente Farmacêutica Pro", layout="wide")

# Initialize session state
if "catalog_chunks" not in st.session_state:
    st.session_state.catalog_chunks = []

if "catalog_embeddings" not in st.session_state:
    st.session_state.catalog_embeddings = None

if "settings" not in st.session_state:
    st.session_state.settings = {
        "name": "Farmácia Central Pro",
        "address": "Rua Principal, 123",
        "phone": "(11) 99999-9999",
        "openingHours": "08:00 - 20:00",
        "services": "Aferição de pressão, Teste de glicemia, Aplicação de injetáveis",
        "delivery_rules": "Entrega grátis para pedidos acima de R$ 50,00. Prazo de 30 a 60 minutos.",
        "payment_methods": "Cartão de Crédito, Débito, PIX e Dinheiro"
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Gemini API Setup ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ Chave de API não encontrada! Configure 'GEMINI_API_KEY' nos Secrets do Streamlit ou como variável de ambiente.")

# --- RAG Logic ---

def get_embeddings(texts):
    """Gera embeddings para uma lista de textos usando o modelo do Google."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'])
    except Exception as e:
        st.error(f"Erro ao gerar embeddings: {e}")
        return None

def find_relevant_context(query, top_k=5):
    """Encontra os trechos mais relevantes do catálogo para a pergunta do usuário."""
    if not st.session_state.catalog_chunks or st.session_state.catalog_embeddings is None:
        return ""

    try:
        # Embed da query
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Cálculo de similaridade de cosseno simples
        similarities = np.dot(st.session_state.catalog_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = [st.session_state.catalog_chunks[i] for i in top_indices]
        return "\n---\n".join(relevant_chunks)
    except Exception as e:
        st.error(f"Erro na busca RAG: {e}")
        return ""

def get_clara_response(user_input, history):
    try:
        model = genai.GenerativeModel('gemini-3.1-pro-preview')
        
        # Busca contexto relevante via RAG
        context = find_relevant_context(user_input)
        
        system_instruction = f"""
        Você é a Clara, uma assistente virtual de atendimento para a farmácia {st.session_state.settings['name']}.
        Seu objetivo é ser prestativa, educada e eficiente.

        INFORMAÇÕES DA FARMÁCIA:
        - Endereço: {st.session_state.settings['address']}
        - Telefone: {st.session_state.settings['phone']}
        - Horário: {st.session_state.settings['openingHours']}
        - Serviços Oferecidos: {st.session_state.settings['services']}
        - Regras de Entrega: {st.session_state.settings['delivery_rules']}
        - Formas de Pagamento: {st.session_state.settings['payment_methods']}

        CONTEXTO RELEVANTE DO CATÁLOGO (RAG):
        {context if context else "Nenhuma informação específica encontrada no catálogo para esta busca."}

        REGRAS CRÍTICAS:
        1. Use APENAS as informações de preços e produtos encontradas no contexto acima ou no histórico.
        2. Se um produto não estiver no catálogo, informe educadamente que não temos no momento.
        3. NUNCA invente preços ou prazos.
        4. Responda sempre em Português do Brasil.
        5. Se o usuário perguntar sobre serviços, entregas ou pagamentos, use as informações de configuração acima.
        """
        
        # Chat session with history
        chat = model.start_chat(history=[
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} 
            for m in history
        ])
        
        response = chat.send_message(f"{system_instruction}\n\nUsuário: {user_input}")
        return response.text
    except Exception as e:
        return f"❌ Erro na Clara Pro: {str(e)}"

def process_file_locally(uploaded_file):
    """Extrai texto localmente para evitar lentidão de envio de arquivo inteiro para API."""
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.type.startswith("text"):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.type.startswith("image"):
            # Para imagens, ainda precisamos da IA para extrair o texto
            model = genai.GenerativeModel('gemini-3.1-pro-preview')
            image = Image.open(uploaded_file)
            prompt = "Extraia todos os produtos e preços desta imagem. Formate como: 'Produto - R$ Preço'."
            response = model.generate_content([prompt, image])
            text = response.text
        return text
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
        return ""

def index_catalog(text):
    """Divide o texto em chunks e gera embeddings para o RAG."""
    if not text.strip():
        return
    
    # Chunking simples por linha ou parágrafo
    chunks = [c.strip() for c in text.split('\n') if len(c.strip()) > 5]
    
    if not chunks:
        return

    with st.spinner("Gerando base de conhecimento (Embeddings)..."):
        embeddings = get_embeddings(chunks)
        if embeddings is not None:
            st.session_state.catalog_chunks = chunks
            st.session_state.catalog_embeddings = embeddings
            st.success(f"Catálogo indexado com sucesso! {len(chunks)} itens processados.")

# --- UI Layout ---
st.title("💊 Clara Pro - RAG & Inteligência")

tabs = st.tabs(["💬 Chat de Atendimento", "📚 Base de Conhecimento", "⚙️ Configurações do Negócio"])

# --- Tab 1: Chat ---
with tabs[0]:
    st.subheader(f"Atendimento: {st.session_state.settings['name']}")
    
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Como posso ajudar hoje?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Clara Pro está analisando..."):
                response = get_clara_response(prompt, st.session_state.messages[:-1])
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --- Tab 2: Knowledge Base (RAG) ---
with tabs[1]:
    st.subheader("Gerenciamento do Catálogo (RAG)")
    st.write("Aqui você carrega os dados da sua farmácia para que a Clara possa consultá-los de forma inteligente.")
    
    uploaded_file = st.file_uploader("Suba seu catálogo (PDF, Imagem ou Texto)", type=["png", "jpg", "jpeg", "pdf", "txt", "csv"])
    
    if uploaded_file is not None:
        if st.button("Processar e Indexar Catálogo"):
            with st.spinner("Extraindo texto..."):
                text_content = process_file_locally(uploaded_file)
                if text_content:
                    index_catalog(text_content)
                else:
                    st.error("Não foi possível extrair texto do arquivo.")

    st.divider()
    st.write(f"**Status da Base:** {len(st.session_state.catalog_chunks)} trechos indexados.")
    if st.session_state.catalog_chunks:
        with st.expander("Ver trechos indexados"):
            for i, chunk in enumerate(st.session_state.catalog_chunks[:20]):
                st.text(f"{i+1}: {chunk}")
            if len(st.session_state.catalog_chunks) > 20:
                st.write("... e mais.")

# --- Tab 3: Settings ---
with tabs[2]:
    st.subheader("Configurações da Farmácia")
    with st.form("settings_form_pro"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nome da Farmácia", value=st.session_state.settings['name'])
            address = st.text_input("Endereço", value=st.session_state.settings['address'])
            phone = st.text_input("Telefone", value=st.session_state.settings['phone'])
            hours = st.text_input("Horário de Funcionamento", value=st.session_state.settings['openingHours'])
        
        with col2:
            services = st.text_area("Serviços Oferecidos", value=st.session_state.settings['services'])
            delivery = st.text_area("Regras de Entrega", value=st.session_state.settings['delivery_rules'])
            payment = st.text_area("Formas de Pagamento", value=st.session_state.settings['payment_methods'])
        
        if st.form_submit_button("Salvar Configurações Pro"):
            st.session_state.settings = {
                "name": name,
                "address": address,
                "phone": phone,
                "openingHours": hours,
                "services": services,
                "delivery_rules": delivery,
                "payment_methods": payment
            }
            st.success("Configurações atualizadas com sucesso!")

st.sidebar.markdown("---")
st.sidebar.info("🚀 **Clara Pro** utiliza RAG (Retrieval-Augmented Generation) para buscas precisas no seu catálogo, reduzindo custos e aumentando a velocidade de resposta.")
st.sidebar.warning("O modelo gemini-3.1-pro-preview é usado para o chat, garantindo maior raciocínio lógico.")
