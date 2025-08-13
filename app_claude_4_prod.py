import streamlit as st
import time
import yaml
from datetime import datetime
import numpy as np
from groq import Groq
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# ==============================================================================
# 1. CONFIGURAÇÃO INICIAL E ESTILOS
# ==============================================================================

# Configuração da página (deve ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="🏓 PingPoli Agent",
    page_icon="🏓",
    layout="centered",
    initial_sidebar_state="expanded"
)

def load_css():
    """Carrega um CSS personalizado para um visual moderno e limpo."""
    st.markdown("""
    <style>
        /* Fonte global */
        html, body, [class*="st-"], .main {
            font-family: 'Inter', sans-serif;
        }

        /* Fundo do app */
        .main {
            background-color: #f0f2f6;
        }

        /* Cabeçalho principal */
        .main-header {
            text-align: center;
            padding: 1.5rem 0;
            color: #1a2a45;
        }
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
        }
        .main-header p {
            font-size: 1.1rem;
            color: #556270;
        }

        /* Banner de compromisso */
        .commitment-banner {
            background-color: #e6f7ff;
            color: #0050b3;
            border-left: 5px solid #1890ff;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0 2rem 0;
            font-weight: 500;
            text-align: center;
        }

        /* Mensagens do chat */
        .stChatMessage {
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        /* Input de chat fixo no final (se possível em futuras versões do Streamlit) */
        .stChatInput {
            background-color: #FFFFFF;
        }
        
        /* Estilo da barra lateral */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            padding: 1.5rem;
        }
        .sidebar-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1a2a45;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f0f2f6;
        }
        .stats-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 0.75rem;
        }
        .stats-number {
            font-size: 1.75rem;
            font-weight: bold;
            color: #1890ff;
        }
        .stats-label {
            font-size: 0.9rem;
            color: #556270;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0 1rem 0;
            color: #8899a6;
            font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Carregar CSS
load_css()

# Carregar segredos e inicializar cliente Groq
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    MONGO_URI = st.secrets["MONGO_URI"]
    groq_client = Groq(api_key=GROQ_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("🚨 API Key do GROQ ou URI do MongoDB não encontradas! Verifique seus segredos.")
    st.stop()

# Inicializar estado da sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()

# ==============================================================================
# 2. FUNÇÕES DE CACHE (Performance)
# ==============================================================================

@st.cache_resource
def get_sentence_transformer_model():
    """Carrega e armazena em cache o modelo de embedding."""
    return SentenceTransformer('PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir')

@st.cache_resource
def get_mongo_collection():
    """Conecta ao MongoDB e retorna a coleção, armazenando a conexão em cache."""
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["pingpoli"]
    return db["members_informations"]

# ==============================================================================
# 3. FUNÇÕES DE BACKEND (Lógica do Agente)
# ==============================================================================

def search_for_documents(input_text_embedding: np.ndarray, collection, k: int = 5) -> list[str]:
    """Busca os k documentos mais similares no MongoDB."""
    similarities = []
    for doc in collection.find({}, {'_id': 0, 'text': 1, 'embedding': 1}):
        if 'embedding' in doc and doc['embedding'] and 'text' in doc:
            cos_sim = 1 - cosine(input_text_embedding, doc['embedding'])
            similarities.append((cos_sim, doc['text']))
    
    # Ordena pela similaridade (decrescente) e pega os top k
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    return [text for sim, text in similarities[:k]]

def load_system_prompt(user_input: str) -> str:
    """Carrega o prompt do sistema a partir de um arquivo YAML."""
    try:
        with open("prompts.yaml", "r", encoding="utf-8") as file:
            prompts = yaml.safe_load(file)
        return prompts.get("system_user_prompt", "").format(user_input=user_input)
    except Exception:
        # Fallback caso o arquivo de prompts não exista ou tenha erro
        return f"Responda à seguinte pergunta de forma detalhada e útil: {user_input}"

def generate_llm_response(user_input: str, model, collection) -> str:
    """Pipeline completo: embedding, busca RAG e geração de resposta com LLM."""
    
    # 1. Gerar embedding da pergunta do usuário
    input_embedding = model.encode(user_input)
    
    # 2. Buscar documentos relevantes (contexto RAG)
    retrieved_docs = search_for_documents(input_embedding, collection)
    context = "\n\n---\n\n".join(retrieved_docs)
    
    # 3. Carregar e formatar o prompt final
    system_user_prompt = load_system_prompt(user_input)
    final_prompt = f"""
    Você é um assistente de IA especialista. Use o CONTEXTO fornecido abaixo para responder à PERGUNTA do usuário de forma precisa e factual.

    CONTEXTO:
    {context}

    PERGUNTA:
    {system_user_prompt}
    """

    # 4. Chamar a API da Groq
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.7,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Erro ao comunicar com a API da Groq: {e}")
        return "Desculpe, não consegui processar sua solicitação no momento."

# ==============================================================================
# 4. COMPONENTES DE UI (Layout do Aplicativo)
# ==============================================================================

def setup_sidebar():
    """Configura o conteúdo da barra lateral."""
    with st.sidebar:
        st.markdown("<h2 class='sidebar-header'>📊 Estatísticas da Sessão</h2>", unsafe_allow_html=True)
        
        # Estatísticas
        st.session_state.total_questions = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
        session_duration = datetime.now() - st.session_state.session_start
        minutes = int(session_duration.total_seconds() // 60)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <p class="stats-number">{st.session_state.total_questions}</p>
                <p class="stats-label">Perguntas</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <p class="stats-number">{minutes}</p>
                <p class="stats-label">Minutos Ativos</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<h2 class='sidebar-header'>🛡️ Nosso Compromisso</h2>", unsafe_allow_html=True)
        st.info("""
        O Agente IA do PingPoli tem um único propósito: fornecer informações 
        precisas e factuais. Ele sempre falará a verdade, com base nos dados 
        disponíveis.
        """, icon="🛡️")

        st.markdown("<h2 class='sidebar-header'>🏓 Sobre o PingPoli</h2>", unsafe_allow_html=True)
        st.success("""
        **PingPoli** é a equipe de tênis de mesa com tradição em excelência esportiva e desenvolvimento de talentos como o Shigueru.
        """, icon="🏓")

        if st.button("🗑️ Limpar Histórico do Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==============================================================================
# 5. LÓGICA PRINCIPAL DO APLICATIVO
# ==============================================================================

# Carregar recursos cacheados
embedding_model = get_sentence_transformer_model()
mongo_collection = get_mongo_collection()

# Configurar a barra lateral
setup_sidebar()

# Cabeçalho Principal
st.markdown("""
<div class="main-header">
    <h1>🏓 PingPoli Agent</h1>
    <p>Seu assistente inteligente especializado em tênis de mesa</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="commitment-banner">
    🛡️ Compromisso com a Verdade: O PingPoli Agent fala apenas a verdade, sempre a verdade. Diga não às mentiras.
</div>
""", unsafe_allow_html=True)


# Exibir histórico do chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar="🙋‍♂️" if message["role"] == "user" else "🏓"):
        st.markdown(message["content"])

# Capturar input do usuário
if prompt := st.chat_input("Faça sua pergunta sobre o PingPoli..."):
    # Adicionar mensagem do usuário ao histórico e à tela
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙋‍♂️"):
        st.markdown(prompt)

    # Mostrar indicador de "digitando" e gerar resposta
    with st.chat_message("assistant", avatar="🏓"):
        with st.spinner("🏓 PingPoli Agent está pensando..."):
            response = generate_llm_response(prompt, embedding_model, mongo_collection)
            
            # Simular digitação para uma melhor UX
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
    # Adicionar resposta do bot ao histórico
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer
st.markdown("""
<div class="footer">
    <p><strong>PingPoli Agent</strong> - Powered by GROQ AI | Desenvolvido com ❤️ e Streamlit</p>
</div>
""", unsafe_allow_html=True)