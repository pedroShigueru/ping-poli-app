import streamlit as st
import time
import os
from groq import Groq
import yaml
from datetime import datetime
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import heapq
from pymongo import MongoClient

# Configuração da página
st.set_page_config(
    page_title="🏓 PingPoli Agent",
    page_icon="🏓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para UI moderna (removido JavaScript)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #4ecdc4 50%, #45b7d1 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        max-width: 80%;
        margin-left: auto;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(240, 147, 251, 0.3);
        max-width: 80%;
        margin-right: auto;
    }
    
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        border-left: 4px solid #4ecdc4;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: bold;
        color: #4ecdc4;
        margin: 0;
    }
    
    .stats-label {
        color: #666;
        margin: 0.5rem 0 0 0;
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #666;
        font-style: italic;
        margin: 1rem 0;
    }
    
    .typing-dots {
        display: inline-flex;
        gap: 3px;
    }
    
    .typing-dots span {
        height: 8px;
        width: 8px;
        background: #4ecdc4;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    .suggestion-chip {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid #bbdefb;
    }
    
    .suggestion-chip:hover {
        background: #1976d2;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(25, 118, 210, 0.3);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* Animação para entrada de mensagens */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .message-animate {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Estilo personalizado para o input */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #4ecdc4;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #45b7d1;
        box-shadow: 0 0 10px rgba(78, 205, 196, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Inicializar variáveis de sessão
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
MONGO_URI = st.secrets["MONGO_URI"]

# Verificar se a API key existe
if not GROQ_API_KEY:
    st.error("🚨 API Key do GROQ não encontrada! Verifique seu arquivo .env")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

def transform_sentence_to_embedding(input_text: str):
    model = SentenceTransformer('PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir')
    embedding = model.encode(input_text)
    
    return embedding

def connection_mongodb():
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["pingpoli"]
    collection = db["members_informations"]
    
    return collection

def search_for_documents(input_text_embedding: np.array) -> list:
    collection = connection_mongodb()

    rag_text_similarity_list = []
    k = 5

    for document in collection.find({}, {'_id': 0, 'text': 1, 'embedding': 1}):
        embedding = document['embedding']
        text = document['text']

        cos_similarity = 1 - cosine(input_text_embedding, embedding)

        if len(rag_text_similarity_list) < k:
            rag_text_similarity_list.append({'text': text, 'similarity': cos_similarity})
        else:
            min_similarity_item = min(rag_text_similarity_list, key=lambda x: x['similarity'])

            if cos_similarity > min_similarity_item['similarity']:
                index_to_replace = rag_text_similarity_list.index(min_similarity_item)
                rag_text_similarity_list.pop(index_to_replace)
                rag_text_similarity_list.append({'text': text, 'similarity': cos_similarity})

    rag_text_similarity_list.sort(key=lambda x: x['similarity'], reverse=True)

    return rag_text_similarity_list

def load_prompt_from_yaml(prompt_name: str, user_input: str) -> str:
    """Carrega prompt do arquivo YAML"""
    try:
        with open("prompts.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config[prompt_name].format(user_input=user_input)
    except FileNotFoundError:
        return f"Você é um especialista em tênis de mesa e conhece tudo sobre a equipe PingPoli. Responda a seguinte pergunta de forma detalhada e útil: {user_input}"
    except Exception as e:
        return f"Erro ao carregar prompt: {str(e)}"

def animate_text_response(text: str, placeholder):
    """Anima a resposta texto por texto"""
    response = ""
    for char in text:
        response += char
        placeholder.markdown(f'<div class="bot-message message-animate">{response}▋</div>', unsafe_allow_html=True)
        time.sleep(0.01)
    placeholder.markdown(f'<div class="bot-message message-animate">{response}</div>', unsafe_allow_html=True)

def generate_answer(user_input: str, system_user_prompt: str, model: str = "llama-3.3-70b-versatile", stream: bool = False, temperature: float = 0.8) -> str:
    """Gera resposta usando a API do GROQ"""

    input_embedding = transform_sentence_to_embedding(user_input)
    retrieved_documents = search_for_documents(input_embedding)

    context = "\n\n---\n\n".join([doc['text'] for doc in retrieved_documents])

    final_prompt = f"""
    Você é um assistente de IA especialista. Use o CONTEXTO fornecido abaixo para responder à PERGUNTA do usuário."

    CONTEXTO:
    {context}

    PERGUNTA:
    {system_user_prompt}
    """

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user", 
                "content": final_prompt
            }],
            stream=stream,
            temperature=temperature
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Erro ao gerar resposta: {str(e)}"


# Header principal
st.markdown("""
<div class="main-header">
    <h1>🏓 PingPoli Agent</h1>
    <p>Seu assistente inteligente especializado em tênis de mesa</p>
</div>
""", unsafe_allow_html=True)

# Layout em colunas
col1, col2 = st.columns([3, 1])

with col1:
    # Container do chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Exibir histórico do chat
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f'<div class="user-message">🙋‍♂️ {question}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-message">🏓 {answer}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Olá! Como posso ajudar você hoje?</h3>
                <p>Faça qualquer pergunta sobre a equipe PingPoli, seus membros, campeonatos ou treinos!</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sugestões de perguntas
    if not st.session_state.chat_history:
        st.markdown("**💡 Sugestões de perguntas:**")
        suggestions_col1, suggestions_col2, suggestions_col3 = st.columns(3)
        
        with suggestions_col1:
            if st.button("Horários de treino", key="sug1"):
                st.session_state.suggested_question = "Qual horário de treino do PingPoli?"
        
        with suggestions_col2:
            if st.button("Membros da equipe", key="sug2"):
                st.session_state.suggested_question = "Quem são os membros da equipe?"
        
        with suggestions_col3:
            if st.button("Quer saber sobre o Grandioso Shigueru?", key="sug3"):
                st.session_state.suggested_question = "Quero saber sobre o Grandioso Shigueru."
    
    # Usando formulário para capturar Enter
    with st.form(key=f"chat_form_{st.session_state.input_key}", clear_on_submit=True):
        user_input = st.text_input(
            "💬 Faça sua pergunta:",
            placeholder="Digite sua pergunta sobre o PingPoli e pressione Enter ou clique em Enviar...",
            key=f"user_input_{st.session_state.input_key}",
            value=st.session_state.get('suggested_question', ''),
        )
        
        # Limpar sugestão após usar

        
        # Botões de ação dentro do form
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            send_button = st.form_submit_button("🚀 Enviar", type="primary", use_container_width=True)
        
        with col_clear:
            clear_button = st.form_submit_button("🗑️ Limpar Chat", use_container_width=True)
    
    # Processar limpeza do chat fora do form
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.total_questions = 0
        st.session_state.input_key += 1
        st.rerun()

with col2:
    # Estatísticas da sessão
    st.markdown("### 📊 Estatísticas da Sessão")
    
    st.markdown(f"""
    <div class="stats-card">
        <p class="stats-number">{st.session_state.total_questions}</p>
        <p class="stats-label">Perguntas Feitas</p>
    </div>
    """, unsafe_allow_html=True)
    
    session_duration = datetime.now() - st.session_state.session_start
    minutes = int(session_duration.total_seconds() // 60)
    
    st.markdown(f"""
    <div class="stats-card">
        <p class="stats-number">{minutes}</p>
        <p class="stats-label">Minutos Ativos</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Configurações (removidas - usando apenas llama padrão)
    # st.markdown("### ⚙️ Configurações")
    
    # Informações da equipe
    st.markdown("### 🏓 Sobre o PingPoli")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; font-size: 0.9rem;">
        <strong>PingPoli</strong> é a equipe de tênis de mesa com tradição em excelência esportiva e desenvolvimento de talentos.
    </div>
    """, unsafe_allow_html=True)

# Processamento da pergunta (apenas quando botão for clicado)
if send_button and user_input.strip():
    if 'suggested_question' in st.session_state:
        del st.session_state.suggested_question
    # Adicionar pergunta ao histórico
    st.session_state.total_questions += 1
    
    # Mostrar indicador de digitação
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="typing-indicator">
        🏓 PingPoli Agent está digitando
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Gerar resposta (usando modelo fixo)
    system_user_prompt = load_prompt_from_yaml("system_user_prompt", user_input)
    answer = generate_answer(user_input, system_user_prompt, model="llama-3.3-70b-versatile", temperature=0.8)
    
    # Remover indicador de digitação
    typing_placeholder.empty()
    
    # Adicionar ao histórico
    st.session_state.chat_history.append((user_input, answer))
    
    # Mostrar nova resposta com animação
    response_placeholder = st.empty()
    animate_text_response(answer, response_placeholder)
    
    # Resetar o input incrementando a key
    st.session_state.input_key += 1
    
    # Recarregar para mostrar histórico atualizado
    time.sleep(1)
    st.rerun()

# Footer
st.markdown("""
<div class="footer">
    <p>🏓 <strong>PingPoli Agent</strong> - Powered by GROQ AI | Desenvolvido com ❤️ e Streamlit</p>
    <p style="font-size: 0.8rem; opacity: 0.7;">
        Versão 2.0 | Última atualização: Hoje
    </p>
</div>
""", unsafe_allow_html=True)