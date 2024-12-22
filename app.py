import streamlit as st
import faiss
import numpy as np
import json
import openai
from langchain.embeddings.openai import OpenAIEmbeddings

# Configuration de l'API OpenAI directement dans le code (Non recommandé mais selon votre demande)
openai.api_key = "saisir votre clé"

if not openai.api_key:
    st.error("Clé API OpenAI non définie.")
    st.stop()

# Chargement sécurisé des fichiers FAISS et métadonnées
def load_files(index_path, metadata_path):
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        return index, metadata
    except FileNotFoundError as e:
        st.error(f"❌ Fichier non trouvé : {e}")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des fichiers : {e}")
    st.stop()

index, metadata = load_files("embeddings_index.faiss", "metadata.json")

# Génération des embeddings via LangChain avec OpenAI
def get_embedding(query):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai.api_key)
        embedding = embeddings.embed_query(query)
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des embeddings : {e}")
        return None

# Recherche FAISS optimisée
def search_faiss(query, k=3):
    embedding = get_embedding(query)
    if embedding is None:
        return []

    distances, indices = index.search(embedding.reshape(1, -1), k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            result = metadata[idx]
            similarity = 1 / (1 + distances[0][i])
            result["relevance"] = round(similarity * 100, 2)
            results.append(result)
    return results

# Modèle d'explications
def explanation_model(query, context):
    try:
        prompt = f"""
        Vous êtes un assistant pédagogique spécialisé en mathématiques et sciences.
        Voici une question à expliquer en détail :

        ### Question :
        {query}

        ### Contexte :
        {context}

        Fournissez une explication claire et détaillée en utilisant des termes simples et des notations mathématiques en LaTeX si nécessaire.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"❌ Erreur lors de l'interrogation du modèle d'explication : {e}")
        return "Erreur lors de l'interrogation du modèle d'explication."

# Modèle pour générer des exemples
def example_model(explanation):
    try:
        prompt = f"""
        Vous êtes un assistant pédagogique qui génère des exemples pour clarifier des concepts.
        Voici une explication d'un concept mathématique ou scientifique :

        ### Explication :
        {explanation}

        Fournissez un exemple concret et clair pour illustrer cette explication.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"❌ Erreur lors de l'interrogation du modèle d'exemples : {e}")
        return "Erreur lors de l'interrogation du modèle d'exemples."

# Initialisation de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_questions" not in st.session_state:
    st.session_state.user_questions = []

st.title("💻 Profi Chatbot")
st.markdown("Posez une question et recevez des réponses contextuelles basées sur les données indexées.")

# Barre latérale pour les questions posées par l'utilisateur
with st.sidebar:
    st.header("📋 Questions posées")
    if st.session_state.user_questions:
        for i, question in enumerate(st.session_state.user_questions, 1):
            st.markdown(f"**{i}.** {question}")
    else:
        st.markdown("Aucune question posée pour le moment.")

for msg in st.session_state.messages:
    role = "👤 Vous" if msg["role"] == "user" else "💬 Profi Chatbot"
    if msg["role"] == "user":
        with st.chat_message(msg["role"]):
            st.write(f"*{role} :* {msg['content']}")
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg['content'], unsafe_allow_html=True)

# Entrée utilisateur
user_input = st.chat_input("Posez votre question ici...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.user_questions.append(user_input)

    with st.spinner("Recherche en cours..."):
        results = search_faiss(user_input, k=3)
        context = "\n\n".join([result["content"] for result in results]) if results else ""

        # Modèle d'explications
        explanation = explanation_model(user_input, context)

        # Modèle d'exemples
        example = example_model(explanation)

        # Combinaison des deux réponses
        response_content = f"### Explication\n{explanation}\n\n### Exemple\n{example}"

    st.session_state.messages.append({"role": "assistant", "content": response_content})

    with st.chat_message("assistant"):
        st.markdown(response_content, unsafe_allow_html=True)
