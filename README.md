# Profi ChatBot

# Teachy: Chatbot Éducatif pour Écoles Privées

## Introduction
**Teachy** est un chatbot éducatif innovant conçu pour assister les élèves dans leurs révisions et leur apprentissage de manière interactive et personnalisée. Avec les cours particuliers désormais limités en dehors des établissements scolaires par le ministère de l'Éducation tunisien, Teachy répond au besoin pressant d’un outil accessible et en phase avec le programme scolaire. Contrairement à des outils génériques comme Google ou ChatGPT, Teachy propose des réponses claires et adaptées au niveau des élèves, en s’appuyant sur les manuels, exercices, et ressources des enseignants.

## ✅ Fonctionnalités
- **Alignement sur le programme scolaire** : Réponses adaptées au programme et au niveau de l'élève.
- **Support en continu** : Questions posées à tout moment et suivies d’explications détaillées.
- **Feedback aux enseignants** : Identification des chapitres problématiques grâce aux questions fréquentes des élèves.
- **Approche proactive** : Suivi de la progression des élèves pour un apprentissage autonome.

## Architecture du Projet
### 1. Extraction de Texte
- **Problème** : Le fichier PDF initial était sous format image.
- **Solution** :
  - Conversion du PDF en images.
  - Extraction du texte à l'aide d'un script Python (voir fichier `Extraction du texte d'un PDF.ipynb`).
  - Transformation du texte extrait en un corpus structuré.

### 2. Génération d’Embeddings
- **Objectif** : Créer une base de connaissances structurée par chapitre.
- **Approche** :
  - Utilisation de LangChain pour générer des embeddings basés sur OpenAI.
  - Indexation des données avec FAISS pour des recherches rapides et pertinentes (voir fichier `Generation_d'un_chatbot_sur_colab.ipynb`).

### 3. Application Frontend
- Développement d'une interface utilisateur avec Streamlit (voir fichier `app.py`).
- **Fonctionnalités incluses** :
  - Chargement sécurisé des index FAISS et métadonnées.
  - Génération de réponses via GPT-4 en utilisant des contextes spécifiques au programme scolaire.
  - Explications détaillées avec exemples pour une meilleure compréhension.

## Installation
### Prérequis
- Python 3.8+
- Bibliothèques nécessaires (voir `requirements.txt`).

### Étapes d'installation
1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-repo/teachy.git
   cd teachy
## Installation

### Étapes d'installation
1. **Installez les dépendances** :
   ```bash
   pip install -r requirements.txt
### Utilisation
Lancez l’application et posez une question dans l’interface utilisateur.
Le chatbot génère une réponse en s’appuyant sur les données indexées et le programme scolaire.
Les enseignants peuvent suivre les points de blocage identifiés via les questions fréquentes.
### Structure du Dépôt
**Extraction du texte d'un PDF.ipynb** : Extraction et structuration du texte des fichiers PDF.
**Generation_d'un_chatbot_sur_colab.ipynb** : Création des embeddings et indexation.
**app.py** : Frontend de l'application Streamlit.
**metadata.json & embeddings_index.faiss** : Base de connaissances indexée.

