import os
import json
from langchain.embeddings.openai import OpenAIEmbeddings

# Définir la clé API OpenAI
os.environ["OPENAI_API_KEY"] = ""

# Initialiser OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Fonction pour fractionner un texte
def split_text(text, max_tokens=8000):
    words = text.split()
    segments = []
    current_segment = []

    for word in words:
        current_segment.append(word)
        if len(" ".join(current_segment)) > max_tokens:
            segments.append(" ".join(current_segment))
            current_segment = []
    
    if current_segment:
        segments.append(" ".join(current_segment))
    
    return segments

# Charger le fichier JSON
file_path = "last_optimize.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Traiter les données pour structurer les chapitres et sous-sections
def process_content(data):
    processed_data = []

    for i, chapter in enumerate(data["data"]):
        try:
            print(f"Traitement du chapitre {i + 1}: {chapter['chapter']}")
            # Fractionner et générer des embeddings pour le contenu principal
            chapter_segments = split_text(chapter["content"])
            chapter_embeddings = [
                embeddings.embed_query(segment) for segment in chapter_segments if segment.strip()
            ]
            
            chapter_entry = {
                "id": f"{chapter['chapter'].replace(' ', '_').lower()}",
                "type": "chapter",
                "title": chapter["chapter"],
                "start_page": chapter.get("start_page", None),
                "end_page": chapter.get("end_page", None),
                "content": chapter["content"],
                "embeddings": chapter_embeddings,
                "subsections": []
            }

            for subsection in chapter.get("subsections", []):
                try:
                    print(f"  - Sous-section: {subsection['title']}")
                    subsection_segments = split_text(subsection["content"])
                    subsection_embeddings = [
                        embeddings.embed_query(segment) for segment in subsection_segments if segment.strip()
                    ]

                    subsection_entry = {
                        "id": f"{chapter_entry['id']}_{subsection['type'].lower()}_{subsection['title'].replace(' ', '_').lower()}",
                        "type": subsection["type"],
                        "title": subsection["title"],
                        "start": subsection.get("start", None),
                        "content": subsection["content"],
                        "embeddings": subsection_embeddings
                    }
                    chapter_entry["subsections"].append(subsection_entry)
                except Exception as e:
                    print(f"Erreur lors du traitement de la sous-section {subsection['title']}: {e}")

            processed_data.append(chapter_entry)
        except Exception as e:
            print(f"Erreur lors du traitement du chapitre {chapter['chapter']}: {e}")

    return processed_data

# Traiter les données
processed_data = process_content(data)

# Sauvegarder les données traitées avec embeddings
output_path = "embeddings.json"
with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(processed_data, out_file, indent=4)

print(f"Les données traitées ont été sauvegardées dans {output_path}.")


#Faiss
import faiss
import numpy as np
import json

# Charger les données générées avec embeddings
with open("embeddings.json", "r", encoding="utf-8") as file:
    processed_data = json.load(file)

# Préparer les vecteurs et les métadonnées
embeddings = []
metadata = []

for chapter in processed_data:
    for i, embedding in enumerate(chapter["embeddings"]):
        embeddings.append(np.array(embedding, dtype=np.float32))
        metadata.append({
            "id": chapter["id"],
            "type": "chapter",
            "title": chapter["title"],
            "content": chapter["content"],
            "index": i,
            "start_page": chapter.get("start_page"),
            "end_page": chapter.get("end_page"),
        })

    for subsection in chapter["subsections"]:
        for i, embedding in enumerate(subsection["embeddings"]):
            embeddings.append(np.array(embedding, dtype=np.float32))
            metadata.append({
                "id": subsection["id"],
                "type": subsection["type"],
                "title": subsection["title"],
                "content": subsection["content"],
                "index": i,
                "start": subsection.get("start"),
            })

# Convertir les embeddings en un tableau NumPy
embeddings = np.array(embeddings)

# Créer l'index FAISS
dimension = embeddings.shape[1]  # Dimension des embeddings
index = faiss.IndexFlatL2(dimension)  # Index basé sur la distance L2 (euclidienne)
index.add(embeddings)

# Sauvegarder l'index et les métadonnées
faiss.write_index(index, "embeddings_index.faiss")
with open("metadata.json", "w", encoding="utf-8") as meta_file:
    json.dump(metadata, meta_file, indent=4)

print("Les embeddings ont été stockés dans FAISS.")