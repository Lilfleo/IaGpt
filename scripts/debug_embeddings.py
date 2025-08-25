#!/usr/bin/env python3
# debug_embeddings.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.filemaker_extractor import FileMakerExtractor
from sentence_transformers import SentenceTransformer
import numpy as np
import json


def test_embeddings():
    # Connexion
    extractor = FileMakerExtractor()
    if not extractor.login():
        print("❌ Impossible de se connecter à FileMaker")
        return

    # Modèle
    print("🤖 Chargement du modèle...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Question
    question = "prix d une part pour cristal life en 2021"
    question_embedding = model.encode([question])

    print(f"🔍 Recherche: {question}\n")

    try:
        # Récupération chunks avec recherche FileMaker
        url = f"{extractor.server}/fmi/data/v1/databases/{extractor.database}/layouts/Chunks/records/_find"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {extractor.token}'
        }

        # Recherche chunks contenant "cristal"
        payload = {
            "query": [{"Text": "*cristal*"}]  # Recherche wildcard
        }

        response = requests.post(url, json=payload, headers=headers, verify=False)

        if response.status_code == 200:
            data = response.json()
            chunks = data['response']['data']

            print(f"📊 {len(chunks)} chunks trouvés avec 'cristal'\n")

            # Analyser chaque chunk
            for i, chunk_record in enumerate(chunks[:10]):  # Limite à 10
                chunk_data = chunk_record['fieldData']

                print(f"--- CHUNK {i + 1} ---")
                print(f"ID: {chunk_data.get('idDocument', 'N/A')}")
                print(f"Texte: {chunk_data.get('Text', '')[:200]}...")

                # Récupérer l'embedding depuis EmbeddingJson
                embedding_str = chunk_data.get('EmbeddingJson', '')

                if embedding_str and embedding_str.strip():
                    try:
                        # Convertir l'embedding JSON en array
                        embedding = json.loads(embedding_str)

                        # Calculer la similarité
                        similarity = np.dot(question_embedding[0], np.array(embedding))

                        print(f"✅ Similarité: {similarity:.4f}")

                        # Mettre en évidence si très similaire
                        if similarity > 0.3:
                            print("🔥 TRÈS SIMILAIRE !")

                    except json.JSONDecodeError:
                        print("❌ Embedding JSON invalide")
                    except Exception as e:
                        print(f"❌ Erreur calcul similarité: {e}")
                else:
                    print("❌ Pas d'embedding trouvé")

                print("-" * 80)

        elif response.status_code == 401:
            print("❌ Pas de résultats trouvés (401 - normal si aucun match)")
        else:
            print(f"❌ Erreur API: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Erreur générale: {e}")

    finally:
        extractor.logout()


if __name__ == '__main__':
    # Import requests ici car il manquait
    import requests

    test_embeddings()
