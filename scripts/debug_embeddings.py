#!/usr/bin/env python3
# debug_embeddings.py

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.filemaker_extractor import FileMakerExtractor
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import requests


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
        # ✅ MÉTHODE 1: Récupérer TOUS les chunks puis filtrer
        url = f"{extractor.server}/fmi/data/v1/databases/{extractor.database}/layouts/Chunks/records"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {extractor.token}'
        }

        # Récupérer les premiers chunks
        params = {
            '_limit': 100  # Limite pour le test
        }

        response = requests.get(url, headers=headers, params=params, verify=False)

        if response.status_code == 200:
            data = response.json()
            all_chunks = data['response']['data']

            print(f"📄 Total chunks récupérés: {len(all_chunks)}")

            # Filtrer ceux qui contiennent "cristal"
            cristal_chunks = []
            for chunk_record in all_chunks:
                chunk_data = chunk_record['fieldData']
                text = chunk_data.get('Text', '')  # ← Attention: "Text" avec majuscule

                if 'cristal' in text.lower():
                    cristal_chunks.append(chunk_data)

            print(f"🎯 {len(cristal_chunks)} chunks avec 'cristal' trouvés\n")

            # Analyser chaque chunk cristal
            for i, chunk_data in enumerate(cristal_chunks[:5]):  # Limite à 5

                print(f"--- CHUNK CRISTAL {i + 1} ---")
                print(f"ID Document: {chunk_data.get('idDocument', 'N/A')}")
                print(f"Texte: {chunk_data.get('Text', '')[:300]}...")

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
                            print("🔥 TRÈS SIMILAIRE À VOTRE QUESTION !")
                        elif similarity > 0.2:
                            print("🟡 Assez similaire")
                        else:
                            print("🔵 Peu similaire")

                    except json.JSONDecodeError as e:
                        print(f"❌ Embedding JSON invalide: {e}")
                    except Exception as e:
                        print(f"❌ Erreur calcul similarité: {e}")
                else:
                    print("❌ Pas d'embedding trouvé pour ce chunk")

                print("-" * 80)

        else:
            print(f"❌ Erreur récupération chunks: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        import traceback
        traceback.print_exc()

    finally:
        extractor.logout()


if __name__ == '__main__':
    test_embeddings()
