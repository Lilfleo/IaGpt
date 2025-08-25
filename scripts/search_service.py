#!/usr/bin/env python3
from flask import Flask, request, jsonify
import sys
import os
import locale

locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.filemaker_extractor import FileMakerExtractor
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import requests

app = Flask(__name__)


class RAGSearcher:
    def __init__(self):
        self.extractor = FileMakerExtractor()
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def search(self, question, top_k=5):
        """Recherche sémantique optimisée avec pagination"""
        print(f"🔍 Recherche: {question}")

        if not self.extractor.login():
            return {"error": "Connexion impossible"}

        try:
            # Recherche avec pagination
            top_chunks = self.search_with_pagination(question, top_k)

            if not top_chunks:
                return {"error": "Aucun résultat trouvé"}

            # Générer réponse
            context = "\n".join([chunk['text'] for chunk in top_chunks])
            response = self.generate_answer(question, context)

            return {
                "question": question,
                "response": response,
                "sources": [chunk['document_name'] for chunk in top_chunks],
                "chunks_analyzed": len(top_chunks)
            }

        finally:
            self.extractor.logout()

    def search_with_pagination(self, question, top_k=5):
        """Recherche avec pagination pour éviter la surcharge mémoire"""
        print(f"🧠 Encoding de la question...")

        # 1. Encoder la question
        question_embedding = self.model.encode([question])
        question_vec = question_embedding[0]

        # 2. Recherche par batch
        all_similarities = []
        batch_size = 100
        offset = 1
        total_processed = 0

        print(f"📊 Début de l'analyse par batch...")

        while True:
            print(f"📄 Traitement batch offset {offset}...")

            # Récupérer un batch
            url = f"{self.extractor.server}/fmi/data/v1/databases/{self.extractor.database}/layouts/Chunks/records"
            headers = {'Authorization': f'Bearer {self.extractor.token}'}
            params = {'_offset': offset, '_limit': batch_size}

            response = requests.get(url, headers=headers, params=params, verify=False)

            if response.status_code != 200:
                print(f"❌ Erreur HTTP: {response.status_code}")
                break

            data = response.json()
            records = data.get('response', {}).get('data', [])

            if not records:
                print("📭 Plus de records")
                break

            print(f"📋 {len(records)} records dans ce batch")

            # Calculer similarités pour ce batch
            batch_similarities = []
            for rec in records:
                text = rec['fieldData'].get('Text', '')
                embedding_json = rec['fieldData'].get('EmbeddingJson', '')

                if text and embedding_json:
                    try:
                        chunk_emb = np.array(json.loads(embedding_json))
                        similarity = np.dot(question_vec, chunk_emb)

                        batch_similarities.append({
                            'similarity': float(similarity),
                            'text': text,
                            'document_id': rec['fieldData'].get('idDocument', ''),
                            'document_name': rec['fieldData'].get('DocumentName',
                                                                  f"Doc_{rec['fieldData'].get('idDocument', 'X')}")
                        })
                    except Exception as e:
                        print(f"⚠️ Erreur embedding: {e}")
                        continue

            all_similarities.extend(batch_similarities)
            total_processed += len(batch_similarities)

            print(f"✅ {len(batch_similarities)} chunks valides dans ce batch")

            # Arrêter si batch incomplet
            if len(records) < batch_size:
                print("📄 Dernier batch atteint")
                break

            offset += batch_size

            # Limitation pour éviter timeout (optionnel)
            if total_processed >= 5000:
                print("⚠️ Limite de 5000 chunks atteinte pour éviter timeout")
                break

        print(f"📊 TOTAL: {total_processed} chunks analysés")

        if not all_similarities:
            return []

        # 3. Trier et prendre le top
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = all_similarities[:top_k]

        print(f"🎯 Top {len(top_chunks)} chunks sélectionnés")
        for i, chunk in enumerate(top_chunks[:3]):  # Afficher les 3 meilleurs
            print(f"   {i + 1}. Similarité: {chunk['similarity']:.4f} - {chunk['document_name']}")

        return top_chunks

    def generate_answer(self, question, context):
        """Génère la réponse avec Ollama"""
        print("🤖 Génération de la réponse avec Ollama...")

        prompt = f"""Contexte: {context}

Question: {question}

Réponds en français en te basant uniquement sur le contexte fourni. Si le contexte ne permet pas de répondre, dis-le clairement."""

        try:
            response = requests.post('http://localhost:11434/api/generate',
                                     json={
                                         'model': 'llama3.2:1b',
                                         'prompt': prompt,
                                         'stream': False
                                     },
                                     timeout=30)

            if response.status_code == 200:
                result = response.json().get('response', 'Erreur génération')
                print("✅ Réponse générée avec succès")
                return result
            else:
                print(f"❌ Erreur Ollama: {response.status_code}")
                return "Erreur: Service IA indisponible"

        except requests.exceptions.Timeout:
            return "Erreur: Timeout du service IA"
        except Exception as e:
            print(f"❌ Exception Ollama: {e}")
            return f"Erreur: {str(e)}"


# Instance globale
searcher = RAGSearcher()


@app.route('/search', methods=['POST'])
def search():
    try:
        print("\n" + "=" * 50)
        print("🔍 NOUVELLE REQUÊTE REÇUE")
        print("=" * 50)

        data = request.get_json()
        question = data.get('question', '').strip()

        print(f"📝 Question: '{question}'")

        if not question:
            return jsonify({"error": "Question manquante"}), 400

        print("🚀 Début de la recherche...")

        # Appel de la recherche
        result = searcher.search(question)

        print("✅ Recherche terminée avec succès")
        print("=" * 50)

        return jsonify(result)

    except Exception as e:
        print(f"❌ ERREUR GLOBALE: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Erreur serveur: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    return jsonify({"status": "OK", "service": "RAG API"})


if __name__ == '__main__':
    print("🚀 Démarrage du serveur RAG...")
    print("📡 URL: http://localhost:9000")
    print("🔍 Endpoint: POST /search")
    print("💚 Health: GET /health")
    app.run(host='0.0.0.0', port=9000, debug=True)
