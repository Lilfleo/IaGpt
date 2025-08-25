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
        """Recherche sémantique"""
        # Connexion FileMaker
        if not self.extractor.login():
            return {"error": "Connexion impossible"}

        try:
            # 1. Encoder la question
            question_embedding = self.model.encode([question])

            # 2. Récupérer les chunks avec embeddings
            chunks = self.get_chunks_with_embeddings()

            # 3. Calculer similarités
            similarities = []
            for chunk in chunks:
                if chunk.get('embedding'):
                    chunk_emb = np.array(json.loads(chunk['embedding']))
                    similarity = np.dot(question_embedding[0], chunk_emb)
                    similarities.append({
                        'chunk': chunk,
                        'similarity': similarity
                    })

            # 4. Trier par pertinence
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_chunks = similarities[:top_k]

            # 5. Générer réponse avec Ollama
            context = "\n".join([s['chunk']['text'] for s in top_chunks])
            response = self.generate_answer(question, context)

            return {
                "question": question,
                "response": response,
                "sources": [s['chunk'].get('document_name', 'Document') for s in top_chunks]
            }

        finally:
            self.extractor.logout()

    def get_chunks_with_embeddings(self):
        """Récupère chunks depuis FileMaker"""
        url = f"{self.extractor.server}/fmi/data/v1/databases/{self.extractor.database}/layouts/Chunks/records"
        headers = {'Authorization': f'Bearer {self.extractor.token}'}

        response = requests.get(url, headers=headers, verify=False)
        if response.status_code == 200:
            records = response.json()['response']['data']
            return [{
                'text': rec['fieldData'].get('Text', ''),
                'embedding': rec['fieldData'].get('EmbeddingJson', ''),
                'document_id': rec['fieldData'].get('idDocument', ''),
                'document_name': rec['fieldData'].get('DocumentName', 'Document')  # ← CORRIGÉ
            } for rec in records]
        return []

    def generate_answer(self, question, context):
        """Génère la réponse avec Ollama"""
        prompt = f"""Contexte: {context}

Question: {question}

Réponds en français en te basant uniquement sur le contexte fourni."""

        try:
            response = requests.post('http://localhost:11434/api/generate',
                                     json={
                                         'model': 'llama3.2:1b',
                                         'prompt': prompt,
                                         'stream': False
                                     })

            if response.status_code == 200:
                return response.json().get('response', 'Erreur génération')
            else:
                return "Erreur: Service IA indisponible"

        except Exception as e:
            return f"Erreur: {str(e)}"


searcher = RAGSearcher()


@app.route('/search', methods=['POST'])
def search():
    try:
        print("🔍 Requête reçue")



        data = request.get_json()
        print(f"📝 Question: {data.get('question')}")

        question = data.get('question')
        if not question:
            return jsonify({"error": "Question manquante"}), 400

        print("🚀 Début recherche...")

        # 🎯 LIGNE MANQUANTE - Appel de la recherche !
        result = searcher.search(question)

        print("✅ Recherche terminée")
        return jsonify(result)

    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Connexion impossible"}), 500


# 🎯 LIGNE MANQUANTE - Démarrage du serveur !
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)
