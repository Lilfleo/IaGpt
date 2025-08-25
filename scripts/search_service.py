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

    def search(self, question):
        """Recherche principale avec logging détaillé"""
        try:
            print(f"🔍 Phase 1: Recherche textuelle dans FileMaker...")
            top_chunks = self.search_with_pagination(question, top_k=20)

            # 🔥 DEBUG : MONTRER LES CHUNKS RÉCUPÉRÉS
            print(f"\n🔍 CHUNKS RÉCUPÉRÉS (TOP 5):")
            for i, chunk in enumerate(top_chunks[:5]):
                chunk_data = chunk['fieldData'] if 'fieldData' in chunk else chunk
                text = chunk_data.get('Text', '')[:200]
                doc_id = chunk_data.get('idDocument', 'N/A')
                print(f"--- CHUNK {i + 1} ---")
                print(f"Doc: {doc_id}")
                print(f"Texte: {text}...")

                # Vérifier la présence des mots-clés
                text_lower = text.lower()
                keywords_found = []
                if 'cristal' in text_lower: keywords_found.append('cristal')
                if 'prix' in text_lower: keywords_found.append('prix')
                if 'souscription' in text_lower: keywords_found.append('souscription')
                if 'rente' in text_lower: keywords_found.append('rente')
                if '2022' in text_lower: keywords_found.append('2022')

                print(f"Mots-clés présents: {keywords_found}")

            # Continue avec la génération...
            context = self.prepare_context(top_chunks[:5])
            # ... reste identique

            response = self.generate_answer(question, context)

            return {
                "question": question,
                "response": response,
                "sources": [chunk['document_name'] for chunk in top_chunks],
                "chunks_analyzed": len(top_chunks)
            }

        finally:
            self.extractor.logout()

    def search_with_pagination(self, question, top_k=20):
        """Recherche hybride: FileMaker + Embedding"""
        print(f"🧠 Recherche hybride pour: '{question}'")

        # 1️⃣ RECHERCHE TEXTUELLE PRÉALABLE
        print(f"🔍 Phase 1: Recherche textuelle dans FileMaker...")
        filtered_chunks = self.extractor.search_chunks_smart(question, limit=2000)

        if not filtered_chunks:
            print("❌ Aucun chunk trouvé avec la recherche textuelle")
            return []

        print(f"📊 {len(filtered_chunks)} chunks pré-filtrés par recherche textuelle")

        # 2️⃣ CALCUL D'EMBEDDING SUR LES RÉSULTATS FILTRÉS
        print(f"🧮 Phase 2: Calcul des similarités sémantiques...")
        question_embedding = self.model.encode([question])
        question_vec = question_embedding[0]

        similarities = []
        processed = 0

        for chunk_record in filtered_chunks:
            chunk_data = chunk_record['fieldData']
            text = chunk_data.get('Text', '')
            embedding_json = chunk_data.get('EmbeddingJson', '')

            if text and embedding_json and embedding_json.strip():
                try:
                    chunk_embedding = np.array(json.loads(embedding_json))
                    similarity = np.dot(question_vec, chunk_embedding)

                    similarities.append({
                        'similarity': float(similarity),
                        'text': text,
                        'document_id': chunk_data.get('idDocument', ''),
                        'document_name': f"Doc_{chunk_data.get('idDocument', 'N/A')}"
                    })
                    processed += 1

                except (json.JSONDecodeError, ValueError) as e:
                    continue

        print(f"✅ {processed} embeddings traités avec succès")

        if not similarities:
            print("❌ Aucun embedding valide trouvé")
            return []

        # 3️⃣ TRIER PAR SIMILARITÉ
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = similarities[:top_k]

        print(f"🎯 Top {len(top_chunks)} chunks sélectionés:")
        for i, chunk in enumerate(top_chunks[:5]):
            print(f"   {i + 1}. Similarité: {chunk['similarity']:.4f} | {chunk['document_name']}")

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
