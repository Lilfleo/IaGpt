#!/usr/bin/env python3
from flask import Flask, request, jsonify
import sys
import os
import locale
import json
import requests
import numpy as np

# Configuration locale
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.filemaker_extractor import FileMakerExtractor
from sentence_transformers import SentenceTransformer

app = Flask(__name__)


class RAGSearcher:
    """Service de recherche RAG avec FileMaker et IA"""

    def __init__(self):
        print("🔧 Initialisation du service RAG...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Modèle d'embedding chargé")

    def connect_filemaker(self):
        """Établit une connexion fraîche à FileMaker"""
        extractor = FileMakerExtractor()
        if extractor.connect():
            print("✅ Connexion FileMaker établie")
            return extractor
        else:
            print("❌ Échec connexion FileMaker")
            return None

    def search(self, question):
        """Recherche principale avec gestion complète"""
        extractor = None

        try:
            print(f"🔍 Début de recherche pour: '{question}'")

            # 1️⃣ CONNEXION FILEMAKER
            extractor = self.connect_filemaker()
            if not extractor:
                return self.error_response(question, "Impossible de se connecter à la base de données")

            # 2️⃣ RECHERCHE TEXTUELLE PRÉALABLE
            print(f"🔍 Phase 1: Recherche textuelle...")
            raw_chunks = extractor.search_chunks_smart(question, limit=1000)

            if not raw_chunks:
                print("❌ Aucun chunk trouvé")
                return self.empty_response(question, "Aucune information trouvée dans la base de données")

            print(f"📊 {len(raw_chunks)} chunks trouvés par recherche textuelle")

            # 3️⃣ CALCUL SIMILARITÉS SÉMANTIQUES
            print(f"🧮 Phase 2: Calcul des similarités...")
            top_chunks = self.calculate_similarities(question, raw_chunks)
            # APRÈS la ligne top_chunks = self.calculate_similarities(...)
            print("🔍 DEBUG - TOP 3 CHUNKS TROUVÉS :")
            for i, chunk in enumerate(top_chunks[:3]):
                chunk_data = chunk['fieldData'] if 'fieldData' in chunk else chunk
                print(f"  Chunk {i + 1}: Doc {chunk_data.get('idDocument', 'N/A')}")
                print(f"    Similarité: {chunk.get('similarity', 'N/A'):.4f}")
                print(f"    Texte (50 chars): '{chunk_data.get('Text', '')[:50]}...'")
                print(f"    A des embeddings: {'OUI' if chunk_data.get('EmbeddingJson') else 'NON'}")
                print()

            if not top_chunks:
                return self.empty_response(question, "Aucun chunk avec embedding valide trouvé")

            # 4️⃣ DEBUG DES CHUNKS SÉLECTIONNÉS
            self.debug_chunks(top_chunks[:5], question)

            # 5️⃣ GÉNÉRATION DE LA RÉPONSE
            context = self.prepare_context(top_chunks[:5])
            response = self.generate_answer(question, context)

            # 6️⃣ RÉSULTAT FINAL
            return {
                "question": question,
                "response": response,
                "sources": [chunk['document_name'] for chunk in top_chunks[:5]],
                "chunks_analyzed": len(top_chunks),
                "status": "success"
            }

        except Exception as e:
            print(f"❌ ERREUR GLOBALE: {e}")
            return self.error_response(question, f"Erreur interne: {str(e)}")

        finally:
            # 7️⃣ NETTOYAGE
            if extractor:
                extractor.logout()
                print("🔌 Connexion FileMaker fermée")

    def calculate_similarities(self, question, raw_chunks, top_k=20):
        """Calcule les similarités sémantiques avec debug détaillé"""
        print(f"\n🔍 DEBUG CALCULATE_SIMILARITIES")
        print(f"📊 Nombre de chunks reçus: {len(raw_chunks)}")

        # Embedding de la question
        question_embedding = self.model.encode([question])
        question_vec = question_embedding[0]
        print(f"🧮 Question embedding shape: {len(question_vec)}")

        similarities = []
        processed = 0
        errors = 0

        for i, chunk_record in enumerate(raw_chunks[:5]):  # Debug sur les 5 premiers
            try:
                print(f"\n--- CHUNK {i + 1} DEBUG ---")

                # Extraction des données du chunk
                chunk_data = chunk_record['fieldData'] if 'fieldData' in chunk_record else chunk_record
                print(f"📄 Record ID: {chunk_record.get('recordId')}")
                print(f"📄 Chunk data keys: {list(chunk_data.keys())}")

                text = chunk_data.get('Text', '').strip()
                embedding_json = chunk_data.get('EmbeddingJson', '')
                doc_id = chunk_data.get('idDocument', 'N/A')

                print(f"📝 Text présent: {'OUI' if text else 'NON'} ({len(text)} chars)")
                print(f"🧮 EmbeddingJson type: {type(embedding_json)}")
                print(f"🧮 EmbeddingJson présent: {'OUI' if embedding_json else 'NON'}")

                if embedding_json:
                    print(f"🧮 EmbeddingJson length: {len(str(embedding_json))} caractères")
                    print(f"🧮 EmbeddingJson preview: {str(embedding_json)[:100]}...")

                    # Test de parsing
                    if isinstance(embedding_json, str):
                        print("🔧 Tentative de parsing JSON string...")
                        chunk_embedding = np.array(json.loads(embedding_json))
                    else:
                        print("🔧 Déjà un objet, conversion directe...")
                        chunk_embedding = np.array(embedding_json)

                    print(f"✅ Embedding parsé OK, shape: {chunk_embedding.shape}")

                    # Validation des dimensions
                    if len(chunk_embedding) != len(question_vec):
                        print(f"❌ ERREUR DIMENSION: chunk={len(chunk_embedding)} vs question={len(question_vec)}")
                        continue

                    # Calcul de similarité cosinus
                    similarity = np.dot(question_vec, chunk_embedding) / (
                            np.linalg.norm(question_vec) * np.linalg.norm(chunk_embedding)
                    )
                    print(f"🎯 Similarité calculée: {similarity:.6f}")

                    similarities.append({
                        'similarity': float(similarity),
                        'text': text,
                        'document_id': doc_id,
                        'document_name': f"Doc_{doc_id}",
                        'raw_data': chunk_data
                    })
                    processed += 1

                else:
                    print(f"❌ Pas d'embedding - SKIPPÉ")

            except json.JSONDecodeError as e:
                print(f"❌ Erreur JSON parsing: {e}")
                errors += 1
            except Exception as e:
                print(f"❌ Erreur générale: {e}")
                errors += 1

        print(f"\n📊 RÉSUMÉ DEBUG:")
        print(f"   ✅ Traités avec succès: {processed}")
        print(f"   ❌ Erreurs: {errors}")

        # Continuez le traitement pour TOUS les chunks (pas juste les 5 premiers)
        print(f"\n🔄 TRAITEMENT COMPLET DE TOUS LES CHUNKS...")

        for chunk_record in raw_chunks:
            try:
                chunk_data = chunk_record['fieldData'] if 'fieldData' in chunk_record else chunk_record
                text = chunk_data.get('Text', '').strip()
                embedding_json = chunk_data.get('EmbeddingJson', '').strip()
                doc_id = chunk_data.get('idDocument', 'N/A')

                if not text or not embedding_json:
                    continue

                # Parsing de l'embedding
                chunk_embedding = np.array(json.loads(embedding_json)) if isinstance(embedding_json, str) else np.array(
                    embedding_json)

                # Calcul de similarité
                similarity = np.dot(question_vec, chunk_embedding) / (
                        np.linalg.norm(question_vec) * np.linalg.norm(chunk_embedding)
                )

                similarities.append({
                    'similarity': float(similarity),
                    'text': text,
                    'document_id': doc_id,
                    'document_name': f"Doc_{doc_id}",
                    'raw_data': chunk_data
                })

            except (json.JSONDecodeError, ValueError, KeyError):
                continue

        if not similarities:
            print("❌ AUCUNE SIMILARITÉ CALCULÉE")
            return []

        # Tri par similarité décroissante
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = similarities[:top_k]

        print(f"🎯 Top {len(top_chunks)} chunks sélectionnés sur {len(similarities)} total")
        return top_chunks

    def debug_chunks(self, chunks, question):
        """Affiche le debug des chunks sélectionnés"""
        print(f"\n🔍 DEBUG - TOP {len(chunks)} CHUNKS:")

        # Mots-clés de recherche pour le debug
        question_words = question.lower().split()

        for i, chunk in enumerate(chunks, 1):
            text_preview = chunk['text'][:200]
            similarity = chunk['similarity']
            doc_name = chunk['document_name']

            print(f"--- CHUNK {i} ---")
            print(f"📄 Document: {doc_name}")
            print(f"🎯 Similarité: {similarity:.4f}")
            print(f"📝 Extrait: {text_preview}...")

            # Recherche de mots-clés dans le texte
            text_lower = chunk['text'].lower()
            keywords_found = [word for word in question_words if word in text_lower and len(word) > 2]

            if keywords_found:
                print(f"🔍 Mots-clés trouvés: {keywords_found}")
            else:
                print(f"⚠️  Aucun mot-clé direct trouvé")

            print()

    def prepare_context(self, chunks):
        """Prépare le contexte pour l'IA à partir des chunks"""
        if not chunks:
            return "Aucun contexte disponible."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk['text'][:600]  # Limite à 600 chars par chunk
            doc_name = chunk['document_name']
            similarity = chunk['similarity']

            context_parts.append(
                f"[Source {i} - {doc_name} (pertinence: {similarity:.3f})]\n{text}"
            )

        return "\n\n" + "=" * 50 + "\n\n".join(context_parts)

    def generate_answer(self, question, context):
        """Génère la réponse avec Ollama"""
        print("🤖 Génération de la réponse avec Ollama...")

        prompt = f"""Contexte: {context}

        Question: {question}

        Réponds en français en te basant uniquement sur les informations du contexte. Si tu dois faire un calcul, montre-le simplement.

        Réponse:"""

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'mistral:7b-instruct',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,  # Plus déterministe
                        'num_ctx': 4096  # Plus de contexte
                    }
                },
                timeout=180
            )

            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                print("✅ Réponse générée avec succès")
                return result if result else "Erreur: Réponse vide générée"
            else:
                print(f"❌ Erreur Ollama: {response.status_code}")
                return "Erreur: Service IA indisponible"

        except requests.exceptions.Timeout:
            print("❌ Timeout Ollama")
            return "Erreur: Le service IA a pris trop de temps à répondre"
        except Exception as e:
            print(f"❌ Exception Ollama: {e}")
            return f"Erreur service IA: {str(e)}"

    def error_response(self, question, message):
        """Génère une réponse d'erreur standardisée"""
        return {
            "question": question,
            "response": message,
            "sources": [],
            "chunks_analyzed": 0,
            "status": "error"
        }

    def empty_response(self, question, message):
        """Génère une réponse vide standardisée"""
        return {
            "question": question,
            "response": message,
            "sources": [],
            "chunks_analyzed": 0,
            "status": "no_results"
        }


# Instance globale du service
print("🚀 Création de l'instance RAG...")
searcher = RAGSearcher()
print("✅ Service RAG initialisé")


@app.route('/search', methods=['POST'])
def search_endpoint():
    """Endpoint principal de recherche"""
    try:
        print("\n" + "=" * 60)
        print("🔍 NOUVELLE REQUÊTE REÇUE")
        print("=" * 60)

        # Récupération de la question
        data = request.get_json()
        if not data:
            return jsonify({"error": "Pas de données JSON reçues"}), 400

        question = data.get('question', '').strip()

        if not question:
            return jsonify({"error": "Question manquante ou vide"}), 400

        print(f"📝 Question: '{question}'")

        # Lancement de la recherche
        result = searcher.search(question)

        print(f"✅ Recherche terminée - Status: {result.get('status', 'unknown')}")
        print("=" * 60)

        return jsonify(result)

    except Exception as e:
        print(f"❌ ERREUR ENDPOINT: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Erreur serveur: {str(e)}",
            "status": "error"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé du service"""
    try:
        # Test de connexion FileMaker
        extractor = FileMakerExtractor()
        fm_ok = extractor.connect()
        if fm_ok:
            extractor.logout()

        # Test Ollama
        try:
            ollama_response = requests.get('http://localhost:11434/api/tags', timeout=5)
            ollama_ok = ollama_response.status_code == 200
        except:
            ollama_ok = False

        return jsonify({
            "status": "OK" if (fm_ok and ollama_ok) else "PARTIAL",
            "service": "RAG API",
            "components": {
                "filemaker": "OK" if fm_ok else "ERROR",
                "ollama": "OK" if ollama_ok else "ERROR",
                "embeddings": "OK"
            },
            "version": "2.0"
        })

    except Exception as e:
        return jsonify({
            "status": "ERROR",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("\n🚀 DÉMARRAGE DU SERVEUR RAG")
    print("=" * 40)
    print("📡 URL: http://localhost:9000")
    print("🔍 Recherche: POST /search")
    print("💚 Santé: GET /health")
    print("=" * 40)

    app.run(host='0.0.0.0', port=9000, debug=True)
