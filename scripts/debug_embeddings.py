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
    extractor.login()

    # Modèle
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Question
    question = "prix d une part pour cristal life en 2021"
    question_embedding = model.encode([question])

    print(f"🔍 Recherche: {question}\n")

    # Récupération chunks avec "cristal"
    chunks = extractor.find_records(
        'chunks',
        {'text': 'cristal'},  # Recherche simple
        skip=0,
        limit=10
    )

    print(f"📊 {len(chunks)} chunks trouvés avec 'cristal'\n")

    # Test embeddings
    for i, chunk in enumerate(chunks):
        print(f"--- CHUNK {i + 1} ---")
        print(f"Document: {chunk.get('document_name', 'N/A')}")
        print(f"Texte: {chunk.get('text', '')[:200]}...")

        # Récupérer l'embedding
        embedding_record = extractor.find_records(
            'EmbeddingJson',
            {'chunk_id': chunk['id']}
        )

        if embedding_record:
            embedding = json.loads(embedding_record[0]['embedding'])
            similarity = np.dot(question_embedding[0], np.array(embedding))
            print(f"✅ Similarité: {similarity:.4f}")
        else:
            print("❌ Pas d'embedding trouvé")

        print("-" * 50)

    extractor.logout()


if __name__ == '__main__':
    test_embeddings()
