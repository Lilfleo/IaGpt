#!/usr/bin/env python3
import sys
import os

# Ajoutez le bon chemin
sys.path.append('/opt/filemaker-ai-poc/IaGpt')
sys.path.append('/opt/filemaker-ai-poc/IaGpt/scripts')

# Maintenant l'import direct
from filemaker_extractor import FileMakerExtractor
import json
import requests

def debug_complete_pipeline():
    print("🔍 DEBUG COMPLET DU PIPELINE")
    # ... reste du code identique

    print("=" * 60)

    extractor = FileMakerExtractor()

    # 1. TEST CONNEXION
    print("\n1️⃣ TEST CONNEXION")
    if not extractor.login():
        print("❌ ÉCHEC CONNEXION")
        return
    print("✅ Connexion OK")

    # 2. RECHERCHE MANUELLE DIRECTE DANS FILEMAKER
    print("\n2️⃣ RECHERCHE DIRECTE 'CRISTAL' DANS FILEMAKER")

    url = f"{extractor.server}/fmi/data/v1/databases/{extractor.database}/layouts/Chunks/_find"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {extractor.token}'
    }

    # Test avec un seul mot
    payload = {
        "query": [{"Text": "*cristal*"}],
        "limit": "50"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, verify=False)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            chunks = data['response']['data']
            print(f"📊 {len(chunks)} chunks trouvés avec 'cristal'")

            # Analyser chaque chunk
            for i, chunk in enumerate(chunks[:5]):  # Les 5 premiers
                chunk_data = chunk['fieldData']
                text = chunk_data.get('Text', '')[:200]  # Premiers 200 chars
                doc_id = chunk_data.get('idDocument', 'N/A')
                has_embedding = bool(chunk_data.get('EmbeddingJson', '').strip())

                print(f"\n--- CHUNK {i + 1} ---")
                print(f"Doc ID: {doc_id}")
                print(f"Has Embedding: {has_embedding}")
                print(f"Text: {text}...")

                # Chercher spécifiquement "prix" et "part"
                if 'prix' in text.lower() and 'part' in text.lower():
                    print("🎯 BINGO! Ce chunk contient 'prix' et 'part'")

        else:
            print(f"❌ Erreur: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

    # 3. TEST RECHERCHE MULTI-MOTS
    print("\n3️⃣ TEST RECHERCHE MULTI-MOTS")

    payload_multi = {
        "query": [
            {"Text": "*cristal*"},
            {"Text": "*prix*"},
            {"Text": "*part*"}
        ],
        "limit": "100"
    }

    try:
        response = requests.post(url, json=payload_multi, headers=headers, verify=False)
        if response.status_code == 200:
            data = response.json()
            chunks = data['response']['data']
            print(f"📊 {len(chunks)} chunks trouvés avec recherche multi-mots")

            # Analyser la pertinence
            for chunk in chunks[:10]:
                text = chunk['fieldData'].get('Text', '').lower()
                score = 0
                keywords_found = []

                if 'cristal' in text:
                    score += 1
                    keywords_found.append('cristal')
                if 'prix' in text:
                    score += 1
                    keywords_found.append('prix')
                if 'part' in text:
                    score += 1
                    keywords_found.append('part')
                if '2021' in text:
                    score += 1
                    keywords_found.append('2021')

                if score >= 2:  # Au moins 2 mots-clés
                    print(f"🔥 CHUNK PERTINENT (score: {score})")
                    print(f"   Mots trouvés: {keywords_found}")
                    print(f"   Doc: {chunk['fieldData'].get('idDocument')}")
                    print(f"   Extrait: {text[:150]}...")
                    print()

    except Exception as e:
        print(f"❌ Exception multi: {e}")

    # 4. VÉRIFICATION TOTALE DES CHUNKS
    print("\n4️⃣ STATISTIQUES GLOBALES")

    url_all = f"{extractor.server}/fmi/data/v1/databases/{extractor.database}/layouts/Chunks"

    try:
        response = requests.get(url_all, headers=headers, params={'_limit': '1'}, verify=False)
        if response.status_code == 200:
            # Obtenir le total via foundCount
            found_count = response.json()['response']['dataInfo']['foundCount']
            print(f"📊 Total chunks dans la base: {found_count}")

        # Test recherche large
        large_search = {
            "query": [{"Text": "*"}],  # Tous les chunks non vides
            "limit": "1000"
        }

        response = requests.post(url, json=large_search, headers=headers, verify=False)
        if response.status_code == 200:
            chunks = response.json()['response']['data']

            cristal_count = 0
            prix_count = 0
            part_count = 0
            embedding_count = 0

            for chunk in chunks:
                text = chunk['fieldData'].get('Text', '').lower()
                if 'cristal' in text:
                    cristal_count += 1
                if 'prix' in text:
                    prix_count += 1
                if 'part' in text:
                    part_count += 1
                if chunk['fieldData'].get('EmbeddingJson', '').strip():
                    embedding_count += 1

            print(f"📈 Dans les 1000 premiers chunks:")
            print(f"   - Contiennent 'cristal': {cristal_count}")
            print(f"   - Contiennent 'prix': {prix_count}")
            print(f"   - Contiennent 'part': {part_count}")
            print(f"   - Ont un embedding: {embedding_count}")

    except Exception as e:
        print(f"❌ Exception stats: {e}")


if __name__ == "__main__":
    debug_complete_pipeline()
