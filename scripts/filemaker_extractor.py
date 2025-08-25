import requests
import base64
import urllib3
import logging
import json
from dotenv import load_dotenv
import os
import re

# !/usr/bin/env python3


# Désactive les avertissements SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FileMakerExtractor:
    def __init__(self):
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.env'))
        self.server = os.getenv('FILEMAKER_SERVER')
        self.database = os.getenv('FILEMAKER_DATABASE')
        self.username = os.getenv('FILEMAKER_USERNAME')
        self.password = os.getenv('FILEMAKER_PASSWORD')
        self.token = None

        # Logger
        self.logger = logging.getLogger(__name__)

    def login(self):
        """Connexion à FileMaker Server"""
        url = f"{self.server}/fmi/data/v1/databases/{self.database}/sessions"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {self._encode_credentials()}'
        }

        try:
            response = requests.post(url, headers=headers, verify=False)

            if response.status_code in [200, 201]:
                data = response.json()
                self.token = data['response']['token']
                self.logger.info("✅ Connexion FileMaker OK")
                return True
            else:
                self.logger.error(f"❌ Erreur connexion: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"💥 Exception connexion: {str(e)}")
            return False

    def _encode_credentials(self):
        """Encode les identifiants en base64"""
        import base64
        credentials = f"{self.username}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()

    def get_documents(self):
        """Récupère TOUS les documents avec pagination"""
        if not self.token:
            self.logger.error("❌ Pas de token, connexion requise")
            return []

        all_documents = []
        offset = 1
        limit = 100  # FileMaker limite

        while True:
            url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Documents/records"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.token}'
            }

            params = {
                '_offset': offset,
                '_limit': limit
            }

            try:
                response = requests.get(url, headers=headers, params=params, verify=False)

                if response.status_code == 200:
                    data = response.json()
                    documents = data['response']['data']
                    all_documents.extend(documents)

                    self.logger.info(f"📄 Récupérés: {len(documents)} docs (offset: {offset})")

                    # Si moins de documents que la limite, on a tout récupéré
                    if len(documents) < limit:
                        break

                    offset += limit

                else:
                    self.logger.error(f"❌ Erreur récupération: {response.text}")
                    break

            except Exception as e:
                self.logger.error(f"💥 Exception: {str(e)}")
                break

        self.logger.info(f"📄 TOTAL: {len(all_documents)} documents")
        return all_documents

    def get_chunks_for_document(self, doc_id):
        """Vérifie si un document a déjà des chunks"""
        if not self.token:
            return []

        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Chunks/records/_find"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        payload = {
            "query": [{"idDocument": str(doc_id)}]
        }

        try:
            response = requests.post(url, json=payload, headers=headers, verify=False)
            if response.status_code == 200:
                data = response.json()
                return data['response']['data']
            else:
                return []
        except:
            return []

    def create_chunk(self, doc_id, text, chunk_index, embedding_json=""):
        """Créer un chunk dans FileMaker"""
        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Chunks/records"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        payload = {
            "fieldData": {
                "idDocument": str(doc_id),
                "Text": text,
                "EmbeddingJson": embedding_json,
                "Norme": f"chunk_{chunk_index}"
            }
        }

        try:
            response = requests.post(url, json=payload, headers=headers, verify=False)

            if response.status_code in [200, 201]:
                return True
            else:
                self.logger.error(f"❌ Erreur chunk: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"💥 Exception chunk: {str(e)}")
            return False

    def download_pdf(self, pdf_url, output_path):
        """Télécharge un PDF depuis FileMaker Server"""
        headers = {
            'Authorization': f'Bearer {self.token}'
        }

        try:
            response = requests.get(pdf_url, headers=headers, verify=False, stream=True)

            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                self.logger.error(f"❌ Erreur téléchargement: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"💥 Exception téléchargement: {str(e)}")
            return False

    def logout(self):
        """Déconnexion"""
        if self.token:
            url = f"{self.server}/fmi/data/v1/databases/{self.database}/sessions/{self.token}"
            try:
                requests.delete(url, verify=False)
                self.logger.info("👋 Déconnexion FileMaker")
            except:
                pass

    def extract_keywords(self, question, min_length=3):
        """Extrait automatiquement les mots-clés d'une question"""
        # Mots vides français/anglais à ignorer
        stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 'sont',
            'dans', 'sur', 'avec', 'pour', 'par', 'ce', 'cette', 'ces', 'qui', 'que', 'quoi',
            'comment', 'combien', 'quand', 'où', 'quel', 'quelle', 'quels', 'quelles',
            'the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'on', 'at', 'for', 'by', 'with'
        }

        # Nettoyer et segmenter
        words = re.findall(r'\b[a-zA-ZÀ-ÿ0-9]+\b', question.lower())

        # Filtrer les mots significatifs
        keywords = [
            word for word in words
            if len(word) >= min_length and word not in stop_words
        ]

        # Retourner les mots uniques
        return list(dict.fromkeys(keywords))

    def search_chunks_smart(self, question, limit=1000):
        """Recherche intelligente avec segmentation automatique"""
        if not self.token:
            self.logger.error("❌ Pas de token pour la recherche")
            return []

        # Extraire les mots-clés
        keywords = self.extract_keywords(question)
        self.logger.info(f"🔍 Mots-clés extraits: {keywords}")

        if not keywords:
            self.logger.warning("Aucun mot-clé trouvé dans la question")
            return []

        # Construire la requête OR pour FileMaker
        query_conditions = []
        for keyword in keywords[:8]:  # Limiter à 8 mots max
            query_conditions.append({"Text": f"*{keyword}*"})

        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Chunks/_find"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        payload = {
            "query": query_conditions,
            "limit": str(limit)
        }

        try:
            response = requests.post(url, json=payload, headers=headers, verify=False)

            if response.status_code == 200:
                data = response.json()
                chunks = data['response']['data']
                self.logger.info(f"✅ {len(chunks)} chunks trouvés avec recherche textuelle")
                return chunks
            elif response.status_code == 401:
                self.logger.error("❌ Erreur 401: Token expiré")
                return []
            else:
                self.logger.error(f"❌ Erreur recherche: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            self.logger.error(f"❌ Exception recherche: {e}")
            return []
