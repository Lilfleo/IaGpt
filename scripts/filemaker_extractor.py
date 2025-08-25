#!/usr/bin/env python3
import requests
import json
import logging
import urllib3

# Désactive les avertissements SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FileMakerExtractor:
    def __init__(self):
        # Configuration - AJUSTEZ vos paramètres
        self.server = "https://votre-serveur-filemaker.com"
        self.database = "votre-base"
        self.username = "votre-username"
        self.password = "votre-password"
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
