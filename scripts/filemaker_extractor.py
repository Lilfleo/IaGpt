import requests
import base64
import urllib3
import logging
import json
from dotenv import load_dotenv
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FileMakerExtractor:
    def __init__(self):
        load_dotenv('config/config.env')
        self.server = os.getenv('FILEMAKER_SERVER')
        self.database = os.getenv('FILEMAKER_DATABASE')
        self.username = os.getenv('FILEMAKER_USERNAME')
        self.password = os.getenv('FILEMAKER_PASSWORD')
        self.token = None

        # Logger
        self.logger = logging.getLogger(__name__)

    def login(self):
        """Connexion à FileMaker Data API"""
        url = f"{self.server}/fmi/data/v1/databases/{self.database}/sessions"

        # Authentification Basic (base64)
        credentials = f"{self.username}:{self.password}"
        credentials_b64 = base64.b64encode(credentials.encode()).decode()

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {credentials_b64}'
        }

        payload = {}  # Payload vide pour FileMaker

        self.logger.info(f"🔗 Connexion à: {url}")

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10, verify=False)

            self.logger.info(f"📊 Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                self.token = data['response']['token']
                self.logger.info("✅ Connexion réussie!")
                return True
            else:
                self.logger.error(f"❌ Erreur connexion: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"💥 Exception: {str(e)}")
            return False

    def get_documents(self):
        """Récupère la liste des documents"""
        if not self.token:
            self.logger.error("❌ Pas de token, connexion requise")
            return []

        # URL pour récupérer les enregistrements (ajustez le nom de la table)
        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Documents/records"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        try:
            response = requests.get(url, headers=headers, verify=False)

            if response.status_code == 200:
                data = response.json()
                documents = data['response']['data']
                self.logger.info(f"📄 {len(documents)} documents trouvés")
                return documents
            else:
                self.logger.error(f"❌ Erreur récupération: {response.text}")
                return []

        except Exception as e:
            self.logger.error(f"💥 Exception: {str(e)}")
            return []

    def logout(self):
        """Déconnexion"""
        if self.token:
            url = f"{self.server}/fmi/data/v1/databases/{self.database}/sessions/{self.token}"
            try:
                requests.delete(url, verify=False)
                self.logger.info("👋 Déconnexion")
            except:
                pass
