import requests
import json
import base64
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileMakerExtractor:
    def __init__(self):
        load_dotenv('config/config.env')
        self.server_url = os.getenv('FILEMAKER_SERVER')
        self.database = os.getenv('FILEMAKER_DATABASE')
        self.username = os.getenv('FILEMAKER_USERNAME')
        self.password = os.getenv('FILEMAKER_PASSWORD')
        self.session_token = None
        self.extraction_path = Path(os.getenv('PDF_EXTRACTION_PATH', './data/extracted_pdfs'))
        self.extraction_path.mkdir(parents=True, exist_ok=True)

    def login(self):
        """Connexion à FileMaker Data API"""
        url = f"{self.server_url}/fmi/data/v1/databases/{self.database}/sessions"
        payload = {"user": self.username, "password": self.password}

        try:
            response = requests.post(url, json=payload, timeout=10)
            logger.info(f"🔗 Connexion à: {url}")
            logger.info(f"📊 Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                self.session_token = data['response']['token']
                logger.info("✅ Connexion FileMaker réussie")
                return True
            else:
                logger.error(f"❌ Erreur connexion: {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Exception connexion: {e}")
            return False

    def logout(self):
        """Déconnexion"""
        if not self.session_token:
            return

        url = f"{self.server_url}/fmi/data/v1/databases/{self.database}/sessions/{self.session_token}"
        try:
            response = requests.delete(url)
            logger.info("🔓 Déconnexion FileMaker")
        except:
            pass

    def get_documents(self):
        """Récupère la liste des documents"""
        if not self.session_token:
            return None

        url = f"{self.server_url}/fmi/data/v1/databases/{self.database}/layouts/document/records"
        headers = {"Authorization": f"Bearer {self.session_token}"}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data['response']['data']
            else:
                logger.error(f"❌ Erreur récupération docs: {response.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Exception récupération: {e}")
            return None
