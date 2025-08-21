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
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                self.session_token = response.json()["response"]["token"]
                logger.info(f"✅ Connexion réussie à {self.database}")
                return True
            else:
                logger.error(f"❌ Erreur connexion: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Erreur réseau: {e}")
            return False

    def get_documents(self):
        """Récupère la liste des documents depuis la table 'document'"""
        if not self.session_token:
            logger.error("❌ Pas de token de session")
            return None

        url = f"{self.server_url}/fmi/data/v1/databases/{self.database}/layouts/document/records"
        headers = {"Authorization": f"Bearer {self.session_token}"}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Récupéré {len(data['response']['data'])} documents")
                return data['response']['data']
            else:
                logger.error(f"❌ Erreur récupération documents: {response.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Erreur: {e}")
            return None

    def extract_pdf(self, record_id, filename_prefix="doc"):
        """Extrait un PDF spécifique par record_id"""
        if not self.session_token:
            logger.error("❌ Pas de token de session")
            return None

        url = f"{self.server_url}/fmi/data/v1/databases/{self.database}/layouts/document/records/{record_id}/containers/fichier/1"
        headers = {"Authorization": f"Bearer {self.session_token}"}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                filename = f"{filename_prefix}_{record_id}.pdf"
                filepath = self.extraction_path / filename

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                logger.info(f"✅ PDF extrait: {filepath}")
                return str(filepath)
            else:
                logger.error(f"❌ Erreur extraction PDF {record_id}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Erreur extraction: {e}")
            return None

    def extract_all_pdfs(self):
        """Extrait tous les PDFs de la base"""
        documents = self.get_documents()
        if not documents:
            return []

        extracted_files = []
        for i, doc in enumerate(documents):
            record_id = doc['recordId']
            filepath = self.extract_pdf(record_id, f"rapport_{i + 1}")
            if filepath:
                extracted_files.append(filepath)

        logger.info(f"✅ {len(extracted_files)} PDFs extraits au total")
        return extracted_files

    def logout(self):
        """Ferme la session FileMaker"""
        if self.session_token:
            url = f"{self.server_url}/fmi/data/v1/databases/{self.database}/sessions/{self.session_token}"
            requests.delete(url)
            logger.info("✅ Session fermée")


# Test simple
if __name__ == "__main__":
    extractor = FileMakerExtractor()
    if extractor.login():
        documents = extractor.get_documents()
        print(f"Nombre de documents trouvés: {len(documents) if documents else 0}")
        extractor.logout()
