#!/usr/bin/env python3
"""
FileMaker Data API Extractor - Version 2.0
Extraction et recherche de données FileMaker pour le système RAG
"""

import requests
import base64
import urllib3
import logging
import json
import os
import re
from dotenv import load_dotenv

# Désactive les avertissements SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FileMakerExtractor:
    """Extracteur de données FileMaker avec recherche intelligente"""

    def __init__(self):
        """Initialisation avec chargement de la configuration"""
        self._load_config()
        self._setup_logging()
        self.token = None
        self.session_active = False

    def _load_config(self):
        """Charge la configuration depuis le fichier .env"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config',
            'config.env'
        )
        load_dotenv(config_path)

        self.server = os.getenv('FILEMAKER_SERVER')
        self.database = os.getenv('FILEMAKER_DATABASE')
        self.username = os.getenv('FILEMAKER_USERNAME')
        self.password = os.getenv('FILEMAKER_PASSWORD')

        # Validation de la configuration
        if not all([self.server, self.database, self.username, self.password]):
            raise ValueError("Configuration FileMaker incomplète dans config.env")

    def _setup_logging(self):
        """Configure le logging"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _encode_credentials(self):
        """Encode les identifiants en Base64 pour l'authentification"""
        credentials = f"{self.username}:{self.password}"
        return base64.b64encode(credentials.encode()).decode()

    def login(self):
        """Établit une session avec FileMaker Server"""
        url = f"{self.server}/fmi/data/v1/databases/{self.database}/sessions"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Basic {self._encode_credentials()}'
        }

        try:
            self.logger.info("🔐 Tentative de connexion à FileMaker...")
            response = requests.post(url, headers=headers, verify=False, timeout=10)

            if response.status_code in [200, 201]:
                data = response.json()
                self.token = data['response']['token']
                self.session_active = True
                self.logger.info("✅ Connexion FileMaker établie")
                return True
            else:
                self.logger.error(f"❌ Échec connexion: {response.status_code} - {response.text}")
                return False

        except requests.RequestException as e:
            self.logger.error(f"❌ Erreur réseau FileMaker: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Erreur inattendue connexion: {str(e)}")
            return False

    def connect(self):
        """Alias pour login() - compatibilité avec les autres services"""
        return self.login()

    def logout(self):
        """Ferme la session FileMaker"""
        if not self.token:
            return True

        url = f"{self.server}/fmi/data/v1/databases/{self.database}/sessions/{self.token}"

        try:
            requests.delete(url, verify=False, timeout=5)
            self.logger.info("👋 Session FileMaker fermée")
        except Exception as e:
            self.logger.warning(f"⚠️ Erreur fermeture session: {str(e)}")
        finally:
            self.token = None
            self.session_active = False

        return True

    def get_all_chunks_sample(self, limit=1000):
        """Récupère un échantillon de tous les chunks sans filtre"""
        if not self._check_connection():
            return []

        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Chunks/records"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        params = {
            '_limit': str(min(limit, 1000))
        }

        try:
            self.logger.info(f"🎯 Récupération échantillon: {limit} chunks")

            response = requests.get(
                url,
                headers=headers,
                params=params,
                verify=False,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                chunks = data['response']['data']
                self.logger.info(f"✅ {len(chunks)} chunks récupérés")
                return chunks
            else:
                self.logger.error(f"❌ Erreur: {response.status_code}")
                return []

        except Exception as e:
            self.logger.error(f"❌ Exception: {str(e)}")
            return []

    def _check_connection(self):
        """Vérifie que la connexion est active"""
        if not self.token or not self.session_active:
            self.logger.error("❌ Pas de session active - connexion requise")
            return False
        return True

    def extract_keywords(self, question, min_length=3):
        """Extrait automatiquement les mots-clés significatifs d'une question"""
        # Mots vides à ignorer
        stop_words = {
            # Français
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'est', 'sont',
            'dans', 'sur', 'avec', 'pour', 'par', 'ce', 'cette', 'ces', 'qui', 'que', 'quoi',
            'comment', 'combien', 'quand', 'où', 'quel', 'quelle', 'quels', 'quelles',
            'avoir', 'être', 'faire', 'dire', 'aller', 'voir', 'savoir', 'pouvoir',
            # Anglais
            'the', 'a', 'an', 'and', 'or', 'is', 'are', 'in', 'on', 'at', 'for', 'by', 'with',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }

        # Extraction des mots (lettres, chiffres, accents)
        words = re.findall(r'\b[a-zA-ZÀ-ÿ0-9]+\b', question.lower())

        # Filtrage des mots significatifs
        keywords = [
            word for word in words
            if len(word) >= min_length and word not in stop_words
        ]

        # Suppression des doublons en gardant l'ordre
        return list(dict.fromkeys(keywords))

    def search_chunks_smart(self, question, limit=1000):
        """
        Recherche intelligente dans les chunks avec extraction automatique de mots-clés

        Args:
            question (str): Question en langage naturel
            limit (int): Nombre maximum de résultats (max 1000)

        Returns:
            list: Liste des chunks trouvés au format FileMaker natif
        """
        if not self._check_connection():
            return []

        # Extraction des mots-clés
        keywords = self.extract_keywords(question)
        self.logger.info(f"🔍 Mots-clés extraits: {keywords[:8]}")  # Affiche max 8

        if not keywords:
            self.logger.warning("❌ Aucun mot-clé significatif trouvé")
            return []

        # Configuration de la requête FileMaker
        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Chunks/_find"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        # Construction de la requête OR (recherche sur plusieurs mots-clés)
        query_conditions = []
        for keyword in keywords[:10]:  # Limite à 10 mots-clés pour éviter la surcharge
            query_conditions.append({"Text": f"*{keyword}*"})

        payload = {
            "query": query_conditions,
            "limit": str(min(limit, 1000))  # FileMaker Data API limite à 1000
        }

        try:
            self.logger.info(f"🎯 Recherche FileMaker: {len(query_conditions)} conditions")

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                verify=False,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                chunks = data['response']['data']

                self.logger.info(f"✅ {len(chunks)} chunks trouvés")
                return chunks

            elif response.status_code == 401:
                self.logger.error("❌ Token expiré - reconnexion nécessaire")
                self.session_active = False
                return []

            else:
                self.logger.error(f"❌ Erreur recherche: {response.status_code}")
                if response.text:
                    self.logger.error(f"Détails: {response.text[:200]}")
                return []

        except requests.Timeout:
            self.logger.error("❌ Timeout de la recherche FileMaker")
            return []
        except requests.RequestException as e:
            self.logger.error(f"❌ Erreur réseau recherche: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"❌ Erreur inattendue recherche: {str(e)}")
            return []

    def get_documents(self, limit=None):
        """
        Récupère tous les documents avec pagination automatique

        Args:
            limit (int, optional): Limite du nombre de documents à récupérer

        Returns:
            list: Liste de tous les documents
        """
        if not self._check_connection():
            return []

        all_documents = []
        offset = 1
        batch_size = 100  # Taille des lots FileMaker
        total_retrieved = 0

        while True:
            url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Documents/records"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.token}'
            }

            params = {
                '_offset': offset,
                '_limit': batch_size
            }

            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    verify=False,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    documents = data['response']['data']

                    if not documents:
                        break

                    all_documents.extend(documents)
                    total_retrieved += len(documents)

                    self.logger.info(f"📄 Lot récupéré: {len(documents)} docs (total: {total_retrieved})")

                    # Vérification des limites
                    if limit and total_retrieved >= limit:
                        all_documents = all_documents[:limit]
                        break

                    if len(documents) < batch_size:
                        break  # Dernier lot

                    offset += batch_size

                else:
                    self.logger.error(f"❌ Erreur récupération documents: {response.status_code}")
                    break

            except Exception as e:
                self.logger.error(f"❌ Exception récupération documents: {str(e)}")
                break

        self.logger.info(f"📄 TOTAL DOCUMENTS: {len(all_documents)}")
        return all_documents

    def get_chunks_for_document(self, doc_id):
        """
        Vérifie si un document possède déjà des chunks

        Args:
            doc_id (str): ID du document

        Returns:
            list: Liste des chunks existants pour ce document
        """
        if not self._check_connection():
            return []

        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Chunks/_find"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        payload = {
            "query": [{"idDocument": doc_id}],
            "limit": "1000"
        }

        try:
            response = requests.post(url, json=payload, headers=headers, verify=False)

            if response.status_code == 200:
                data = response.json()
                return data['response']['data']
            else:
                return []

        except Exception as e:
            self.logger.error(f"❌ Erreur récupération chunks: {str(e)}")
            return []

    def create_chunk(self, idDocument, chunk_text, chunk_index, embeddings=None):
        """Crée un nouveau chunk dans FileMaker avec tous les champs"""
        if not self._check_connection():
            return False

        url = f"{self.server}/fmi/data/v1/databases/{self.database}/layouts/Chunks/records"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        # ✅ TOUS les champs corrects maintenant !
        field_data = {
            "idDocument": str(idDocument),
            "Text": chunk_text,
            "ChunkIndex": chunk_index  # ✅ Maintenant ça existe !
        }

        # ✅ Nom de champ corrigé
        if embeddings:
            if isinstance(embeddings, str):
                field_data["EmbeddingJson"] = embeddings
            else:
                field_data["EmbeddingJson"] = json.dumps(embeddings)

        payload = {"fieldData": field_data}

        try:
            response = requests.post(url, json=payload, headers=headers, verify=False, timeout=30)

            if response.status_code in [200, 201]:
                self.logger.debug(f"✅ Chunk créé: doc={idDocument}, index={chunk_index}")
                return True
            else:
                self.logger.error(f"❌ Erreur création chunk: {response.status_code}")
                if response.text:
                    self.logger.error(f"Détails: {response.text[:500]}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Exception création chunk: {str(e)}")
            return False

    def download_pdf(self, pdf_url, output_path):
        """
        Télécharge un PDF depuis FileMaker Server

        Args:
            pdf_url (str): URL du PDF sur FileMaker Server
            output_path (str): Chemin de sauvegarde local

        Returns:
            bool: Succès du téléchargement
        """
        if not self._check_connection():
            return False

        headers = {
            'Authorization': f'Bearer {self.token}'
        }

        try:
            self.logger.info(f"📥 Téléchargement PDF: {os.path.basename(output_path)}")

            response = requests.get(
                pdf_url,
                headers=headers,
                verify=False,
                stream=True,
                timeout=60
            )

            if response.status_code == 200:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                self.logger.info(f"✅ PDF téléchargé: {output_path}")
                return True
            else:
                self.logger.error(f"❌ Erreur téléchargement: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Exception téléchargement: {str(e)}")
            return False

    def __enter__(self):
        """Support du context manager (with statement)"""
        if self.login():
            return self
        else:
            raise ConnectionError("Impossible de se connecter à FileMaker")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage automatique à la sortie du context manager"""
        self.logout()

# Test rapide si le script est exécuté directement
if __name__ == "__main__":
    print("🧪 Test FileMaker Extractor...")

    try:
        with FileMakerExtractor() as fm:
            print("✅ Connexion réussie")

            # Test de recherche
            chunks = fm.search_chunks_smart("prix souscription", limit=5)
            print(f"📊 Chunks trouvés: {len(chunks)}")

    except Exception as e:
        print(f"❌ Erreur: {e}")
