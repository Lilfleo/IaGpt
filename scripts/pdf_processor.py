#!/usr/bin/env python3
import os
import re
import json
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from filemaker_extractor import FileMakerExtractor
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self):
        self.extractor = FileMakerExtractor()
        # Modèle d'embeddings léger et efficace
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def clean_text(self, text):
        """Nettoie le texte extrait"""
        # Supprime les caractères indésirables
        text = re.sub(r'\s+', ' ', text)  # Espaces multiples
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\€\%\(\)\-]', '', text)  # Caractères spéciaux
        return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        """Extrait le texte d'un PDF"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text

            return self.clean_text(text)

        except Exception as e:
            logger.error(f"❌ Erreur extraction PDF {pdf_path}: {str(e)}")
            return ""

    def chunk_text(self, text, chunk_size=800, overlap=100):
        """Découpe intelligemment le texte en chunks"""
        # Priorité aux sections financières
        sections = re.split(r'\n(?=.*(?:ÉVOLUTION|RÉSULTATS|BILAN|CAPITAUX|ACTIF|PASSIF).*)\n', text)

        chunks = []
        for section in sections:
            if len(section) <= chunk_size:
                chunks.append(section)
            else:
                # Découpe par phrases si trop long
                sentences = re.split(r'(?<=[.!?])\s+', section)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk + sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "

                if current_chunk:
                    chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]

    def generate_embeddings(self, texts):
        """Génère les embeddings pour une liste de textes"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings

    def process_document(self, document_record):
        """Traite un document complet"""
        record_id = document_record['recordId']
        field_data = document_record['fieldData']
        filename = field_data.get('Nom_fichier', '')
        pdf_url = field_data.get('fichier', '')
        existing_text = field_data.get('text', '')

        logger.info(f"🔄 Traitement: {filename}")

        # Si pas de texte extrait, on utilise le PDF
        if not existing_text or len(existing_text.strip()) < 100:
            if pdf_url:
                # Télécharger et extraire
                pdf_path = f"/tmp/{filename}"
                if self.extractor.download_pdf(pdf_url, pdf_path):
                    text = self.extract_text_from_pdf(pdf_path)
                    os.remove(pdf_path)  # Nettoie
                else:
                    logger.error(f"❌ Impossible de télécharger {filename}")
                    return False
            else:
                logger.error(f"❌ Pas d'URL PDF pour {filename}")
                return False
        else:
            text = existing_text

        if not text or len(text.strip()) < 100:
            logger.warning(f"⚠️ Texte insuffisant pour {filename}")
            return False

        # Chunking
        chunks = self.chunk_text(text)
        logger.info(f"📝 {len(chunks)} chunks créés pour {filename}")

        # Génération des embeddings
        embeddings = self.generate_embeddings(chunks)

        # Sauvegarde dans FileMaker
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedding_json = json.dumps(embedding.tolist())
            self.extractor.create_chunk(record_id, chunk, i + 1, embedding_json)

        return True


def main(start_index=0, batch_size=50):
    """Traitement principal avec pagination"""
    processor = PDFProcessor()

    if not processor.extractor.login():
        logger.error("❌ Connexion FileMaker échouée")
        return

    documents = processor.extractor.get_documents()
    logger.info(f"📁 {len(documents)} documents total")

    # Traitement par batch
    end_index = min(start_index + batch_size, len(documents))
    batch_docs = documents[start_index:end_index]

    logger.info(f"🔄 Traitement documents {start_index + 1} à {end_index}")

    for i, doc in enumerate(batch_docs):
        logger.info(f"🔄 Document {start_index + i + 1}/{len(documents)}")
        processor.process_document(doc)

    processor.extractor.logout()
    logger.info(f"✅ Batch {start_index}-{end_index} terminé")


if __name__ == "__main__":


    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 450
    main(start, batch)

if __name__ == "__main__":
    main()
