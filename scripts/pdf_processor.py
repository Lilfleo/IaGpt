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

    def process_document(self, document_record, doc_index, total_docs):
        """Traite un document complet"""
        record_id = document_record['recordId']
        field_data = document_record['fieldData']
        filename = field_data.get('Nom_fichier', 'Inconnu')
        pdf_url = field_data.get('fichier', '')
        existing_text = field_data.get('text', '')

        # ✅ VÉRIFICATION ANTI-DOUBLON
        existing_chunks = self.extractor.get_chunks_for_document(record_id)
        if existing_chunks and len(existing_chunks) > 0:
            logger.info(f"⏭️ [{doc_index}/{total_docs}] {filename} - Déjà traité ({len(existing_chunks)} chunks)")
            return True

        logger.info(f"🔄 [{doc_index}/{total_docs}] Nouveau traitement: {filename}")

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
        logger.info(f"📝 {len(chunks)} chunks créés")

        if not chunks:
            logger.warning(f"⚠️ Aucun chunk créé pour {filename}")
            return False

        # Génération des embeddings
        try:
            embeddings = self.generate_embeddings(chunks)
            logger.info(f"🧮 Embeddings générés")
        except Exception as e:
            logger.error(f"❌ Erreur embeddings: {str(e)}")
            return False

        # Sauvegarde dans FileMaker
        success_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                embedding_json = json.dumps(embedding.tolist())
                if self.extractor.create_chunk(record_id, chunk, i + 1, embedding_json):
                    success_count += 1
            except Exception as e:
                logger.error(f"❌ Erreur sauvegarde chunk {i + 1}: {str(e)}")

        logger.info(f"✅ {success_count}/{len(chunks)} chunks sauvegardés")
        return success_count > 0


def main(start_index=0, batch_size=450):
    """Traitement principal avec pagination"""
    processor = PDFProcessor()

    if not processor.extractor.login():
        logger.error("❌ Connexion FileMaker échouée")
        return

    # Récupère TOUS les documents
    documents = processor.extractor.get_documents()
    total_docs = len(documents)

    if total_docs == 0:
        logger.error("❌ Aucun document trouvé")
        processor.extractor.logout()
        return

    logger.info(f"📁 {total_docs} documents trouvés au total")

    # Calcul de la plage de traitement
    end_index = min(start_index + batch_size, total_docs)
    batch_docs = documents[start_index:end_index]

    logger.info(f"🎯 Traitement: documents {start_index + 1} à {end_index}")
    logger.info(f"📊 Batch: {len(batch_docs)} documents à traiter")

    # Traitement
    processed = 0
    skipped = 0
    errors = 0

    for i, doc in enumerate(batch_docs):
        doc_index = start_index + i + 1

        try:
            success = processor.process_document(doc, doc_index, total_docs)
            if success:
                processed += 1
            else:
                errors += 1
        except Exception as e:
            logger.error(f"💥 Erreur document {doc_index}: {str(e)}")
            errors += 1

    # Résumé final
    logger.info(f"🏁 RÉSUMÉ du batch {start_index}-{end_index}:")
    logger.info(f"   ✅ Traités: {processed}")
    logger.info(f"   ⏭️ Déjà faits: {skipped}")
    logger.info(f"   ❌ Erreurs: {errors}")

    processor.extractor.logout()


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 450
    main(start, batch)
