#!/usr/bin/env python3
import os
import re
import json
import fitz  # PyMuPDF
import pdfplumber
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
        # Modèle d'embeddings spécialisé français
        try:
            self.embedding_model = SentenceTransformer('dangvantuan/sentence-camembert-large')
        except:
            # Fallback vers le modèle original si le spécialisé n'est pas disponible
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("📝 Utilisation du modèle d'embedding par défaut")

    def clean_text(self, text):
        """Nettoyage intelligent préservant les informations financières"""
        # Normalisation des nombres et devises
        text = re.sub(r'(\d+)\s*[,\.]\s*(\d+)', r'\1,\2', text)  # Normalise les nombres
        text = re.sub(r'(\d+)\s*€', r'\1€', text)  # Colle € aux nombres
        text = re.sub(r'(\d+)\s*%', r'\1%', text)  # Colle % aux nombres
        text = re.sub(r'(\d+)\s*M€', r'\1M€', text)  # Millions d'euros

        # Réparation des mots coupés par les sauts de ligne
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'([^.!?])\n([a-zà-ÿ])', r'\1 \2', text)

        # Nettoyage des espaces
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\t+', ' ', text)

        # Suppression des caractères indésirables mais préservation de la ponctuation financière
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\€\%\(\)\-\n°/]', '', text)

        return text.strip()

    def extract_text_from_pdf(self, pdf_path):
        """Extraction améliorée avec PyMuPDF et fallback pdfplumber"""
        try:
            text_content = []

            # Tentative avec PyMuPDF (meilleur pour la structure)
            try:
                doc = fitz.open(pdf_path)
                for page_num, page in enumerate(doc):
                    # Extraction avec préservation de la mise en page
                    text_dict = page.get_text("dict")
                    page_text = self.reconstruct_text_with_structure(text_dict)
                    if page_text.strip():
                        text_content.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                doc.close()

                final_text = "\n".join(text_content)
                if len(final_text.strip()) > 100:
                    return self.clean_text(final_text)

            except Exception as e:
                logger.warning(f"⚠️ PyMuPDF failed, trying pdfplumber: {str(e)}")

            # Fallback avec pdfplumber (meilleur pour les tableaux)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages_text = []
                    for page_num, page in enumerate(pdf.pages):
                        page_text = page.extract_text(layout=True)
                        if page_text and page_text.strip():
                            pages_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")

                    final_text = "\n".join(pages_text)
                    if len(final_text.strip()) > 100:
                        return self.clean_text(final_text)

            except Exception as e:
                logger.error(f"❌ pdfplumber aussi échoué: {str(e)}")

            logger.error(f"❌ Toutes les méthodes d'extraction ont échoué pour {pdf_path}")
            return ""

        except Exception as e:
            logger.error(f"❌ Erreur extraction PDF {pdf_path}: {str(e)}")
            return ""

    def reconstruct_text_with_structure(self, text_dict):
        """Reconstruit le texte en préservant la structure des colonnes"""
        lines = []

        try:
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "") + " "
                        if line_text.strip():
                            lines.append(line_text.strip())

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"⚠️ Erreur reconstruction structure: {str(e)}")
            return ""

    def classify_content_type(self, text):
        """Classifie le type de contenu du chunk"""
        text_lower = text.lower()

        # Mots-clés par catégorie
        categories = {
            'pricing': ['prix', 'souscription', 'commission', 'frais', 'tarif', 'modalités'],
            'performance': ['rendement', 'distribution', 'tri', 'rgi', 'performance', 'dividende'],
            'assets': ['acquisition', 'patrimoine', 'actif', 'immobilier', 'surface', 'locataire'],
            'financial_data': ['capitalisation', 'collecte', 'parts', 'euros', 'bilan', 'résultat'],
            'editorial': ['éditorial', 'message', 'président', 'directeur'],
            'legal': ['conditions', 'cession', 'retrait', 'fiscalité', 'règlement']
        }

        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score / len(keywords)

        if not scores:
            return 'general'

        return max(scores, key=scores.get)

    def extract_financial_entities(self, text):
        """Extrait les entités financières du texte"""
        return {
            'prices': re.findall(r'(\d+(?:[,\.]\d+)*)\s*€', text),
            'percentages': re.findall(r'(\d+(?:[,\.]\d+)*)\s*%', text),
            'dates': re.findall(r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})', text),
            'years': re.findall(r'\b(20\d{2})\b', text),
            'quarters': re.findall(r'(\d+[er]*\s*trimestre\s*\d{4})', text, re.IGNORECASE),
            'amounts_millions': re.findall(r'(\d+(?:[,\.]\d+)*)\s*[Mm]€', text),
            'surfaces': re.findall(r'(\d+(?:[,\.]\d+)*)\s*m[²2]', text)
        }

    def calculate_financial_importance(self, text):
        """Calcule un score d'importance financière"""
        score = 0

        # Présence d'entités financières
        entities = self.extract_financial_entities(text)
        score += len(entities['prices']) * 2
        score += len(entities['percentages']) * 2
        score += len(entities['amounts_millions']) * 3

        # Mots-clés importants
        important_keywords = [
            'prix de souscription', 'rendement', 'distribution', 'capitalisation',
            'tri', 'rgi', 'performance', 'acquisition', 'collecte'
        ]

        text_lower = text.lower()
        for keyword in important_keywords:
            if keyword in text_lower:
                score += 3

        return min(10, score)  # Score max de 10

    def chunk_text_intelligent(self, text, chunk_size=800, overlap=100):
        """Chunking intelligent par sections financières"""

        # Patterns pour identifier les sections importantes
        section_patterns = {
            'chiffres_cles': r'(?i)(chiffres?\s+cl[ée]s?|situation\s+au|r[ée]sum[ée]|les\s+chiffres)',
            'prix_tarifs': r'(?i)(prix\s+de\s+souscription|modalit[ée]s\s+de\s+souscription|tarifs?)',
            'performance': r'(?i)(performance|rendement|distribution|r[ée]sultats?|tri|rgi)',
            'patrimoine': r'(?i)(patrimoine|acquisitions?|actifs?|portefeuille|zoom\s+sur)',
            'editorial': r'(?i)([ée]ditorial|message|pr[ée]sident|directeur\s+g[ée]n[ée]ral)',
            'evolution': r'(?i)([ée]volution|coup\s+d.oeil|nouvelles?\s+acquisitions?)',
            'conditions': r'(?i)(conditions?\s+de\s+cession|modalit[ée]s|fiscalit[ée])',
            'actualite': r'(?i)(actualit[ée]|news|informations?\s+g[ée]n[ée]rales?)'
        }

        chunks = []
        processed_ranges = []

        # Premier passage : sections identifiées
        for section_name, pattern in section_patterns.items():
            matches = list(re.finditer(pattern, text))

            for match in matches:
                # Contexte étendu autour de la section
                start = max(0, match.start() - 150)
                end = min(len(text), match.start() + chunk_size + 200)

                # Éviter les chevauchements avec des sections déjà traitées
                if any(start < proc_end and end > proc_start for proc_start, proc_end in processed_ranges):
                    continue

                # Chercher une fin de section ou de phrase naturelle
                chunk_end = self.find_natural_boundary(text, end)
                chunk_text = text[start:chunk_end].strip()

                if len(chunk_text) > 100:  # Chunk significatif
                    # Enrichissement avec métadonnées
                    chunk_info = {
                        'text': chunk_text,
                        'section': section_name,
                        'content_type': self.classify_content_type(chunk_text),
                        'financial_entities': self.extract_financial_entities(chunk_text),
                        'financial_score': self.calculate_financial_importance(chunk_text),
                        'word_count': len(chunk_text.split())
                    }

                    chunks.append(chunk_info)
                    processed_ranges.append((start, chunk_end))

        # Deuxième passage : chunking traditionnel pour le reste
        remaining_text = text
        for proc_start, proc_end in sorted(processed_ranges, reverse=True):
            remaining_text = remaining_text[:proc_start] + remaining_text[proc_end:]

        if remaining_text.strip():
            traditional_chunks = self.traditional_chunking(remaining_text, chunk_size, overlap)
            for chunk_text in traditional_chunks:
                chunk_info = {
                    'text': chunk_text,
                    'section': 'general',
                    'content_type': self.classify_content_type(chunk_text),
                    'financial_entities': self.extract_financial_entities(chunk_text),
                    'financial_score': self.calculate_financial_importance(chunk_text),
                    'word_count': len(chunk_text.split())
                }
                chunks.append(chunk_info)

        # Tri par importance financière et suppression des doublons
        chunks = self.deduplicate_and_sort_chunks(chunks)

        # Retour du texte simple pour compatibilité avec le système existant
        return [chunk['text'] for chunk in chunks if len(chunk['text'].strip()) > 50]

    def find_natural_boundary(self, text, position):
        """Trouve une frontière naturelle pour découper le texte"""
        if position >= len(text):
            return len(text)

        # Chercher la fin du paragraphe le plus proche
        for i in range(position, min(position + 200, len(text))):
            if text[i:i + 2] == '\n\n':
                return i

        # Sinon, chercher la fin de phrase
        for i in range(position, min(position + 100, len(text))):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] in ' \n':
                return i + 1

        return position

    def traditional_chunking(self, text, chunk_size, overlap):
        """Chunking traditionnel amélioré"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Ajuster pour ne pas couper au milieu d'une phrase
            if end < len(text):
                # Chercher la fin de phrase la plus proche
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] in ' \n':
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk and len(chunk) > 50:
                chunks.append(chunk)

            start = max(start + chunk_size - overlap, end - overlap)
            if start >= len(text):
                break

        return chunks

    def deduplicate_and_sort_chunks(self, chunks):
        """Déduplique et trie les chunks par importance"""
        # Déduplication basée sur la similarité du contenu
        unique_chunks = []

        for chunk in chunks:
            is_duplicate = False
            chunk_words = set(chunk['text'].lower().split())

            for existing in unique_chunks:
                existing_words = set(existing['text'].lower().split())
                similarity = len(chunk_words & existing_words) / len(chunk_words | existing_words)

                if similarity > 0.8:  # 80% de similarité = doublon
                    is_duplicate = True
                    # Garder le chunk avec le meilleur score financier
                    if chunk['financial_score'] > existing['financial_score']:
                        unique_chunks.remove(existing)
                        unique_chunks.append(chunk)
                    break

            if not is_duplicate:
                unique_chunks.append(chunk)

        # Tri par score financier décroissant
        return sorted(unique_chunks, key=lambda x: x['financial_score'], reverse=True)

    def generate_embeddings(self, texts):
        """Génère les embeddings pour une liste de textes"""
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            logger.error(f"❌ Erreur génération embeddings: {str(e)}")
            raise

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

        # Extraction du texte
        text = ""
        if existing_text and len(existing_text.strip()) >= 100:
            text = existing_text
            logger.info(f"📝 Utilisation du texte existant ({len(text)} caractères)")
        elif pdf_url:
            # Télécharger et extraire avec la nouvelle méthode
            pdf_path = f"/tmp/{filename}"
            if self.extractor.download_pdf(pdf_url, pdf_path):
                text = self.extract_text_from_pdf(pdf_path)
                os.remove(pdf_path)  # Nettoie
                logger.info(f"📄 Texte extrait du PDF ({len(text)} caractères)")
            else:
                logger.error(f"❌ Impossible de télécharger {filename}")
                return False
        else:
            logger.error(f"❌ Pas de texte ni d'URL PDF pour {filename}")
            return False

        if not text or len(text.strip()) < 100:
            logger.warning(f"⚠️ Texte insuffisant pour {filename}")
            return False

        # Chunking intelligent
        try:
            chunks = self.chunk_text_intelligent(text)
            logger.info(f"📝 {len(chunks)} chunks créés avec chunking intelligent")
        except Exception as e:
            logger.warning(f"⚠️ Chunking intelligent échoué, fallback traditionnel: {str(e)}")
            chunks = self.traditional_chunking(text, 800, 100)
            logger.info(f"📝 {len(chunks)} chunks créés avec chunking traditionnel")

        if not chunks:
            logger.warning(f"⚠️ Aucun chunk créé pour {filename}")
            return False

        # Génération des embeddings
        try:
            embeddings = self.generate_embeddings(chunks)
            logger.info(f"🧮 Embeddings générés pour {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"❌ Erreur embardings: {str(e)}")
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

        success_rate = (success_count / len(chunks)) * 100
        logger.info(f"✅ {success_count}/{len(chunks)} chunks sauvegardés ({success_rate:.1f}%)")

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
    logger.info(f"   ✅ Traités avec succès: {processed}")
    logger.info(f"   ❌ Erreurs: {errors}")
    logger.info(
        f"   📊 Taux de succès: {(processed / (processed + errors) * 100):.1f}%" if (processed + errors) > 0 else "")

    processor.extractor.logout()


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 450
    main(start, batch)
