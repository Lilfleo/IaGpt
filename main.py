#!/usr/bin/env python3
from scripts.filemaker_extractor import FileMakerExtractor


def test_connection():
    print("🔍 Test de connexion FileMaker...")
    extractor = FileMakerExtractor()

    if extractor.login():
        print("✅ Connexion OK")
        documents = extractor.get_documents()
        if documents:
            print(f"📄 {len(documents)} documents trouvés")
            # Test extraction du premier PDF
            if len(documents) > 0:
                record_id = documents[0]['recordId']
                pdf_path = extractor.extract_pdf(record_id, "test")
                if pdf_path:
                    print(f"✅ Test PDF extrait: {pdf_path}")
        extractor.logout()
    else:
        print("❌ Connexion échouée")


if __name__ == "__main__":
    test_connection()
