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
        extractor.logout()
    else:
        print("❌ Connexion échouée")


if __name__ == "__main__":
    test_connection()
