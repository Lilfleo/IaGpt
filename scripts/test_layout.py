#!/usr/bin/env python3
"""
Diagnostic des noms de champs FileMaker - VERSION SIMPLE
"""
from filemaker_extractor import FileMakerExtractor


def diagnose_fields():
    extractor = FileMakerExtractor()
    if not extractor.login():
        print("❌ Connexion échouée")
        return

    print("🔍 DIAGNOSTIC STRUCTURE FILEMAKER")
    print("=" * 40)

    # Utilise une méthode qui existe déjà
    try:
        # Essaye avec un document existant
        documents = extractor.get_documents()
        if documents:
            first_doc = documents[0]
            doc_id = first_doc['recordId']

            # Récupère les chunks de ce document
            chunks = extractor.get_chunks_for_document(doc_id)

            if chunks:
                first_chunk = chunks[0]
                print("📋 STRUCTURE DU PREMIER CHUNK :")
                print(f"   recordId: {first_chunk.get('recordId')}")
                print(f"   fieldData keys: {list(first_chunk.get('fieldData', {}).keys())}")
                print()

                # Affiche tous les champs
                field_data = first_chunk.get('fieldData', {})
                for field_name, field_value in field_data.items():
                    if isinstance(field_value, str) and len(field_value) > 100:
                        value_preview = field_value[:100] + "..."
                    else:
                        value_preview = str(field_value) if field_value else "VIDE"
                    print(f"   🏷️ {field_name}: {value_preview}")

                print("\n" + "=" * 50)
                print("🔍 ANALYSE SPÉCIFIQUE:")

                # Vérifications spécifiques
                embedding_field = field_data.get('EmbeddingJson')
                text_field = field_data.get('Text')

                print(f"🎯 Champ 'EmbeddingJson': {'EXISTE' if 'EmbeddingJson' in field_data else 'MANQUANT'}")
                if embedding_field:
                    print(f"   Contenu: {'VIDE' if not embedding_field else f'{len(str(embedding_field))} caractères'}")

                print(f"🎯 Champ 'Text': {'EXISTE' if 'Text' in field_data else 'MANQUANT'}")
                if text_field:
                    print(f"   Contenu: {'VIDE' if not text_field else f'{len(str(text_field))} caractères'}")

            else:
                print("❌ Aucun chunk trouvé pour ce document")
        else:
            print("❌ Aucun document trouvé")

    except Exception as e:
        print(f"❌ Erreur: {e}")

    extractor.logout()


if __name__ == "__main__":
    diagnose_fields()
