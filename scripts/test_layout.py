#!/usr/bin/env python3
"""
Diagnostic des noms de champs FileMaker
"""
from filemaker_extractor import FileMakerExtractor


def diagnose_fields():
    extractor = FileMakerExtractor()
    if not extractor.login():
        print("❌ Connexion échouée")
        return

    print("🔍 DIAGNOSTIC STRUCTURE FILEMAKER")
    print("=" * 40)

    # Récupère UN chunk pour voir la structure
    chunks = extractor.get_all_chunks()
    if chunks:
        first_chunk = chunks[0]
        print("📋 STRUCTURE DU PREMIER CHUNK :")
        print(f"   recordId: {first_chunk.get('recordId')}")
        print(f"   fieldData keys: {list(first_chunk.get('fieldData', {}).keys())}")
        print()

        # Affiche tous les champs
        field_data = first_chunk.get('fieldData', {})
        for field_name, field_value in field_data.items():
            value_preview = str(field_value)[:100] if field_value else "VIDE"
            print(f"   🏷️ {field_name}: {value_preview}")

    else:
        print("❌ Aucun chunk trouvé")

    extractor.logout()


if __name__ == "__main__":
    diagnose_fields()
