# KNN Projekt - Potlačení halucinací modelu Whisper Large-v3 - Checkpoint

## Složky a soubory
* checkpoint_report.pdf - Technická zpráva (verze pro checkpoint).
* /src/ - Zdrojové kódy v jazyce Python
    * data_augmentation.py - Obsahuje metody na augmentaci dat.
    * hallucination_detection.py - obsahuje metody pro detekci halucinací.
    * transcription.py - Skript pro přepis zvukové stopy vybraných youtube videí (aktuálně nevyužito).
    * baseline_solution.py - Skript pro otestování aktuálního baseline řešení.
    * requirements.txt - Seznam požadovaných Python balíčků.
* /data/ - Složka obsahující data se kterými aktuálně pracujeme. - Součástí odevzdání
je jen malá podmnožina kterou skript baseline_solution.py využije, zbytek je dostupný
v repozitáři (případně v nejnovějším commitu ke dni 1.4.2024).
* /tmp/ - Složka kde skript ukládá augmentované nahrávky ze složky data (pro případnou manuální kontrolu).

## Návod na otestování aktuálního (checkpoint) stavu projektu.
1. Nainstalujte potřebné balíčky (`pip install -r src/requirements.txt`).
2. Spusťte testovací skript z kořenové složky (`python3 src/baseline_solution.py`).
3. Volitelné - Poslechněte si augmentované nahrávky ve složce `/tmp`.
4. Na výstupu (stdout) lze vidět přepisy označené jako podezřelé (či jako obsahující běžné halucinace), je velice možné že v něm najdete
zřetelné halucinace modelu.
