# KNN Projekt - Potlačení halucinací ASR modelu Whisper Large-v3

## Složky a soubory
* final_technical_report.pdf - Výsledná technická zpráva.
* final_presentation_slides.pdf - Slidy k prezentaci (obhajobě) projektu.
* article_presentation_slides.pdf - Slidy k prezentaci článku v průběhu semestru.
* /src/ - Zdrojové kódy v jazyce Python (Pozor! vše je psáno tak, aby se zpouštělo z kořenového adresáře)
    * solution.py - Impementace třídy WhisperLargeV3Wrapped, obsahující implementované metody pro potlačení halucinací.
    * main.py - Skript pro spuštění samotného experimentu včetně uložení výstupů.
    * data_augmentation.py - Obsahuje metody na augmentaci dat.
    * hallucination_detection.py - obsahuje metody pro detekci halucinací.
    * plot_results.py - Skript pro vykreslení výsledků experimentu (pracuje s výsledky ve složce /results_stash/).
    * transcription.py - Skript pro přepis zvukové stopy vybraných youtube videí (v konečném řešení nevyužito).
    * requirements.txt - Seznam požadovaných Python balíčků.
* /data/ - Složka obsahující data se kterými pracujeme. - Součástí odevzdání je jen malá podmnožina celého datasetu.
* /results_stash/ - Podsložky obsahují výstupy experimentů, dále se zde nachází různé mezivýsledky z naší práce.

## Návod na otestování projektu (výsledný experiment).
1. Nainstalujte potřebné balíčky (`pip install -r src/requirements.txt`).
2. Stáhněte si náš repozitář namísto odevzdaného zipu (chcete-li celý dataset).
3. Spusťte hlavní skript z kořenové složky (`python3 src/main.py`) - pozor, tímto se přepíšou výsledky experimentu.
4. Volitelně můžete také vygenerovat grafy z výsledků spuštěním (`python3 src/plot_results.py`).
