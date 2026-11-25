THA FILE CSV VAO DAY NEU MUON CHAY RIENG TREN CSV DA TIEN XU LY (D0..D4, ground_truth, clean_sales)!
Hoac chay truc tiep tren data/raw/ cua repo (giong Random Forest):
  python models/HGB/run.py validate --config models/HGB/configs/hgb.json --data-dir data/raw --output-dir reports/hgb/validate_001
  python models/HGB/run.py run      --config models/HGB/configs/hgb.json --data-dir data/raw --output-dir reports/hgb/run_001
