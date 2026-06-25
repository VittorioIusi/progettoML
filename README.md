# progettoML — Miglioramento di TabPFN-2.5 su classificazione binaria

Progetto universitario: migliorare le prestazioni di **TabPFN-2.5** su task di
classificazione binaria combinando modifiche esterne (preprocessing, feature
selection, ensembling, post-processing) con un **LoRA fine-tuning** leggero sui
layer di attenzione, senza riaddestrare i pesi principali del modello.

## Struttura

```
.
├── utils/data_loader.py      # Caricamento dataset OpenML (cache, encoding, split)
├── evaluation/metrics.py     # AUC-ROC, F1 macro, Brier, ECE + tabelle/CSV
├── models/lora.py            # Adapter LoRA per i layer di attenzione di TabPFN
├── TabPFN-main/              # Sorgente di TabPFN (pacchetto `tabpfn`, v8.0.8)
├── data/raw/                 # Cache dataset (ignorata da git)
├── results/                  # Output esperimenti (ignorata da git)
├── requirements.txt
└── .env.example              # Template per TABPFN_TOKEN (la chiave NON va committata)
```

## Setup su Google Colab (GPU)

1. **Runtime → Cambia tipo di runtime → GPU**.
2. Clona il repo e installa le dipendenze:
   ```python
   !git clone https://github.com/VittorioIusi/progettoML.git
   %cd progettoML
   !pip install -q -e ./TabPFN-main
   !pip install -q -r requirements.txt
   ```
3. Imposta il token Prior Labs (necessario solo al primo download dei pesi).
   Aggiungilo nei **Secrets** di Colab (icona 🔑) con nome `TABPFN_TOKEN`, poi:
   ```python
   import os
   from google.colab import userdata
   os.environ["TABPFN_TOKEN"] = userdata.get("TABPFN_TOKEN")
   ```
4. Verifica accesso al modello v2.5:
   ```python
   from tabpfn import TabPFNClassifier
   from tabpfn.constants import ModelVersion
   clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5)
   clf._initialize_model_variables()
   print("blocchi:", len(clf.model_.blocks))   # atteso 24
   ```

## Sviluppo in locale (CPU)

I moduli `utils`, `evaluation` e `models` sono testabili su CPU senza GPU ne'
token; ognuno ha un blocco `if __name__ == "__main__"` con i propri test:

```bash
python utils/data_loader.py
python evaluation/metrics.py
python models/lora.py
```

## Dataset

- **Fine-tuning LoRA** (medici): diabetes (37), breast_cancer (1510),
  heart_disease (53), chronic_kidney (42972), hepatitis (55).
- **Valutazione**: thyroid (40082), adult (1590), credit_g (31),
  blood_transfusion (1464).

> Nota: per `chronic_kidney` si usa l'OpenML ID **42972** (`chronic-kidney-disease`).
> L'ID 40922 inizialmente indicato corrisponde a un dataset diverso
> (`Run_or_walk_information`) e non va usato.
