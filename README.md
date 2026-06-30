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

## Dataset ed esperimenti LoRA

La configurazione del fine-tuning LoRA è stata spostata dal **dominio medico**
(dataset troppo piccoli, 155–768 righe) a un **dominio finanziario/creditizio**,
con due obiettivi: (1) usare dataset abbastanza grandi da fornire molti passi di
gradiente; (2) valutare **in-domain** (finanziario → finanziario) invece che
cross-domain, così da rispondere correttamente alla domanda *"il LoRA
specializza TabPFN su uno specifico settore?"*.

- **Fine-tuning LoRA** (finanziari): `bank_marketing` (1461, 45.211),
  `default_credit` (42477, 30.000), `polish_bankruptcy_2` (42984, 10.173),
  `polish_bankruptcy_3` (42985, 10.503), `polish_bankruptcy_4` (42986, 9.792).
- **Valutazione** (finanziari, in-domain, mai visti in training):
  `polish_bankruptcy_1` (42880, 7.027), `polish_bankruptcy_5` (42987, 5.910),
  `australian_credit` (40981, 690), `credit_g` (31, 1.000).

Al caricamento, ogni dataset stampa una riga di **verifica** con nome, numero di
righe/feature e distribuzione del target (presidio contro gli ID errati).

### Esperimenti

1. **Esp. 1 — generalizzazione in-domain** (`run_lora_exp5`): un LoRA allenato
   sui 5 finanziari, valutato sui 4 di test finanziari.
2. **Esp. 2 — ablation sulla capacità** (`run_lora_ablation`): r8/r16/QKVO, ora
   eseguito sui nuovi dataset finanziari.
3. **Esp. 3 — ablation sulla dimensione dei dati** (`run_lora_datasize_financial`):
   usa `gmsc` (GiveMeSomeCredit, 45577, ~150.000 righe), test set **fisso** di
   8.000 righe e training set crescenti (500 → 40.000) campionati stratificati;
   produce tabella + grafico della curva AUC vs dimensione (base vs LoRA).
   Sostituisce il vecchio esperimento a singolo `cardiovascular`.

> **ID OpenML corretti** (diversi ID iniziali erano errati — verificare sempre):
> - Polish bankruptcy: gli ID 40474–40478 erano dataset `thyroid` a **5 classi**;
>   i corretti `polish-bankruptcy-Nyear` sono **42880/42984/42985/42986/42987**.
> - `gmsc`: l'ID 44089 aveva solo 16.714 righe (insufficiente per training da
>   40k); quello giusto è **45577** (`Give-Me-Some-Credit`, 150.000 righe).
> - (storici, configurazione medica) `chronic_kidney` → 42972, `thyroid` → 1000.

> ⚠️ **Versione di TabPFN per il LoRA.** L'iniezione degli adapter richiede
> TabPFN **8.0.8** (attenzione con `q_projection`/`v_projection` separati). Le
> versioni più vecchie (es. 7.1.1) fondono le proiezioni in un unico parametro
> `_w_qkv` e l'iniezione fallisce. Su Colab è già 8.0.8; in locale assicurati di
> installare `./TabPFN-main` (`pip install -e ./TabPFN-main`).
