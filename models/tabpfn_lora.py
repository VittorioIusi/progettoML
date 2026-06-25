"""
tabpfn_lora.py
==============

Integrazione tra il modulo :mod:`models.lora` e TabPFN-2.5: costruzione di un
classificatore con adapter LoRA iniettati e un **loop di training trasparente**
che allena *solo* gli adapter, lasciando congelati i pesi pre-addestrati.

Approccio (B): invece di usare la "scatola nera" ``FinetunedTabPFNClassifier``,
scriviamo il nostro ciclo di ottimizzazione. Riutilizziamo pero' le primitive di
preprocessing collaudate di TabPFN (``get_preprocessed_dataset_chunks``,
``meta_dataset_collator``, ``fit_from_preprocessed``, ``forward``), perche' e' li'
che si annida la complessita' del modello in-context. Cosi' otteniamo:

    * controllo completo del loop (epoche, loss, quali parametri si allenano);
    * il preprocessing corretto del modello, senza reimplementarlo.

Meccanica di un passo (identica a quella del fine-tuner ufficiale):

    1. ``clf.fit_from_preprocessed(context...)``  -> imposta il contesto in-context
    2. ``logits = clf.forward(X_query, return_raw_logits=True)``  -> shape (Q,B,E,L)
    3. cross-entropy tra i logits delle query e le label vere
    4. backward + step  (solo sui parametri LoRA)

NOTA: questo modulo richiede ``tabpfn`` installato e va eseguito su GPU
(Colab). Gli import di ``tabpfn`` sono "lazy" (dentro le funzioni) cosi' il file
resta importabile anche in un ambiente senza tabpfn (per controlli di sintassi).
"""

from __future__ import annotations

from functools import partial
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Import del nostro modulo LoRA (torch-only). Gestiamo sia l'uso come package
# (``from models.tabpfn_lora import ...``) sia l'esecuzione diretta.
try:
    from .lora import (
        LoRAConfig,
        count_trainable_parameters,
        inject_lora_adapters,
        lora_state_dict,
        mark_only_lora_as_trainable,
        save_lora_adapters,
    )
except ImportError:  # pragma: no cover - fallback per esecuzione come script
    from lora import (  # type: ignore
        LoRAConfig,
        count_trainable_parameters,
        inject_lora_adapters,
        lora_state_dict,
        mark_only_lora_as_trainable,
        save_lora_adapters,
    )


# ---------------------------------------------------------------------------
# Costruzione del classificatore con LoRA
# ---------------------------------------------------------------------------

def create_lora_classifier(
    lora_config: LoRAConfig,
    *,
    device: str = "cuda",
    n_estimators: int = 2,
    random_state: int = 42,
) -> Tuple[Any, List[str]]:
    """Crea un ``TabPFNClassifier`` v2.5 con adapter LoRA iniettati e congelati.

    Il classificatore viene creato in modalita' ``batched`` (necessaria per il
    fine-tuning con gradienti), i pesi vengono caricati, gli adapter LoRA
    iniettati nei layer di attenzione e tutti i parametri non-LoRA congelati.

    Args:
        lora_config: Configurazione degli adapter LoRA.
        device: Device su cui collocare il modello (``"cuda"`` su Colab).
        n_estimators: Numero di membri dell'ensemble usati durante il
            fine-tuning (default ``2``, come il fine-tuner ufficiale).
        random_state: Seed per la riproducibilita'.

    Returns:
        Tupla ``(clf, injected)`` dove ``clf`` e' il ``TabPFNClassifier`` pronto
        per il training e ``injected`` la lista dei moduli adattati.
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    clf = TabPFNClassifier.create_default_for_version(
        version=ModelVersion.V2_5,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        fit_mode="batched",
        differentiable_input=False,
        ignore_pretraining_limits=True,
    )
    # Carica i pesi pre-addestrati e materializza il modello PyTorch.
    clf._initialize_model_variables()

    net = clf.model_
    injected = inject_lora_adapters(net, lora_config)
    mark_only_lora_as_trainable(net)
    net.to(device)

    return clf, injected


# ---------------------------------------------------------------------------
# Helper interni
# ---------------------------------------------------------------------------

def _should_skip_batch(batch: Any) -> bool:
    """True se qualche label delle query non e' presente nel contesto.

    La cross-entropy in-context richiede che ogni classe da predire compaia tra
    le label di contesto; in caso contrario il batch va saltato (come nel
    fine-tuner ufficiale).
    """
    ctx_unique = torch.unique(
        torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_context])
    )
    qry_unique = torch.unique(
        torch.cat([torch.unique(t.reshape(-1)) for t in batch.y_query])
    )
    query_in_context = torch.isin(qry_unique, ctx_unique)
    return not bool(query_in_context.all())


# ---------------------------------------------------------------------------
# Loop di training (solo adapter LoRA)
# ---------------------------------------------------------------------------

def train_lora(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    device: str = "cuda",
    n_ctx_plus_query: int = 10_000,
    query_ratio: float = 0.2,
    grad_clip: float | None = 1.0,
    random_state: int = 42,
    early_stopping: bool = True,
    es_val_ratio: float = 0.2,
    patience: int = 10,
    eval_every: int = 1,
    n_estimators_eval: int = 2,
    verbose: bool = True,
) -> List[float]:
    """Allena i soli adapter LoRA di ``clf``, con early stopping su validation.

    Per ogni epoca i dati vengono ri-mescolati, preprocessati con le primitive
    di TabPFN e suddivisi in batch contesto/query; per ogni batch si calcola la
    cross-entropy sulle query e si aggiornano unicamente i parametri LoRA.

    Se ``early_stopping`` e' True, una porzione di ``X_train`` viene tenuta da
    parte (split interno) e usata per valutare l'AUC ad ogni epoca: si tengono i
    pesi LoRA con AUC migliore e si interrompe dopo ``patience`` epoche senza
    miglioramento. Questo split interno e' **separato** dall'eventuale validation
    set usato dal chiamante per il confronto finale (niente data leakage).

    Args:
        clf: Il ``TabPFNClassifier`` con LoRA iniettato (output di
            :func:`create_lora_classifier`).
        X_train: Feature di training, shape ``(n_samples, n_features)``.
        y_train: Label di training (0/1), shape ``(n_samples,)``.
        epochs: Numero massimo di epoche.
        learning_rate: Learning rate di AdamW (default ``1e-4`` per stabilita').
        weight_decay: Weight decay di AdamW.
        device: Device di calcolo.
        n_ctx_plus_query: Numero massimo di campioni per meta-dataset
            (contesto + query) prima dello split.
        query_ratio: Frazione di ogni meta-dataset usata come query per la loss.
        grad_clip: Norma massima per il gradient clipping (``None`` per
            disabilitarlo).
        random_state: Seed base (per epoca si usa ``random_state + epoch``).
        early_stopping: Se True, abilita la validazione per-epoca e l'early
            stopping con ripristino dei pesi migliori.
        es_val_ratio: Frazione di ``X_train`` riservata alla validazione interna
            di early stopping.
        patience: Numero di epoche senza miglioramento prima di fermarsi.
        eval_every: Ogni quante epoche valutare l'AUC di validazione.
        n_estimators_eval: Numero di estimatori usati nella valutazione di
            early stopping (basso = piu' veloce).
        verbose: Se True, stampa loss e AUC per epoca.

    Returns:
        Lista delle loss medie di training per epoca.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from tabpfn import TabPFNClassifier
    from tabpfn.architectures.interface import PerformanceOptions
    from tabpfn.finetuning.data_util import (
        get_preprocessed_dataset_chunks,
        meta_dataset_collator,
    )
    from tabpfn.finetuning.train_util import clone_model_for_evaluation

    net = clf.model_
    net.to(device)

    # Ottimizzatore sui SOLI parametri allenabili (gli adapter LoRA).
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "Nessun parametro allenabile: hai iniettato il LoRA e chiamato "
            "mark_only_lora_as_trainable()? Usa create_lora_classifier()."
        )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate, weight_decay=weight_decay
    )

    # Niente activation checkpointing per semplicita'/robustezza alla prima
    # versione; i dataset medici sono piccoli e stanno in memoria su una T4.
    perf = PerformanceOptions(
        force_recompute_layer=False,
        use_chunkwise_inference=False,
    )

    # Split interno per l'early stopping (tiene pulito il validation del
    # chiamante). Il training avviene su X_tr; l'AUC si misura su X_es.
    if early_stopping:
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_train,
            y_train,
            test_size=es_val_ratio,
            random_state=random_state,
            stratify=y_train,
        )
    else:
        X_tr, y_tr = X_train, y_train
        X_es, y_es = None, None

    n_classes = int(len(np.unique(y_tr)))
    max_data = min(n_ctx_plus_query, len(y_tr))
    query_size = max(int(max_data * query_ratio), n_classes)

    def _validation_auc() -> float:
        """AUC del modello LoRA corrente: clona, fitta su X_tr, predice X_es."""
        eval_args = {
            "device": device,
            "n_estimators": n_estimators_eval,
            "random_state": random_state,
        }
        ev = clone_model_for_evaluation(clf, eval_args, TabPFNClassifier)
        ev.fit(X_tr, y_tr)
        proba = ev.predict_proba(X_es)[:, 1]
        return float(roc_auc_score(y_es, proba))

    history: List[float] = []
    best_auc = -float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        seed = random_state + epoch
        net.train()

        # Split contesto/query rigenerato ogni epoca con seed diverso.
        splitter = partial(
            train_test_split, test_size=query_size, random_state=seed
        )
        datasets = get_preprocessed_dataset_chunks(
            calling_instance=clf,
            X_raw=X_tr,
            y_raw=y_tr,
            split_fn=splitter,
            max_data_size=max_data,
            model_type="classifier",
            equal_split_size=False,
            data_shuffle_seed=seed,
            preprocessing_random_state=seed,
        )
        loader = torch.utils.data.DataLoader(
            datasets,
            batch_size=1,
            collate_fn=meta_dataset_collator,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            if _should_skip_batch(batch):
                continue

            optimizer.zero_grad()

            # 1) Imposta il contesto in-context (no_refit interno).
            clf.fit_from_preprocessed(
                batch.X_context,
                batch.y_context,
                batch.cat_indices,
                batch.configs,
                performance_options=perf,
            )

            # 2) Forward sulle query -> logits grezzi (Q, B, E, L).
            logits_QBEL = clf.forward(batch.X_query, return_raw_logits=True)
            Q, B, E, L = logits_QBEL.shape

            # 3) Riarrangia per la cross-entropy: (B*E, L, Q) vs target (B*E, Q).
            logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
            targets_BQ = batch.y_query.repeat(B * E, 1).to(device)
            loss = F.cross_entropy(logits_BLQ, targets_BQ)

            # 4) Backward + step (solo LoRA).
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()

            epoch_loss += float(loss.detach().item())
            n_batches += 1

        mean_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")
        history.append(mean_loss)
        msg = f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.4f}"

        # --- Early stopping su validation interno ---
        if early_stopping and (epoch + 1) % eval_every == 0:
            net.eval()
            with torch.no_grad():
                val_auc = _validation_auc()
            msg += f" | val AUC: {val_auc:.4f}"
            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_state = lora_state_dict(net)  # copia CPU dei soli pesi LoRA
                epochs_no_improve = 0
                msg += " *"
            else:
                epochs_no_improve += 1

        if verbose:
            print(msg)

        if early_stopping and epochs_no_improve >= patience:
            if verbose:
                print(
                    f"Early stopping all'epoca {epoch + 1} "
                    f"(miglior val AUC: {best_auc:.4f})"
                )
            break

    # Ripristina i pesi LoRA migliori visti durante il training.
    if early_stopping and best_state is not None:
        net.load_state_dict(best_state, strict=False)
        if verbose:
            print(f"Ripristinati i pesi LoRA migliori (val AUC: {best_auc:.4f})")

    return history


# ---------------------------------------------------------------------------
# Orchestrazione di alto livello
# ---------------------------------------------------------------------------

def finetune_lora_on_dataset(
    dataset_name: str,
    *,
    lora_config: LoRAConfig | None = None,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    n_estimators: int = 2,
    random_state: int = 42,
    save_path: str | None = None,
    verbose: bool = True,
) -> Tuple[Any, List[float]]:
    """Pipeline completa: carica un dataset di fine-tuning, inietta e allena LoRA.

    Usa :func:`utils.data_loader.get_finetune_data` per ottenere lo split, crea
    il classificatore con LoRA, allena i soli adapter e (opzionalmente) salva i
    pesi LoRA su disco.

    Args:
        dataset_name: Nome del dataset in ``FINETUNE_DATASETS`` (es.
            ``"diabetes"``).
        lora_config: Configurazione LoRA. Se ``None``, usa i default
            (``r=8, alpha=16``, target q/v).
        epochs: Numero di epoche di training.
        learning_rate: Learning rate per gli adapter.
        device: Device di calcolo.
        n_estimators: Numero di estimatori dell'ensemble in training.
        random_state: Seed.
        save_path: Se fornito, salva i pesi LoRA in questo file ``.pt``.
        verbose: Se True, stampa avanzamento e statistiche.

    Returns:
        Tupla ``(clf, history)`` con il classificatore allenato e la lista delle
        loss per epoca.
    """
    try:
        from utils.data_loader import get_finetune_data
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Impossibile importare utils.data_loader. Assicurati che la root del "
            "progetto sia nel sys.path (es. sys.path.insert(0, '/content/progettoML'))."
        ) from exc

    if lora_config is None:
        lora_config = LoRAConfig()

    X_train, y_train, X_val, y_val = get_finetune_data(
        dataset_name, random_state=random_state
    )

    clf, injected = create_lora_classifier(
        lora_config,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    if verbose:
        tr, tot, pct = count_trainable_parameters(clf.model_)
        print(
            f"Dataset '{dataset_name}': train={len(y_train)}, val={len(y_val)} | "
            f"adapter={len(injected)} | allenabili={tr:,}/{tot:,} ({pct:.3f}%)"
        )

    history = train_lora(
        clf,
        X_train,
        y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        random_state=random_state,
        verbose=verbose,
    )

    if save_path is not None:
        save_lora_adapters(clf.model_, save_path)
        if verbose:
            print(f"Pesi LoRA salvati in '{save_path}'")

    return clf, history


def evaluate_lora_vs_baseline(
    dataset_name: str,
    *,
    lora_config: LoRAConfig | None = None,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    n_estimators_train: int = 2,
    n_estimators_eval: int = 8,
    random_state: int = 42,
    save_path: str | None = None,
    verbose: bool = True,
) -> Tuple[Any, List[float]]:
    """Confronta TabPFN base vs TabPFN+LoRA su un dataset di fine-tuning.

    Esegue, nell'ordine:
        1. Valuta il TabPFN **base** (senza LoRA) sul validation set.
        2. Allena gli adapter LoRA sul training set.
        3. Clona il modello allenato per l'inferenza standard e lo valuta sullo
           stesso validation set.
        4. Costruisce e stampa la tabella di confronto con le metriche del
           progetto (AUC-ROC, F1 macro, Brier, ECE).

    Baseline e modello LoRA usano lo **stesso** numero di estimatori in
    inferenza (``n_estimators_eval``) per un confronto equo; il training del
    LoRA puo' usare meno estimatori (``n_estimators_train``) per velocita'.

    Args:
        dataset_name: Nome del dataset in ``FINETUNE_DATASETS`` (es. ``"diabetes"``).
        lora_config: Configurazione LoRA (default ``r=8, alpha=16``, target q/v).
        epochs: Numero di epoche di training degli adapter.
        learning_rate: Learning rate per gli adapter.
        device: Device di calcolo.
        n_estimators_train: Numero di estimatori usati durante il training LoRA.
        n_estimators_eval: Numero di estimatori usati in inferenza per ENTRAMBI
            i modelli (confronto equo).
        random_state: Seed.
        save_path: Se fornito, salva i pesi LoRA in questo file ``.pt``.
        verbose: Se True, stampa avanzamento, tabella di confronto e statistiche.

    Returns:
        Tupla ``(df, history)`` dove ``df`` e' il DataFrame di confronto (una riga
        per modello) e ``history`` la lista delle loss di training per epoca.
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.train_util import clone_model_for_evaluation

    try:
        from utils.data_loader import get_finetune_data
        from evaluation.metrics import (
            evaluate_model,
            evaluate_multiple_datasets,
            print_comparison_table,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Assicurati che la root del progetto sia nel sys.path "
            "(es. sys.path.insert(0, '/content/progettoML'))."
        ) from exc

    if lora_config is None:
        lora_config = LoRAConfig()

    X_train, y_train, X_val, y_val = get_finetune_data(
        dataset_name, random_state=random_state
    )

    # --- 1. Baseline: TabPFN v2.5 senza LoRA ------------------------------
    if verbose:
        print(f"\n[1/3] Valuto TabPFN base su '{dataset_name}'...")
    base = TabPFNClassifier.create_default_for_version(
        version=ModelVersion.V2_5,
        device=device,
        n_estimators=n_estimators_eval,
        random_state=random_state,
    )
    base.fit(X_train, y_train)
    prob_base = base.predict_proba(X_val)[:, 1]
    res_base = evaluate_model(y_val, prob_base, "TabPFN-base", dataset_name)

    # Libera memoria GPU prima del training.
    del base
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. Training degli adapter LoRA -----------------------------------
    if verbose:
        print(f"\n[2/3] Alleno gli adapter LoRA ({epochs} epoche)...")
    clf, injected = create_lora_classifier(
        lora_config,
        device=device,
        n_estimators=n_estimators_train,
        random_state=random_state,
    )
    if verbose:
        tr, tot, pct = count_trainable_parameters(clf.model_)
        print(f"  adapter={len(injected)} | allenabili={tr:,}/{tot:,} ({pct:.3f}%)")

    history = train_lora(
        clf,
        X_train,
        y_train,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        random_state=random_state,
        n_estimators_eval=n_estimators_train,
        verbose=verbose,
    )

    # --- 3. Valutazione del modello LoRA (clone per inferenza) ------------
    if verbose:
        print(f"\n[3/3] Valuto TabPFN+LoRA su '{dataset_name}'...")
    eval_args = {
        "device": device,
        "n_estimators": n_estimators_eval,
        "random_state": random_state,
    }
    lora_infer = clone_model_for_evaluation(clf, eval_args, TabPFNClassifier)
    lora_infer.fit(X_train, y_train)
    prob_lora = lora_infer.predict_proba(X_val)[:, 1]
    res_lora = evaluate_model(y_val, prob_lora, "TabPFN-LoRA", dataset_name)

    # --- 4. Confronto -----------------------------------------------------
    df = evaluate_multiple_datasets([res_base, res_lora])
    if verbose:
        print_comparison_table(df)

    if save_path is not None:
        save_lora_adapters(clf.model_, save_path)
        if verbose:
            print(f"Pesi LoRA salvati in '{save_path}'")

    return df, history


def train_lora_multi(
    clf: Any,
    train_datasets: List[Tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    device: str = "cuda",
    n_ctx_plus_query: int = 10_000,
    query_ratio: float = 0.2,
    grad_clip: float | None = 1.0,
    random_state: int = 42,
    early_stopping: bool = True,
    es_val_ratio: float = 0.2,
    patience: int = 10,
    eval_every: int = 2,
    n_estimators_eval: int = 2,
    verbose: bool = True,
) -> List[float]:
    """Allena i soli adapter LoRA su PIU' dataset congiuntamente.

    Ad ogni epoca il modello vede un batch per ciascun dataset (un passo di
    ottimizzazione per dataset), cosi' gli adapter imparano qualcosa di comune
    ai dataset. L'early stopping usa la media dell'AUC sui validation interni
    (una porzione tenuta da parte per ogni dataset).

    Args:
        clf: Il ``TabPFNClassifier`` con LoRA iniettato.
        train_datasets: Lista di coppie ``(X, y)``, una per dataset.
        (gli altri argomenti sono come in :func:`train_lora`.)

    Returns:
        Lista delle loss medie di training per epoca (media sui batch/dataset).
    """
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from tabpfn import TabPFNClassifier
    from tabpfn.architectures.interface import PerformanceOptions
    from tabpfn.finetuning.data_util import (
        get_preprocessed_dataset_chunks,
        meta_dataset_collator,
    )
    from tabpfn.finetuning.train_util import clone_model_for_evaluation

    net = clf.model_
    net.to(device)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("Nessun parametro allenabile (LoRA iniettato?).")
    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate, weight_decay=weight_decay
    )
    perf = PerformanceOptions(
        force_recompute_layer=False, use_chunkwise_inference=False
    )

    # Split train/early-stopping per ciascun dataset.
    train_parts: List[Tuple[np.ndarray, np.ndarray]] = []
    es_parts: List[Tuple[np.ndarray, np.ndarray]] = []
    for X, y in train_datasets:
        if early_stopping:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X, y, test_size=es_val_ratio, random_state=random_state, stratify=y
            )
            train_parts.append((X_tr, y_tr))
            es_parts.append((X_es, y_es))
        else:
            train_parts.append((X, y))

    X_list = [p[0] for p in train_parts]
    y_list = [p[1] for p in train_parts]

    def _mean_val_auc() -> float:
        eval_args = {
            "device": device,
            "n_estimators": n_estimators_eval,
            "random_state": random_state,
        }
        aucs = []
        for (X_tr, y_tr), (X_es, y_es) in zip(train_parts, es_parts):
            ev = clone_model_for_evaluation(clf, eval_args, TabPFNClassifier)
            ev.fit(X_tr, y_tr)
            proba = ev.predict_proba(X_es)[:, 1]
            aucs.append(roc_auc_score(y_es, proba))
        return float(np.mean(aucs))

    history: List[float] = []
    best_auc = -float("inf")
    best_state: dict | None = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        seed = random_state + epoch
        net.train()

        # Un batch per dataset: split contesto/query per frazione.
        splitter = partial(train_test_split, test_size=query_ratio, random_state=seed)
        datasets = get_preprocessed_dataset_chunks(
            calling_instance=clf,
            X_raw=X_list,
            y_raw=y_list,
            split_fn=splitter,
            max_data_size=n_ctx_plus_query,
            model_type="classifier",
            equal_split_size=False,
            data_shuffle_seed=seed,
            preprocessing_random_state=seed,
        )
        loader = torch.utils.data.DataLoader(
            datasets,
            batch_size=1,
            collate_fn=meta_dataset_collator,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            if _should_skip_batch(batch):
                continue
            optimizer.zero_grad()
            clf.fit_from_preprocessed(
                batch.X_context,
                batch.y_context,
                batch.cat_indices,
                batch.configs,
                performance_options=perf,
            )
            logits_QBEL = clf.forward(batch.X_query, return_raw_logits=True)
            Q, B, E, L = logits_QBEL.shape
            logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)
            targets_BQ = batch.y_query.repeat(B * E, 1).to(device)
            loss = F.cross_entropy(logits_BLQ, targets_BQ)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1

        mean_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")
        history.append(mean_loss)
        msg = f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.4f} ({n_batches} batch)"

        if early_stopping and (epoch + 1) % eval_every == 0:
            net.eval()
            with torch.no_grad():
                val_auc = _mean_val_auc()
            msg += f" | media val AUC: {val_auc:.4f}"
            if val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_state = lora_state_dict(net)
                epochs_no_improve = 0
                msg += " *"
            else:
                epochs_no_improve += 1

        if verbose:
            print(msg)

        if early_stopping and epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping all'epoca {epoch + 1} (best AUC: {best_auc:.4f})")
            break

    if early_stopping and best_state is not None:
        net.load_state_dict(best_state, strict=False)
        if verbose:
            print(f"Ripristinati i pesi LoRA migliori (media val AUC: {best_auc:.4f})")

    return history


def run_lora_exp5(
    *,
    finetune_names: List[str] | None = None,
    eval_names: List[str] | None = None,
    lora_config: LoRAConfig | None = None,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    n_estimators_train: int = 2,
    n_estimators_eval: int = 8,
    random_state: int = 42,
    save_path: str | None = None,
    verbose: bool = True,
) -> Tuple[Any, List[float]]:
    """Esperimento 5: LoRA addestrato sui dataset medici, valutato su quelli di test.

    Allena UN solo LoRA congiuntamente sui dataset di fine-tuning (medici) e poi
    confronta TabPFN base vs TabPFN+LoRA sui dataset di valutazione (mai visti in
    training), misurando cosi' la **generalizzazione** dell'adattamento LoRA.

    Args:
        finetune_names: Nomi dei dataset di training (default: tutti i
            ``FINETUNE_DATASETS``).
        eval_names: Nomi dei dataset di valutazione (default: tutti gli
            ``EVALUATION_DATASETS``).
        lora_config: Configurazione LoRA.
        epochs: Epoche di training.
        learning_rate: Learning rate degli adapter.
        device: Device di calcolo.
        n_estimators_train: Estimatori usati in training/early-stopping.
        n_estimators_eval: Estimatori usati nella valutazione finale.
        random_state: Seed.
        save_path: Se fornito, salva i pesi LoRA in questo file ``.pt``.
        verbose: Se True, stampa avanzamento e tabella finale.

    Returns:
        Tupla ``(df, history)`` con il DataFrame di confronto (due righe per
        dataset di valutazione: base e LoRA) e la curva di loss del training.
    """
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning.train_util import clone_model_for_evaluation

    try:
        from utils.data_loader import (
            EVALUATION_DATASETS,
            FINETUNE_DATASETS,
            get_evaluation_data,
            get_finetune_data,
        )
        from evaluation.metrics import (
            evaluate_model,
            evaluate_multiple_datasets,
            print_comparison_table,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Assicurati che la root del progetto sia nel sys.path "
            "(es. sys.path.insert(0, '/content/progettoML'))."
        ) from exc

    if finetune_names is None:
        finetune_names = list(FINETUNE_DATASETS)
    if eval_names is None:
        eval_names = list(EVALUATION_DATASETS)
    if lora_config is None:
        lora_config = LoRAConfig()

    # --- 1. Carica i dataset medici di training --------------------------
    if verbose:
        print(f"Carico {len(finetune_names)} dataset di fine-tuning: {finetune_names}")
    train_datasets = []
    for name in finetune_names:
        X_tr, y_tr, _, _ = get_finetune_data(name, random_state=random_state)
        train_datasets.append((X_tr, y_tr))

    # --- 2. Crea e allena il LoRA congiunto ------------------------------
    clf, injected = create_lora_classifier(
        lora_config,
        device=device,
        n_estimators=n_estimators_train,
        random_state=random_state,
    )
    if verbose:
        tr, tot, pct = count_trainable_parameters(clf.model_)
        print(f"adapter={len(injected)} | allenabili={tr:,}/{tot:,} ({pct:.3f}%)")
        print(f"Alleno UN LoRA su {len(train_datasets)} dataset medici insieme...\n")

    history = train_lora_multi(
        clf,
        train_datasets,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        random_state=random_state,
        n_estimators_eval=n_estimators_train,
        verbose=verbose,
    )

    if save_path is not None:
        save_lora_adapters(clf.model_, save_path)
        if verbose:
            print(f"\nPesi LoRA salvati in '{save_path}'")

    # --- 3. Confronto base vs LoRA sui dataset di valutazione ------------
    results = []
    eval_args = {
        "device": device,
        "n_estimators": n_estimators_eval,
        "random_state": random_state,
    }
    for name in eval_names:
        if verbose:
            print(f"\nValuto base vs LoRA su '{name}' (mai visto in training)...")
        X_tr, y_tr, X_te, y_te = get_evaluation_data(name, random_state=random_state)

        # Baseline
        base = TabPFNClassifier.create_default_for_version(
            version=ModelVersion.V2_5,
            device=device,
            n_estimators=n_estimators_eval,
            random_state=random_state,
        )
        base.fit(X_tr, y_tr)
        prob_base = base.predict_proba(X_te)[:, 1]
        results.append(evaluate_model(y_te, prob_base, "TabPFN-base", name))
        del base
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # LoRA (stessi adapter, dataset di test come contesto)
        lora_infer = clone_model_for_evaluation(clf, eval_args, TabPFNClassifier)
        lora_infer.fit(X_tr, y_tr)
        prob_lora = lora_infer.predict_proba(X_te)[:, 1]
        results.append(evaluate_model(y_te, prob_lora, "TabPFN-LoRA", name))
        del lora_infer
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = evaluate_multiple_datasets(results)
    if verbose:
        print_comparison_table(df)

    return df, history


if __name__ == "__main__":
    # Questo modulo richiede tabpfn + GPU e va eseguito su Colab.
    # In locale serve solo a verificare che la sintassi sia corretta.
    print(
        "tabpfn_lora.py: modulo di training LoRA per TabPFN-2.5.\n"
        "Eseguire su Colab. Esempio:\n"
        "    from models.tabpfn_lora import finetune_lora_on_dataset\n"
        "    from models.lora import LoRAConfig\n"
        "    clf, hist = finetune_lora_on_dataset('diabetes', epochs=10)\n"
    )
