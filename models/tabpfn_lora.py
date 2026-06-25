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
        mark_only_lora_as_trainable,
        save_lora_adapters,
    )
except ImportError:  # pragma: no cover - fallback per esecuzione come script
    from lora import (  # type: ignore
        LoRAConfig,
        count_trainable_parameters,
        inject_lora_adapters,
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
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cuda",
    n_ctx_plus_query: int = 10_000,
    query_ratio: float = 0.2,
    grad_clip: float | None = 1.0,
    random_state: int = 42,
    verbose: bool = True,
) -> List[float]:
    """Allena i soli adapter LoRA di ``clf`` sui dati forniti.

    Per ogni epoca i dati vengono ri-mescolati, preprocessati con le primitive
    di TabPFN e suddivisi in batch contesto/query; per ogni batch si calcola la
    cross-entropy sulle query e si aggiornano unicamente i parametri LoRA.

    Args:
        clf: Il ``TabPFNClassifier`` con LoRA iniettato (output di
            :func:`create_lora_classifier`).
        X_train: Feature di training, shape ``(n_samples, n_features)``.
        y_train: Label di training (0/1), shape ``(n_samples,)``.
        epochs: Numero di epoche.
        learning_rate: Learning rate dell'ottimizzatore AdamW. Per il LoRA si
            usano valori piu' alti che per il full fine-tuning (default ``1e-3``).
        weight_decay: Weight decay di AdamW.
        device: Device di calcolo.
        n_ctx_plus_query: Numero massimo di campioni per meta-dataset
            (contesto + query) prima dello split.
        query_ratio: Frazione di ogni meta-dataset usata come query per la loss.
        grad_clip: Norma massima per il gradient clipping (``None`` per
            disabilitarlo).
        random_state: Seed base (per epoca si usa ``random_state + epoch``).
        verbose: Se True, stampa la loss media a fine epoca.

    Returns:
        Lista delle loss medie per epoca (utile per tracciare la curva).
    """
    from sklearn.model_selection import train_test_split

    from tabpfn.architectures.interface import PerformanceOptions
    from tabpfn.finetuning.data_util import (
        get_preprocessed_dataset_chunks,
        meta_dataset_collator,
    )

    net = clf.model_
    net.to(device)
    net.train()

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

    n_classes = int(len(np.unique(y_train)))
    max_data = min(n_ctx_plus_query, len(y_train))
    query_size = max(int(max_data * query_ratio), n_classes)

    history: List[float] = []

    for epoch in range(epochs):
        seed = random_state + epoch

        # Split contesto/query rigenerato ogni epoca con seed diverso.
        splitter = partial(
            train_test_split, test_size=query_size, random_state=seed
        )
        datasets = get_preprocessed_dataset_chunks(
            calling_instance=clf,
            X_raw=X_train,
            y_raw=y_train,
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
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} - loss media: {mean_loss:.4f} "
                f"({n_batches} batch)"
            )

    return history


# ---------------------------------------------------------------------------
# Orchestrazione di alto livello
# ---------------------------------------------------------------------------

def finetune_lora_on_dataset(
    dataset_name: str,
    *,
    lora_config: LoRAConfig | None = None,
    epochs: int = 10,
    learning_rate: float = 1e-3,
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
