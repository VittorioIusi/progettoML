"""
lora.py
=======

Implementazione leggera di **LoRA** (Low-Rank Adaptation) per i layer di
attenzione di TabPFN-2.5, senza dipendenze esterne oltre a PyTorch.

Idea di base
------------
Un layer lineare ``y = W x`` viene adattato aggiungendo un termine a basso
rango::

    y = W x + (alpha / r) * (B A) x

dove ``A`` ha shape ``(r, in)`` e ``B`` ha shape ``(out, r)``, con ``r``
(il rango) molto piccolo. I pesi originali ``W`` restano **congelati**: si
allenano solo ``A`` e ``B``. All'inizializzazione ``B = 0``, quindi il modello
parte numericamente identico a quello pre-addestrato e l'adattamento emerge
solo durante il fine-tuning.

Nel caso di TabPFN-2.5 i target naturali sono le proiezioni
``q_projection`` / ``k_projection`` / ``v_projection`` / ``out_projection``
(tutte ``nn.Linear(192, 192, bias=False)``) presenti in ogni blocco dentro:

    * ``per_sample_attention_between_features``  (AlongRowAttention)
    * ``per_column_attention_between_cells``     (AlongColumnAttention)

Vedi ``tabpfn.architectures.tabpfn_v2_5``.

Questo modulo e' indipendente da TabPFN: opera su un qualsiasi ``nn.Module``
individuando gli ``nn.Linear`` il cui nome foglia compare in
``LoRAConfig.target_modules``. Puo' quindi essere testato in isolamento.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LoRAConfig:
    """Iperparametri per l'iniezione di adapter LoRA.

    Attributes:
        r: Rango delle matrici a basso rango. Valori tipici: 4, 8, 16. Piu' alto
            => piu' capacita' di adattamento ma piu' parametri. Default ``8``.
        alpha: Fattore di scala dell'adapter. Il contributo viene moltiplicato
            per ``alpha / r``. Default ``16``.
        dropout: Dropout applicato all'input del ramo LoRA durante il training.
            Default ``0.0``.
        target_modules: Nomi foglia degli ``nn.Linear`` da adattare. Per TabPFN
            i candidati sono ``"q_projection"``, ``"k_projection"``,
            ``"v_projection"``, ``"out_projection"``. Default ``("q", "v")``
            (LoRA "QV" classico).
    """

    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ("q_projection", "v_projection")

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"r deve essere un intero positivo, ricevuto {self.r}.")
        if self.alpha <= 0:
            raise ValueError(f"alpha deve essere positivo, ricevuto {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout deve essere in [0, 1), ricevuto {self.dropout}."
            )
        if len(self.target_modules) == 0:
            raise ValueError("target_modules non puo' essere vuoto.")


# ---------------------------------------------------------------------------
# Layer LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wrapper LoRA attorno a un ``nn.Linear`` esistente.

    Il layer originale viene mantenuto come sottomodulo ``base`` e congelato
    (``requires_grad=False``). Vengono aggiunte due matrici allenabili
    ``lora_A`` e ``lora_B`` che implementano il termine a basso rango.

    Forward::

        y = base(x) + scaling * (dropout(x) @ A^T) @ B^T

    con ``scaling = alpha / r``.

    Args:
        base_layer: Il ``nn.Linear`` da adattare (i suoi pesi vengono congelati).
        r: Rango dell'adapter.
        alpha: Fattore di scala (il contributo e' scalato per ``alpha / r``).
        dropout: Probabilita' di dropout sull'input del ramo LoRA.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"LoRALinear richiede un nn.Linear, ricevuto {type(base_layer).__name__}."
            )

        self.base = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        # Congela i pesi del layer originale.
        for param in self.base.parameters():
            param.requires_grad = False

        # Matrici a basso rango, sullo stesso device/dtype del layer base.
        weight = base_layer.weight
        self.lora_A = nn.Parameter(
            torch.empty((r, self.in_features), device=weight.device, dtype=weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty((self.out_features, r), device=weight.device, dtype=weight.dtype)
        )
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """Inizializza A (kaiming uniform) e B (zeri).

        Con ``B = 0`` il termine LoRA e' inizialmente nullo, quindi il modulo
        riproduce esattamente il layer base finche' non inizia il training.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applica il layer base piu' (se non gia' fuso) il termine LoRA."""
        base_out = self.base(x)
        if self.merged:
            # I pesi LoRA sono gia' stati sommati ai pesi base.
            return base_out
        lora_out = (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """Fonde i pesi LoRA dentro il layer base (per inferenza veloce).

        Dopo il merge il ramo a basso rango non viene piu' calcolato nel
        forward. Idempotente: una seconda chiamata non ha effetto.
        """
        if self.merged:
            return
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data += delta_w.to(self.base.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Annulla un precedente :meth:`merge`, ripristinando i pesi base."""
        if not self.merged:
            return
        delta_w = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data -= delta_w.to(self.base.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, merged={self.merged}"
        )


# ---------------------------------------------------------------------------
# Helper per la sostituzione dei sottomoduli
# ---------------------------------------------------------------------------

def _get_parent_and_attr(model: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    """Restituisce ``(modulo_genitore, nome_attributo)`` per un nome puntato.

    Es. ``"blocks.0.q_projection"`` -> (modulo ``blocks.0``, ``"q_projection"``).
    """
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


# ---------------------------------------------------------------------------
# Iniezione / gestione degli adapter
# ---------------------------------------------------------------------------

def inject_lora_adapters(model: nn.Module, config: LoRAConfig) -> List[str]:
    """Sostituisce gli ``nn.Linear`` target con :class:`LoRALinear`.

    Percorre ricorsivamente ``model`` e, per ogni ``nn.Linear`` il cui nome
    foglia compare in ``config.target_modules``, lo rimpiazza in-place con un
    wrapper LoRA. La modifica avviene direttamente sul modello passato.

    Args:
        model: Il modulo da adattare (es. ``TabPFNV2p5``).
        config: Configurazione LoRA.

    Returns:
        La lista dei nomi (puntati) dei moduli effettivamente adattati.

    Raises:
        ValueError: Se nessun modulo corrisponde a ``config.target_modules``.
    """
    # Raccoglie prima i target, poi sostituisce (non si modifica un modello
    # mentre lo si itera con named_modules()).
    targets: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue  # gia' adattato
        leaf = name.split(".")[-1]
        if isinstance(module, nn.Linear) and leaf in config.target_modules:
            targets.append((name, module))

    if not targets:
        raise ValueError(
            f"Nessun nn.Linear corrisponde a target_modules={config.target_modules}. "
            f"Controlla i nomi dei layer del modello."
        )

    for name, linear in targets:
        parent, attr = _get_parent_and_attr(model, name)
        setattr(
            parent,
            attr,
            LoRALinear(linear, r=config.r, alpha=config.alpha, dropout=config.dropout),
        )

    return [name for name, _ in targets]


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Congela tutti i parametri tranne quelli LoRA (``lora_A`` / ``lora_B``).

    Args:
        model: Il modello (con adapter gia' iniettati).
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Conta i parametri allenabili e totali del modello.

    Args:
        model: Il modello da ispezionare.

    Returns:
        Tupla ``(trainable, total, percentuale)`` dove ``percentuale`` e' la
        frazione di parametri allenabili sul totale, in percentuale.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = (100.0 * trainable / total) if total > 0 else 0.0
    return trainable, total, pct


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Estrae il solo state dict degli adapter LoRA.

    Utile per salvare unicamente i pesi addestrati (pochi MB) invece dell'intero
    modello.

    Args:
        model: Il modello con adapter iniettati.

    Returns:
        Dizionario ``{nome_parametro: tensore}`` contenente solo i parametri
        ``lora_A`` / ``lora_B``.
    """
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def save_lora_adapters(model: nn.Module, filepath: str) -> None:
    """Salva su disco i soli pesi LoRA.

    Args:
        model: Il modello con adapter iniettati.
        filepath: Percorso del file ``.pt`` di destinazione.
    """
    torch.save(lora_state_dict(model), filepath)


def load_lora_adapters(
    model: nn.Module, filepath: str, *, strict: bool = True
) -> None:
    """Carica pesi LoRA salvati con :func:`save_lora_adapters`.

    Args:
        model: Il modello con adapter gia' iniettati (stessa configurazione).
        filepath: Percorso del file ``.pt`` con i pesi LoRA.
        strict: Se True, richiede che le chiavi LoRA combacino esattamente.

    Raises:
        RuntimeError: Se ``strict`` e mancano/avanzano chiavi LoRA.
    """
    state = torch.load(filepath, map_location="cpu")
    incompatible = model.load_state_dict(state, strict=False)
    if strict:
        missing_lora = [k for k in incompatible.missing_keys if "lora_" in k]
        if missing_lora or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Mismatch nel caricamento LoRA: mancanti={missing_lora}, "
                f"inattesi={list(incompatible.unexpected_keys)}."
            )


def merge_lora_adapters(model: nn.Module) -> int:
    """Fonde tutti gli adapter LoRA nei rispettivi layer base.

    Da usare prima dell'inferenza per azzerare l'overhead del ramo a basso
    rango. Vedi :meth:`LoRALinear.merge`.

    Args:
        model: Il modello con adapter iniettati.

    Returns:
        Il numero di adapter fusi.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            n += 1
    return n


# ---------------------------------------------------------------------------
# Blocco di test (CPU, modello sintetico che imita TabPFN v2.5)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 70)
    print("TEST DEL MODULO lora.py (modello sintetico, CPU)")
    print("=" * 70)

    EMB = 192  # come TabPFN v2.5

    class _FakeAttention(nn.Module):
        """Imita la classe Attention di TabPFN v2.5 (q/k/v/out projection)."""

        def __init__(self) -> None:
            super().__init__()
            self.q_projection = nn.Linear(EMB, EMB, bias=False)
            self.k_projection = nn.Linear(EMB, EMB, bias=False)
            self.v_projection = nn.Linear(EMB, EMB, bias=False)
            self.out_projection = nn.Linear(EMB, EMB, bias=False)

        def forward(self, x):
            return self.out_projection(self.v_projection(x))

    class _FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.per_sample_attention_between_features = _FakeAttention()
            self.per_column_attention_between_cells = _FakeAttention()
            self.mlp = nn.Sequential(nn.Linear(EMB, EMB * 2), nn.GELU(), nn.Linear(EMB * 2, EMB))

        def forward(self, x):
            x = self.per_sample_attention_between_features(x)
            x = self.per_column_attention_between_cells(x)
            return self.mlp(x)

    class _FakeTabPFN(nn.Module):
        def __init__(self, n_layers: int = 24) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(_FakeBlock() for _ in range(n_layers))

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    model = _FakeTabPFN(n_layers=24)
    x = torch.randn(4, EMB)

    # --- Output di riferimento PRIMA del LoRA ---
    model.eval()
    with torch.no_grad():
        out_before = model(x)

    # --- Test 1: iniezione adapter ---
    print("\n[TEST 1] inject_lora_adapters() (target: q, v)")
    cfg = LoRAConfig(r=8, alpha=16, target_modules=("q_projection", "v_projection"))
    injected = inject_lora_adapters(model, cfg)
    print(f"  -> adapter iniettati: {len(injected)} (atteso 24 layer x 2 attn x 2 proj = 96)")
    assert len(injected) == 96, "numero di adapter inatteso"

    # --- Test 2: invarianza dell'output all'inizializzazione (B=0) ---
    print("\n[TEST 2] Output identico subito dopo l'iniezione (B=0)")
    model.eval()
    with torch.no_grad():
        out_after = model(x)
    max_diff = (out_before - out_after).abs().max().item()
    print(f"  -> differenza massima: {max_diff:.2e} (attesa ~0)")
    assert max_diff < 1e-5, "l'output non e' invariato all'inizializzazione!"

    # --- Test 3: congelamento e conteggio parametri ---
    print("\n[TEST 3] mark_only_lora_as_trainable() + conteggio")
    mark_only_lora_as_trainable(model)
    trainable, total, pct = count_trainable_parameters(model)
    print(f"  -> allenabili: {trainable:,} / totali: {total:,} ({pct:.3f}%)")
    # Solo i parametri lora_A/lora_B devono essere allenabili.
    non_lora_trainable = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "lora_" not in n
    ]
    assert not non_lora_trainable, f"parametri non-LoRA allenabili: {non_lora_trainable}"
    assert pct < 5.0, "i parametri allenabili dovrebbero essere una piccola frazione"

    # --- Test 4: un passo di training cambia gli adapter ---
    print("\n[TEST 4] Un passo di ottimizzazione aggiorna i pesi LoRA")
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-1)
    model.train()
    target = torch.randn(4, EMB)
    loss1 = F.mse_loss(model(x), target)
    opt.zero_grad()
    loss1.backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        loss2 = F.mse_loss(model(x), target)
    print(f"  -> loss: {loss1.item():.4f} -> {loss2.item():.4f}")
    assert loss2.item() < loss1.item(), "la loss non e' diminuita dopo un passo"

    # --- Test 5: salvataggio / caricamento dei soli adapter ---
    print("\n[TEST 5] save/load dei soli adapter LoRA")
    import os
    import tempfile

    sd = lora_state_dict(model)
    print(f"  -> chiavi nel lora_state_dict: {len(sd)} (atteso 96 x 2 = 192)")
    assert len(sd) == 192, "numero di tensori LoRA inatteso"

    tmp = os.path.join(tempfile.gettempdir(), "lora_test.pt")
    save_lora_adapters(model, tmp)

    fresh = _FakeTabPFN(n_layers=24)
    inject_lora_adapters(fresh, cfg)
    # Stesso backbone congelato di `model` (come avere lo stesso checkpoint
    # TabPFN): copiamo i soli pesi non-LoRA, poi carichiamo gli adapter dal file.
    fresh.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "lora_" not in k},
        strict=False,
    )
    load_lora_adapters(fresh, tmp)
    with torch.no_grad():
        out_fresh_after = fresh.eval()(x)
    with torch.no_grad():
        out_trained = model.eval()(x)
    reload_diff = (out_fresh_after - out_trained).abs().max().item()
    print(f"  -> differenza dopo reload vs modello allenato: {reload_diff:.2e}")
    assert reload_diff < 1e-5, "il reload degli adapter non riproduce l'output"
    os.remove(tmp)

    # --- Test 6: merge per inferenza ---
    print("\n[TEST 6] merge_lora_adapters() preserva l'output")
    with torch.no_grad():
        out_pre_merge = model.eval()(x)
    n_merged = merge_lora_adapters(model)
    with torch.no_grad():
        out_post_merge = model.eval()(x)
    merge_diff = (out_pre_merge - out_post_merge).abs().max().item()
    print(f"  -> adapter fusi: {n_merged} | differenza output: {merge_diff:.2e}")
    assert merge_diff < 1e-4, "il merge ha alterato l'output"

    print("\n" + "=" * 70)
    print("TUTTI I TEST SUPERATI")
    print("=" * 70)
