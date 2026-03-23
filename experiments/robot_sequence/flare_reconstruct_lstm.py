import argparse
import statistics
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


VOCAB = {"0": 0, "1": 1, "+": 2, "=": 3, "#": 4}
PAD_TOKEN = 5
VOCAB_SIZE = 6
EOS_INDEX = len(VOCAB)
NS_OUTPUT_DIM = len(VOCAB) + 1

ALL_FLARE_TASKS = [
    "binary-addition",
    "binary-multiplication",
    "bucket-sort",
    "compute-sqrt",
    "cycle-navigation",
    "dyck-2-3",
    "even-pairs",
    "first",
    "majority",
    "marked-copy",
    "marked-reversal",
    "missing-duplicate-string",
    "modular-arithmetic-simple",
    "odds-first",
    "parity",
    "repeat-01",
    "stack-manipulation",
    "unmarked-reversal",
]


@dataclass
class Split:
    name: str
    strings: List[str]
    sequences: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    next_symbol_targets: Optional[torch.Tensor] = None  # [N, T+1, NS_OUTPUT_DIM]
    next_symbol_valid: Optional[torch.Tensor] = None    # [N, T+1] bool


@dataclass
class RawSplit:
    name: str
    token_sequences: List[List[str]]
    labels: List[int]
    next_symbol_sequences: Optional[List[List[dict]]] = None


def is_in_parity_language(w: str) -> bool:
    """Parity language: binary strings with an odd number of 1s."""
    return w.count("1") % 2 == 1


def is_in_even_pairs_language(w: str) -> bool:
    """Even Pairs language: c01(w) + c10(w) is even (equivalently, first==last)."""
    if len(w) <= 1:
        return True
    return w[0] == w[-1]


def sample_positive_parity(min_len: int, max_len: int, rng: random.Random) -> str:
    valid_lengths = [n for n in range(min_len, max_len + 1) if n >= 1]
    if not valid_lengths:
        raise ValueError("No valid lengths for positive parity sample.")

    n = rng.choice(valid_lengths)
    odd_counts = [k for k in range(1, n + 1, 2)]
    c1 = rng.choice(odd_counts)
    c0 = n - c1
    chars = ["1"] * c1 + ["0"] * c0
    rng.shuffle(chars)
    return "".join(chars)


def sample_positive_even_pairs(min_len: int, max_len: int, rng: random.Random) -> str:
    n = rng.randint(min_len, max_len)
    if n == 0:
        return ""
    if n == 1:
        return rng.choice("01")
    first = rng.choice("01")
    middle = "".join(rng.choice("01") for _ in range(n - 2))
    return first + middle + first


def sample_uniform_binary_string(min_len: int, max_len: int, rng: random.Random) -> str:
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice("01") for _ in range(n))


def apply_random_edit(w: str, rng: random.Random) -> str:
    op = rng.choice(["insert", "replace", "delete"])

    if op == "insert":
        i = rng.randint(0, len(w))
        c = rng.choice("01")
        return w[:i] + c + w[i:]

    if op == "replace":
        if len(w) == 0:
            return w
        i = rng.randrange(len(w))
        c = rng.choice("01")
        return w[:i] + c + w[i + 1 :]

    if op == "delete":
        if len(w) == 0:
            return w
        i = rng.randrange(len(w))
        return w[:i] + w[i + 1 :]

    raise RuntimeError("Unknown edit operation")


def sample_num_edits(rng: random.Random, p: float = 0.5) -> int:
    # Geometric distribution over {1,2,3,...}; smaller values are more likely.
    k = 1
    while rng.random() > p:
        k += 1
    return k


def sample_negative_parity(min_len: int, max_len: int, rng: random.Random) -> str:
    while True:
        if rng.random() < 0.5:
            w = sample_uniform_binary_string(min_len, max_len, rng)
        else:
            w = sample_positive_parity(min_len, max_len, rng)
            k = sample_num_edits(rng)
            for _ in range(k):
                w = apply_random_edit(w, rng)
            if len(w) < min_len or len(w) > max_len:
                continue

        if not is_in_parity_language(w):
            return w


def sample_negative_even_pairs(min_len: int, max_len: int, rng: random.Random) -> str:
    while True:
        if rng.random() < 0.5:
            w = sample_uniform_binary_string(min_len, max_len, rng)
        else:
            w = sample_positive_even_pairs(min_len, max_len, rng)
            k = sample_num_edits(rng)
            for _ in range(k):
                w = apply_random_edit(w, rng)
            if len(w) < min_len or len(w) > max_len:
                continue

        if not is_in_even_pairs_language(w):
            return w


def encode_strings(strings: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in strings], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(strings) > 0 else 0
    max_len = max(1, max_len)

    sequences = torch.full((len(strings), max_len), PAD_TOKEN, dtype=torch.long)
    for i, s in enumerate(strings):
        if not s:
            continue
        try:
            seq = torch.tensor([VOCAB[c] for c in s], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"Unsupported token in input string: {exc.args[0]!r}") from exc
        sequences[i, : len(s)] = seq

    return sequences, lengths


def generate_parity_split(
    n_samples: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> Split:
    rng = random.Random(seed)

    labels = [rng.randint(0, 1) for _ in range(n_samples)]
    strings = []
    for y in labels:
        if y == 1:
            w = sample_positive_parity(min_len, max_len, rng)
        else:
            w = sample_negative_parity(min_len, max_len, rng)
        strings.append(w)

    sequences, lengths = encode_strings(strings)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    return Split(
        name=f"parity len[{min_len},{max_len}]",
        strings=strings,
        sequences=sequences,
        lengths=lengths,
        labels=labels_t,
    )


def generate_even_pairs_split(
    n_samples: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> Split:
    rng = random.Random(seed)

    labels = [rng.randint(0, 1) for _ in range(n_samples)]
    strings = []
    for y in labels:
        if y == 1:
            w = sample_positive_even_pairs(min_len, max_len, rng)
        else:
            w = sample_negative_even_pairs(min_len, max_len, rng)
        strings.append(w)

    sequences, lengths = encode_strings(strings)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    return Split(
        name=f"even-pairs len[{min_len},{max_len}]",
        strings=strings,
        sequences=sequences,
        lengths=lengths,
        labels=labels_t,
    )


def generate_synthetic_split(
    language: str,
    n_samples: int,
    min_len: int,
    max_len: int,
    seed: int,
) -> Split:
    if language == "parity":
        return generate_parity_split(n_samples, min_len, max_len, seed)
    if language == "even-pairs":
        return generate_even_pairs_split(n_samples, min_len, max_len, seed)
    if language == "binary-addition":
        raise ValueError(
            "Synthetic generation for binary-addition is not implemented. "
            "Use --dataset-source official for binary-addition."
        )
    if language == "odds-first":
        raise ValueError(
            "Synthetic generation for odds-first is not implemented. "
            "Use --dataset-source official for odds-first."
        )
    raise ValueError(f"Unsupported language: {language}")


def _parse_main_tok_line(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped:
        return []
    # Official FLARE .tok lines are whitespace-separated tokens.
    return stripped.split()


def _read_text_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return text.splitlines()


def _parse_next_symbols_json_line(line: str) -> list[dict]:
    raw_steps = json.loads(line)
    steps = []
    for step in raw_steps:
        syms = str(step.get("s", "")).strip()
        tokens = [] if not syms else syms.split()
        steps.append({"tokens": tokens, "e": bool(step.get("e", False))})
    return steps


def _load_raw_split_from_files(
    name: str,
    main_tok_path: Path,
    labels_path: Path,
    next_symbols_path: Optional[Path] = None,
) -> RawSplit:
    tok_lines = _read_text_lines(main_tok_path)
    label_lines = _read_text_lines(labels_path)
    if len(tok_lines) != len(label_lines):
        raise ValueError(
            f"Mismatch between examples and labels in {name}: "
            f"{len(tok_lines)} vs {len(label_lines)}"
        )

    token_sequences = [_parse_main_tok_line(line) for line in tok_lines]
    labels = [int(line.strip()) for line in label_lines]

    next_symbol_sequences: Optional[List[List[dict]]] = None
    if next_symbols_path is not None and next_symbols_path.exists():
        ns_lines = _read_text_lines(next_symbols_path)
        parsed_lines = [_parse_next_symbols_json_line(line) for line in ns_lines]

        n_examples = len(token_sequences)
        n_pos = int(sum(labels))
        if len(parsed_lines) not in {n_examples, n_pos}:
            raise ValueError(
                "Mismatch between number of next-symbol lines and examples/positives: "
                f"{len(parsed_lines)} vs examples={n_examples}, positives={n_pos}"
            )

        if len(parsed_lines) == n_examples:
            next_symbol_sequences = parsed_lines
        else:
            # Official FLARE train splits store next-symbol labels only for positives.
            next_symbol_sequences = [[] for _ in range(n_examples)]
            j = 0
            for i, y in enumerate(labels):
                if y == 1:
                    next_symbol_sequences[i] = parsed_lines[j]
                    j += 1

    return RawSplit(
        name=name,
        token_sequences=token_sequences,
        labels=labels,
        next_symbol_sequences=next_symbol_sequences,
    )


def _build_token_vocab(raw_splits: List[RawSplit]) -> dict[str, int]:
    tokens = set()
    for split in raw_splits:
        for seq in split.token_sequences:
            tokens.update(seq)
        if split.next_symbol_sequences is not None:
            for ns_seq in split.next_symbol_sequences:
                for step in ns_seq:
                    tokens.update(step.get("tokens", []))

    # Stable ordering across runs/filesystems.
    return {tok: i for i, tok in enumerate(sorted(tokens))}


def _encode_token_sequences(
    token_sequences: List[List[str]], token_to_id: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(seq) for seq in token_sequences], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(token_sequences) > 0 else 0
    max_len = max(1, max_len)
    pad_idx = len(token_to_id)

    sequences = torch.full((len(token_sequences), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(token_sequences):
        if not seq:
            continue
        sequences[i, : len(seq)] = torch.tensor([token_to_id[tok] for tok in seq], dtype=torch.long)
    return sequences, lengths


def _materialize_split_from_raw(raw: RawSplit, token_to_id: dict[str, int]) -> Split:
    sequences, lengths = _encode_token_sequences(raw.token_sequences, token_to_id)
    labels_t = torch.tensor(raw.labels, dtype=torch.float32)

    ns_targets = None
    ns_valid = None
    if raw.next_symbol_sequences is not None:
        eos_idx = len(token_to_id)
        ns_dim = len(token_to_id) + 1
        max_t = max((len(x) for x in raw.next_symbol_sequences), default=1)
        max_t = max(1, max_t)

        ns_targets = torch.zeros((len(raw.token_sequences), max_t, ns_dim), dtype=torch.float32)
        ns_valid = torch.zeros((len(raw.token_sequences), max_t), dtype=torch.bool)

        for i, ns_seq in enumerate(raw.next_symbol_sequences):
            for t, step in enumerate(ns_seq):
                ns_valid[i, t] = True
                for tok in step.get("tokens", []):
                    if tok in token_to_id:
                        ns_targets[i, t, token_to_id[tok]] = 1.0
                if bool(step.get("e", False)):
                    ns_targets[i, t, eos_idx] = 1.0

    return Split(
        name=raw.name,
        strings=[" ".join(seq) for seq in raw.token_sequences],
        sequences=sequences,
        lengths=lengths,
        labels=labels_t,
        next_symbol_targets=ns_targets,
        next_symbol_valid=ns_valid,
    )


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)


def download_official_flare_language(flare_root: Path, language: str) -> None:
    if language not in ALL_FLARE_TASKS:
        raise ValueError(f"Unsupported official FLARE language: {language}")

    base_url = f"https://raw.githubusercontent.com/rycolab/flare/main/{language}"
    files = [
        (f"{language}/main.tok", f"{base_url}/main.tok"),
        (f"{language}/labels.txt", f"{base_url}/labels.txt"),
        (f"{language}/next-symbols.jsonl", f"{base_url}/next-symbols.jsonl"),
        (
            f"{language}/datasets/validation-short/main.tok",
            f"{base_url}/datasets/validation-short/main.tok",
        ),
        (
            f"{language}/datasets/validation-short/labels.txt",
            f"{base_url}/datasets/validation-short/labels.txt",
        ),
        (
            f"{language}/datasets/validation-short/next-symbols.jsonl",
            f"{base_url}/datasets/validation-short/next-symbols.jsonl",
        ),
        (
            f"{language}/datasets/test/main.tok",
            f"{base_url}/datasets/test/main.tok",
        ),
        (
            f"{language}/datasets/test/labels.txt",
            f"{base_url}/datasets/test/labels.txt",
        ),
        (
            f"{language}/datasets/test/next-symbols.jsonl",
            f"{base_url}/datasets/test/next-symbols.jsonl",
        ),
    ]

    for rel_path, url in files:
        out_path = flare_root / rel_path
        if out_path.exists():
            continue
        print(f"Downloading {rel_path}")
        _download_file(url, out_path)


def load_official_language_splits(
    flare_root: Path, language: str
) -> tuple[Split, Split, Split, dict[str, int]]:
    if language not in ALL_FLARE_TASKS:
        raise ValueError(f"Unsupported official FLARE language: {language}")

    base = flare_root / language
    raw_train = _load_raw_split_from_files(
        name=f"{language} official train",
        main_tok_path=base / "main.tok",
        labels_path=base / "labels.txt",
        next_symbols_path=base / "next-symbols.jsonl",
    )
    raw_val_short = _load_raw_split_from_files(
        name=f"{language} official validation-short",
        main_tok_path=base / "datasets" / "validation-short" / "main.tok",
        labels_path=base / "datasets" / "validation-short" / "labels.txt",
        next_symbols_path=base / "datasets" / "validation-short" / "next-symbols.jsonl",
    )
    raw_test = _load_raw_split_from_files(
        name=f"{language} official test",
        main_tok_path=base / "datasets" / "test" / "main.tok",
        labels_path=base / "datasets" / "test" / "labels.txt",
        next_symbols_path=base / "datasets" / "test" / "next-symbols.jsonl",
    )
    token_to_id = _build_token_vocab([raw_train, raw_val_short, raw_test])
    train = _materialize_split_from_raw(raw_train, token_to_id)
    val_short = _materialize_split_from_raw(raw_val_short, token_to_id)
    test = _materialize_split_from_raw(raw_test, token_to_id)
    return train, val_short, test, token_to_id


class PaperLSTMLayer(nn.Module):
    """LSTM layer with decoupled input/forget gates and single bias per gate."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        cat_dim = input_dim + hidden_dim

        self.W_i = nn.Parameter(torch.empty(hidden_dim, cat_dim))
        self.b_i = nn.Parameter(torch.empty(hidden_dim))
        self.W_f = nn.Parameter(torch.empty(hidden_dim, cat_dim))
        self.b_f = nn.Parameter(torch.empty(hidden_dim))
        self.W_g = nn.Parameter(torch.empty(hidden_dim, cat_dim))
        self.b_g = nn.Parameter(torch.empty(hidden_dim))
        self.W_o = nn.Parameter(torch.empty(hidden_dim, cat_dim))
        self.b_o = nn.Parameter(torch.empty(hidden_dim))

    def reset_parameters(self) -> None:
        for p in self.parameters():
            nn.init.uniform_(p, -0.1, 0.1)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([x_t, h_prev], dim=1)
        i_t = torch.sigmoid(F.linear(concat, self.W_i, self.b_i))
        f_t = torch.sigmoid(F.linear(concat, self.W_f, self.b_f))
        g_t = torch.tanh(F.linear(concat, self.W_g, self.b_g))
        o_t = torch.sigmoid(F.linear(concat, self.W_o, self.b_o))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class PaperLSTMClassifier(nn.Module):
    """
    LSTM recognizer aligned with the paper's architecture:
    - multi-layer LSTM
    - learned initial hidden state per layer (c0 = 0)
    - dropout applied on layer inputs and top-layer outputs
    - single recognition head on final timestep representation
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        ns_output_dim: int,
        d_model: int = 64,
        num_layers: int = 5,
        dropout: float = 0.1,
        use_ns_head: bool = False,
    ):
        super().__init__()
        self.repr_dim = d_model
        self.num_layers = num_layers
        self.hidden_dim = d_model
        self.pad_idx = pad_idx
        self.ns_output_dim = ns_output_dim
        self.use_ns_head = use_ns_head
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.layers = nn.ModuleList(
            [PaperLSTMLayer(d_model, d_model) for _ in range(num_layers)]
        )
        self.init_h = nn.Parameter(torch.empty(num_layers, d_model))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        self.ns_head = nn.Linear(d_model, ns_output_dim) if use_ns_head else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Matches the paper's initialization choices.
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[self.pad_idx].zero_()

        for layer in self.layers:
            layer.reset_parameters()

        nn.init.uniform_(self.init_h, -0.1, 0.1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.ns_head is not None:
            nn.init.xavier_uniform_(self.ns_head.weight)
            nn.init.zeros_(self.ns_head.bias)

    def encode_with_history(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)  # [B, T, d]
        batch_size, max_len, _ = emb.shape
        device = emb.device

        h_states = [
            torch.tanh(self.init_h[layer_idx]).unsqueeze(0).expand(batch_size, -1).clone()
            for layer_idx in range(self.num_layers)
        ]
        c_states = [
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]
        history = [self.dropout(h_states[-1])]

        for t in range(max_len):
            active_mask = (lengths > t).unsqueeze(1).to(dtype=emb.dtype)
            layer_input = emb[:, t, :]

            for layer_idx, layer in enumerate(self.layers):
                dropped_input = self.dropout(layer_input)
                new_h, new_c = layer(
                    dropped_input,
                    h_states[layer_idx],
                    c_states[layer_idx],
                )

                # Keep states unchanged after sequence end.
                h_states[layer_idx] = new_h * active_mask + h_states[layer_idx] * (1.0 - active_mask)
                c_states[layer_idx] = new_c * active_mask + c_states[layer_idx] * (1.0 - active_mask)
                layer_input = h_states[layer_idx]
            history.append(self.dropout(h_states[-1]))

        final_features = self.dropout(h_states[-1])
        return final_features, torch.stack(history, dim=1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        final_features, _ = self.encode_with_history(x, lengths)
        return final_features

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)

    def next_symbol_logits(self, history: torch.Tensor) -> torch.Tensor:
        if self.ns_head is None:
            raise RuntimeError("next_symbol_logits called but ns_head is disabled.")
        return self.ns_head(history)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x, lengths))


def to_loader(
    split: Split, batch_size: int, shuffle: bool, ns_output_dim: int
) -> DataLoader:
    if split.next_symbol_targets is None:
        ns_targets = torch.zeros((split.sequences.size(0), 1, ns_output_dim), dtype=torch.float32)
        ns_valid = torch.zeros((split.sequences.size(0), 1), dtype=torch.bool)
    else:
        ns_targets = split.next_symbol_targets
        ns_valid = split.next_symbol_valid
    ds = TensorDataset(split.sequences, split.lengths, split.labels, ns_targets, ns_valid)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    count = 0
    ce_sum = 0.0
    ce_count = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for seq, lengths, labels, _, _ in loader:
            seq = seq.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(seq, lengths)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            count += labels.numel()
            ce_sum += criterion(logits, labels).item()
            ce_count += labels.numel()
    return correct / max(count, 1), ce_sum / max(ce_count, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip_norm: float = 5.0,
    use_ns_loss: bool = False,
    lambda_ns: float = 1.0,
    lr_patience: int = 5,
    early_stop_patience: int = 10,
    lr_decay: float = 0.5,
) -> nn.Module:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    criterion_ns = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_ce = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_main_loss = 0.0
        epoch_ns_loss = 0.0
        epoch_acc = 0.0
        batches = 0

        for seq, lengths, labels, ns_targets, ns_valid in train_loader:
            seq = seq.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            ns_targets = ns_targets.to(device)
            ns_valid = ns_valid.to(device)

            optimizer.zero_grad(set_to_none=True)
            final_features, history = model.encode_with_history(seq, lengths)
            logits = model.classify(final_features)
            main_loss = criterion(logits, labels)
            loss = main_loss

            ns_loss = torch.tensor(0.0, device=device)
            if use_ns_loss:
                ns_logits = model.next_symbol_logits(history)
                max_t = min(ns_logits.size(1), ns_targets.size(1))
                ns_logits_b = ns_logits[:, :max_t, :]
                ns_targets_b = ns_targets[:, :max_t, :]
                ns_valid_b = ns_valid[:, :max_t]

                # In FLARE, auxiliary losses are used only for positive examples.
                pos_mask = labels > 0.5
                joint_mask = ns_valid_b & pos_mask.unsqueeze(1)
                if joint_mask.any():
                    per_symbol = criterion_ns(ns_logits_b, ns_targets_b).mean(dim=-1)
                    ns_loss = per_symbol[joint_mask].mean()
                    loss = loss + lambda_ns * ns_loss

            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_main_loss += main_loss.item()
            epoch_ns_loss += ns_loss.item()
            epoch_acc += accuracy_from_logits(logits.detach(), labels)
            batches += 1

        val_acc, val_ce = evaluate(model, val_loader, device)
        curr_lr = optimizer.param_groups[0]["lr"]
        if use_ns_loss:
            print(
                f"epoch {epoch:02d} | train_loss={epoch_loss / batches:.4f} "
                f"main_loss={epoch_main_loss / batches:.4f} ns_loss={epoch_ns_loss / batches:.4f} "
                f"train_acc={epoch_acc / batches:.4f} val_acc={val_acc:.4f} val_ce={val_ce:.4f} lr={curr_lr:.6g}"
            )
        else:
            print(
                f"epoch {epoch:02d} | train_loss={epoch_loss / batches:.4f} "
                f"train_acc={epoch_acc / batches:.4f} val_acc={val_acc:.4f} val_ce={val_ce:.4f} lr={curr_lr:.6g}"
            )

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if lr_patience > 0 and no_improve % lr_patience == 0:
                for group in optimizer.param_groups:
                    group["lr"] *= lr_decay
            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch:02d} after {no_improve} checkpoints with no val CE improvement."
                )
                break

    model.load_state_dict(best_state)

    return model


def run(args: argparse.Namespace) -> None:
    language = args.language.lower()
    if language not in ALL_FLARE_TASKS:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu_only
        else "mps"
        if torch.backends.mps.is_available() and not args.cpu_only
        else "cpu"
    )
    print(f"Using device: {device}")

    if args.dataset_source == "official":
        flare_root = Path(args.flare_root).expanduser().resolve()
        if args.download_official:
            download_official_flare_language(flare_root, language)
        train_split, val_split, test_split, token_to_id = load_official_language_splits(
            flare_root, language
        )
        ns_output_dim = len(token_to_id) + 1
        pad_idx = len(token_to_id)
        vocab_size = len(token_to_id) + 1
        print(f"Loaded official FLARE {language} data from: {flare_root}")
        print(f"Token vocab size (without PAD/EOS): {len(token_to_id)}")
    else:
        raise ValueError(
            "Synthetic mode is not supported in flare_reconstruct_lstm.py. "
            "Use --dataset-source official."
        )

    print(
        f"Positive fractions | train={train_split.labels.mean().item():.3f} "
        f"val={val_split.labels.mean().item():.3f} "
        f"test={test_split.labels.mean().item():.3f}"
    )
    print(
        f"Length ranges | train=[{int(train_split.lengths.min())},{int(train_split.lengths.max())}] "
        f"val=[{int(val_split.lengths.min())},{int(val_split.lengths.max())}] "
        f"test=[{int(test_split.lengths.min())},{int(test_split.lengths.max())}]"
    )
    if args.use_ns_loss:
        if (
            train_split.next_symbol_targets is None
            or val_split.next_symbol_targets is None
            or test_split.next_symbol_targets is None
        ):
            raise ValueError(
                "Next-symbol labels are required for --use-ns-loss. "
                "Use --dataset-source official and ensure next-symbols.jsonl files are downloaded."
            )

    train_loader = to_loader(
        train_split,
        batch_size=args.batch_size,
        shuffle=True,
        ns_output_dim=ns_output_dim,
    )
    val_loader = to_loader(
        val_split,
        batch_size=args.batch_size,
        shuffle=False,
        ns_output_dim=ns_output_dim,
    )
    test_loader = to_loader(
        test_split,
        batch_size=args.batch_size,
        shuffle=False,
        ns_output_dim=ns_output_dim,
    )

    all_test_acc = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        torch.manual_seed(run_seed)
        random.seed(run_seed)

        model = PaperLSTMClassifier(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            ns_output_dim=ns_output_dim,
            d_model=args.lstm_d_model,
            num_layers=args.lstm_layers,
            dropout=args.lstm_dropout,
            use_ns_head=args.use_ns_loss,
        )

        print(
            f"Training LSTM on FLARE language: {language} | "
            f"run {run_idx + 1}/{args.num_runs} (seed={run_seed})"
        )
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            grad_clip_norm=args.grad_clip_norm,
            use_ns_loss=args.use_ns_loss,
            lambda_ns=args.lambda_ns,
            lr_patience=args.lr_patience,
            early_stop_patience=args.early_stop_patience,
            lr_decay=args.lr_decay,
        )

        test_acc, test_ce = evaluate(model, test_loader, device)
        all_test_acc.append(test_acc)
        print(
            f"\nRun {run_idx + 1} inductive-bias test accuracy (len 0..500): "
            f"{test_acc:.4f} | test_ce={test_ce:.4f}"
        )

    if args.num_runs == 1:
        print(f"\nInductive-bias test accuracy (len 0..500): {all_test_acc[0]:.4f}")
    else:
        mean_acc = statistics.mean(all_test_acc)
        std_acc = statistics.pstdev(all_test_acc)
        print(
            f"\nInductive-bias test accuracy over {args.num_runs} runs "
            f"(len 0..500): mean={mean_acc:.4f} std={std_acc:.4f} max={max(all_test_acc):.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train a paper-style LSTM recognizer on FLARE "
            "using the inductive-bias protocol."
        )
    )
    p.add_argument(
        "--language",
        type=str,
        choices=ALL_FLARE_TASKS,
        default="parity",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument(
        "--dataset-source",
        type=str,
        choices=["official"],
        default="official",
        help="Use official FLARE files.",
    )
    p.add_argument(
        "--flare-root",
        type=str,
        default="/Users/aviv/Repositories/Codex/experiments/robot_sequence/flare_data",
        help="Root directory containing downloaded FLARE data folders.",
    )
    p.add_argument(
        "--download-official",
        action="store_true",
        help="Download required official FLARE files for the selected language into --flare-root if missing.",
    )

    # FLARE defaults from the paper.
    p.add_argument("--train-samples", type=int, default=10000)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--test-samples", type=int, default=5010)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=5.0,
        help="L2 gradient clipping threshold (paper uses 5). Set <=0 to disable.",
    )
    p.add_argument(
        "--lstm-d-model",
        type=int,
        default=64,
        help="LSTM state size d and embedding size d.",
    )
    p.add_argument(
        "--lstm-layers",
        type=int,
        default=5,
        help="Number of LSTM layers (paper default is 5).",
    )
    p.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.1,
        help="Dropout rate applied in recurrent stack (paper default is 0.1).",
    )
    p.add_argument(
        "--use-ns-loss",
        action="store_true",
        help="Include next-symbol prediction auxiliary loss (LNS) on positive examples.",
    )
    p.add_argument(
        "--lambda-ns",
        type=float,
        default=1.0,
        help="Coefficient for next-symbol loss when --use-ns-loss is enabled.",
    )
    p.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Number of validation checkpoints without CE improvement before LR decay.",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Number of validation checkpoints without CE improvement before early stopping.",
    )
    p.add_argument(
        "--lr-decay",
        type=float,
        default=0.5,
        help="LR multiplicative decay factor when patience is reached.",
    )
    p.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of independent training runs with different seeds.",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
