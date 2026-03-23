import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset
import statistics


VOCAB = {"0": 0, "1": 1, "+": 2, "=": 3}
PAD_TOKEN = 4
VOCAB_SIZE = 5


@dataclass
class Split:
    name: str
    strings: List[str]
    sequences: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


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
    raise ValueError(f"Unsupported language: {language}")


def _parse_main_tok_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    # Official FLARE .tok lines are whitespace-separated symbols.
    return "".join(stripped.split())


def _read_text_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return text.splitlines()


def load_split_from_files(name: str, main_tok_path: Path, labels_path: Path) -> Split:
    tok_lines = _read_text_lines(main_tok_path)
    label_lines = _read_text_lines(labels_path)
    if len(tok_lines) != len(label_lines):
        raise ValueError(
            f"Mismatch between examples and labels in {name}: "
            f"{len(tok_lines)} vs {len(label_lines)}"
        )

    strings = [_parse_main_tok_line(line) for line in tok_lines]
    labels = [int(line.strip()) for line in label_lines]
    labels_t = torch.tensor(labels, dtype=torch.float32)
    sequences, lengths = encode_strings(strings)
    return Split(
        name=name,
        strings=strings,
        sequences=sequences,
        lengths=lengths,
        labels=labels_t,
    )


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)


def download_official_flare_language(flare_root: Path, language: str) -> None:
    if language not in {"parity", "even-pairs", "binary-addition"}:
        raise ValueError(f"Unsupported official FLARE language: {language}")

    base_url = f"https://raw.githubusercontent.com/rycolab/flare/main/{language}"
    files = [
        (f"{language}/main.tok", f"{base_url}/main.tok"),
        (f"{language}/labels.txt", f"{base_url}/labels.txt"),
        (
            f"{language}/datasets/validation-short/main.tok",
            f"{base_url}/datasets/validation-short/main.tok",
        ),
        (
            f"{language}/datasets/validation-short/labels.txt",
            f"{base_url}/datasets/validation-short/labels.txt",
        ),
        (
            f"{language}/datasets/test/main.tok",
            f"{base_url}/datasets/test/main.tok",
        ),
        (
            f"{language}/datasets/test/labels.txt",
            f"{base_url}/datasets/test/labels.txt",
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
) -> tuple[Split, Split, Split]:
    if language not in {"parity", "even-pairs", "binary-addition"}:
        raise ValueError(f"Unsupported official FLARE language: {language}")

    base = flare_root / language
    train = load_split_from_files(
        name=f"{language} official train",
        main_tok_path=base / "main.tok",
        labels_path=base / "labels.txt",
    )
    val_short = load_split_from_files(
        name=f"{language} official validation-short",
        main_tok_path=base / "datasets" / "validation-short" / "main.tok",
        labels_path=base / "datasets" / "validation-short" / "labels.txt",
    )
    test = load_split_from_files(
        name=f"{language} official test",
        main_tok_path=base / "datasets" / "test" / "main.tok",
        labels_path=base / "datasets" / "test" / "labels.txt",
    )
    return train, val_short, test


class GRUClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.repr_dim = hidden_size
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(hidden_size) if use_layernorm else nn.Identity()
        self.head = nn.Linear(hidden_size, 1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        lengths_safe = lengths.clamp_min(1)
        packed = pack_padded_sequence(
            emb,
            lengths_safe.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return self.layernorm(h_n[-1])

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x, lengths))


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class LengthDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def to_loader(split: Split, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(split.sequences, split.lengths, split.labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for seq, lengths, labels in loader:
            seq = seq.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(seq, lengths)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            count += labels.numel()
    return correct / max(count, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    discriminator: Optional[nn.Module] = None,
    adv_weight: float = 0.1,
    adv_grl_lambda: float = 1.0,
) -> nn.Module:
    model.to(device)
    criterion_main = nn.BCEWithLogitsLoss()
    criterion_adv = nn.CrossEntropyLoss() if discriminator is not None else None

    params = list(model.parameters())
    if discriminator is not None:
        params += list(discriminator.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    if discriminator is not None:
        discriminator.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
        epoch_loss = 0.0
        epoch_main_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_acc = 0.0
        epoch_disc_acc = 0.0
        batches = 0

        for seq, lengths, labels in train_loader:
            seq = seq.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            features = model.encode(seq, lengths)
            logits = model.classify(features)
            main_loss = criterion_main(logits, labels)
            loss = main_loss

            if discriminator is not None and criterion_adv is not None:
                length_targets = lengths
                disc_logits = discriminator(grad_reverse(features, adv_grl_lambda))
                adv_loss = criterion_adv(disc_logits, length_targets)
                loss = loss + adv_weight * adv_loss

                disc_preds = disc_logits.argmax(dim=1)
                epoch_disc_acc += (
                    (disc_preds == length_targets).float().mean().item()
                )
                epoch_adv_loss += adv_loss.item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_main_loss += main_loss.item()
            epoch_acc += accuracy_from_logits(logits.detach(), labels)
            batches += 1

        val_acc = evaluate(model, val_loader, device)
        if discriminator is None:
            print(
                f"epoch {epoch:02d} | train_loss={epoch_loss / batches:.4f} "
                f"train_acc={epoch_acc / batches:.4f} val_acc={val_acc:.4f}"
            )
        else:
            print(
                f"epoch {epoch:02d} | train_loss={epoch_loss / batches:.4f} "
                f"main_loss={epoch_main_loss / batches:.4f} "
                f"adv_loss={epoch_adv_loss / batches:.4f} "
                f"train_acc={epoch_acc / batches:.4f} "
                f"disc_acc={epoch_disc_acc / batches:.4f} "
                f"val_acc={val_acc:.4f}"
            )

    return model


def run(args: argparse.Namespace) -> None:
    language = args.language.lower()
    if language not in {"parity", "even-pairs", "binary-addition"}:
        raise ValueError(
            "Supported values for --language are: parity, even-pairs, binary-addition."
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
        train_split, val_split, test_split = load_official_language_splits(
            flare_root, language
        )
        print(f"Loaded official FLARE {language} data from: {flare_root}")
    else:
        # FLARE inductive-bias protocol (single language):
        # train in [0,40], short validation in [0,40], test in [0,500].
        train_split = generate_synthetic_split(
            language=language,
            n_samples=args.train_samples,
            min_len=0,
            max_len=40,
            seed=args.seed,
        )
        val_split = generate_synthetic_split(
            language=language,
            n_samples=args.val_samples,
            min_len=0,
            max_len=40,
            seed=args.seed + 1,
        )
        test_split = generate_synthetic_split(
            language=language,
            n_samples=args.test_samples,
            min_len=0,
            max_len=500,
            seed=args.seed + 2,
        )
        print(
            f"Using synthetic {language} data with FLARE inductive-bias length ranges."
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

    train_loader = to_loader(train_split, batch_size=args.batch_size, shuffle=True)
    val_loader = to_loader(val_split, batch_size=args.batch_size, shuffle=False)
    test_loader = to_loader(test_split, batch_size=args.batch_size, shuffle=False)

    all_test_acc = []
    for run_idx in range(args.num_runs):
        run_seed = args.seed + run_idx
        torch.manual_seed(run_seed)
        random.seed(run_seed)

        model = GRUClassifier(
            d_model=args.gru_d_model,
            hidden_size=args.gru_hidden,
            num_layers=args.gru_layers,
            use_layernorm=args.gru_layernorm,
        )
        discriminator = None
        if args.adv_length_discriminator:
            discriminator = LengthDiscriminator(
                input_dim=model.repr_dim,
                hidden_dim=args.adv_hidden_dim,
                num_classes=int(train_split.lengths.max().item()) + 1,
            )

        print(
            f"Training GRU on FLARE regular language: {language} | "
            f"run {run_idx + 1}/{args.num_runs} (seed={run_seed})"
        )
        if discriminator is not None:
            print(
                "Using adversarial length discriminator "
                f"(weight={args.adv_weight}, grl_lambda={args.adv_grl_lambda})"
            )
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            discriminator=discriminator,
            adv_weight=args.adv_weight,
            adv_grl_lambda=args.adv_grl_lambda,
        )

        test_acc = evaluate(model, test_loader, device)
        all_test_acc.append(test_acc)
        print(f"\nRun {run_idx + 1} inductive-bias test accuracy (len 0..500): {test_acc:.4f}")

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
            "Train a GRU recognizer for one regular FLARE language "
            "using the inductive-bias protocol."
        )
    )
    p.add_argument(
        "--language",
        type=str,
        choices=["parity", "even-pairs", "binary-addition"],
        default="parity",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument(
        "--dataset-source",
        type=str,
        choices=["official", "synthetic"],
        default="official",
        help="Use official FLARE files or synthetic data generated with the same protocol.",
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
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--adv-length-discriminator",
        action="store_true",
        help="Adversarially remove sequence-length information from GRU hidden states.",
    )
    p.add_argument(
        "--adv-weight",
        type=float,
        default=0.1,
        help="Weight for adversarial length-discriminator loss.",
    )
    p.add_argument(
        "--adv-grl-lambda",
        type=float,
        default=1.0,
        help="Gradient-reversal scale for adversarial training.",
    )
    p.add_argument(
        "--adv-hidden-dim",
        type=int,
        default=64,
        help="Hidden width of the length discriminator MLP.",
    )

    p.add_argument("--gru-d-model", type=int, default=64)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument("--gru-layernorm", action="store_true")
    p.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of independent training runs with different seeds.",
    )
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
