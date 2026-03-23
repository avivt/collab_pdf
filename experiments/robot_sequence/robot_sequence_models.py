import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


F_TOKEN = 1
B_TOKEN = 0
PAD_TOKEN = 2
VOCAB_SIZE = 3


@dataclass
class Split:
    name: str
    sequences: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


def generate_split(
    n_samples: int,
    min_len: int,
    max_len: int,
    seed: int,
    target_pos_fraction: Optional[float] = None,
) -> Split:
    if min_len < 1:
        raise ValueError("min_len must be >= 1")
    if max_len < min_len:
        raise ValueError("max_len must be >= min_len")

    g = torch.Generator().manual_seed(seed)

    if target_pos_fraction is None:
        lengths = torch.randint(min_len, max_len + 1, (n_samples,), generator=g)
        max_batch_len = int(lengths.max().item())

        sequences = torch.randint(0, 2, (n_samples, max_batch_len), generator=g)
        pos = torch.arange(max_batch_len).unsqueeze(0)
        valid_mask = pos < lengths.unsqueeze(1)
        sequences = sequences.masked_fill(~valid_mask, PAD_TOKEN)

        n_forward = ((sequences == F_TOKEN) & valid_mask).sum(dim=1)
        n_backward = ((sequences == B_TOKEN) & valid_mask).sum(dim=1)
        labels = (n_forward - n_backward == 2).float()
    else:
        if not 0.0 <= target_pos_fraction <= 1.0:
            raise ValueError("target_pos_fraction must be in [0, 1]")

        n_pos = int(round(n_samples * target_pos_fraction))
        labels = torch.zeros(n_samples, dtype=torch.float32)
        labels[:n_pos] = 1.0
        labels = labels[torch.randperm(n_samples, generator=g)]

        all_lengths = torch.arange(min_len, max_len + 1, dtype=torch.long)
        pos_lengths = all_lengths[(all_lengths >= 2) & (all_lengths % 2 == 0)]
        if n_pos > 0 and pos_lengths.numel() == 0:
            raise ValueError("No valid lengths for positive label (need even length >= 2).")

        lengths = torch.empty(n_samples, dtype=torch.long)
        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == 0)[0]

        if pos_idx.numel() > 0:
            pick = torch.randint(0, pos_lengths.numel(), (pos_idx.numel(),), generator=g)
            lengths[pos_idx] = pos_lengths[pick]
        if neg_idx.numel() > 0:
            lengths[neg_idx] = torch.randint(
                min_len, max_len + 1, (neg_idx.numel(),), generator=g
            )

        max_batch_len = int(lengths.max().item())
        sequences = torch.full((n_samples, max_batch_len), PAD_TOKEN, dtype=torch.long)

        for i in range(n_samples):
            seq_len = int(lengths[i].item())
            if labels[i].item() == 1.0:
                n_forward = (seq_len + 2) // 2
                seq = torch.zeros(seq_len, dtype=torch.long)
                forward_idx = torch.randperm(seq_len, generator=g)[:n_forward]
                seq[forward_idx] = F_TOKEN
            else:
                if seq_len % 2 == 1:
                    seq = torch.randint(0, 2, (seq_len,), generator=g)
                else:
                    target_forward = (seq_len + 2) // 2
                    while True:
                        seq = torch.randint(0, 2, (seq_len,), generator=g)
                        if int((seq == F_TOKEN).sum().item()) != target_forward:
                            break

            sequences[i, :seq_len] = seq

    return Split(
        name=f"len[{min_len},{max_len}]",
        sequences=sequences,
        lengths=lengths,
        labels=labels,
    )


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
        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return self.layernorm(h_n[-1])

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classify(self.encode(x, lengths))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.repr_dim = d_model
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=2048)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        try:
            # Newer PyTorch: disable nested tensors directly at construction.
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            # Older PyTorch: build first, then force-disable nested tensor paths.
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            if hasattr(self.encoder, "enable_nested_tensor"):
                self.encoder.enable_nested_tensor = False
            if hasattr(self.encoder, "use_nested_tensor"):
                self.encoder.use_nested_tensor = False
        self.head = nn.Linear(d_model, 1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        padding_mask = x.eq(PAD_TOKEN)
        h = self.embedding(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1)
        return (h * valid_mask).sum(dim=1) / lengths.unsqueeze(1).clamp_min(1)

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
    dataset = TensorDataset(split.sequences, split.lengths, split.labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_correct = 0
    all_count = 0
    with torch.no_grad():
        for seq, lengths, labels in loader:
            seq = seq.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(seq, lengths)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_correct += (preds == labels).sum().item()
            all_count += labels.numel()
    return all_correct / max(all_count, 1)


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
    criterion_main = nn.BCEWithLogitsLoss()
    criterion_adv = nn.CrossEntropyLoss() if discriminator is not None else None

    params = list(model.parameters())
    if discriminator is not None:
        params += list(discriminator.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    model.to(device)
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
                length_targets = lengths - 1
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


def run_experiment(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu_only
        else "mps"
        if torch.backends.mps.is_available() and not args.cpu_only
        else "cpu"
    )
    print(f"Using device: {device}")

    train_split = generate_split(
        n_samples=args.train_samples,
        min_len=1,
        max_len=30,
        seed=args.seed,
        target_pos_fraction=0.5 if not args.unbalanced_train else None,
    )
    val_split = generate_split(
        n_samples=args.val_samples,
        min_len=1,
        max_len=30,
        seed=args.seed + 1,
        target_pos_fraction=0.5 if not args.unbalanced_eval else None,
    )

    test_a = generate_split(
        n_samples=args.test_samples,
        min_len=1,
        max_len=30,
        seed=args.seed + 2,
        target_pos_fraction=0.5 if not args.unbalanced_eval else None,
    )
    test_b = generate_split(
        n_samples=args.test_samples,
        min_len=30,
        max_len=100,
        seed=args.seed + 3,
        target_pos_fraction=0.5 if not args.unbalanced_eval else None,
    )
    test_c = generate_split(
        n_samples=args.test_samples,
        min_len=100,
        max_len=200,
        seed=args.seed + 4,
        target_pos_fraction=0.5 if not args.unbalanced_eval else None,
    )

    print(
        f"Train positive-label fraction: {train_split.labels.mean().item():.4f} "
        f"({'balanced' if not args.unbalanced_train else 'unbalanced'})"
    )
    print(
        f"Val positive-label fraction: {val_split.labels.mean().item():.4f} "
        f"({'balanced' if not args.unbalanced_eval else 'unbalanced'})"
    )
    print(
        f"Test positive-label fractions: "
        f"(a) {test_a.labels.mean().item():.4f}, "
        f"(b) {test_b.labels.mean().item():.4f}, "
        f"(c) {test_c.labels.mean().item():.4f} "
        f"({'balanced' if not args.unbalanced_eval else 'unbalanced'})"
    )

    train_loader = to_loader(train_split, batch_size=args.batch_size, shuffle=True)
    val_loader = to_loader(val_split, batch_size=args.batch_size, shuffle=False)
    test_loaders = [
        ("(a) len<=30", to_loader(test_a, batch_size=args.batch_size, shuffle=False)),
        ("(b) 30<=len<=100", to_loader(test_b, batch_size=args.batch_size, shuffle=False)),
        ("(c) 100<=len<=200", to_loader(test_c, batch_size=args.batch_size, shuffle=False)),
    ]

    models = {
        "GRU": GRUClassifier(
            d_model=args.gru_d_model,
            hidden_size=args.gru_hidden,
            num_layers=args.gru_layers,
            use_layernorm=args.gru_layernorm,
        ),
        "Transformer": TransformerClassifier(
            d_model=args.tf_d_model,
            nhead=args.tf_heads,
            num_layers=args.tf_layers,
            dim_feedforward=args.tf_ffn,
            dropout=args.tf_dropout,
        ),
    }

    model_epochs = {
        "GRU": args.gru_epochs if args.gru_epochs is not None else args.epochs,
        "Transformer": (
            args.tf_epochs if args.tf_epochs is not None else args.epochs
        ),
    }

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training {model_name} for {model_epochs[model_name]} epochs")
        discriminator = None
        if args.adv_length_discriminator:
            discriminator = LengthDiscriminator(
                input_dim=model.repr_dim,
                hidden_dim=args.adv_hidden_dim,
                num_classes=int(train_split.lengths.max().item()),
            )
            print(
                "Using adversarial length discriminator "
                f"(weight={args.adv_weight}, grl_lambda={args.adv_grl_lambda})"
            )

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=model_epochs[model_name],
            lr=args.lr,
            discriminator=discriminator,
            adv_weight=args.adv_weight,
            adv_grl_lambda=args.adv_grl_lambda,
        )

        print(f"\n{model_name} test accuracy")
        for label, loader in test_loaders:
            acc = evaluate(model, loader, device)
            print(f"  {label}: {acc:.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train GRU and Transformer models to predict Y=1 iff the action sequence "
            "has exactly two more F actions than B actions."
        )
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-only", action="store_true")

    p.add_argument("--train-samples", type=int, default=50000)
    p.add_argument("--val-samples", type=int, default=5000)
    p.add_argument("--test-samples", type=int, default=10000)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--gru-epochs", type=int, default=None)
    p.add_argument("--tf-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--adv-length-discriminator",
        action="store_true",
        help="Adversarially remove sequence-length information from hidden states.",
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
    p.add_argument(
        "--unbalanced-train",
        action="store_true",
        help="Use natural label distribution for training data instead of ~50/50 balance.",
    )
    p.add_argument(
        "--unbalanced-eval",
        action="store_true",
        help="Use natural label distribution for validation/test instead of ~50/50 balance.",
    )

    p.add_argument("--gru-d-model", type=int, default=64)
    p.add_argument("--gru-hidden", type=int, default=128)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument(
        "--gru-layernorm",
        action="store_true",
        help="Apply LayerNorm to the GRU final hidden state before classification.",
    )

    p.add_argument("--tf-d-model", type=int, default=64)
    p.add_argument("--tf-heads", type=int, default=4)
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--tf-ffn", type=int, default=128)
    p.add_argument("--tf-dropout", type=float, default=0.1)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_experiment(args)
