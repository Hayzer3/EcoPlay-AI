#!/usr/bin/env python3
"""Train, evaluate, and export the EcoPlay MobileNetV2 image classifier."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


IMAGE_SIZE = 224
SEED = 42


class ImageFolderWithTransform(Dataset):
    """Apply a split-specific transform while sharing ImageFolder samples."""

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.dataset.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, label


def stratified_indices(
    samples: list[tuple[str, int]], seed: int = SEED
) -> tuple[list[int], list[int], list[int]]:
    """Return disjoint 70/15/15 indices while preserving class proportions."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, label) in enumerate(samples):
        by_class[label].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        train_end = round(len(class_indices) * 0.70)
        validation_end = train_end + round(len(class_indices) * 0.15)
        train_indices.extend(class_indices[:train_end])
        validation_indices.extend(class_indices[train_end:validation_end])
        test_indices.extend(class_indices[validation_end:])

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    rng.shuffle(test_indices)
    return train_indices, validation_indices, test_indices


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    return train_transform, evaluation_transform


def make_loaders(
    data_dir: Path, batch_size: int, workers: int
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], list[int]]:
    train_transform, evaluation_transform = build_transforms()
    train_dataset = ImageFolderWithTransform(data_dir, train_transform)
    validation_dataset = ImageFolderWithTransform(data_dir, evaluation_transform)
    test_dataset = ImageFolderWithTransform(data_dir, evaluation_transform)
    train_indices, validation_indices, test_indices = stratified_indices(
        train_dataset.dataset.samples
    )
    loader_options = {"batch_size": batch_size, "num_workers": workers}
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), shuffle=True, **loader_options
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices), shuffle=False, **loader_options
    )
    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset.classes,
        [len(train_indices), len(validation_indices), len(test_indices)],
    )


def create_model(class_count: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, class_count)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, object]:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            expected.extend(labels.tolist())
    return {
        "accuracy": float(np.mean(np.asarray(expected) == np.asarray(predicted))),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=list(range(len(class_names)))
        ).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root")
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate and export an existing best checkpoint without training",
    )
    parser.add_argument(
        "--real-data",
        type=Path,
        help="Optional labeled real-photo ImageFolder root for separate evaluation",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    if not args.data.is_dir():
        parser.error(f"Dataset folder not found: {args.data.resolve()}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_loader, validation_loader, test_loader, class_names, sizes = make_loaders(
        args.data, args.batch_size, args.workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        len(class_names),
        pretrained=not args.no_pretrained and not args.evaluate_only,
    ).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "mobilenetv2_best.pth"

    if not args.evaluate_only:
        train_subset = train_loader.dataset
        train_base = train_subset.dataset
        training_labels = [
            train_base.dataset.samples[index][1] for index in train_subset.indices
        ]
        class_counts = torch.bincount(
            torch.tensor(training_labels), minlength=len(class_names)
        ).float()
        class_weights = len(training_labels) / (len(class_names) * class_counts)
        loss_function = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        best_validation_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, loss_function, device, optimizer
            )
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, loss_function, device
            )
            print(
                f"Epoch {epoch:02}/{args.epochs} | "
                f"train loss={train_loss:.4f} acc={train_accuracy:.4f} | "
                f"validation loss={validation_loss:.4f} acc={validation_accuracy:.4f}"
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), checkpoint_path)

    if not checkpoint_path.is_file():
        parser.error(
            f"Checkpoint not found: {checkpoint_path}. Run training before --evaluate-only."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, class_names)
    real_metrics: dict[str, object] | None = None
    if args.real_data:
        if not args.real_data.is_dir():
            parser.error(f"Real-photo folder not found: {args.real_data.resolve()}")
        real_dataset = ImageFolderWithTransform(
            args.real_data, build_transforms()[1]
        )
        if real_dataset.classes != class_names:
            parser.error(
                "Real-photo classes must match TrashNet classes. "
                f"Expected {class_names}; received {real_dataset.classes}."
            )
        real_loader = DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        real_metrics = evaluate(model, real_loader, device, class_names)
    report = {
        "dataset": str(args.data.resolve()),
        "seed": SEED,
        "split_sizes": {
            "train": sizes[0],
            "validation": sizes[1],
            "test": sizes[2],
        },
        "class_names": class_names,
        "test": test_metrics,
        "real_photos": {
            "metrics": real_metrics,
            "dataset": str(args.real_data.resolve()) if args.real_data else None,
            "status": "evaluated" if real_metrics is not None else "not_evaluated",
        },
        "limitations": [
            "TrashNet images have controlled backgrounds and may not represent real use.",
            "The test metrics describe only the held-out TrashNet split.",
        ],
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model = model.cpu().eval()
    example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        example,
        args.output / "mobilenetv2.onnx",
        input_names=["imagem"],
        output_names=["logits"],
        dynamic_axes={"imagem": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    metadata = {
        "architecture": "MobileNetV2",
        "input_name": "imagem",
        "output_name": "logits",
        "image_size": IMAGE_SIZE,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "class_names": class_names,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Artifacts saved in: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
