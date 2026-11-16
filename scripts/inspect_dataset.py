#!/usr/bin/env python3
"""Count EcoPlay image files by dataset source and TrashNet class."""

from __future__ import annotations

import argparse
from pathlib import Path


CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
SOURCES = ("trashnet", "real")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def inspect(root: Path) -> int:
    if not root.exists():
        print(f"Pasta de dados ainda não existe: {root}")
        print("Crie-a seguindo a estrutura documentada no README.")
        return 2

    print(f"Dataset: {root.resolve()}")
    print("Formatos contados: " + ", ".join(sorted(IMAGE_SUFFIXES)))
    print()

    for source in SOURCES:
        source_folder = root / source
        print(f"{source}:")
        for class_name in CLASSES:
            class_folder = source_folder / class_name
            if not class_folder.is_dir():
                print(f"  {class_name:10}  pasta ausente")
                continue
            print(f"  {class_name:10}  {count_images(class_folder):>4} imagem(ns)")
        print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conta imagens do TrashNet e das fotos reais por classe."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw"),
        help="Pasta que contém as subpastas trashnet/ e real/ (padrão: data/raw).",
    )
    args = parser.parse_args()
    return inspect(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
