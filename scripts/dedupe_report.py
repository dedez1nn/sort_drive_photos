#!/usr/bin/env python3
"""Gera um relatório de arquivos duplicados (hash sha256 idêntico) dentro da
biblioteca local baixada. Não apaga nada — só reporta, para decisão manual.

Uso:
    python scripts/dedupe_report.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent


def sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir", default=str(REPO_ROOT / "data" / "library"),
        help="Pasta com as fotos baixadas (default: data/library)",
    )
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "data" / "duplicates_report.json"),
        help="Arquivo de saída com o relatório (default: data/duplicates_report.json)",
    )
    args = parser.parse_args()

    library_dir = Path(args.library_dir)
    files = [p for p in library_dir.rglob("*") if p.is_file()]

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in tqdm(files, desc="Calculando hashes", unit="arquivo"):
        try:
            by_hash[sha256_of(path)].append(path)
        except OSError as exc:
            tqdm.write(f"Não consegui ler {path}: {exc}")

    duplicate_groups = [paths for paths in by_hash.values() if len(paths) > 1]

    report = []
    total_wasted = 0
    for paths in duplicate_groups:
        size = paths[0].stat().st_size
        wasted = size * (len(paths) - 1)
        total_wasted += wasted
        report.append(
            {
                "files": [str(p) for p in paths],
                "size_bytes": size,
                "wasted_bytes": wasted,
            }
        )

    report.sort(key=lambda g: g["wasted_bytes"], reverse=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(
        f"{len(files)} arquivos analisados, {len(duplicate_groups)} grupos de "
        f"duplicatas, {human_size(total_wasted)} desperdiçados."
    )
    print(f"Relatório salvo em {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
