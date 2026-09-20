#!/usr/bin/env python3
"""Baixa a biblioteca do Proton Photos (não `/my-files`) para uma pasta local,
usando o CLI oficial (`proton-drive`), de forma idempotente.

Uso:
    python scripts/proton_sync.py
    python scripts/proton_sync.py --limit 200   # só as 200 fotos mais recentes (teste)

Requer o binário oficial `proton-drive` (proton.me/download/drive/cli) já
autenticado (`proton-drive auth login`). Em sistemas com o Secret Service
do D-Bus desativado, autenticar com:

    PROTON_DRIVE_CREDENTIALS_STORE=pass dbus-run-session -- proton-drive auth login

Proton Photos é uma seção separada de `/my-files` (o `filesystem list`
retorna "Path type photos is not supported" para `/photos`). A listagem e
o download usam os comandos dedicados `photo timeline` e `photo download`,
confirmados rodando o CLI de verdade nesta conta:

- `photo timeline --json` (sem `-d`) devolve rápido um array plano
  {nodeUid, captureTime, tags} para a biblioteca inteira. Com `-d` ele
  carrega os detalhes completos (nome, tamanho) de cada foto uma por uma,
  o que para ~21 mil fotos leva vários minutos — por isso este script usa
  a versão sem `-d` e não depende do nome/tamanho para nada.
- `photo download -c rename /photos/<uid> <pasta-local>` baixa um nó pelo
  UID direto (sem precisar buscar o nome antes). O prefixo `/photos/` é
  obrigatório mesmo com o UID completo (só o UID sozinho dá "Path ... not
  supported"). `-c rename` é necessário porque a timeline pode ter várias
  fotos com o mesmo nome (ex.: nomes gerados pelo iOS) — sem isso o CLI
  pergunta interativamente o que fazer e trava em uso não interativo.

O manifesto (`data/manifest.json`) só guarda o UID + captureTime de cada
foto já baixada, sem checagem de mudança de conteúdo: itens do Proton
Photos são efetivamente imutáveis (uma edição gera um nó novo), então
"já está no manifesto" é suficiente para pular no rerun.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_proton(proton_bin: str, *args: str) -> str:
    result = subprocess.run(
        [proton_bin, *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`{proton_bin} {' '.join(args)}` falhou (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def list_timeline(proton_bin: str) -> list[dict]:
    output = run_proton(proton_bin, "photo", "timeline", "--json")
    return json.loads(output)


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def download_batch(
    proton_bin: str, uids: list[str], local_dir: Path, dry_run: bool
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    remote_paths = [f"/photos/{uid}" for uid in uids]
    run_proton(proton_bin, "photo", "download", "-c", "rename", *remote_paths, str(local_dir))


def prune_unwanted_extensions(local_dir: Path, keep_ext: set[str], before: set[str]) -> int:
    """Apaga arquivos baixados nesta chamada (não presentes em `before`) cuja
    extensão não está em `keep_ext`. Retorna quantos foram mantidos."""
    kept = 0
    for path in local_dir.iterdir():
        if not path.is_file() or path in before:
            continue
        if path.suffix.lower() in keep_ext:
            kept += 1
        else:
            path.unlink()
    return kept


def new_files_size(local_dir: Path, before: set[Path]) -> int:
    """Soma em bytes dos arquivos que existem agora em `local_dir` e não
    estavam em `before` — chamado depois de qualquer poda por extensão,
    então reflete só o que ficou de fato em disco."""
    total = 0
    for path in local_dir.iterdir():
        if path.is_file() and path not in before:
            total += path.stat().st_size
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir",
        default=str(REPO_ROOT / "data" / "library"),
        help="Pasta local de destino (default: data/library)",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data" / "manifest.json"),
        help="Arquivo de controle do que já foi baixado (default: data/manifest.json)",
    )
    parser.add_argument(
        "--proton-bin",
        default="proton-drive",
        help="Caminho do binário proton-drive (default: procura no PATH)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Quantas fotos baixar por chamada do CLI (default: 25)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Escaneia no máximo N fotos ainda não sincronizadas nesta execução (default: sem limite)",
    )
    parser.add_argument(
        "--keep-ext",
        default=None,
        help="Lista separada por vírgula de extensões a manter (ex.: .heic,.heif); "
             "o resto do lote baixado é apagado na hora. Útil pra filtrar só fotos "
             "de câmera de verdade (a Proton Photos mistura screenshots, stickers etc.)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Com --keep-ext: para assim que essa quantidade de arquivos combinando "
             "for baixada, mesmo sem esgotar --limit",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help="Para de baixar assim que os arquivos mantidos nesta execução somarem esse "
             "tanto de MB, mesmo sem esgotar --limit/--target-count. Pensado pra baixar a "
             "biblioteca inteira em lotes de tamanho fixo (ex.: 3000 = ~3GB por lote).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Só lista o que faria, sem baixar nada"
    )
    args = parser.parse_args()

    keep_ext = (
        {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}"
         for e in args.keep_ext.split(",") if e.strip()}
        if args.keep_ext else None
    )

    local_dir = Path(args.local_dir)
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    print("Listando a timeline do Proton Photos...")
    try:
        timeline = list_timeline(args.proton_bin)
    except Exception as exc:
        print(f"Erro ao listar o Proton Photos: {exc}", file=sys.stderr)
        return 1

    all_pending = [item for item in timeline if item["nodeUid"] not in manifest]
    pending = all_pending if args.limit is None else all_pending[: args.limit]
    already_synced = len(timeline) - len(all_pending)
    print(
        f"{len(timeline)} fotos no total, {already_synced} já sincronizadas, "
        f"{len(pending)} para baixar nesta execução."
    )

    downloaded = 0
    kept = 0
    failed = 0
    scanned = 0
    kept_bytes = 0
    max_bytes = int(args.max_size_mb * 1024 * 1024) if args.max_size_mb else None
    batches = [
        pending[i : i + args.batch_size] for i in range(0, len(pending), args.batch_size)
    ]
    unit = "match" if keep_ext else "foto"
    bar_total = args.target_count if (keep_ext and args.target_count) else len(pending)
    with tqdm(total=bar_total, desc="Baixando", unit=unit) as bar:
        for batch in batches:
            if keep_ext and args.target_count and kept >= args.target_count:
                break
            if max_bytes and kept_bytes >= max_bytes:
                tqdm.write(f"Atingiu --max-size-mb ({args.max_size_mb} MB), parando.")
                break
            uids = [item["nodeUid"] for item in batch]
            before = set(local_dir.iterdir()) if local_dir.exists() else set()
            try:
                download_batch(args.proton_bin, uids, local_dir, args.dry_run)
            except Exception as exc:
                # Um item do lote pode ter falhado sozinho (nome não decifrável,
                # etc.); tenta um por um para não perder o lote inteiro.
                tqdm.write(f"Lote falhou ({exc}), tentando individualmente...")
                for item in batch:
                    item_before = set(local_dir.iterdir()) if local_dir.exists() else set()
                    try:
                        download_batch(args.proton_bin, [item["nodeUid"]], local_dir, args.dry_run)
                    except Exception as item_exc:
                        tqdm.write(f"Falhou: {item['nodeUid']} ({item_exc})")
                        failed += 1
                        scanned += 1
                        continue
                    if not args.dry_run:
                        manifest[item["nodeUid"]] = {"captureTime": item.get("captureTime")}
                        save_manifest(manifest_path, manifest)
                        if keep_ext:
                            item_kept = prune_unwanted_extensions(local_dir, keep_ext, item_before)
                            kept += item_kept
                            if item_kept:
                                bar.update(item_kept)
                        else:
                            bar.update(1)
                        kept_bytes += new_files_size(local_dir, item_before)
                    downloaded += 1
                    scanned += 1
                continue

            scanned += len(batch)
            if not args.dry_run:
                for item in batch:
                    manifest[item["nodeUid"]] = {"captureTime": item.get("captureTime")}
                save_manifest(manifest_path, manifest)
                if keep_ext:
                    batch_kept = prune_unwanted_extensions(local_dir, keep_ext, before)
                    kept += batch_kept
                    if batch_kept:
                        bar.update(batch_kept)
                else:
                    bar.update(len(batch))
                kept_bytes += new_files_size(local_dir, before)
            downloaded += len(batch)

    kept_mb = kept_bytes / (1024 * 1024)
    if keep_ext:
        print(f"Concluído: {scanned} fotos escaneadas, {kept} mantidas ({kept_mb:.0f} MB, "
              f"extensão em {sorted(keep_ext)}), {failed} falharam, "
              f"{already_synced} já estavam sincronizadas antes desta execução.")
    else:
        print(f"Concluído: {downloaded} baixadas ({kept_mb:.0f} MB), {failed} falharam, "
              f"{already_synced} já estavam sincronizadas antes desta execução.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
