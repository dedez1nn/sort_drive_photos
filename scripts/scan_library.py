#!/usr/bin/env python3
"""Varredura fria da biblioteca inteira: extrai os rostos e descarta as fotos.

O problema que isto resolve: a biblioteca tem ~21 mil fotos (~27 GB) e
esta máquina tem menos de 2 GB livres. Não dá para ter tudo em disco, e o
`sweep_library.sh` — que baixa, agrupa e sincroniza um lote por vez — não
serve para responder "quem são as 20 pessoas mais frequentes de TODA a
biblioteca", porque ele decide as identidades lote a lote, na ordem em que
as fotos chegam.

A saída é separar as duas coisas. Esta varredura só colhe, em uma passada:

    baixa lote -> detecta rostos -> guarda (embedding, uid, recorte)
    -> apaga o lote -> próximo

O que sobra ocupa ~90 MB para a biblioteca inteira e não depende mais das
fotos: os embeddings agrupam, os recortes deixam revisar quem é quem
visualmente, e o `uid` deixa rebaixar a foto original das pessoas que
interessarem. O ranqueamento vem depois, offline, em
`scripts/rank_people.py`.

Uso:
    python scripts/scan_library.py                 # varre tudo, em lotes de 1 GB
    python scripts/scan_library.py --max-photos 500  # amostra, para conferir antes
    python scripts/scan_library.py --refresh-catalog # refaz o catálogo do Proton

É seguro interromper (Ctrl-C) e rodar de novo: o que já foi varrido está
em `data/faces/scanned.jsonl` e não é refeito.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

import face_embedder
import identity_store
import proton_catalog
from face_store import FaceStore, make_thumb

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_catalog(path: Path, proton_bin: str, refresh: bool) -> list[dict]:
    if refresh or not path.exists():
        print("Listando a timeline do Proton Photos com detalhes (leva alguns minutos)...")
        entries = proton_catalog.fetch_catalog(proton_bin)
        proton_catalog.save_catalog(path, entries)
        print(f"Catálogo salvo: {len(entries)} fotos em {path}")
        return entries
    entries = proton_catalog.load_catalog(path)
    print(f"Catálogo: {len(entries)} fotos ({path}; use --refresh-catalog para atualizar)")
    return entries


def download_batch(proton_bin: str, uids: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    proton_catalog.run_proton(
        proton_bin, "photo", "download", "-c", "rename",
        *[f"/photos/{uid}" for uid in uids], str(dest),
    )


def scan_batch(
    batch: list[dict], work_dir: Path, store: FaceStore, det_size: int,
    min_face_px: int, min_det_score: float, keep_thumbs: bool,
) -> tuple[int, int]:
    """Processa um lote já baixado em `work_dir`. Devolve (fotos, rostos).

    O arquivo é casado com o `uid` pelo nome: o lote foi montado sem nomes
    repetidos (ver `proton_catalog.plan_batches`) e baixado numa pasta
    vazia, então a correspondência é exata."""
    por_nome = {entry["name"]: entry for entry in batch}
    fotos = rostos = 0
    for path in sorted(work_dir.iterdir()):
        if not path.is_file():
            continue
        entry = por_nome.get(path.name)
        if entry is None:
            # O CLI renomeou (colisão inesperada) ou baixou algo a mais.
            # Sem saber de qual uid veio, o rosto não serviria para
            # rebaixar a foto depois — melhor pular e registrar.
            tqdm.write(f"Sem correspondência de uid: {path.name} (pulado)")
            continue
        try:
            image = face_embedder.load_image(path)
            faces = face_embedder.extract_faces(image, det_size)
        except Exception as exc:
            tqdm.write(f"Falhou: {path.name} ({exc})")
            continue

        photo_id = identity_store.hash_file(path)
        guardados = 0
        for face in faces:
            if face["face_px"] < min_face_px or face["det_score"] < min_det_score:
                continue
            store.append(
                {
                    "photo_id": photo_id,
                    "face_index": face["face_id"],
                    "uid": entry["uid"],
                    "name": entry["name"],
                    "capture_time": entry.get("capture_time"),
                    "bbox": face["bbox"],
                    "det_score": face["det_score"],
                    "face_px": face["face_px"],
                },
                np.array(face["encoding"]),
                make_thumb(image, face["bbox"]) if keep_thumbs else None,
            )
            guardados += 1
        store.mark_scanned(entry["uid"], photo_id, guardados)
        fotos += 1
        rostos += guardados
    return fotos, rostos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(REPO_ROOT / "data" / "faces"),
                        help="Pasta do acervo de rostos (default: data/faces)")
    parser.add_argument("--catalog", default=str(proton_catalog.DEFAULT_CATALOG),
                        help="Catálogo da timeline (default: data/catalog.jsonl)")
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "data" / "scan_tmp"),
                        help="Pasta temporária dos lotes, esvaziada a cada ciclo")
    parser.add_argument("--proton-bin", default="proton-drive")
    parser.add_argument("--refresh-catalog", action="store_true",
                        help="Refaz o catálogo mesmo que já exista")
    parser.add_argument("--batch-mb", type=float, default=1000,
                        help="Teto de disco por lote, em MB (default 1000). O lote é "
                             "apagado antes do próximo, então este é o pico de uso")
    parser.add_argument("--batch-items", type=int, default=250,
                        help="Máximo de fotos por chamada do CLI (default 250). A média é 0,7 MB por "
                             "foto, então 250 dão ~170 MB — bem dentro do teto de disco")
    parser.add_argument("--max-photos", type=int, default=None,
                        help="Para depois de varrer N fotos (para amostrar antes de rodar tudo)")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--min-face-px", type=int, default=50)
    parser.add_argument("--min-det-score", type=float, default=0.6)
    parser.add_argument("--no-thumbs", action="store_true",
                        help="Não guarda o recorte de cada rosto (economiza ~250 MB, mas "
                             "aí só dá pra ver quem é cada pessoa rebaixando as fotos)")
    parser.add_argument("--keep-screenshots", action="store_true",
                        help="Não descarta screenshot/sticker pelo catálogo")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Não descarta foto com contentHash repetido")
    args = parser.parse_args()

    store = FaceStore(Path(args.store))
    store.prepare()
    work_dir = Path(args.work_dir)

    entries = ensure_catalog(Path(args.catalog), args.proton_bin, args.refresh_catalog)
    total = len(entries)

    if not args.keep_screenshots:
        entries = [e for e in entries if proton_catalog.is_camera_photo(e)]
        print(f"  {total - len(entries)} descartadas por serem screenshot/sticker/vídeo")
    if not args.keep_duplicates:
        entries, dropped = proton_catalog.dedupe_by_content(entries)
        print(f"  {dropped} descartadas por serem duplicata exata (mesmo contentHash)")

    ja_vistas = store.scanned_uids()
    pendentes = [e for e in entries if e["uid"] not in ja_vistas]
    print(f"  {len(ja_vistas)} já varridas, {len(pendentes)} pendentes "
          f"({sum(e['size'] for e in pendentes) / 1e9:.1f} GB a baixar no total)")
    if args.max_photos:
        pendentes = pendentes[: args.max_photos]
        print(f"  limitado a {len(pendentes)} nesta execução (--max-photos)")
    if not pendentes:
        print("Nada a fazer.")
        return 0

    batches = proton_catalog.plan_batches(
        pendentes, int(args.batch_mb * 1e6), args.batch_items
    )
    fotos = rostos = 0
    with tqdm(total=len(pendentes), desc="Varrendo", unit="foto") as bar:
        for batch in batches:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            try:
                download_batch(args.proton_bin, [e["uid"] for e in batch], work_dir)
            except Exception as exc:
                tqdm.write(f"Lote falhou no download ({exc}); pulando {len(batch)} fotos")
                bar.update(len(batch))
                continue
            f, r = scan_batch(
                batch, work_dir, store, args.det_size,
                args.min_face_px, args.min_det_score, not args.no_thumbs,
            )
            fotos += f
            rostos += r
            bar.update(len(batch))
            bar.set_postfix(rostos=rostos)
    if work_dir.exists():
        shutil.rmtree(work_dir)

    print(f"\nVarridas {fotos} fotos, {rostos} rostos guardados.")
    print(f"Acervo: {store.count()} rostos em {args.store} "
          f"({sum(p.stat().st_size for p in Path(args.store).rglob('*') if p.is_file()) / 1e6:.0f} MB)")
    print("Próximo passo: python scripts/rank_people.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
