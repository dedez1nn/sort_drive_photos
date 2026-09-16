#!/usr/bin/env python3
"""Detecta rostos nas fotos baixadas, resolve identidades persistentes de
pessoa (não clusters efêmeros) e organiza por symlinks em pastas (sem
duplicar os arquivos em disco).

Uso:
    python scripts/face_cluster.py             # fase 1: fotos com 1-2 rostos
    python scripts/face_cluster.py --phase2    # fase 2: fotos com 3+ rostos adiadas

Roda uma vez em lote (sem servidor). É seguro interromper (Ctrl-C) e rodar
de novo: fotos já indexadas em data/faces_index.json não são reprocessadas
na detecção, e fotos já resolvidas em data/faces.db não são reprocessadas
na identificação.

Por que duas fases: fotos com 3+ rostos têm muito mais chance de casar um
rosto com a pessoa errada (mais candidatos, mais ambiguidade). Processar
primeiro só fotos com 1-2 rostos consolida as identidades (pessoa_001,
pessoa_002, ...) com evidência mais confiável, e só depois --phase2 usa
essas identidades já firmes pra resolver as fotos de grupo maiores, em vez
de arriscar criar várias pessoas novas erradas de uma vez.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import face_recognition
import numpy as np
from tqdm import tqdm

import identity_store
from identity_store import PersonGraph

try:
    import pillow_heif

    pillow_heif.register_heif_opener()  # permite ler .heic/.heif (fotos de iPhone)
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
# .png fica de fora de propósito: no Proton Photos vindo do iPhone, .png é
# sempre screenshot (a câmera só grava .heic/.jpg), nunca foto de verdade.
# Rostos em miniaturas de avatar dentro de prints de conversa (Instagram/
# WhatsApp) se repetiam entre capturas de tela do mesmo contato e o DBSCAN
# agrupava isso como se fosse "uma pessoa", inflando os clusters com ruído
# em vez de fotos de verdade da mesma pessoa. Confirmado abrindo as imagens.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".heif", ".bmp", ".tiff"}


def find_images(library_dir: Path):
    for path in sorted(library_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_index(index_path: Path) -> dict:
    if index_path.exists():
        return json.loads(index_path.read_text())
    return {"faces": []}


def save_index(index_path: Path, index: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))


def already_indexed(index: dict, image_path: Path) -> bool:
    key = str(image_path)
    mtime = image_path.stat().st_mtime
    return any(
        f["image"] == key and f.get("mtime") == mtime for f in index["faces"]
    )


def extract_faces(image_path: Path, model: str) -> list[np.ndarray]:
    image = face_recognition.load_image_file(str(image_path))
    locations = face_recognition.face_locations(image, model=model)
    return face_recognition.face_encodings(image, locations)


def _faces_by_image(index: dict) -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = {}
    for face in index["faces"]:
        by_image.setdefault(face["image"], []).append(face)
    return by_image


def run_phase1(
    conn, graph: PersonGraph, faces_by_image: dict[str, list[dict]],
    match_threshold: float, uncertain_threshold: float,
) -> None:
    """Fotos com 1 rosto: identifica direto. Com 2 rostos: identifica os
    dois independentemente (nunca assume que são pessoas novas só por
    serem duas) e registra que elas aparecem juntas. Com 3+: adia pra
    --phase2, sem tentar identificar nada agora."""
    for image_path in tqdm(sorted(faces_by_image), desc="Resolvendo identidades", unit="foto"):
        path = Path(image_path)
        if not path.exists():
            continue
        faces = faces_by_image[image_path]
        n = len(faces)
        photo_id = identity_store.hash_file(path)
        if identity_store.is_excluded(conn, photo_id):
            continue
        status = identity_store.get_photo_status(conn, photo_id)
        if status in ("phase1_done", "phase2_done", "deferred_multi_face"):
            continue
        mtime = path.stat().st_mtime

        if n >= 3:
            identity_store.upsert_photo(conn, photo_id, image_path, mtime, n, "deferred_multi_face")
            for face in faces:
                face_id = f"{photo_id}:{face['face_id']}"
                embedding = np.array(face["encoding"])
                identity_store.insert_face(
                    conn, face_id, photo_id, face["face_id"], embedding, None, None, "deferred"
                )
            continue

        resolved: list[str | None] = []
        for face in faces:
            embedding = np.array(face["encoding"])
            face_id = f"{photo_id}:{face['face_id']}"
            person_id, dist = graph.best_match(embedding)
            if person_id is not None and dist <= match_threshold:
                graph.confirm(person_id, embedding)
                identity_store.insert_face(
                    conn, face_id, photo_id, face["face_id"], embedding, person_id, dist, "assigned"
                )
            elif person_id is not None and dist <= uncertain_threshold:
                identity_store.insert_face(
                    conn, face_id, photo_id, face["face_id"], embedding, None, dist, "uncertain"
                )
                person_id = None
            else:
                person_id = graph.create_person(embedding)
                identity_store.insert_face(
                    conn, face_id, photo_id, face["face_id"], embedding, person_id, 0.0, "assigned"
                )
            resolved.append(person_id)

        identity_store.upsert_photo(conn, photo_id, image_path, mtime, n, "phase1_done")
        if n == 2 and resolved[0] and resolved[1]:
            identity_store.add_relationship(conn, resolved[0], resolved[1], photo_id)

    rediscover_uncertain(conn, graph, match_threshold)


def rediscover_uncertain(conn, graph: PersonGraph, match_threshold: float) -> None:
    """Rostos que ficaram sem pessoa (zona ambígua) podem, se aparecerem
    juntos repetidas vezes, formar uma pessoa nova recorrente — sem
    contaminar quem já está confirmado, já que nunca comparamos com
    pessoas existentes aqui, só entre os próprios incertos."""
    rows = conn.execute(
        "SELECT face_id, embedding FROM faces WHERE status = 'uncertain' AND person_id IS NULL"
    ).fetchall()
    pool = [(row["face_id"], identity_store.unpack(row["embedding"])) for row in rows]
    used: set[str] = set()
    for i, (face_id_a, emb_a) in enumerate(pool):
        if face_id_a in used:
            continue
        group = [i]
        for j in range(i + 1, len(pool)):
            face_id_b, emb_b = pool[j]
            if face_id_b in used:
                continue
            if float(np.linalg.norm(emb_a - emb_b)) <= match_threshold:
                group.append(j)
        if len(group) < 2:
            continue
        person_id = graph.create_person(pool[group[0]][1])
        for idx in group:
            face_id, embedding = pool[idx]
            if idx != group[0]:
                graph.confirm(person_id, embedding)
            conn.execute(
                "UPDATE faces SET person_id = ?, status = 'assigned' WHERE face_id = ?",
                (person_id, face_id),
            )
            used.add(face_id)


def run_phase2(
    conn, graph: PersonGraph, faces_by_image: dict[str, list[dict]], match_threshold: float
) -> None:
    """Fotos com 3+ rostos, adiadas na fase 1: casa cada rosto com as
    pessoas já consolidadas (só match confiante, nunca cria pessoa nova
    aqui — o risco de errar com muitos candidatos na foto é maior)."""
    deferred = conn.execute(
        "SELECT photo_id, path FROM photos WHERE status = 'deferred_multi_face'"
    ).fetchall()
    for row in tqdm(deferred, desc="Fase 2 (fotos com 3+ rostos)", unit="foto"):
        photo_id, path = row["photo_id"], row["path"]
        faces = faces_by_image.get(path, [])
        resolved: list[str] = []
        for face in faces:
            face_id = f"{photo_id}:{face['face_id']}"
            embedding = np.array(face["encoding"])
            person_id, dist = graph.best_match(embedding)
            if person_id is not None and dist <= match_threshold:
                graph.confirm(person_id, embedding)
                conn.execute(
                    "UPDATE faces SET person_id = ?, confidence = ?, status = 'assigned' "
                    "WHERE face_id = ?",
                    (person_id, dist, face_id),
                )
                resolved.append(person_id)
        identity_store.set_photo_status(conn, photo_id, "phase2_done")
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                identity_store.add_relationship(conn, resolved[i], resolved[j], photo_id)


def resolve_identities(
    conn, index: dict, match_threshold: float, uncertain_threshold: float, phase2: bool
) -> None:
    faces_by_image = _faces_by_image(index)
    graph = PersonGraph.load(conn)

    if phase2:
        run_phase2(conn, graph, faces_by_image, match_threshold)
    else:
        run_phase1(conn, graph, faces_by_image, match_threshold, uncertain_threshold)

    for keep_id, removed_id, dist in graph.reconcile(match_threshold):
        tqdm.write(f"Pessoas fundidas (mesma identidade): {removed_id} -> {keep_id} (distância {dist:.3f})")

    conn.commit()


def build_person_folders_from_db(conn, output_dir: Path) -> int:
    """Reconstrói data/by_person a partir do banco (fonte da verdade é o
    person_id no SQLite, a pasta é só uma visão derivada e descartável).

    Pasta pessoa_XXX só tem foto onde essa pessoa aparece SOZINHA (1 rosto
    na foto). Foto com 2+ pessoas identificadas vai pra uma pasta com o
    nome da combinação exata de pessoas (ex.: person_002_e_person_004),
    reaproveitada automaticamente sempre que essa mesma dupla/trio
    reaparecer numa foto futura — sem duplicar a pessoa em pasta separada
    nem misturar a foto de grupo na pasta solo dela."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    photo_paths = {r["photo_id"]: r["path"] for r in conn.execute("SELECT photo_id, path FROM photos")}
    num_faces_by_photo = {
        r["photo_id"]: r["num_faces"] for r in conn.execute("SELECT photo_id, num_faces FROM photos")
    }
    rows = conn.execute("SELECT photo_id, person_id, status FROM faces").fetchall()

    by_photo: dict[str, list] = {}
    for row in rows:
        by_photo.setdefault(row["photo_id"], []).append(row)

    all_person_ids: set[str] = set()
    for photo_id, photo_rows in by_photo.items():
        src_str = photo_paths.get(photo_id)
        if not src_str:
            continue
        src = Path(src_str)
        resolved = sorted({r["person_id"] for r in photo_rows if r["person_id"]})
        all_person_ids.update(resolved)
        num_faces = num_faces_by_photo.get(photo_id, len(photo_rows))

        if not resolved:
            folder = "aguardando_fase2" if any(r["status"] == "deferred" for r in photo_rows) else "revisao_manual"
        elif len(resolved) == 1 and num_faces == 1:
            folder = resolved[0]  # solo de verdade: só um rosto na foto inteira
        elif len(resolved) == 1:
            folder = f"{resolved[0]}_em_grupo"  # tem gente não identificada junto
        else:
            folder = "_e_".join(resolved)  # combinação exata de pessoas

        person_dir = output_dir / folder
        person_dir.mkdir(parents=True, exist_ok=True)
        dest = person_dir / src.name
        if not dest.exists():
            dest.symlink_to(src.resolve())

    return len(all_person_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir", default=str(REPO_ROOT / "data" / "library"),
        help="Pasta com as fotos baixadas (default: data/library)",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "data" / "by_person"),
        help="Pasta onde criar os symlinks organizados por pessoa (default: data/by_person)",
    )
    parser.add_argument(
        "--index", default=str(REPO_ROOT / "data" / "faces_index.json"),
        help="Arquivo de índice de rostos já processados (default: data/faces_index.json)",
    )
    parser.add_argument(
        "--db", default=str(REPO_ROOT / "data" / "faces.db"),
        help="Banco SQLite com o grafo de identidades (default: data/faces.db)",
    )
    parser.add_argument(
        "--model", choices=["hog", "cnn"], default="hog",
        help="Modelo de detecção facial: hog é bem mais leve em CPU (default)",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.5,
        help="Distância máxima pra considerar 'mesma pessoa' com confiança (default 0.5)",
    )
    parser.add_argument(
        "--uncertain-threshold", type=float, default=0.58,
        help="Entre --match-threshold e este valor fica 'incerto' (revisão manual) em vez "
             "de forçar o match errado; acima disso vira pessoa nova (default 0.58)",
    )
    parser.add_argument(
        "--save-every", type=int, default=25,
        help="Salva o índice a cada N imagens processadas (default 25)",
    )
    parser.add_argument(
        "--phase2", action="store_true",
        help="Processa as fotos com 3+ rostos adiadas na fase 1, casando com as pessoas "
             "já consolidadas (não roda a fase 1 nem cria pessoa nova)",
    )
    parser.add_argument(
        "--exclude", nargs=2, metavar=("ARQUIVO", "MOTIVO"), action="append", default=[],
        help="Marca uma foto como 'não é rosto de verdade' (foto de tela, objeto, etc.) — "
             "desfaz qualquer pessoa/relação já criada a partir dela e nunca mais reprocessa. "
             "Repete a flag pra excluir várias de uma vez. Não roda detecção nem fases, só "
             "aplica a exclusão e reconstrói as pastas.",
    )
    args = parser.parse_args()

    library_dir = Path(args.library_dir)
    index_path = Path(args.index)
    output_dir = Path(args.output_dir)
    db_path = Path(args.db)

    if args.exclude:
        conn = identity_store.connect(db_path)
        try:
            for file_arg, reason in args.exclude:
                photo_path = Path(file_arg)
                if not photo_path.is_absolute():
                    photo_path = library_dir / photo_path
                photo_id = identity_store.exclude_photo_by_path(conn, photo_path, reason)
                print(f"Excluída: {photo_path.name} ({reason})")
            conn.commit()
            num_people = build_person_folders_from_db(conn, output_dir)
        finally:
            conn.close()
        print(f"Pastas reconstruídas: {num_people} pessoas em {output_dir}")
        return 0

    index = load_index(index_path)
    images = list(find_images(library_dir))
    pending = [img for img in images if not already_indexed(index, img)]
    print(f"{len(images)} imagens no total, {len(pending)} para processar.")

    processed_since_save = 0
    for image_path in tqdm(pending, desc="Detectando rostos", unit="imagem"):
        try:
            encodings = extract_faces(image_path, args.model)
        except Exception as exc:
            tqdm.write(f"Falhou: {image_path} ({exc})")
            continue

        mtime = image_path.stat().st_mtime
        for face_id, encoding in enumerate(encodings):
            index["faces"].append(
                {
                    "image": str(image_path),
                    "face_id": face_id,
                    "mtime": mtime,
                    "encoding": encoding.tolist(),
                }
            )

        processed_since_save += 1
        if processed_since_save >= args.save_every:
            save_index(index_path, index)
            processed_since_save = 0

    save_index(index_path, index)

    print(f"{len(index['faces'])} rostos indexados no total. Resolvendo identidades...")
    conn = identity_store.connect(db_path)
    try:
        resolve_identities(
            conn, index,
            match_threshold=args.match_threshold,
            uncertain_threshold=args.uncertain_threshold,
            phase2=args.phase2,
        )
        num_people = build_person_folders_from_db(conn, output_dir)
    finally:
        conn.close()

    fase = "fase 2 (fotos com 3+ rostos)" if args.phase2 else "fase 1 (fotos com 1-2 rostos)"
    print(f"Pronto ({fase}): {num_people} pessoas em {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
