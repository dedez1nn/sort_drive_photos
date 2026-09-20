#!/usr/bin/env python3
"""Detecta rostos nas fotos baixadas, resolve identidades persistentes de
pessoa (não clusters efêmeros) e organiza por symlinks em pastas (sem
duplicar os arquivos em disco).

Uso:
    python scripts/face_cluster.py             # fase 1: fotos com 1-2 rostos
    python scripts/face_cluster.py --phase2    # fase 2: fotos com 3+ rostos adiadas
    python scripts/face_cluster.py --recluster # refaz todas as identidades do zero

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

Quatro defesas contra os dois modos de falha que apareceram aqui — uma
identidade engolindo pessoas diferentes por encadeamento, e (depois de
corrigir isso) a mesma pessoa fragmentada em três identidades:

1. Embedding ArcFace 512-d + detector SCRFD (`face_embedder.py`), no
   lugar do HOG + embedding 128-d do dlib. Medido nos mesmos rostos, com
   o dlib as distâncias de "mesma pessoa" (máx 0.727) e "pessoas
   diferentes" (mín 0.554) se INVERTEM — nenhum limiar separa. É a razão
   de o rapaz de óculos aparecer em três pastas.
2. Portão de qualidade na detecção (`--min-det-score`, `--min-face-px`):
   agora usando a confiança do próprio detector, não uma aproximação de
   nitidez calculada por fora.
3. Restrição de exclusão mútua: dois rostos da MESMA foto nunca viram a
   mesma pessoa. Informação que estava nos dados e não era usada, e que
   é o que permite subir o limiar o bastante pra reunir a mesma pessoa
   em condições diferentes sem fundir quem aparece ao lado dela.
4. Match por quórum com freio de coerência, e separação automática de
   identidade contaminada (`.split_incoherent`), no lugar do antigo "só
   renomeia a pasta pra _dispersa e segue".
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import face_embedder
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
# WhatsApp) se repetiam entre capturas de tela do mesmo contato e o
# agrupamento juntava isso como se fosse "uma pessoa", inflando os grupos
# com ruído em vez de fotos de verdade da mesma pessoa. Confirmado abrindo
# as imagens.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".heic", ".heif", ".bmp", ".tiff"}

# Screenshot do Android é .jpg (ao contrário do iPhone, que só gera .png),
# então a exclusão por extensão acima não pega. Mas o nome é padronizado
# pelo próprio Android: "Screenshot_AAAA-MM-DD-HH-MM-SS-mmm_pacote.app.jpg".
# Confirmado num lote real: 365 de 1115 fotos (33%) eram esse padrão — a
# maioria prints de Discord/Twitter com avatarzinho genérico ao lado de
# cada mensagem, que o detector pega como rosto e agrupa tudo numa "pessoa"
# só (person_001 chegou a ter 401 rostos, de 444 detectados no lote todo).
SCREENSHOT_NAME_RE = re.compile(r"^screenshot_", re.IGNORECASE)

# Tolerância pra considerar uma imagem "16:9". Fotos de câmera de iPhone
# são 4:3 (3024x4032); o que chega em 16:9 exato (3840x2160, 3664x2062) é
# frame de vídeo/Live Photo extraído pelo Proton Photos. Só usado com
# --skip-video-frames, porque uma foto panorâmica legítima também pode
# cair perto de 16:9.
VIDEO_FRAME_RATIO = 16 / 9
VIDEO_FRAME_TOLERANCE = 0.02

# Rosto menor que isso (menor lado da caixa, em pixels) é descartado antes
# de virar identidade. O SCRFD detecta rosto pequeno bem melhor que o HOG,
# então o piso pode ser mais baixo que os 80px que o HOG exigia — mas um
# rosto de 30px continua sendo alguém de fundo que não dá pra reconhecer.
DEFAULT_MIN_FACE_PX = 50

# Confiança do próprio detector. É o sinal que faltava no HOG: ele não
# devolvia confiança nenhuma, e a única defesa era medir tamanho e nitidez
# do recorte por fora (variância do laplaciano), que era uma aproximação
# grosseira. Abaixo de 0.6 o que aparece é majoritariamente desenho,
# textura e rosto em cartaz.
DEFAULT_MIN_DET_SCORE = 0.6


def is_video_frame(path: Path) -> bool:
    try:
        width, height = Image.open(path).size
    except Exception:
        return False
    if not height:
        return False
    return abs(width / height - VIDEO_FRAME_RATIO) <= VIDEO_FRAME_TOLERANCE


def find_images(library_dir: Path, skip_video_frames: bool = False):
    """Sempre devolve caminho absoluto resolvido: o índice é chaveado pelo
    caminho como string, então rodar com `--library-dir data/library` e com
    o caminho absoluto geraria duas entradas pra mesma foto e redetectaria
    a biblioteca inteira."""
    for path in sorted(library_dir.resolve().rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if SCREENSHOT_NAME_RE.match(path.name):
            continue
        if skip_video_frames and is_video_frame(path):
            continue
        yield path


def load_index(index_path: Path) -> dict:
    """O índice guarda qual modelo gerou os embeddings. Vetores de
    modelos diferentes não são comparáveis (dimensão e escala diferentes),
    então misturá-los produziria agrupamento silenciosamente errado —
    um índice de outro modelo é descartado, não migrado."""
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index.setdefault("scanned", {})
        if index.get("embedder") != face_embedder.EMBEDDER:
            anterior = index.get("embedder", "dlib (versão anterior, sem marca)")
            print(
                f"Índice gerado por outro modelo ({anterior}); o atual é "
                f"{face_embedder.EMBEDDER}. Embeddings de modelos diferentes não são "
                f"comparáveis, então o índice será refeito do zero."
            )
            return {"faces": [], "scanned": {}, "embedder": face_embedder.EMBEDDER}
        return index
    return {"faces": [], "scanned": {}, "embedder": face_embedder.EMBEDDER}


def save_index(index_path: Path, index: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))


def already_indexed(index: dict, index_by_image: dict[str, list[dict]], image_path: Path) -> bool:
    """Índice de uma versão anterior conta como não indexado: sem
    `det_score` não dá pra aplicar o portão de qualidade, e é melhor
    redetectar a foto uma vez do que deixá-la passando por cima do filtro
    pra sempre.

    `index["scanned"]` existe pra que foto SEM rosto nenhum também conte
    como já processada. Sem isso ela não deixa rastro no índice (que só
    tem rostos) e é redetectada em toda execução — na biblioteca de teste eram
    168 das 243 fotos, o passo mais caro do pipeline rodando à toa toda
    vez que se ajusta um threshold."""
    key = str(image_path)
    mtime = image_path.stat().st_mtime
    faces = index_by_image.get(key)
    if faces is None:
        return index.get("scanned", {}).get(key) == mtime
    return all(f.get("mtime") == mtime and "det_score" in f for f in faces)


def extract_faces(image_path: Path, det_size: int) -> list[dict]:
    """Detecta e descreve cada rosto da foto, junto da confiança do
    detector que o portão de qualidade usa depois. Medir aqui e guardar no
    índice (em vez de filtrar na hora) deixa mudar `--min-det-score`/
    `--min-face-px` e reprocessar sem ter que redetectar tudo de novo."""
    image = face_embedder.load_image(image_path)
    return face_embedder.extract_faces(image, det_size)


def passes_quality(face: dict, min_face_px: int, min_det_score: float) -> bool:
    return face.get("face_px", 0) >= min_face_px and face.get("det_score", 0.0) >= min_det_score


def faces_by_image(
    index: dict, min_face_px: int = 0, min_det_score: float = 0.0
) -> dict[str, list[dict]]:
    """Agrupa o índice por foto, já aplicando o portão de qualidade. Uma
    foto cujos rostos foram todos reprovados some do resultado — não vira
    "foto sem pessoa identificada", ela simplesmente não tem rosto
    aproveitável."""
    by_image: dict[str, list[dict]] = {}
    for face in index["faces"]:
        if not passes_quality(face, min_face_px, min_det_score):
            continue
        by_image.setdefault(face["image"], []).append(face)
    return by_image


def assign_face(
    conn,
    graph: PersonGraph,
    face_id: str,
    photo_id: str,
    face: dict,
    embedding: np.ndarray,
    match_threshold: float,
    uncertain_threshold: float,
    coherence_limit: float,
    allow_new_person: bool,
    excluir: set[str] | None = None,
) -> str | None:
    """Decide de quem é um rosto e grava a decisão.

    O match só é aceito se passar por duas portas: a distância de quórum
    (`graph.score`) estar dentro de `match_threshold` E absorver esse
    rosto não estragar a coerência interna da pessoa. A segunda porta é o
    freio que impede a bola-de-neve de começar — sem ela, basta uma
    sequência de matches "no limite" pra uma identidade ir derivando até
    virar um balaio de gente diferente.

    `excluir` são as pessoas já identificadas nesta mesma foto: ninguém
    aparece duas vezes numa foto, então elas ficam fora da disputa. Isso
    protege exatamente o caso mais difícil — quem convive com você
    aparece ao seu lado nas fotos, e é com essas pessoas que o embedding
    mais se confunde."""
    person_id, dist = graph.best_match(embedding, excluir=excluir)

    if person_id is not None and dist <= match_threshold:
        if graph.would_stay_coherent(person_id, embedding, coherence_limit):
            graph.confirm(person_id, face_id, embedding)
            identity_store.insert_face(
                conn, face_id, photo_id, face["face_id"], embedding, person_id, dist, "assigned"
            )
            return person_id
        # Casou, mas juntar degradaria a identidade: guarda como incerto
        # em vez de forçar. Se for mesmo alguém recorrente, o
        # `rediscover_uncertain` monta a pessoa depois, a partir dos
        # próprios incertos.
        identity_store.insert_face(
            conn, face_id, photo_id, face["face_id"], embedding, None, dist, "uncertain"
        )
        return None

    if person_id is not None and dist <= uncertain_threshold:
        identity_store.insert_face(
            conn, face_id, photo_id, face["face_id"], embedding, None, dist, "uncertain"
        )
        return None

    if not allow_new_person:
        identity_store.insert_face(
            conn, face_id, photo_id, face["face_id"], embedding, None, dist, "uncertain"
        )
        return None

    new_id = graph.create_person(face_id, embedding)
    identity_store.insert_face(
        conn, face_id, photo_id, face["face_id"], embedding, new_id, 0.0, "assigned"
    )
    return new_id


def run_phase1(
    conn, graph: PersonGraph, by_image: dict[str, list[dict]],
    match_threshold: float, uncertain_threshold: float, coherence_limit: float,
) -> None:
    """Fotos com 1 rosto: identifica direto. Com 2 rostos: identifica os
    dois independentemente (nunca assume que são pessoas novas só por
    serem duas) e registra que elas aparecem juntas. Com 3+: adia pra
    --phase2, sem tentar identificar nada agora."""
    for image_path in tqdm(sorted(by_image), desc="Resolvendo identidades", unit="foto"):
        path = Path(image_path)
        if not path.exists():
            continue
        faces = by_image[image_path]
        n = len(faces)
        photo_id = identity_store.hash_file(path)
        if identity_store.is_excluded(conn, photo_id):
            continue
        status = identity_store.get_photo_status(conn, photo_id)
        if status in ("phase1_done", "phase2_done", "deferred_multi_face", "reclustered"):
            continue
        mtime = path.stat().st_mtime

        if n >= 3:
            identity_store.upsert_photo(conn, photo_id, image_path, mtime, n, "deferred_multi_face")
            for face in faces:
                identity_store.insert_face(
                    conn, f"{photo_id}:{face['face_id']}", photo_id, face["face_id"],
                    np.array(face["encoding"]), None, None, "deferred",
                )
            continue

        resolved: list[str | None] = []
        for face in faces:
            resolved.append(assign_face(
                conn, graph, f"{photo_id}:{face['face_id']}", photo_id, face,
                np.array(face["encoding"]), match_threshold, uncertain_threshold,
                coherence_limit, allow_new_person=True,
                excluir={p for p in resolved if p},
            ))
        # Marca a foto como pronta só depois de resolver todos os rostos
        # dela: interrompida no meio, ela é reprocessada inteira no rerun
        # (os inserts de rosto são idempotentes) em vez de ficar registrada
        # como pronta com metade dos rostos.
        identity_store.upsert_photo(conn, photo_id, image_path, mtime, n, "phase1_done")
        if n == 2 and resolved[0] and resolved[1]:
            identity_store.add_relationship(conn, resolved[0], resolved[1], photo_id)

    rediscover_uncertain(conn, graph, match_threshold, coherence_limit)


def rediscover_uncertain(
    conn, graph: PersonGraph, match_threshold: float, coherence_limit: float
) -> None:
    """Rostos que ficaram sem pessoa (zona ambígua, ou recusados pelo
    freio de coerência) podem, se aparecerem juntos repetidas vezes,
    formar uma pessoa nova recorrente — sem contaminar quem já está
    confirmado, já que nunca comparamos com pessoas existentes aqui, só
    entre os próprios incertos.

    Antes isso era um agrupamento guloso "todo mundo a menos de X do
    primeiro da fila", que tem o mesmo defeito de encadeamento do resto
    do pipeline antigo. Agora é a mesma clusterização hierárquica com
    average linkage usada no `--recluster`, e o grupo ainda precisa
    nascer coerente pra virar pessoa."""
    rows = conn.execute(
        "SELECT face_id, embedding FROM faces WHERE status = 'uncertain' AND person_id IS NULL"
    ).fetchall()
    if len(rows) < 2:
        return
    face_ids = [row["face_id"] for row in rows]
    embeddings = [identity_store.unpack(row["embedding"]) for row in rows]

    labels = identity_store.cluster_labels(
        embeddings, match_threshold, groups=[f.rsplit(":", 1)[0] for f in face_ids]
    )
    groups: dict[int, list[int]] = {}
    for pos, label in enumerate(labels):
        groups.setdefault(int(label), []).append(pos)

    for group in groups.values():
        if len(group) < 2:
            continue
        members = [embeddings[i] for i in group]
        if identity_store.mean_pairwise(members) > coherence_limit:
            continue
        person_id = graph.create_person(face_ids[group[0]], embeddings[group[0]])
        for i in group[1:]:
            graph.confirm(person_id, face_ids[i], embeddings[i])
        for i in group:
            conn.execute(
                "UPDATE faces SET person_id = ?, status = 'assigned' WHERE face_id = ?",
                (person_id, face_ids[i]),
            )


def run_phase2(
    conn, graph: PersonGraph, by_image: dict[str, list[dict]],
    match_threshold: float, uncertain_threshold: float, coherence_limit: float,
) -> None:
    """Fotos com 3+ rostos, adiadas na fase 1: casa cada rosto com as
    pessoas já consolidadas (só match confiante, nunca cria pessoa nova
    aqui — o risco de errar com muitos candidatos na foto é maior)."""
    deferred = conn.execute(
        "SELECT photo_id, path FROM photos WHERE status = 'deferred_multi_face'"
    ).fetchall()
    for row in tqdm(deferred, desc="Fase 2 (fotos com 3+ rostos)", unit="foto"):
        photo_id, path = row["photo_id"], row["path"]
        resolved: list[str] = []
        for face in by_image.get(path, []):
            got = assign_face(
                conn, graph, f"{photo_id}:{face['face_id']}", photo_id, face,
                np.array(face["encoding"]), match_threshold, uncertain_threshold,
                coherence_limit, allow_new_person=False, excluir=set(resolved),
            )
            if got:
                resolved.append(got)
        identity_store.set_photo_status(conn, photo_id, "phase2_done")
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                identity_store.add_relationship(conn, resolved[i], resolved[j], photo_id)


def run_recluster(
    conn, by_image: dict[str, list[dict]], cluster_threshold: float, coherence_limit: float
) -> PersonGraph:
    """Refaz TODAS as identidades do zero, a partir de todos os rostos do
    índice de uma vez, com clusterização hierárquica global.

    O caminho incremental (fase 1/fase 2) é o que permite processar a
    biblioteca em lotes sem guardar tudo em disco, mas ele decide rosto a
    rosto e na ordem em que as fotos chegam — uma decisão ruim no começo
    fica para sempre. Isso aqui é a alternativa de tempo em tempo: olha
    todos os rostos juntos, não depende de ordem, e usa average linkage,
    que não encadeia. Foi o que desfez a identidade de 266 rostos.

    Só as exclusões manuais (`--exclude`) sobrevivem; pessoas, rostos e
    relações são reconstruídos."""
    excluded = identity_store.excluded_photo_ids(conn)
    # O sweep apaga o lote local depois de processar (ver
    # scripts/sweep_library.sh), então a maioria das fotos do índice não
    # existe mais em disco pra ser hasheada. O photo_id delas já está no
    # banco da execução em que foram baixadas — reaproveitar isso é o que
    # permite reclusterizar a biblioteca inteira, não só o lote atual.
    photo_id_by_path = {r["path"]: r["photo_id"] for r in conn.execute("SELECT path, photo_id FROM photos")}

    entries: list[tuple[str, str, dict]] = []  # (photo_id, path, face)
    orphans = 0
    duplicates = 0
    seen: set[str] = set()
    for image_path in tqdm(sorted(by_image), desc="Identificando fotos", unit="foto"):
        path = Path(image_path)
        photo_id = photo_id_by_path.get(image_path)
        if photo_id is None:
            if not path.exists():
                # Nunca passou por uma fase (não está no banco) e o
                # arquivo já sumiu: não dá pra identificar a foto.
                orphans += 1
                continue
            photo_id = identity_store.hash_file(path)
        if photo_id in excluded:
            continue
        if photo_id in seen:
            # Mesmo conteúdo em dois caminhos (ver scripts/dedupe_report.py).
            # O photo_id é o hash do conteúdo, então os face_id colidiriam e
            # o banco guardaria um rosto onde o grafo em memória teria dois.
            duplicates += 1
            continue
        seen.add(photo_id)
        faces = by_image[image_path]
        identity_store.upsert_photo(
            conn, photo_id, image_path,
            path.stat().st_mtime if path.exists() else None,
            len(faces), "reclustered",
        )
        for face in faces:
            entries.append((photo_id, image_path, face))

    if orphans:
        tqdm.write(f"{orphans} fotos do índice ignoradas (sem registro no banco e sem arquivo local)")
    if duplicates:
        tqdm.write(f"{duplicates} fotos ignoradas por serem cópia exata de outra já processada")

    conn.execute("DELETE FROM relationships")
    conn.execute("DELETE FROM faces")
    conn.execute("DELETE FROM persons")
    conn.execute("DELETE FROM meta WHERE key = 'person_seq'")

    graph = PersonGraph(conn=conn)
    if not entries:
        return graph

    embeddings = [np.array(face["encoding"]) for _, _, face in entries]
    labels = identity_store.cluster_labels(
        embeddings, cluster_threshold, groups=[photo_id for photo_id, _, _ in entries]
    )
    groups: dict[int, list[int]] = {}
    for pos, label in enumerate(labels):
        groups.setdefault(int(label), []).append(pos)

    # Ordem por tamanho pra que person_001 seja a pessoa com mais fotos —
    # é ela que se quer conferir primeiro ao revisar o resultado.
    for group in sorted(groups.values(), key=len, reverse=True):
        person_id = None
        for pos in group:
            photo_id, _, face = entries[pos]
            face_id = f"{photo_id}:{face['face_id']}"
            if person_id is None:
                person_id = graph.create_person(face_id, embeddings[pos])
            else:
                graph.confirm(person_id, face_id, embeddings[pos])
            identity_store.insert_face(
                conn, face_id, photo_id, face["face_id"], embeddings[pos], person_id, None, "assigned"
            )

    for person_id, created in graph.split_incoherent(coherence_limit, cluster_threshold):
        tqdm.write(f"Identidade separada: {person_id} -> +{len(created)} pessoa(s)")

    rebuild_relationships(conn)
    return graph


def rebuild_relationships(conn) -> None:
    """Recria a tabela de "apareceram juntos" a partir de quem está em
    cada foto agora."""
    conn.execute("DELETE FROM relationships")
    by_photo: dict[str, set[str]] = {}
    for row in conn.execute(
        "SELECT photo_id, person_id FROM faces WHERE person_id IS NOT NULL"
    ):
        by_photo.setdefault(row["photo_id"], set()).add(row["person_id"])
    for photo_id, people in by_photo.items():
        ordered = sorted(people)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                identity_store.add_relationship(conn, ordered[i], ordered[j], photo_id)


def resolve_identities(
    conn, by_image: dict[str, list[dict]], match_threshold: float, uncertain_threshold: float,
    coherence_limit: float, phase2: bool,
) -> None:
    graph = PersonGraph.load(conn)

    if phase2:
        run_phase2(conn, graph, by_image, match_threshold, uncertain_threshold, coherence_limit)
    else:
        run_phase1(conn, graph, by_image, match_threshold, uncertain_threshold, coherence_limit)

    for person_id, created in graph.split_incoherent(coherence_limit, match_threshold):
        tqdm.write(
            f"Identidade contaminada separada: {person_id} -> "
            f"{', '.join(created)} ({len(created)} pessoa(s) nova(s))"
        )

    for keep_id, removed_id, dist in graph.reconcile(match_threshold, coherence_limit):
        tqdm.write(
            f"Pessoas fundidas (mesma identidade): {removed_id} -> {keep_id} (distância {dist:.3f})"
        )

    conn.commit()


def build_person_folders_from_db(
    conn, output_dir: Path, coherence_limit: float = 0.5, min_person_photos: int = 3
) -> dict[str, int]:
    """Reconstrói data/by_person a partir do banco (fonte da verdade é o
    person_id no SQLite, a pasta é só uma visão derivada e descartável).

    Pasta pessoa_XXX só tem foto onde essa pessoa aparece SOZINHA (1 rosto
    na foto). Foto com 2+ pessoas identificadas vai pra uma pasta com o
    nome da combinação exata de pessoas (ex.: person_002_e_person_004),
    reaproveitada automaticamente sempre que essa mesma dupla/trio
    reaparecer numa foto futura — sem duplicar a pessoa em pasta separada
    nem misturar a foto de grupo na pasta solo dela.

    Pessoa com menos de `min_person_photos` rostos vai pra
    `pessoas_raras/` em vez do nível de cima. Numa biblioteca de verdade
    a maioria dos rostos é gente que aparece uma vez só (desconhecido ao
    fundo, garçom, pessoa na rua): sem isso, as dezenas de pastas de uma
    foto só afogam as poucas pessoas que realmente importam. Elas não
    somem, só saem da frente.

    O sufixo `_dispersa` continua existindo pro caso raro de uma
    identidade seguir incoerente mesmo depois da separação automática —
    é o aviso de "não confie nessa pasta"."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    graph = PersonGraph.load(conn)
    diffuse = {pid for pid in graph.all_person_ids() if graph.coherence(pid) > coherence_limit}
    rare = {pid for pid in graph.all_person_ids() if graph.size(pid) < min_person_photos}

    def folder_name(person_id: str) -> str:
        return f"{person_id}_dispersa" if person_id in diffuse else person_id

    photos = {
        r["photo_id"]: (r["path"], r["num_faces"])
        for r in conn.execute("SELECT photo_id, path, num_faces FROM photos")
    }
    by_photo: dict[str, list] = {}
    for row in conn.execute("SELECT photo_id, person_id, status FROM faces"):
        by_photo.setdefault(row["photo_id"], []).append(row)

    counts = {"pessoas": 0, "raras": 0, "dispersas": len(diffuse), "links": 0}
    all_person_ids: set[str] = set()

    for photo_id, photo_rows in by_photo.items():
        entry = photos.get(photo_id)
        if not entry:
            continue
        src_str, num_faces = entry
        src = Path(src_str)
        resolved = sorted({r["person_id"] for r in photo_rows if r["person_id"]})
        all_person_ids.update(resolved)

        if not resolved:
            folder = (
                "aguardando_fase2"
                if any(r["status"] == "deferred" for r in photo_rows)
                else "revisao_manual"
            )
        elif len(resolved) == 1 and num_faces == 1:
            folder = folder_name(resolved[0])  # solo de verdade: só um rosto na foto inteira
        elif len(resolved) == 1:
            folder = f"{folder_name(resolved[0])}_em_grupo"  # tem gente não identificada junto
        else:
            folder = "_e_".join(folder_name(p) for p in resolved)  # combinação exata de pessoas

        if resolved and all(p in rare for p in resolved):
            folder = f"pessoas_raras/{folder}"

        if not src.exists():
            # Lote antigo já sincronizado e apagado localmente (ver
            # scripts/sweep_library.sh) — a identidade continua no banco,
            # só não tem mais o arquivo local pra linkar.
            continue

        person_dir = output_dir / folder
        person_dir.mkdir(parents=True, exist_ok=True)
        dest = person_dir / src.name
        if not dest.exists():
            dest.symlink_to(src.resolve())
            counts["links"] += 1

    counts["pessoas"] = len(all_person_ids)
    counts["raras"] = len(rare & all_person_ids)
    return counts


def report(conn, graph_coherence_limit: float) -> None:
    """Resumo do que saiu — dá pra ver de relance se voltou a existir uma
    identidade engolindo a biblioteca."""
    graph = PersonGraph.load(conn)
    sizes = sorted(
        ((graph.size(p), p, graph.coherence(p)) for p in graph.all_person_ids()), reverse=True
    )
    total = sum(n for n, _, _ in sizes)
    if not total:
        print("Nenhum rosto atribuído.")
        return
    print(f"\n{len(sizes)} pessoas, {total} rostos atribuídos.")
    print(f"Maior identidade: {sizes[0][1]} com {sizes[0][0]} rostos "
          f"({sizes[0][0] / total:.0%} da biblioteca, coerência {sizes[0][2]:.3f})")
    print("Top 10 por número de rostos:")
    for n, person_id, coherence in sizes[:10]:
        flag = "  <- incoerente" if coherence > graph_coherence_limit else ""
        print(f"  {person_id}: {n} rostos, coerência {coherence:.3f}{flag}")


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
        "--det-size", type=int, default=640,
        help="Resolução em que o detector trabalha (default 640). Valores maiores acham "
             "rostos menores e custam mais tempo",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=1.00,
        help="Distância máxima pra considerar 'mesma pessoa' com confiança (default 1.00). "
             "É a média das distâncias aos rostos mais próximos já atribuídos à pessoa, "
             "não a distância a um centroide. Escala do ArcFace (embedding normalizado, "
             "distância em [0,2]) — não é comparável com os limiares da versão dlib",
    )
    parser.add_argument(
        "--uncertain-threshold", type=float, default=1.15,
        help="Entre --match-threshold e este valor fica 'incerto' (revisão manual) em vez "
             "de forçar o match errado; acima disso vira pessoa nova (default 1.15)",
    )
    parser.add_argument(
        "--coherence-limit", type=float, default=1.10,
        help="Distância média máxima entre os rostos de uma mesma pessoa (default 1.10). "
             "Acima disso a identidade é considerada contaminada e separada automaticamente; "
             "também impede que um match novo degrade uma identidade já existente",
    )
    parser.add_argument(
        "--min-face-px", type=int, default=DEFAULT_MIN_FACE_PX,
        help=f"Descarta rosto menor que isso (menor lado da caixa, em pixels) "
             f"(default {DEFAULT_MIN_FACE_PX}). Use 0 pra desligar o filtro",
    )
    parser.add_argument(
        "--min-det-score", type=float, default=DEFAULT_MIN_DET_SCORE,
        help=f"Descarta rosto com confiança de detecção abaixo disso "
             f"(default {DEFAULT_MIN_DET_SCORE}). Abaixo de 0.6 o que aparece é "
             f"majoritariamente desenho, textura e rosto em cartaz. Use 0 pra desligar",
    )
    parser.add_argument(
        "--min-person-photos", type=int, default=3,
        help="Pessoa com menos rostos que isso vai pra pessoas_raras/ em vez do nível de cima "
             "(default 3). Use 1 pra deixar todas no mesmo nível",
    )
    parser.add_argument(
        "--skip-video-frames", action="store_true",
        help="Ignora imagens em 16:9 exato (3840x2160, 3664x2062, ...). No Proton Photos vindo "
             "do iPhone, foto de câmera é 4:3 — 16:9 é frame de vídeo/Live Photo. Normalmente "
             "não é preciso: medido na biblioteca de teste, o portão de qualidade já descarta 92%% "
             "dos rostos que vinham desses arquivos, e mantém os poucos que prestam. Só vale "
             "pra economizar tempo de detecção; cuidado que panorâmica legítima também cai aqui",
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
        "--recluster", action="store_true",
        help="Refaz todas as identidades do zero a partir do índice inteiro, com "
             "clusterização hierárquica global (não depende da ordem das fotos, ao contrário "
             "das fases incrementais). Preserva só as exclusões manuais. Use depois de mudar "
             "os thresholds ou quando as identidades tiverem degradado",
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
                identity_store.exclude_photo_by_path(conn, photo_path, reason)
                print(f"Excluída: {photo_path.name} ({reason})")
            conn.commit()
            counts = build_person_folders_from_db(
                conn, output_dir, args.coherence_limit, args.min_person_photos
            )
        finally:
            conn.close()
        print(f"Pastas reconstruídas: {counts['pessoas']} pessoas em {output_dir}")
        return 0

    index = load_index(index_path)

    if not args.recluster:
        index_by_image: dict[str, list[dict]] = {}
        for face in index["faces"]:
            index_by_image.setdefault(face["image"], []).append(face)

        images = list(find_images(library_dir, args.skip_video_frames))
        pending = [img for img in images if not already_indexed(index, index_by_image, img)]
        print(f"{len(images)} imagens no total, {len(pending)} para processar.")

        processed_since_save = 0
        for image_path in tqdm(pending, desc="Detectando rostos", unit="imagem"):
            try:
                detected = extract_faces(image_path, args.det_size)
            except Exception as exc:
                tqdm.write(f"Falhou: {image_path} ({exc})")
                continue

            key = str(image_path)
            mtime = image_path.stat().st_mtime
            index["faces"] = [f for f in index["faces"] if f["image"] != key]
            for face in detected:
                index["faces"].append({"image": key, "mtime": mtime, **face})
            index["scanned"][key] = mtime

            processed_since_save += 1
            if processed_since_save >= args.save_every:
                save_index(index_path, index)
                processed_since_save = 0

        save_index(index_path, index)

    by_image = faces_by_image(index, args.min_face_px, args.min_det_score)
    kept = sum(len(v) for v in by_image.values())
    print(
        f"{len(index['faces'])} rostos indexados, {kept} passaram no filtro de qualidade "
        f"(>= {args.min_face_px}px, confiança >= {args.min_det_score}). Resolvendo identidades..."
    )

    conn = identity_store.connect(db_path)
    try:
        anterior = identity_store.check_embedder(conn, face_embedder.EMBEDDER)
        if anterior:
            print(
                f"Banco tem identidades de outro modelo ({anterior}); o atual é "
                f"{face_embedder.EMBEDDER}. Como os embeddings não são comparáveis, as "
                f"identidades serão refeitas do zero (as exclusões manuais são preservadas)."
            )
            identity_store.reset_identities(conn)
        if args.recluster:
            run_recluster(conn, by_image, args.match_threshold, args.coherence_limit)
            conn.commit()
        else:
            resolve_identities(
                conn, by_image,
                match_threshold=args.match_threshold,
                uncertain_threshold=args.uncertain_threshold,
                coherence_limit=args.coherence_limit,
                phase2=args.phase2,
            )
        counts = build_person_folders_from_db(
            conn, output_dir, args.coherence_limit, args.min_person_photos
        )
        report(conn, args.coherence_limit)
    finally:
        conn.close()

    if args.recluster:
        fase = "reclusterização global"
    else:
        fase = "fase 2 (fotos com 3+ rostos)" if args.phase2 else "fase 1 (fotos com 1-2 rostos)"
    print(
        f"\nPronto ({fase}): {counts['pessoas']} pessoas, {counts['links']} fotos linkadas "
        f"em {output_dir} ({counts['raras']} pessoas em pessoas_raras/, "
        f"{counts['dispersas']} ainda dispersas)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
