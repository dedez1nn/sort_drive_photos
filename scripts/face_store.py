"""Armazenamento append-only dos rostos extraídos da biblioteca.

Substitui o `data/faces_index.json` da versão anterior, que guardava os
embeddings como listas de float em JSON e reescrevia o arquivo inteiro a
cada N fotos. Isso funciona para algumas centenas de rostos e desmorona
na biblioteca real: 25 mil rostos dariam ~400 MB de JSON, reescritos
umas mil vezes ao longo de uma varredura — centenas de GB de escrita
para guardar 50 MB de dado.

Aqui os embeddings vão para um arquivo binário plano (float32, append),
e os metadados para um JSONL (uma linha por rosto, append). As duas
escritas são O(1) e a ordem das linhas corresponde à ordem dos vetores.

O que fica guardado é o suficiente para nunca mais precisar da foto:

- `embedding`  — para agrupar
- `uid`        — o nó no Proton Photos, para rebaixar a foto original
                 depois de decidir quem interessa
- `thumb`      — recorte do rosto (~200px), para revisar quem é quem sem
                 rebaixar nada

As fotos completas são descartáveis; estes três artefatos não.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

EMBEDDING_DIM = 512
DTYPE = np.float32
THUMB_PX = 200


class FaceStore:
    def __init__(self, root: Path, dim: int = EMBEDDING_DIM):
        self.root = Path(root)
        self.dim = dim
        self.meta_path = self.root / "faces.jsonl"
        self.vec_path = self.root / "embeddings.f32"
        self.thumb_dir = self.root / "thumbs"
        self.scanned_path = self.root / "scanned.jsonl"

    def prepare(self) -> None:
        self.thumb_dir.mkdir(parents=True, exist_ok=True)

    # ---------- escrita ----------

    def append(self, meta: dict, embedding: np.ndarray, thumb: Image.Image | None) -> None:
        """Grava um rosto. O vetor e a linha de metadado são acrescentados
        na mesma posição relativa nos dois arquivos — é essa correspondência
        posicional que dispensa guardar índice."""
        vec = np.asarray(embedding, dtype=DTYPE)
        if vec.shape != (self.dim,):
            raise ValueError(f"embedding com shape {vec.shape}, esperado ({self.dim},)")
        if thumb is not None:
            name = f"{meta['photo_id'][:16]}_{meta['face_index']}.jpg"
            thumb.save(self.thumb_dir / name, quality=82)
            meta = {**meta, "thumb": name}
        with open(self.vec_path, "ab") as fh:
            fh.write(vec.tobytes())
        with open(self.meta_path, "a") as fh:
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def mark_scanned(self, uid: str, photo_id: str, num_faces: int) -> None:
        """Registra que a foto já passou pela detecção, inclusive quando
        ela não tem rosto nenhum — senão ela não deixa rastro e seria
        redetectada a cada execução."""
        with open(self.scanned_path, "a") as fh:
            fh.write(json.dumps({"uid": uid, "photo_id": photo_id, "faces": num_faces}) + "\n")

    # ---------- leitura ----------

    def scanned_uids(self) -> set[str]:
        if not self.scanned_path.exists():
            return set()
        with open(self.scanned_path) as fh:
            return {json.loads(line)["uid"] for line in fh if line.strip()}

    def load_meta(self) -> list[dict]:
        if not self.meta_path.exists():
            return []
        with open(self.meta_path) as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def load_embeddings(self) -> np.ndarray:
        """Lê os vetores como memmap: a biblioteca inteira dá ~50 MB, mas
        assim nem isso precisa ser copiado para a RAM de uma vez."""
        if not self.vec_path.exists():
            return np.empty((0, self.dim), dtype=DTYPE)
        return np.memmap(self.vec_path, dtype=DTYPE, mode="r").reshape(-1, self.dim)

    def count(self) -> int:
        if not self.vec_path.exists():
            return 0
        return self.vec_path.stat().st_size // (self.dim * np.dtype(DTYPE).itemsize)


def prancha(meta, membros, thumb_dir: Path, destino: Path, cols: int = 12,
            cell: int = 110, max_rostos: int = 48):
    """Folha de contato de uma identidade, para responder a olho a
    pergunta "isto é uma pessoa só?".

    A amostra é espalhada ao longo do tempo, não os primeiros N. Rostos
    consecutivos tendem a ser do mesmo dia — mesma luz, mesma roupa,
    mesmo ângulo — e é exatamente a amostra que parece coerente mesmo
    quando o grupo mistura gente diferente. Espalhando, um grupo que não
    é uma pessoa só se denuncia.

    A legenda diz quantos rostos apareceram de quantos: sem ela, quem
    olha 48 de 670 não tem como saber que está vendo 7% da identidade."""
    com_thumb = [m for m in membros if meta[m].get("thumb")]
    if not com_thumb:
        return False
    total = len(com_thumb)
    ordenados = sorted(com_thumb, key=lambda m: meta[m].get("capture_time") or "")
    if total > max_rostos:
        passo = total / max_rostos
        escolhidos = [ordenados[int(i * passo)] for i in range(max_rostos)]
    else:
        escolhidos = ordenados
    linhas = (len(escolhidos) + cols - 1) // cols
    rodape = 18
    folha = Image.new("RGB", (cell * cols, cell * linhas + rodape), (22, 22, 22))
    for i, m in enumerate(escolhidos):
        caminho = thumb_dir / meta[m]["thumb"]
        if not caminho.exists():
            continue
        r, c = divmod(i, cols)
        folha.paste(Image.open(caminho).resize((cell, cell)), (c * cell, r * cell))
    legenda = f"{len(escolhidos)} de {total} rostos"
    if total > max_rostos:
        datas = [meta[m].get("capture_time") or "" for m in ordenados]
        periodo = f"{datas[0][:7]} a {datas[-1][:7]}" if datas[0] and datas[-1] else "?"
        legenda += f", amostrados ao longo de {periodo}"
    ImageDraw.Draw(folha).text((5, cell * linhas + 4), legenda, fill=(190, 190, 190))
    folha.save(destino)
    return True


def make_thumb(image: np.ndarray, bbox, px: int = THUMB_PX) -> Image.Image:
    """Recorte quadrado do rosto com uma folga de 30%, para dar contexto
    suficiente para reconhecer a pessoa na revisão visual."""
    x1, y1, x2, y2 = bbox
    pad = (y2 - y1) * 0.3
    h, w = image.shape[:2]
    box = (
        max(int(x1 - pad), 0), max(int(y1 - pad), 0),
        min(int(x2 + pad), w), min(int(y2 + pad), h),
    )
    return Image.fromarray(image).crop(box).resize((px, px))
