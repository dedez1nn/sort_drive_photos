"""Detecção e descrição de rostos com InsightFace (SCRFD + ArcFace).

Substitui o par HOG + embedding 128-d do `face_recognition` (dlib), que
não era discriminativo o bastante para esta biblioteca. Medido nos
mesmos rostos, conferidos visualmente um a um:

                          mesma pessoa (máx)   pessoas diferentes (mín)
    dlib 128-d                    0.727                 0.554
    ArcFace 512-d                 1.045                 1.292

Com o dlib as duas distribuições se invertem — o mesmo rapaz em duas
fotos (retrato de estúdio em P&B vs. selfie na praia) ficava a 0.727,
mais longe do que ele e uma pessoa diferente (0.554). Nenhum limiar
separa isso, e era a causa de uma pessoa aparecer em três pastas. Com o
ArcFace sobra uma margem real entre as duas distribuições.

De quebra é mais rápido: 0.32s por imagem contra 1.2s do HOG, medido
nesta máquina (CPU, sem GPU).

Os embeddings vêm L2-normalizados, então a distância euclidiana fica em
[0, 2] e se relaciona com a similaridade de cosseno por
`d² = 2 - 2·cos`. Os limiares default de `face_cluster.py` estão nessa
escala — não são comparáveis com os do dlib.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    # Registrado aqui, e não em cada script que abre foto, porque este
    # módulo é o único ponto de leitura de imagem do projeto. Quando o
    # registro ficava a cargo de quem chamava, bastava um script novo
    # esquecer dele para as fotos .heic do iPhone falharem — foi o que
    # aconteceu ao verificar as fotos rebaixadas.
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Nome do backend, gravado no índice: embeddings de modelos diferentes não
# são comparáveis entre si, então misturá-los no mesmo índice produziria
# agrupamento silenciosamente errado.
EMBEDDER = "insightface-buffalo_l"
EMBEDDING_DIM = 512

_app = None


def load_image(path) -> np.ndarray:
    """Abre a foto em RGB **já com a orientação EXIF aplicada**.

    Quase toda foto de celular é gravada pelo sensor sempre na mesma
    orientação física, acompanhada de uma tag EXIF dizendo como girar na
    exibição. Quem lê os pixels crus recebe a foto deitada.

    Medido na biblioteca de teste: 70% das fotos têm essa tag, e alimentar o
    detector com elas deitadas custava 40% dos rostos — 35 detectados
    contra 57, numa amostra de 120 fotos. O SCRFD tolera alguma
    inclinação, mas não 90 graus.

    Toda leitura de imagem do projeto passa por aqui, justamente para que
    esse erro não volte a existir em um caminho e não no outro."""
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def _get_app(det_size: int = 640):
    """Carrega o modelo uma vez por processo (leva alguns segundos)."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        _app = app
    return _app


def extract_faces(image: np.ndarray, det_size: int = 640) -> list[dict]:
    """Recebe a imagem em RGB e devolve um dicionário por rosto.

    `det_score` é a confiança do próprio detector (SCRFD), que é o sinal
    de qualidade que faltava no HOG: ele não dava confiança nenhuma, e a
    única defesa possível era medir tamanho e nitidez do recorte por
    fora."""
    app = _get_app(det_size)
    faces = app.get(image[:, :, ::-1])  # insightface espera BGR
    out = []
    for face_id, face in enumerate(faces):
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        out.append(
            {
                "face_id": face_id,
                "bbox": [x1, y1, x2, y2],
                "det_score": float(face.det_score),
                "face_px": float(min(x2 - x1, y2 - y1)),
                "encoding": face.normed_embedding.astype(float).tolist(),
            }
        )
    return out
