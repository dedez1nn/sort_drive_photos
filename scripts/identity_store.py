"""Camada de persistência do grafo de identidades (pessoa/foto/rosto/relação).

Guarda em SQLite (data/faces.db) o que o `face_cluster.py` descobre:
quem é cada pessoa (`person_id` persistente, nunca reindexado a cada
execução — ao contrário dos labels de cluster do DBSCAN antigo), qual
foto é qual (`photo_id` = sha256 do conteúdo do arquivo, então a mesma
foto nunca gera dois registros mesmo se estiver em dois caminhos), quais
rostos foram vistos e a quem foram atribuídos, e quais pessoas aparecem
juntas em quais fotos.

Não existe uma coluna "centroide" congelada por pessoa: a cada pergunta
"esse rosto é de quem?" recalculamos a média dos embeddings já
confirmados daquela pessoa. Nessa escala (dezenas a poucas centenas de
rostos) isso é rápido o bastante e evita bugs de média incremental —
além de fazer o merge de duas pessoas virar uma troca trivial de
`person_id`, sem precisar recombinar médias manualmente.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS photos (
    photo_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    mtime REAL,
    num_faces INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS faces (
    face_id TEXT PRIMARY KEY,
    photo_id TEXT NOT NULL REFERENCES photos(photo_id),
    face_index INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    person_id TEXT REFERENCES persons(person_id),
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'uncertain'
);

CREATE TABLE IF NOT EXISTS relationships (
    person_a TEXT NOT NULL REFERENCES persons(person_id),
    person_b TEXT NOT NULL REFERENCES persons(person_id),
    photo_id TEXT NOT NULL REFERENCES photos(photo_id),
    PRIMARY KEY (person_a, person_b, photo_id)
);

-- Fotos marcadas manualmente como "não é rosto de verdade" (foto de tela,
-- objeto confundido com rosto, etc.) — não existe sinal automático
-- confiável pra isso (EXIF de uma foto de tela é idêntico ao de uma foto
-- normal), então é sempre uma decisão visual, registrada aqui pra nunca
-- mais entrar no agrupamento, mesmo reprocessando o índice do zero.
CREATE TABLE IF NOT EXISTS excluded_photos (
    photo_id TEXT PRIMARY KEY REFERENCES photos(photo_id),
    reason TEXT NOT NULL,
    excluded_at TEXT NOT NULL
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float64).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)


def upsert_photo(
    conn: sqlite3.Connection, photo_id: str, path: str, mtime: float, num_faces: int, status: str
) -> None:
    conn.execute(
        """INSERT INTO photos (photo_id, path, mtime, num_faces, status)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(photo_id) DO UPDATE SET
               path=excluded.path, mtime=excluded.mtime,
               num_faces=excluded.num_faces, status=excluded.status""",
        (photo_id, path, mtime, num_faces, status),
    )


def get_photo_status(conn: sqlite3.Connection, photo_id: str) -> str | None:
    row = conn.execute("SELECT status FROM photos WHERE photo_id = ?", (photo_id,)).fetchone()
    return row["status"] if row else None


def set_photo_status(conn: sqlite3.Connection, photo_id: str, status: str) -> None:
    conn.execute("UPDATE photos SET status = ? WHERE photo_id = ?", (status, photo_id))


def is_excluded(conn: sqlite3.Connection, photo_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM excluded_photos WHERE photo_id = ?", (photo_id,)).fetchone()
    return row is not None


def exclude_photo(conn: sqlite3.Connection, photo_id: str, reason: str) -> None:
    """Marca a foto como excluída pra sempre (foto de tela, objeto
    confundido com rosto, etc. — sempre uma decisão visual, não dá pra
    automatizar com confiança) e desfaz qualquer atribuição de pessoa e
    relação que já tinha sido feita a partir dela. Pessoas que ficarem
    sem nenhum rosto depois disso são removidas, já que só existiam por
    causa dessa foto."""
    conn.execute(
        "INSERT INTO excluded_photos (photo_id, reason, excluded_at) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(photo_id) DO UPDATE SET reason = excluded.reason",
        (photo_id, reason),
    )
    affected_persons = {
        row["person_id"]
        for row in conn.execute(
            "SELECT DISTINCT person_id FROM faces WHERE photo_id = ? AND person_id IS NOT NULL",
            (photo_id,),
        ).fetchall()
    }
    conn.execute("DELETE FROM relationships WHERE photo_id = ?", (photo_id,))
    conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
    conn.execute("UPDATE photos SET status = 'excluded' WHERE photo_id = ?", (photo_id,))
    for person_id in affected_persons:
        remaining = conn.execute(
            "SELECT COUNT(*) AS n FROM faces WHERE person_id = ?", (person_id,)
        ).fetchone()["n"]
        if remaining == 0:
            conn.execute(
                "DELETE FROM relationships WHERE person_a = ? OR person_b = ?", (person_id, person_id)
            )
            conn.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))


def exclude_photo_by_path(conn: sqlite3.Connection, path: Path, reason: str) -> str:
    photo_id = hash_file(path)
    exclude_photo(conn, photo_id, reason)
    return photo_id


def next_person_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT value FROM meta WHERE key = 'person_seq'").fetchone()
    seq = int(row["value"]) + 1 if row else 1
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('person_seq', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(seq),),
    )
    return f"person_{seq:03d}"


def insert_face(
    conn: sqlite3.Connection,
    face_id: str,
    photo_id: str,
    face_index: int,
    embedding: np.ndarray,
    person_id: str | None,
    confidence: float | None,
    status: str,
) -> None:
    conn.execute(
        """INSERT INTO faces (face_id, photo_id, face_index, embedding, person_id, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(face_id) DO UPDATE SET
               person_id=excluded.person_id, confidence=excluded.confidence, status=excluded.status""",
        (face_id, photo_id, face_index, pack(embedding), person_id, confidence, status),
    )


def add_relationship(conn: sqlite3.Connection, person_a: str, person_b: str, photo_id: str) -> None:
    if person_a == person_b:
        return
    a, b = sorted((person_a, person_b))
    conn.execute(
        "INSERT OR IGNORE INTO relationships (person_a, person_b, photo_id) VALUES (?, ?, ?)",
        (a, b, photo_id),
    )


@dataclass
class PersonGraph:
    """Índice em memória dos embeddings já confirmados de cada pessoa,
    pra achar o melhor match sem bater no banco a cada rosto comparado."""

    conn: sqlite3.Connection
    _embeddings: dict[str, list[np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> "PersonGraph":
        graph = cls(conn=conn)
        rows = conn.execute(
            "SELECT person_id, embedding FROM faces "
            "WHERE status = 'assigned' AND person_id IS NOT NULL"
        ).fetchall()
        for row in rows:
            graph._embeddings.setdefault(row["person_id"], []).append(unpack(row["embedding"]))
        return graph

    def centroid(self, person_id: str) -> np.ndarray:
        return np.mean(self._embeddings[person_id], axis=0)

    def best_match(self, embedding: np.ndarray) -> tuple[str | None, float | None]:
        best_id, best_dist = None, None
        for person_id in self._embeddings:
            dist = float(np.linalg.norm(self.centroid(person_id) - embedding))
            if best_dist is None or dist < best_dist:
                best_id, best_dist = person_id, dist
        return best_id, best_dist

    def create_person(self, embedding: np.ndarray) -> str:
        person_id = next_person_id(self.conn)
        self.conn.execute(
            "INSERT INTO persons (person_id, created_at) VALUES (?, datetime('now'))",
            (person_id,),
        )
        self._embeddings[person_id] = [embedding]
        return person_id

    def confirm(self, person_id: str, embedding: np.ndarray) -> None:
        self._embeddings.setdefault(person_id, []).append(embedding)

    def all_person_ids(self) -> list[str]:
        return list(self._embeddings)

    def merge(self, keep_id: str, remove_id: str) -> None:
        """Funde `remove_id` em `keep_id`: reatribui rostos e relações,
        apaga a pessoa removida. Nunca cria uma identidade nova."""
        if keep_id == remove_id:
            return
        self.conn.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (keep_id, remove_id))
        for row in self.conn.execute(
            "SELECT person_a, person_b, photo_id FROM relationships WHERE person_a = ? OR person_b = ?",
            (remove_id, remove_id),
        ).fetchall():
            a = keep_id if row["person_a"] == remove_id else row["person_a"]
            b = keep_id if row["person_b"] == remove_id else row["person_b"]
            add_relationship(self.conn, a, b, row["photo_id"])
        self.conn.execute(
            "DELETE FROM relationships WHERE person_a = ? OR person_b = ?", (remove_id, remove_id)
        )
        self.conn.execute("DELETE FROM persons WHERE person_id = ?", (remove_id,))
        self._embeddings[keep_id] = self._embeddings.get(keep_id, []) + self._embeddings.pop(remove_id, [])

    def reconcile(self, match_threshold: float) -> list[tuple[str, str, float]]:
        """Funde pessoas cujo centroide ficou perto demais uma da outra —
        fragmentação de identidade que sobrou de execuções passadas ou de
        rostos "redescobertos" que na verdade já eram uma pessoa conhecida.
        Retorna os merges feitos, pra log."""
        merged: list[tuple[str, str, float]] = []
        changed = True
        while changed:
            changed = False
            ids = sorted(self.all_person_ids())
            for i, a in enumerate(ids):
                if a not in self._embeddings:
                    continue
                for b in ids[i + 1 :]:
                    if b not in self._embeddings:
                        continue
                    dist = float(np.linalg.norm(self.centroid(a) - self.centroid(b)))
                    if dist <= match_threshold:
                        merged.append((a, b, dist))
                        self.merge(a, b)
                        changed = True
                        break
                if changed:
                    break
        return merged
