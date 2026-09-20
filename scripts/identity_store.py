"""Camada de persistência do grafo de identidades (pessoa/foto/rosto/relação).

Guarda em SQLite (data/faces.db) o que o `face_cluster.py` descobre:
quem é cada pessoa (`person_id` persistente, nunca reindexado a cada
execução — ao contrário dos labels de cluster do DBSCAN antigo), qual
foto é qual (`photo_id` = sha256 do conteúdo do arquivo, então a mesma
foto nunca gera dois registros mesmo se estiver em dois caminhos), quais
rostos foram vistos e a quem foram atribuídos, e quais pessoas aparecem
juntas em quais fotos.

Uma pessoa não tem centroide congelado nem centroide nenhum no caminho
de decisão: a pergunta "esse rosto é de quem?" é respondida comparando o
rosto novo com os rostos concretos já atribuídos a cada pessoa (ver
`PersonGraph.score`). Média de embeddings só aparece como métrica de
diagnóstico. Nessa escala (dezenas a poucas centenas de milhares de
rostos por lote) isso é rápido o bastante, e faz o merge de duas pessoas
virar uma troca trivial de `person_id`.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

# Quantos rostos concretos da pessoa precisam estar perto do rosto novo
# pra considerar que é ela. Ver `PersonGraph.score` pro porquê de não ser
# comparação com o centroide.
QUORUM = 3

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
-- objeto confundido com rosto, etc.) — o filtro automático de qualidade
-- (tamanho/nitidez do rosto, em face_cluster.py) pega o grosso do lixo,
-- mas não tem como distinguir um desenho bem enquadrado de uma pessoa,
-- então isso continua sendo uma decisão visual, registrada aqui pra nunca
-- mais entrar no agrupamento, mesmo reprocessando o índice do zero.
CREATE TABLE IF NOT EXISTS excluded_photos (
    photo_id TEXT PRIMARY KEY REFERENCES photos(photo_id),
    reason TEXT NOT NULL,
    excluded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def check_embedder(conn: sqlite3.Connection, embedder: str) -> str | None:
    """Garante que o banco só contenha embeddings de um modelo. Vetores de
    modelos diferentes têm dimensão e escala diferentes, então misturá-los
    faria o agrupamento errar em silêncio. Devolve o modelo anterior se o
    banco tinha outro (e já grava o novo), ou None se não havia conflito."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
    atual = row["value"] if row else None
    if atual == embedder:
        return None
    tem_rosto = conn.execute("SELECT 1 FROM faces LIMIT 1").fetchone() is not None
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('embedder', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (embedder,),
    )
    if not tem_rosto:
        return None
    return atual or "dlib (versão anterior, sem marca)"


def reset_identities(conn: sqlite3.Connection) -> None:
    """Apaga pessoas, rostos e relações, preservando as exclusões manuais
    e o registro das fotos. Usado quando o modelo de embedding muda."""
    conn.execute("DELETE FROM relationships")
    conn.execute("DELETE FROM faces")
    conn.execute("DELETE FROM persons")
    conn.execute("DELETE FROM meta WHERE key = 'person_seq'")
    conn.execute("UPDATE photos SET status = 'pending' WHERE status != 'excluded'")


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


def mean_pairwise(embeddings) -> float:
    """Distância média entre todos os pares de um conjunto de rostos."""
    if len(embeddings) < 2:
        return 0.0
    return float(np.mean(pdist(np.asarray(embeddings))))


# Acima disso, a matriz de distâncias n×n da clusterização global não cabe
# confortavelmente na memória desta máquina (n=12000 já são ~1.1GB em
# float64). O caminho incremental por lotes não tem esse limite.
MAX_GLOBAL_CLUSTER_FACES = 12000


def cluster_labels(embeddings, threshold: float, groups=None) -> np.ndarray:
    """Clusteriza rostos por aglomeração hierárquica com average linkage,
    cortando a árvore na distância `threshold`.

    Average linkage de propósito: single linkage (o critério implícito de
    quem vai encadeando "A parece com B, B parece com C") é justamente o
    que colou gente diferente numa identidade só; complete linkage vai pro
    outro extremo e fragmenta a mesma pessoa a cada mudança de ângulo ou
    iluminação. Average fica no meio e foi o que separou melhor esta
    biblioteca.

    `groups` (uma foto por rosto, na mesma ordem) ativa a restrição de
    exclusão mútua: dois rostos da MESMA foto nunca caem na mesma pessoa,
    porque ninguém aparece duas vezes numa foto. É informação de graça que
    estava nos dados e não era usada, e ela importa porque quem convive
    com você aparece nas suas fotos junto de você — exatamente os pares
    que o embedding mais tende a confundir. Sem ela, subir o limiar o
    bastante pra reunir a mesma pessoa em condições diferentes também
    começava a fundir quem aparecia ao lado dela."""
    arr = np.asarray(embeddings)
    n = len(arr)
    if n < 2:
        return np.zeros(n, dtype=int)
    if n > MAX_GLOBAL_CLUSTER_FACES:
        raise MemoryError(
            f"{n} rostos passam do limite de {MAX_GLOBAL_CLUSTER_FACES} para "
            f"clusterização global (a matriz n×n não caberia na memória). "
            f"Use o caminho incremental (sem --recluster) ou aperte o filtro "
            f"de qualidade."
        )
    if groups is None:
        return fcluster(linkage(pdist(arr), method="average"), threshold, criterion="distance")
    return _cluster_constrained(arr, list(groups), threshold)


def _cluster_constrained(arr: np.ndarray, groups: list, threshold: float) -> np.ndarray:
    """Average linkage com restrição de exclusão mútua, implementado na
    mão porque o scipy não aceita restrições.

    Usa a fórmula de Lance-Williams pra atualizar as distâncias a cada
    fusão (mesma recorrência que o scipy usa internamente para average
    linkage), e guarda por cluster o conjunto de fotos que ele cobre — a
    fusão é proibida quando os dois clusters compartilham uma foto."""
    n = len(arr)
    dist = squareform(pdist(arr)).astype(np.float64)
    np.fill_diagonal(dist, np.inf)

    photos_of = [{g} for g in groups]
    size = np.ones(n)
    alive = np.ones(n, dtype=bool)
    label = np.arange(n)

    while True:
        flat = int(np.argmin(dist))
        i, j = divmod(flat, n)
        best = dist[i, j]
        if not np.isfinite(best) or best > threshold:
            break
        if photos_of[i] & photos_of[j]:
            # Mesma foto: não podem ser a mesma pessoa. Marca o par como
            # impossível e procura o próximo mais próximo.
            dist[i, j] = dist[j, i] = np.inf
            continue
        ni, nj = size[i], size[j]
        dist[i, :] = (ni * dist[i, :] + nj * dist[j, :]) / (ni + nj)
        dist[:, i] = dist[i, :]
        dist[i, i] = np.inf
        dist[j, :] = np.inf
        dist[:, j] = np.inf
        photos_of[i] |= photos_of[j]
        size[i] = ni + nj
        alive[j] = False
        label[label == j] = i

    _, out = np.unique(label, return_inverse=True)
    return out


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


def excluded_photo_ids(conn: sqlite3.Connection) -> set[str]:
    return {row["photo_id"] for row in conn.execute("SELECT photo_id FROM excluded_photos")}


def exclude_photo(conn: sqlite3.Connection, photo_id: str, reason: str) -> None:
    """Marca a foto como excluída pra sempre e desfaz qualquer atribuição
    de pessoa e relação que já tinha sido feita a partir dela. Pessoas que
    ficarem sem nenhum rosto depois disso são removidas, já que só
    existiam por causa dessa foto."""
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
    """Índice em memória dos rostos já confirmados de cada pessoa, pra
    decidir identidade sem bater no banco a cada comparação. Guarda o
    `face_id` junto do embedding porque separar uma identidade
    contaminada (`split`) precisa saber exatamente qual linha da tabela
    `faces` reatribuir."""

    conn: sqlite3.Connection
    quorum: int = QUORUM
    _faces: dict[str, list[tuple[str, np.ndarray]]] = field(default_factory=dict)

    @classmethod
    def load(cls, conn: sqlite3.Connection, quorum: int = QUORUM) -> "PersonGraph":
        graph = cls(conn=conn, quorum=quorum)
        rows = conn.execute(
            "SELECT face_id, person_id, embedding FROM faces "
            "WHERE status = 'assigned' AND person_id IS NOT NULL"
        ).fetchall()
        for row in rows:
            graph._faces.setdefault(row["person_id"], []).append(
                (row["face_id"], unpack(row["embedding"]))
            )
        return graph

    def embeddings(self, person_id: str) -> list[np.ndarray]:
        return [emb for _, emb in self._faces.get(person_id, [])]

    def all_person_ids(self) -> list[str]:
        return list(self._faces)

    def size(self, person_id: str) -> int:
        return len(self._faces.get(person_id, []))

    def score(self, person_id: str, embedding: np.ndarray) -> float:
        """Distância média do rosto novo aos `quorum` rostos MAIS PRÓXIMOS
        já atribuídos a essa pessoa. Isso era a distância ao centroide.

        A diferença NÃO é que o centroide fique mais perto: medido no blob
        real de 266 rostos que motivou esta mudança, a distância média ao
        centroide (0.597) e o score de quórum (0.582) são praticamente
        iguais. O ganho é outro — o centroide é um ponto que se move a
        cada rosto absorvido, então a identidade vai derivando pelo espaço
        um rosto de cada vez e depois de N passos aceita alguém que
        estaria longe demais do ponto de partida. Exigir proximidade a
        `quorum` rostos CONCRETOS ancora a decisão em evidência que não se
        move: o rosto novo tem que se parecer com várias fotos reais
        daquela pessoa, não com um resumo delas.

        Sozinha, a troca vale pouco (na ablação, o maior grupo cai de 37%
        para 31% da biblioteca). Ela importa combinada com o freio de
        coerência e com o portão de qualidade — ver
        docs/agrupamento_facial.md, seção "O que cada mudança resolveu"."""
        faces = self._faces.get(person_id)
        if not faces:
            return float("inf")
        dists = np.linalg.norm(np.asarray([emb for _, emb in faces]) - embedding, axis=1)
        k = min(self.quorum, len(dists))
        return float(np.mean(np.sort(dists)[:k]))

    def best_match(
        self, embedding: np.ndarray, excluir: set[str] | None = None
    ) -> tuple[str | None, float | None]:
        """`excluir` tira da disputa pessoas que já foram identificadas na
        mesma foto — ninguém aparece duas vezes numa foto."""
        best_id, best_dist = None, None
        for person_id in self._faces:
            if excluir and person_id in excluir:
                continue
            dist = self.score(person_id, embedding)
            if best_dist is None or dist < best_dist:
                best_id, best_dist = person_id, dist
        return best_id, best_dist

    def create_person(self, face_id: str, embedding: np.ndarray) -> str:
        person_id = next_person_id(self.conn)
        self.conn.execute(
            "INSERT INTO persons (person_id, created_at) VALUES (?, datetime('now'))",
            (person_id,),
        )
        self._faces[person_id] = [(face_id, embedding)]
        return person_id

    def confirm(self, person_id: str, face_id: str, embedding: np.ndarray) -> None:
        self._faces.setdefault(person_id, []).append((face_id, embedding))

    def coherence(self, person_id: str) -> float:
        """Distância média entre cada par de rostos dessa pessoa. Uma
        identidade real e coesa fica bem abaixo do match_threshold (o
        mesmo rosto em ângulos diferentes ainda se parece consigo). Uma
        identidade formada por encadeamento mistura gente diferente e
        essa média dispara."""
        return mean_pairwise(self.embeddings(person_id))

    def would_stay_coherent(self, person_id: str, embedding: np.ndarray, limit: float) -> bool:
        """Simula adicionar o rosto e responde se a pessoa continuaria
        coesa. É o freio que impede a bola-de-neve de começar: mesmo com
        o match parecendo bom, se absorver o rosto já degrada a
        identidade inteira, é mais provável ser gente diferente do que a
        mesma pessoa num ângulo novo."""
        current = self.embeddings(person_id)
        if len(current) < 2:
            return True
        return mean_pairwise(current + [embedding]) <= limit

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
        self._faces[keep_id] = self._faces.get(keep_id, []) + self._faces.pop(remove_id, [])

    def linkage_distance(self, person_a: str, person_b: str) -> float:
        """Distância média entre TODOS os pares de rostos das duas pessoas
        (average linkage), não entre os dois centroides.

        Average linkage só aproxima duas pessoas se os rostos concretos de
        uma se parecerem com os da outra — dois grupos podem ter
        centroides próximos sem que nenhum par de rostos reais seja
        parecido."""
        a = np.asarray(self.embeddings(person_a))
        b = np.asarray(self.embeddings(person_b))
        if a.size == 0 or b.size == 0:
            return float("inf")
        return float(np.mean(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))

    def reconcile(
        self, merge_threshold: float, coherence_limit: float
    ) -> list[tuple[str, str, float]]:
        """Funde pessoas que na verdade são a mesma — fragmentação que
        sobrou de execuções passadas ou de rostos redescobertos que já
        eram alguém conhecido.

        Duas guardas contra o merge errado (que antes ia fundindo em
        cascata até sobrar um blob só): a distância é average linkage
        entre os rostos de verdade, e o merge só acontece se a pessoa
        resultante continuar coerente. Cada rodada funde o par mais
        próximo de todos, não o primeiro que aparecer na ordem alfabética,
        pra que o resultado não dependa da ordem dos `person_id`."""
        merged: list[tuple[str, str, float]] = []
        rejected: set[tuple[str, str]] = set()
        while True:
            ids = sorted(self.all_person_ids())
            best: tuple[float, str, str] | None = None
            for i, a in enumerate(ids):
                for b in ids[i + 1 :]:
                    if (a, b) in rejected:
                        continue
                    dist = self.linkage_distance(a, b)
                    if dist <= merge_threshold and (best is None or dist < best[0]):
                        best = (dist, a, b)
            if best is None:
                return merged
            dist, a, b = best
            if mean_pairwise(self.embeddings(a) + self.embeddings(b)) > coherence_limit:
                # Juntas ficariam incoerentes: são duas pessoas parecidas,
                # não uma pessoa fragmentada. Marca o par como recusado
                # pra não reavaliar pra sempre, e segue pros outros pares.
                rejected.add((a, b))
                continue
            merged.append((a, b, dist))
            self.merge(a, b)
            # `a` absorveu os rostos de `b`, então os pares recusados que
            # envolviam qualquer um dos dois foram decididos sobre um
            # conjunto que não existe mais — voltam a ser avaliáveis. Isso
            # não gera laço infinito: cada volta ou funde (e o número de
            # pessoas cai) ou só acrescenta recusas.
            rejected = {p for p in rejected if a not in p and b not in p}

    def split(self, person_id: str, threshold: float) -> list[str]:
        """Quebra uma identidade contaminada nos subgrupos que ela de fato
        contém, clusterizando os rostos dela com average linkage.

        Antes o `coherence()` só renomeava a pasta pra `_dispersa` e ia
        embora: a pessoa contaminada continuava no grafo, continuava
        recebendo rostos novos e continuava participando do `reconcile`,
        contaminando o resto. Marcar não resolvia — agora separa de fato.
        O maior subgrupo fica com o `person_id` original (as pastas já
        sincronizadas dele continuam fazendo sentido) e os outros viram
        pessoas novas. Retorna os `person_id` criados."""
        faces = self._faces.get(person_id, [])
        if len(faces) < 2:
            return []
        labels = cluster_labels(
            [emb for _, emb in faces], threshold,
            groups=[fid.rsplit(":", 1)[0] for fid, _ in faces],
        )
        groups: dict[int, list[int]] = {}
        for pos, label in enumerate(labels):
            groups.setdefault(int(label), []).append(pos)
        if len(groups) < 2:
            return []

        ordered = sorted(groups.values(), key=len, reverse=True)
        keep, rest = ordered[0], ordered[1:]
        self._faces[person_id] = [faces[i] for i in keep]

        created: list[str] = []
        for group in rest:
            head_face_id, head_embedding = faces[group[0]]
            new_id = self.create_person(head_face_id, head_embedding)
            for i in group[1:]:
                self.confirm(new_id, *faces[i])
            for i in group:
                self.conn.execute(
                    "UPDATE faces SET person_id = ? WHERE face_id = ?", (new_id, faces[i][0])
                )
            created.append(new_id)

        # As relações "apareceram juntos" foram gravadas com o person_id
        # antigo e não dá pra saber a qual subgrupo cada uma pertencia.
        # Apagar é mais honesto que manter relação de gente errada; elas
        # voltam a ser gravadas conforme as fases rodam de novo.
        self.conn.execute(
            "DELETE FROM relationships WHERE person_a = ? OR person_b = ?", (person_id, person_id)
        )
        return created

    def split_incoherent(
        self, coherence_limit: float, threshold: float
    ) -> list[tuple[str, list[str]]]:
        """Separa toda pessoa incoerente do grafo. Repete enquanto houver
        o que separar, porque um subgrupo recém-criado ainda pode estar
        contaminado por dentro."""
        done: list[tuple[str, list[str]]] = []
        pending = [p for p in self.all_person_ids() if self.coherence(p) > coherence_limit]
        while pending:
            person_id = pending.pop()
            created = self.split(person_id, threshold)
            if not created:
                continue
            done.append((person_id, created))
            for candidate in (person_id, *created):
                if self.size(candidate) > 1 and self.coherence(candidate) > coherence_limit:
                    pending.append(candidate)
        return done
