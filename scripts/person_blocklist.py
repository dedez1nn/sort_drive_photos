#!/usr/bin/env python3
"""Pessoas que o ranking encontra mas que não devem virar álbum.

Duas coisas diferentes aparecem no topo do ranking e não são "uma pessoa
da sua vida que você quer num álbum":

- **Grupos que não são uma pessoa só.** Uma identidade do topo pode ser um
  monte de gente de máscara: com metade do rosto coberto, o que sobra do
  embedding é parecido demais entre pessoas diferentes e elas colapsam num
  grupo só. Um álbum disso não serve para nada — e nenhum ajuste de limiar
  conserta, porque o sinal que separaria essas pessoas não está na foto.
- **Pessoas de verdade que você não quer organizar.** Nada de errado com o
  agrupamento; você só não quer um álbum daquela pessoa, nem hoje nem
  depois de revarrer a biblioteca.

Daí os dois níveis:

    sem-album   fica no ranking, nunca vira álbum
    ignorar     sai do ranking e nunca vira álbum, em toda aparição futura

## Por que não basta guardar "pessoa_NN" numa lista

`pessoa_NN` é a colocação, e a colocação troca de dono quando o acervo
cresce. O `id` do ranking também não serve: ele é o hash dos uids das
fotos, então **muda quando a pessoa ganha uma foto nova** — que é
exatamente o caso de "toda aparição futura".

O que acompanha a pessoa é o rosto dela. Então o perfil guardado aqui é o
banco de embeddings da identidade bloqueada (`<chave>.f32`) mais a lista
de rostos dela (`photo_id:face_index`).

## Como o reconhecimento futuro é feito — e como ele deu errado primeiro

A primeira versão testava rosto a rosto, com o mesmo quórum que o nível 3
do ranqueamento usa para atribuir um rosto solto: média das 3 menores
distâncias até o banco, bloqueia se <= 1.00. Medido na biblioteca de
teste, o perfil de UMA identidade bloqueada pegou 1149 rostos — dos quais
**645 eram de outra pessoa**, justamente a que mais aparece no acervo
inteiro.

O motivo é que o teste do nível 3 não é absoluto, é uma **disputa**: lá o
rosto vai para a identidade mais próxima entre todas, e só o vencedor
importa. Reaproveitado sozinho, sem ninguém para perder a disputa, o
mesmo limiar vira uma rede que pega meio mundo.

A versão que vale usa o critério do nível 2, que é o que o projeto já usa
para decidir "estes dois grupos são a mesma pessoa", e o aplica à
identidade inteira, não a rostos avulsos:

1. **Nenhum conflito de foto.** Se a identidade tem um rosto numa foto
   onde o perfil tem OUTRO rosto, são duas pessoas na mesma foto — e
   duas pessoas na mesma foto nunca são a mesma pessoa. É a regra que
   salva a vizinha mais próxima do perfil: ela está a 0.992 dele, e o
   perfil está a 0.964 de si mesmo — margem nenhuma —, mas as duas
   aparecem juntas em 5 fotos, e isso decide.
2. **Distância dentro do limiar.** RMS entre os centroides <= 1.10, que é
   a média exata das distâncias ao quadrado entre todos os pares dos dois
   conjuntos (ver `rank_people.nivel2`).

Medido no ranking da biblioteca de teste, cada perfil bloqueia exatamente a
própria identidade e nada mais; a segunda colocada de cada perfil é
barrada pela regra 1.

Uso:
    python scripts/person_blocklist.py add pessoa_07 --nivel sem-album \
        --nome "Máscaras" --motivo "grupo de gente de máscara, não é uma pessoa"
    python scripts/person_blocklist.py add pessoa_03 pessoa_09 --nivel ignorar
    python scripts/person_blocklist.py list
    python scripts/person_blocklist.py check     # o que cada perfil pega no ranking atual
    python scripts/person_blocklist.py remove b2c3d4e5f601
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from face_store import FaceStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "blocklist"
DEFAULT_RANKING = REPO_ROOT / "data" / "top_pessoas" / "ranking.json"

NIVEIS = ("sem-album", "ignorar")

# Fonte única do limiar de "estes dois conjuntos de rostos são a mesma
# pessoa". O `--merge-threshold` do rank_people usa este valor como
# default em vez de repetir o número: com duas constantes, ajustar o
# ranqueamento e esquecer a blocklist faria o bloqueio do álbum divergir
# do bloqueio do ranking em silêncio.
MERGE_THRESHOLD = 1.10
EMBEDDING_DIM = 512


def rms(a: np.ndarray, b: np.ndarray) -> float:
    """Raiz da média dos quadrados das distâncias entre todos os pares de
    dois conjuntos, calculada a partir dos centroides. Vale porque os
    embeddings do ArcFace são L2-normalizados — a dedução está em
    `rank_people.nivel2`."""
    return float(np.sqrt(max(2.0 - 2.0 * float(np.dot(a, b)), 0.0)))


def por_foto(faces) -> dict[str, set[str]]:
    """`photo_id:face_index` -> {face_index} agrupado por foto."""
    d: dict[str, set[str]] = defaultdict(set)
    for chave in faces:
        pid, _, fi = chave.rpartition(":")
        d[pid].add(fi)
    return d


class Perfil:
    """Uma identidade bloqueada: os rostos dela, o banco de embeddings e o
    motivo."""

    def __init__(self, dados: dict, vetores: np.ndarray):
        self.chave: str = dados["chave"]
        self.nome: str = dados.get("nome") or dados["chave"]
        self.nivel: str = dados["nivel"]
        self.motivo: str = dados.get("motivo") or ""
        self.origem: dict = dados.get("origem") or {}
        self.criado_em: str = dados.get("criado_em") or ""
        self.faces: list[str] = dados.get("faces") or []
        self.thumbs: list[str] = dados.get("thumbs") or []
        self.vetores = np.asarray(vetores, dtype=np.float64)
        self.centroide = self.vetores.mean(axis=0) if len(self.vetores) else np.zeros(EMBEDDING_DIM)
        self._por_foto = por_foto(self.faces)

    def __str__(self) -> str:
        return f"{self.nome} [{self.chave}]"

    def conflitos(self, faces) -> int:
        """Em quantas fotos esta identidade e o perfil têm rostos
        DIFERENTES. Qualquer número acima de zero já diz que são duas
        pessoas — ninguém aparece duas vezes na mesma foto.

        Tem que ser por rosto, não por foto: a própria pessoa bloqueada
        compartilha todas as fotos com o perfil (são os mesmos rostos), e
        uma regra por foto a deixaria passar."""
        outros = 0
        for pid, indices in por_foto(faces).items():
            meus = self._por_foto.get(pid)
            if meus and (indices - meus):
                outros += 1
        return outros

    def to_json(self) -> dict:
        return {"chave": self.chave, "nome": self.nome, "nivel": self.nivel,
                "motivo": self.motivo, "origem": self.origem,
                "criado_em": self.criado_em, "rostos": len(self.vetores),
                "faces": self.faces, "thumbs": self.thumbs}


class Blocklist:
    def __init__(self, root: Path, perfis: list[Perfil]):
        self.root = Path(root)
        self.perfis = perfis

    # ---------- persistência ----------

    @property
    def index_path(self) -> Path:
        return self.root / "perfis.json"

    @classmethod
    def load(cls, root: Path = DEFAULT_ROOT) -> "Blocklist":
        root = Path(root)
        index = root / "perfis.json"
        if not index.exists():
            return cls(root, [])
        dados = json.loads(index.read_text())
        perfis = []
        for item in dados.get("perfis", []):
            vec = root / f"{item['chave']}.f32"
            if not vec.exists():
                print(f"AVISO: perfil {item['chave']} sem embeddings ({vec}); ignorado")
                continue
            vetores = np.fromfile(vec, dtype=np.float32).reshape(-1, EMBEDDING_DIM)
            perfis.append(Perfil(item, vetores))
        return cls(root, perfis)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for p in self.perfis:
            (self.root / f"{p.chave}.f32").write_bytes(
                np.asarray(p.vetores, dtype=np.float32).tobytes()
            )
        self.index_path.write_text(json.dumps(
            {"atualizado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "perfis": [p.to_json() for p in self.perfis]}, indent=2, ensure_ascii=False))

    def upsert(self, perfil: Perfil) -> bool:
        """Devolve True se substituiu um perfil existente."""
        for i, p in enumerate(self.perfis):
            if p.chave == perfil.chave:
                self.perfis[i] = perfil
                return True
        self.perfis.append(perfil)
        return False

    def remove(self, chave: str) -> Perfil | None:
        for i, p in enumerate(self.perfis):
            if p.chave == chave or p.nome == chave:
                self.perfis.pop(i)
                (self.root / f"{p.chave}.f32").unlink(missing_ok=True)
                self.save()
                return p
        return None

    # ---------- a decisão ----------

    def do_nivel(self, nivel: str) -> list[Perfil]:
        return [p for p in self.perfis if p.nivel == nivel]

    def classificar(self, faces, vetores, nivel: str | None = None,
                    limiar: float = MERGE_THRESHOLD) -> tuple[Perfil | None, float]:
        """Esta identidade é uma pessoa bloqueada?

        Recebe a identidade inteira — as chaves `photo_id:face_index` dos
        rostos e os embeddings deles — porque é a identidade que vira (ou
        não) álbum, e porque é no conjunto que existe a informação de
        co-ocorrência que decide os casos difíceis.

        Devolve o perfil que bloqueia e a distância; quando nada bloqueia,
        devolve (None, menor distância vista), que serve de diagnóstico."""
        perfis = self.perfis if nivel is None else self.do_nivel(nivel)
        vetores = np.asarray(vetores, dtype=np.float64)
        if not perfis or not len(vetores):
            return None, float("inf")
        centro = vetores.mean(axis=0)
        faces = list(faces)
        melhor, melhor_d, menor = None, float("inf"), float("inf")
        for p in perfis:
            d = rms(centro, p.centroide)
            menor = min(menor, d)
            if d > limiar or p.conflitos(faces):
                continue
            if d < melhor_d:
                melhor, melhor_d = p, d
        return (melhor, melhor_d) if melhor else (None, menor)


# ---------- ligação com o ranking e o acervo ----------


def carregar_ranking(path: Path) -> tuple[dict, dict[str, dict]]:
    dados = json.loads(Path(path).read_text())
    pessoas = dados["pessoas"] if isinstance(dados, dict) else dados
    indice = {p["pessoa"]: p for p in pessoas}
    indice.update({p["id"]: p for p in pessoas if p.get("id")})
    return (dados if isinstance(dados, dict) else {}), indice


def posicoes_no_acervo(meta: list[dict]) -> dict[str, int]:
    """`photo_id:face_index` -> posição no acervo (que é a linha do
    embedding). É a mesma chave que o ranking guarda em `faces`."""
    return {f"{m['photo_id']}:{m['face_index']}": i for i, m in enumerate(meta)}


def vetores_da_pessoa(pessoa: dict, indice: dict[str, int], vetores) -> np.ndarray:
    pos = [indice[f] for f in pessoa.get("faces", []) if f in indice]
    if not pos:
        return np.empty((0, EMBEDDING_DIM))
    return np.asarray(vetores[pos], dtype=np.float64)


# ---------- CLI ----------


def cmd_add(args) -> int:
    store = FaceStore(Path(args.store))
    meta = store.load_meta()
    vetores = store.load_embeddings()
    if not meta:
        print("Acervo vazio — rode scripts/scan_library.py primeiro.")
        return 1
    indice = posicoes_no_acervo(meta)

    cabecalho, ranking = carregar_ranking(Path(args.ranking))
    faltando = [n for n in args.pessoas if n not in ranking]
    if faltando:
        print(f"Não estão no ranking: {', '.join(faltando)}")
        return 1
    if args.nome and len(args.pessoas) > 1:
        print("--nome só funciona com uma pessoa por vez.")
        return 1

    bl = Blocklist.load(Path(args.root))
    for chave_pedida in args.pessoas:
        pessoa = ranking[chave_pedida]
        v = vetores_da_pessoa(pessoa, indice, vetores)
        if not len(v):
            print(f"{pessoa['pessoa']}: nenhum dos rostos está no acervo atual; pulando")
            continue
        perfil = Perfil({
            "chave": pessoa["id"],
            "nome": args.nome or pessoa["pessoa"],
            "nivel": args.nivel,
            "motivo": args.motivo or "",
            "origem": {"pessoa": pessoa["pessoa"], "id": pessoa["id"],
                       "fotos": pessoa.get("fotos"),
                       "ranking_gerado_em": cabecalho.get("gerado_em")},
            "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # As chaves dos rostos são o que permite a regra de
            # co-ocorrência lá na frente — sem elas, sobra só a distância,
            # que na biblioteca de teste não separa duas pessoas parecidas.
            "faces": [f for f in pessoa.get("faces", []) if f in indice],
            "thumbs": pessoa.get("thumbs", [])[:48],
        }, v)
        subst = bl.upsert(perfil)
        print(f"{'atualizado' if subst else 'bloqueado'}: {perfil}  "
              f"nível={perfil.nivel}  {len(v)} rostos de referência"
              + (f"  ({perfil.motivo})" if perfil.motivo else ""))
    bl.save()
    print(f"\n{len(bl.perfis)} perfis em {bl.index_path}")
    return 0


def cmd_list(args) -> int:
    bl = Blocklist.load(Path(args.root))
    if not bl.perfis:
        print("Blocklist vazia.")
        return 0
    for nivel in NIVEIS:
        perfis = bl.do_nivel(nivel)
        if not perfis:
            continue
        print(f"\n{nivel}:")
        for p in perfis:
            print(f"  {p.nome:<24} [{p.chave}]  {len(p.vetores):>4} rostos   "
                  f"(era {p.origem.get('pessoa', '?')} no ranking de "
                  f"{(p.origem.get('ranking_gerado_em') or '?')[:10]})")
            if p.motivo:
                print(f"      {p.motivo}")
    return 0


def cmd_remove(args) -> int:
    bl = Blocklist.load(Path(args.root))
    p = bl.remove(args.chave)
    if not p:
        print(f"Não achei '{args.chave}' na blocklist.")
        return 1
    print(f"removido: {p} (nível {p.nivel})")
    return 0


def cmd_check(args) -> int:
    """Confere a blocklist contra o ranking atual: quem cada perfil
    bloqueia, e com que folga sobre quem ele NÃO bloqueia. É aqui que se
    vê se um bloqueio está prestes a levar junto alguém que você quer."""
    bl = Blocklist.load(Path(args.root))
    if not bl.perfis:
        print("Blocklist vazia.")
        return 0
    store = FaceStore(Path(args.store))
    meta = store.load_meta()
    if not meta:
        print("Acervo vazio — rode scripts/scan_library.py primeiro.")
        return 1
    indice = posicoes_no_acervo(meta)
    vetores = np.asarray(store.load_embeddings(), dtype=np.float64)
    _, ranking = carregar_ranking(Path(args.ranking))
    pessoas = [ranking[n] for n in sorted(ranking) if n.startswith("pessoa_")]

    dados = [(p, vetores_da_pessoa(p, indice, vetores)) for p in pessoas]
    print(f"{len(pessoas)} identidades no ranking, {len(bl.perfis)} perfis "
          f"(limiar {args.threshold}):\n")
    for perfil in bl.perfis:
        print(f"{perfil}  nível={perfil.nivel}")
        linhas = []
        for pessoa, v in dados:
            if not len(v):
                continue
            d = rms(v.mean(axis=0), perfil.centroide)
            if d > args.threshold + 0.15:
                continue
            conf = perfil.conflitos(pessoa["faces"])
            linhas.append((d, pessoa["pessoa"], conf, d <= args.threshold and not conf))
        for d, nome, conf, bloqueia in sorted(linhas):
            marca = "BLOQUEIA" if bloqueia else (
                f"passa (aparece em {conf} fotos com o perfil)" if conf else "passa (longe)")
            print(f"    {nome:<12} rms={d:.3f}  {marca}")
        if not linhas:
            print("    nenhuma identidade perto deste perfil no ranking atual")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--store", default=str(REPO_ROOT / "data" / "faces"))
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Bloqueia uma pessoa do ranking")
    a.add_argument("pessoas", nargs="+", help="pessoa_NN ou o id estável")
    a.add_argument("--nivel", choices=NIVEIS, default="sem-album",
                   help="sem-album: fica no ranking mas nunca vira álbum. "
                        "ignorar: sai do ranking também (default: sem-album)")
    a.add_argument("--nome", default=None, help="Rótulo legível (uma pessoa por vez)")
    a.add_argument("--motivo", default=None)
    a.add_argument("--ranking", default=str(DEFAULT_RANKING))
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="Mostra a blocklist")
    l.set_defaults(func=cmd_list)

    r = sub.add_parser("remove", help="Desbloqueia (chave ou nome)")
    r.add_argument("chave")
    r.set_defaults(func=cmd_remove)

    c = sub.add_parser("check", help="Confere os perfis contra o ranking atual")
    c.add_argument("--threshold", type=float, default=MERGE_THRESHOLD)
    c.add_argument("--ranking", default=str(DEFAULT_RANKING))
    c.set_defaults(func=cmd_check)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
