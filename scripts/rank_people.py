#!/usr/bin/env python3
"""Ranqueia as pessoas mais frequentes da biblioteca inteira, offline.

Roda sobre o acervo colhido por `scripts/scan_library.py` — embeddings,
uids e recortes — sem precisar de nenhuma foto em disco.

## Por que não dá para simplesmente agrupar tudo de uma vez

A clusterização hierárquica precisa da matriz n×n de distâncias, que não
é esparsa. Para 25 mil rostos são 5 GB; para 40 mil, 14 GB. Não cabe.

## A saída: agrupar em dois níveis

**Nível 1 — dentro de blocos.** Os rostos são divididos em blocos por
ordem cronológica (foto da mesma pessoa tende a se concentrar no tempo,
então o bloco já nasce com material coerente) e cada bloco é agrupado
exatamente, com a mesma clusterização restrita usada no resto do
projeto. Um bloco de 4000 rostos custa 128 MB de matriz.

A mesma pessoa aparece em vários blocos, e isso é esperado — o nível 2
junta. O que importa é que **quem é frequente forma um grupo de 2+
rostos em cada bloco onde aparece**: alguém com 200 fotos numa
biblioteca de 40 mil rostos tem ~20 por bloco.

**Nível 2 — entre blocos.** Só os grupos com 2+ rostos entram, o que
descarta de uma vez as centenas de rostos que aparecem uma única vez
(cartaz, desconhecido ao fundo) e derruba o tamanho do problema em uma
ordem de grandeza. Cada grupo é representado por alguns embeddings que
cobrem a variação interna dele, e os representantes são agrupados entre
si — de novo com a restrição de que dois rostos da mesma foto nunca são
a mesma pessoa, propagada dos grupos do nível 1.

**Nível 3 — atribuição.** Com as identidades finais formadas, TODOS os
rostos (inclusive os que ficaram sozinhos no nível 1) são atribuídos por
quórum. É isso que dá a contagem exata: uma aparição isolada de alguém
frequente volta para a conta dele.

**Nível 4 — blocklist.** Com as identidades prontas, as que são uma
pessoa marcada como `ignorar` em `scripts/person_blocklist.py` saem do
ranking, e as marcadas como `sem-album` ficam, carimbadas com `bloqueio`
no `ranking.json` — é esse carimbo, junto com a checagem que o próprio
`album_person.py` refaz, que impede um álbum delas. O reconhecimento é
por rosto e não pelo nome, então continua valendo depois de revarrer a
biblioteca; o porquê do teste ser sobre a identidade inteira está no
cabeçalho do person_blocklist.

Uso:
    python scripts/rank_people.py                # top 20
    python scripts/rank_people.py --top 50
    python scripts/rank_people.py --block 6000   # blocos maiores, mais RAM
    python scripts/rank_people.py --no-blocklist # sem os bloqueios, para comparar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import person_blocklist
from face_store import FaceStore, prancha
from identity_store import cluster_labels

REPO_ROOT = Path(__file__).resolve().parent.parent


def _grupos(labels) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for pos, label in enumerate(labels):
        out[int(label)].append(pos)
    return out


def nivel1(vetores, fotos, ordem, bloco: int, limiar: float) -> list[list[int]]:
    """Agrupa dentro de cada bloco. Devolve os grupos como listas de
    índices globais."""
    grupos: list[list[int]] = []
    for inicio in tqdm(range(0, len(ordem), bloco), desc="Nível 1 (blocos)", unit="bloco"):
        idx = ordem[inicio : inicio + bloco]
        labels = cluster_labels(vetores[idx], limiar, groups=[fotos[i] for i in idx])
        for membros in _grupos(labels).values():
            grupos.append([idx[m] for m in membros])
    return grupos


def nivel2(vetores, fotos, grupos, limiar: float) -> list[list[int]]:
    """Junta os grupos do nível 1 que são a mesma pessoa. Só entram os que
    têm 2+ rostos; os solitários voltam no nível 3.

    A distância entre dois grupos é a média das distâncias ao quadrado
    entre TODOS os pares de rostos deles — average linkage exato, sem
    amostrar representantes. Isso é possível de graça porque os
    embeddings do ArcFace são L2-normalizados, e nesse caso vale a
    identidade

        média_{a∈A, b∈B} ||a − b||²  =  2 − 2·⟨centroide(A), centroide(B)⟩

    (conferida numericamente contra a força bruta). Ou seja: o par de
    centroides dá o valor EXATO da média sobre todos os pares, em tempo
    constante. Não é a distância entre centroides, que seria uma
    aproximação ruim — é a média real, escrita de outro jeito.

    Uma primeira versão comparava 4 representantes por grupo, escolhidos
    nos extremos para cobrir a variação interna. Deu errado do jeito
    esperado em retrospecto: o average linkage sobre pontos extremos
    infla a distância, e uma pessoa com 460 rostos acabou dividida em
    dois grupos no ranking.

    O limiar é comparado contra a raiz da média dos quadrados (RMS), que
    é sempre ≥ a média das distâncias — daí `--merge-threshold` ser um
    pouco mais folgado que `--threshold`."""
    candidatos = [g for g in grupos if len(g) > 1]
    if not candidatos:
        return []
    print(f"Nível 2: {len(candidatos)} grupos recorrentes entram na fusão")

    centroides = np.array([vetores[g].mean(axis=0) for g in candidatos])
    tamanhos = np.array([len(g) for g in candidatos], dtype=float)
    membros = [list(g) for g in candidatos]
    fotos_de = [{fotos[i] for i in g} for g in candidatos]
    vivo = np.ones(len(candidatos), dtype=bool)

    while True:
        idx = np.flatnonzero(vivo)
        if len(idx) < 2:
            break
        c = centroides[idx]
        rms = np.sqrt(np.maximum(2.0 - 2.0 * (c @ c.T), 0.0))
        np.fill_diagonal(rms, np.inf)
        # Grupos que dividem uma foto são pessoas diferentes.
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if fotos_de[idx[a]] & fotos_de[idx[b]]:
                    rms[a, b] = rms[b, a] = np.inf
        pos = int(np.argmin(rms))
        a, b = divmod(pos, len(idx))
        if not np.isfinite(rms[a, b]) or rms[a, b] > limiar:
            break
        i, j = int(idx[a]), int(idx[b])
        na, nb = tamanhos[i], tamanhos[j]
        centroides[i] = (na * centroides[i] + nb * centroides[j]) / (na + nb)
        tamanhos[i] = na + nb
        membros[i].extend(membros[j])
        fotos_de[i] |= fotos_de[j]
        vivo[j] = False

    return [membros[i] for i in np.flatnonzero(vivo)]


def nivel3(vetores, fotos, identidades, limiar, quorum: int) -> dict[int, list[int]]:
    """Atribui todos os rostos às identidades formadas, por quórum. Um
    rosto só entra se ainda não houver outro rosto DELE mesmo nessa
    identidade vindo da mesma foto."""
    final: dict[int, list[int]] = {i: list(m) for i, m in enumerate(identidades)}
    fotos_por_id = {i: {fotos[m] for m in membros} for i, membros in final.items()}
    ja = {m for membros in final.values() for m in membros}

    bancos = {i: vetores[membros] for i, membros in final.items()}
    soltos = [i for i in range(len(vetores)) if i not in ja]
    for face in tqdm(soltos, desc="Nível 3 (atribuição)", unit="rosto"):
        v = vetores[face]
        melhor, melhor_d = None, None
        for i, banco in bancos.items():
            if fotos[face] in fotos_por_id[i]:
                continue
            d = np.sort(np.linalg.norm(banco - v, axis=1))
            score = float(d[: min(quorum, len(d))].mean())
            if melhor_d is None or score < melhor_d:
                melhor, melhor_d = i, score
        if melhor is not None and melhor_d <= limiar:
            final[melhor].append(face)
            fotos_por_id[melhor].add(fotos[face])
    return final


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default=str(REPO_ROOT / "data" / "faces"))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "top_pessoas"))
    p.add_argument("--top", type=int, default=20, help="Quantas pessoas reportar (default 20)")
    p.add_argument("--block", type=int, default=4000,
                   help="Rostos por bloco no nível 1 (default 4000; a matriz do bloco "
                        "custa block² × 8 bytes de RAM)")
    p.add_argument("--merge-threshold", type=float,
                   default=person_blocklist.MERGE_THRESHOLD,
                   help="Limiar para fundir grupos no nível 2 (default 1.10). Comparado "
                        "contra a raiz da média dos quadrados, que é sempre maior que a "
                        "média simples — por isso é mais folgado que --threshold")
    p.add_argument("--threshold", type=float, default=1.00,
                   help="Limiar de agrupamento, escala do ArcFace (default 1.00)")
    p.add_argument("--quorum", type=int, default=3)
    p.add_argument("--blocklist", default=str(person_blocklist.DEFAULT_ROOT),
                   help="Pasta com as pessoas bloqueadas (default data/blocklist). "
                        "Veja scripts/person_blocklist.py")
    p.add_argument("--no-blocklist", action="store_true",
                   help="Ranqueia como se a blocklist estivesse vazia — para conferir "
                        "o que ela está tirando")
    p.add_argument("--min-det-score", type=float, default=0.70,
                   help="Ignora rostos abaixo desta confiança de detecção (default 0.70). "
                        "A varredura guarda tudo acima de 0.60 justamente para este corte "
                        "ser ajustável aqui, que é a etapa barata — reprocessar o "
                        "ranqueamento leva um minuto, revarrer a biblioteca leva duas horas. "
                        "Medido: 0.70 mantém 90%% dos rostos e limpa parte do lixo "
                        "(desenho, rosto em tela, cartaz) sem custar as pessoas de verdade")
    p.add_argument("--min-face-px", type=float, default=0,
                   help="Ignora rostos menores que isso (default 0, sem corte adicional)")
    args = p.parse_args()

    store = FaceStore(Path(args.store))
    todos = store.load_meta()
    todos_vet = np.array(store.load_embeddings(), dtype=np.float64)
    if len(todos) != len(todos_vet):
        print(f"Acervo inconsistente: {len(todos)} metadados x {len(todos_vet)} vetores.")
        return 1
    if not len(todos):
        print("Acervo vazio — rode scripts/scan_library.py primeiro.")
        return 1

    manter = [
        i for i, m in enumerate(todos)
        if m.get("det_score", 0) >= args.min_det_score
        and m.get("face_px", 0) >= args.min_face_px
    ]
    meta = [todos[i] for i in manter]
    vetores = todos_vet[manter]
    print(f"{len(todos)} rostos no acervo, {len(meta)} acima do corte de qualidade "
          f"(confiança >= {args.min_det_score}"
          + (f", >= {args.min_face_px:.0f}px" if args.min_face_px else "") + ").")
    if len(meta) < 2:
        print("Rostos de menos para ranquear.")
        return 1

    bl = (person_blocklist.Blocklist(Path(args.blocklist), [])
          if args.no_blocklist else person_blocklist.Blocklist.load(Path(args.blocklist)))
    if bl.perfis:
        print(f"Blocklist: {len(bl.perfis)} pessoas bloqueadas "
              f"({len(bl.do_nivel('ignorar'))} ignoradas, "
              f"{len(bl.do_nivel('sem-album'))} sem álbum).")

    fotos = [m["photo_id"] for m in meta]
    ordem = sorted(range(len(meta)), key=lambda i: meta[i].get("capture_time") or "")

    grupos = nivel1(vetores, fotos, ordem, args.block, args.threshold)
    print(f"Nível 1: {len(grupos)} grupos "
          f"({sum(1 for g in grupos if len(g) > 1)} com 2+ rostos)")
    identidades = nivel2(vetores, fotos, grupos, args.merge_threshold)
    print(f"Nível 2: {len(identidades)} identidades recorrentes")
    if not identidades:
        print("Nenhuma pessoa recorrente encontrada.")
        return 0
    final = nivel3(vetores, fotos, identidades, args.threshold, args.quorum)

    def chaves(membros):
        return [f"{meta[m]['photo_id']}:{meta[m]['face_index']}" for m in membros]

    # Nível 4 — blocklist. Depois do agrupamento, e não antes, porque o
    # reconhecimento de uma pessoa bloqueada precisa da identidade
    # inteira: é no conjunto que estão a média que o limiar compara e a
    # co-ocorrência que separa duas pessoas parecidas. Filtrar rosto a
    # rosto na entrada foi tentado e pegou 645 rostos da pessoa mais
    # frequente do acervo junto com os da bloqueada (ver
    # person_blocklist).
    #
    # De quebra, os rostos da pessoa ignorada continuam disputando o nível
    # 3: ela segue "puxando para si" os rostos que são dela, em vez de
    # deixá-los livres para serem absorvidos por quem ficou.
    ordenadas = sorted(final.values(), key=len, reverse=True)
    ranking, bloqueios = [], {}
    for membros in ordenadas:
        ks = chaves(membros)
        perfil, d = bl.classificar(ks, vetores[membros], "ignorar", args.merge_threshold)
        if perfil:
            print(f"  blocklist/ignorar: {len(membros)} rostos são {perfil} "
                  f"(rms {d:.3f}) — fora do ranking"
                  + (f": {perfil.motivo}" if perfil.motivo else ""))
            continue
        perfil, d = bl.classificar(ks, vetores[membros], "sem-album", args.merge_threshold)
        if perfil:
            bloqueios[len(ranking)] = {"nivel": "sem-album", "perfil": perfil.chave,
                                       "nome": perfil.nome, "motivo": perfil.motivo,
                                       "rms": round(d, 3)}
        ranking.append(membros)
        if len(ranking) >= args.top:
            break
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    resumo = []
    print(f"\nTop {len(ranking)} pessoas mais frequentes:")
    for pos, membros in enumerate(ranking, 1):
        uids = sorted({meta[m]["uid"] for m in membros})
        datas = sorted(d for d in (meta[m].get("capture_time") for m in membros) if d)
        nome = f"pessoa_{pos:02d}"
        # `pessoa_NN` é só a posição no ranking desta execução, e a posição
        # troca de dono quando o acervo cresce ou o corte muda — foi o que
        # aconteceu na prática: uma colocação virou outra pessoa entre
        # duas execuções durante a varredura. O `id` abaixo é derivado das
        # fotos da identidade, então acompanha a PESSOA, não a colocação;
        # é ele que o fetch_person usa para detectar que uma pasta já
        # baixada é de outra pessoa.
        ident_id = hashlib.sha256("|".join(uids).encode()).hexdigest()[:12]
        prancha(meta, membros, store.thumb_dir, out / f"{nome}.jpg")
        periodo = f"{datas[0][:7]} a {datas[-1][:7]}" if datas else "?"
        # "sem-album" continua no ranking de propósito: é informação útil
        # ("este grupo existe e é gente de máscara"), e a pessoa segue
        # disponível para o fetch_person. Só o álbum está barrado — e a
        # marca viaja no ranking.json para o album_person ver.
        bloqueio = bloqueios.get(pos - 1)
        print(f"  {nome} [{ident_id}]: {len(membros)} rostos em {len(uids)} fotos   ({periodo})"
              + (f"   <-- sem álbum: {bloqueio['nome']}" if bloqueio else ""))
        # `uids` diz em QUE FOTOS a pessoa aparece (é o que o fetch_person
        # usa). `faces` diz QUAL ROSTO de cada foto é dela — sem isso,
        # qualquer conferência posterior tem que adivinhar, e numa foto de
        # grupo acaba pegando o rosto de outra pessoa. Custou três análises
        # erradas antes de virar campo do arquivo.
        resumo.append({"pessoa": nome, "id": ident_id, "rostos": len(membros),
                       "fotos": len(uids), "bloqueio": bloqueio,
                       "primeiro": datas[0] if datas else None,
                       "ultimo": datas[-1] if datas else None,
                       "uids": uids,
                       "faces": [f"{meta[m]['photo_id']}:{meta[m]['face_index']}"
                                 for m in membros],
                       "thumbs": [meta[m]["thumb"] for m in membros if meta[m].get("thumb")]})
    (out / "ranking.json").write_text(json.dumps(
        {"gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "rostos_no_acervo": len(todos), "rostos_usados": len(meta),
         "min_det_score": args.min_det_score, "threshold": args.threshold,
         "blocklist": [{"chave": p.chave, "nome": p.nome, "nivel": p.nivel,
                        "motivo": p.motivo} for p in bl.perfis],
         "pessoas": resumo}, indent=2, ensure_ascii=False))
    print(f"\nPranchas de rosto e ranking.json em {out}")
    print(f"Para rebaixar as fotos de alguém: python scripts/fetch_person.py pessoa_01")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
