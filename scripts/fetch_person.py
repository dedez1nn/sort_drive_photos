#!/usr/bin/env python3
"""Rebaixa do Proton Photos as fotos de uma pessoa do ranking.

Fecha o ciclo da varredura fria: `scan_library.py` apaga as fotos depois
de extrair os rostos, mas guarda o `uid` de cada uma. Este script pega os
uids de uma pessoa em `data/top_pessoas/ranking.json` e baixa só aquelas
fotos de volta.

É o passo que justifica ter guardado o uid. Sem ele, identificar as
pessoas mais frequentes da biblioteca seria um resultado só de papel —
você saberia quem são e não teria as fotos.

Uso:
    python scripts/fetch_person.py pessoa_01
    python scripts/fetch_person.py pessoa_01 pessoa_03 --out ~/fotos-familia
    python scripts/fetch_person.py pessoa_01 --dry-run

Baixa em lotes com teto de disco, como a varredura — mas aqui os arquivos
NÃO são apagados, são o resultado que você quer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import proton_catalog

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pessoas", nargs="+",
                   help="Nomes do ranking (ex.: pessoa_01) ou os ids estáveis que ele "
                        "imprime entre colchetes. O nome é a posição e muda de dono quando "
                        "o acervo cresce; o id acompanha a pessoa")
    p.add_argument("--ranking", default=str(REPO_ROOT / "data" / "top_pessoas" / "ranking.json"))
    p.add_argument("--catalog", default=str(proton_catalog.DEFAULT_CATALOG))
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "por_pessoa"),
                   help="Pasta destino; cada pessoa vira uma subpasta (default: data/por_pessoa)")
    p.add_argument("--proton-bin", default="proton-drive")
    p.add_argument("--batch-items", type=int, default=100)
    p.add_argument("--batch-mb", type=float, default=500)
    p.add_argument("--limit", type=int, default=None,
                   help="Baixa só as N fotos mais antigas da pessoa. Serve para conferir "
                        "o agrupamento sem ocupar disco com a pessoa inteira")
    p.add_argument("--dry-run", action="store_true",
                   help="Só diz quantas fotos e quantos MB baixaria")
    args = p.parse_args()

    ranking_path = Path(args.ranking)
    if not ranking_path.exists():
        print(f"Ranking não encontrado em {ranking_path} — rode scripts/rank_people.py antes.")
        return 1
    dados = json.loads(ranking_path.read_text())
    pessoas = dados["pessoas"] if isinstance(dados, dict) else dados
    ranking = {r["pessoa"]: r for r in pessoas}
    # Aceita tanto o nome de posição (pessoa_03) quanto o id estável.
    ranking.update({r["id"]: r for r in pessoas if r.get("id")})
    catalogo = {e["uid"]: e for e in proton_catalog.load_catalog(Path(args.catalog))}

    faltando = [nome for nome in args.pessoas if nome not in ranking]
    if faltando:
        print(f"Não estão no ranking: {', '.join(faltando)}")
        print(f"Disponíveis: {', '.join(sorted(ranking))}")
        return 1

    for nome in args.pessoas:
        uids = ranking[nome]["uids"]
        entradas = [catalogo[u] for u in uids if u in catalogo]
        # Contado antes do --limit: senão o corte pedido apareceria como
        # se fossem fotos sumidas do catálogo.
        perdidas = len(uids) - len(entradas)
        entradas.sort(key=lambda e: e.get("capture_time") or "")
        if args.limit:
            entradas = entradas[: args.limit]
        mb = sum(e["size"] for e in entradas) / 1e6
        print(f"\n{nome} [{ranking[nome].get('id','?')}]: {len(entradas)} fotos, {mb:.0f} MB"
              + (f" ({perdidas} uids não estão mais no catálogo)" if perdidas else ""))
        if args.dry_run:
            continue

        destino = Path(args.out) / nome
        destino.mkdir(parents=True, exist_ok=True)

        # `pessoa_NN` é a posição no ranking, e a posição troca de dono
        # quando o acervo cresce. Sem esta checagem, rebaixar depois de um
        # ranqueamento novo misturaria duas pessoas na mesma pasta.
        marca = destino / "_origem.json"
        atual = {"id": ranking[nome].get("id"), "pessoa": ranking[nome]["pessoa"],
                 "ranking_gerado_em": dados.get("gerado_em") if isinstance(dados, dict) else None}
        if marca.exists():
            antes = json.loads(marca.read_text())
            if antes.get("id") and atual["id"] and antes["id"] != atual["id"]:
                print(f"  ATENÇÃO: esta pasta foi baixada para outra pessoa "
                      f"(id {antes['id']}, ranking de {antes.get('ranking_gerado_em')}). "
                      f"O ranking atual tem {atual['id']} nessa posição. Pulando — apague a "
                      f"pasta ou use o id estável: python scripts/fetch_person.py {atual['id']}")
                continue
        marca.write_text(json.dumps(atual, indent=2, ensure_ascii=False))

        ja = {p.name for p in destino.iterdir() if p.is_file() and p.name != "_origem.json"}
        pendentes = [e for e in entradas if e["name"] not in ja]
        if not pendentes:
            print("  já está tudo baixado.")
            continue

        lotes = proton_catalog.plan_batches(
            pendentes, int(args.batch_mb * 1e6), args.batch_items
        )
        with tqdm(total=len(pendentes), desc=f"  {nome}", unit="foto") as bar:
            for lote in lotes:
                try:
                    proton_catalog.run_proton(
                        args.proton_bin, "photo", "download", "-c", "rename",
                        *[f"/photos/{e['uid']}" for e in lote], str(destino),
                    )
                except Exception as exc:
                    tqdm.write(f"  lote falhou ({exc})")
                bar.update(len(lote))
        print(f"  em {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
