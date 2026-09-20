#!/usr/bin/env python3
"""Cria um álbum no Proton Photos para cada pessoa do ranking.

Por que álbum e não pasta: o Proton Photos é uma seção separada de
`/my-files` e não aceita pastas (`filesystem list /photos` responde "Path
type photos is not supported"). Mas o CLI tem `album`, que é o
equivalente nativo — e melhor do que pasta para este caso:

- **Não duplica byte nenhum.** `album add-photo` referencia a foto que já
  está na sua biblioteca. Copiar as fotos das 20 pessoas para pastas em
  `/my-files` custaria alguns GB de armazenamento pago, repetidos.
- **Aparece no app.** O álbum é uma entidade de primeira classe no Proton
  Photos, no celular e na web; uma pasta em `/my-files` não apareceria
  junto das fotos.
- **Não mexe na biblioteca.** A foto continua onde está, na timeline. Sair
  do álbum não apaga nada.

Uso:
    python scripts/album_person.py pessoa_01
    python scripts/album_person.py a1b2c3d4e5f6 --name "Nome da pessoa"
    python scripts/album_person.py pessoa_01 --dry-run
    python scripts/album_person.py pessoa_01 pessoa_02 pessoa_03

É seguro rodar de novo: o álbum é reaproveitado pelo nome e as fotos que
já estão nele são puladas.

Pessoas na blocklist (`scripts/person_blocklist.py`) são recusadas aqui,
e a checagem é feita contra os ROSTOS da pessoa, não contra o nome nem o
id do ranking — os dois mudam quando a biblioteca cresce, e "nunca mais
crie um álbum dessa pessoa" só vale se continuar valendo depois da
próxima varredura. Rodar com um `ranking.json` antigo, de antes do
bloqueio, também não passa: o carimbo do ranking é só um atalho, a
decisão é refeita aqui.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import person_blocklist
import proton_catalog
from face_store import FaceStore

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RANKING = REPO_ROOT / "data" / "top_pessoas" / "ranking.json"


def listar_albuns(proton_bin: str) -> dict[str, str]:
    """Nome do álbum -> uid."""
    saida = proton_catalog.run_proton(proton_bin, "album", "list", "--json")
    albuns = {}
    for node in json.loads(saida or "[]"):
        nome = node.get("name")
        nome = nome.get("value") if isinstance(nome, dict) else nome
        if nome:
            albuns[nome] = node["uid"]
    return albuns


def fotos_do_album(proton_bin: str, album_uid: str) -> set[str]:
    saida = proton_catalog.run_proton(
        proton_bin, "album", "photos", f"/photos/{album_uid}"
    )
    return {linha.strip() for linha in saida.splitlines() if linha.strip()}


def carregar_ranking(path: Path) -> dict[str, dict]:
    dados = json.loads(path.read_text())
    pessoas = dados["pessoas"] if isinstance(dados, dict) else dados
    indice = {p["pessoa"]: p for p in pessoas}
    indice.update({p["id"]: p for p in pessoas if p.get("id")})
    return indice


def bloqueio(pessoa: dict, bl: person_blocklist.Blocklist,
             indice: dict[str, int] | None, vetores) -> tuple[str, str] | None:
    """(nível, descrição) se esta pessoa não pode virar álbum, senão None.

    Os dois níveis da blocklist barram o álbum igual; a diferença entre
    eles está em outro lugar (o `ignorar` também some do ranqueamento).
    Uma pessoa `ignorar` normalmente nem chega aqui, porque o rank_people
    já a tirou — a checagem existe para o ranking mais antigo que o
    bloqueio.

    A decisão é pelos rostos sempre que o acervo estiver em disco. O
    carimbo do `ranking.json` é só o plano B, para quando os embeddings
    não estão disponíveis."""
    if not bl.perfis:
        return None
    if indice is not None:
        v = person_blocklist.vetores_da_pessoa(pessoa, indice, vetores)
        if len(v):
            perfil, d = bl.classificar(pessoa.get("faces", []), v)
            if perfil:
                return perfil.nivel, f"{perfil} — rms {d:.3f}" + (
                    f", {perfil.motivo}" if perfil.motivo else "")
            return None
    marca = pessoa.get("bloqueio")
    if marca:
        return marca.get("nivel", "sem-album"), (
            f"{marca.get('nome')} (carimbo do ranking)"
            + (f", {marca['motivo']}" if marca.get("motivo") else ""))
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pessoas", nargs="+",
                   help="Nomes do ranking (pessoa_01) ou os ids estáveis")
    p.add_argument("--ranking", default=str(DEFAULT_RANKING))
    p.add_argument("--name", default=None,
                   help="Nome do álbum. Só vale quando você passa UMA pessoa; "
                        "sem isso o nome é 'Pessoa NN'")
    p.add_argument("--prefix", default="Pessoa",
                   help="Prefixo do nome automático do álbum (default: 'Pessoa')")
    p.add_argument("--proton-bin", default="proton-drive")
    p.add_argument("--blocklist", default=str(person_blocklist.DEFAULT_ROOT))
    p.add_argument("--store", default=str(REPO_ROOT / "data" / "faces"))
    p.add_argument("--batch", type=int, default=50,
                   help="Fotos por chamada de add-photo (default 50)")
    p.add_argument("--dry-run", action="store_true",
                   help="Diz o que faria sem criar álbum nem adicionar foto")
    args = p.parse_args()

    if args.name and len(args.pessoas) > 1:
        print("--name só funciona com uma pessoa por vez.")
        return 1

    ranking_path = Path(args.ranking)
    if not ranking_path.exists():
        print(f"Ranking não encontrado em {ranking_path} — rode scripts/rank_people.py antes.")
        return 1
    ranking = carregar_ranking(ranking_path)

    faltando = [n for n in args.pessoas if n not in ranking]
    if faltando:
        print(f"Não estão no ranking: {', '.join(faltando)}")
        return 1

    bl = person_blocklist.Blocklist.load(Path(args.blocklist))
    indice, vetores = None, None
    if bl.perfis:
        store = FaceStore(Path(args.store))
        meta = store.load_meta()
        if meta:
            indice = person_blocklist.posicoes_no_acervo(meta)
            vetores = np.asarray(store.load_embeddings(), dtype=np.float64)
        print(f"Blocklist: {len(bl.perfis)} pessoas bloqueadas"
              + ("" if indice else " (acervo indisponível; valendo só o carimbo do ranking)"))

    pedidas = []
    for chave in args.pessoas:
        motivo = bloqueio(ranking[chave], bl, indice, vetores)
        if motivo:
            print(f"{chave}: BLOQUEADA ({motivo[0]}) — {motivo[1]}. Nenhum álbum criado.")
            continue
        pedidas.append(chave)
    if not pedidas:
        return 1

    albuns = listar_albuns(args.proton_bin)

    for chave in pedidas:
        pessoa = ranking[chave]
        nome = args.name or f"{args.prefix} {pessoa['pessoa'].split('_')[-1]}"
        uids = pessoa["uids"]
        print(f"\n{nome}  <-  {pessoa['pessoa']} [{pessoa.get('id','?')}], "
              f"{len(uids)} fotos ({pessoa.get('primeiro','?')[:7]} a {pessoa.get('ultimo','?')[:7]})")

        album_uid = albuns.get(nome)
        if album_uid:
            print(f"  álbum já existe ({album_uid[-12:]})")
        elif args.dry_run:
            print("  criaria o álbum")
        else:
            proton_catalog.run_proton(args.proton_bin, "album", "create", nome)
            albuns = listar_albuns(args.proton_bin)
            album_uid = albuns.get(nome)
            if not album_uid:
                print(f"  ERRO: criei o álbum mas não o encontrei na listagem; pulando")
                continue
            print(f"  álbum criado ({album_uid[-12:]})")

        if album_uid:
            ja = fotos_do_album(args.proton_bin, album_uid)
            pendentes = [u for u in uids if u not in ja]
            print(f"  {len(ja)} fotos já no álbum, {len(pendentes)} a adicionar")
        else:
            pendentes = list(uids)
        if args.dry_run or not pendentes:
            continue

        with tqdm(total=len(pendentes), desc="  adicionando", unit="foto") as bar:
            for i in range(0, len(pendentes), args.batch):
                lote = pendentes[i : i + args.batch]
                try:
                    proton_catalog.run_proton(
                        args.proton_bin, "album", "add-photo", f"/photos/{album_uid}",
                        *[f"/photos/{u}" for u in lote],
                    )
                except Exception as exc:
                    tqdm.write(f"  lote falhou ({str(exc)[:160]})")
                bar.update(len(lote))

        # O CLI pode devolver sucesso com uma foto do lote faltando — visto
        # na prática: um lote de 50 retornou exit 0 e só 49 entraram, e a
        # mesma foto entrou sozinha na tentativa seguinte. Então o que vale
        # é reconferir o conteúdo do álbum, não o código de saída.
        faltando = [u for u in uids if u not in fotos_do_album(args.proton_bin, album_uid)]
        if faltando:
            print(f"  {len(faltando)} não entraram no lote; tentando uma a uma")
            for u in faltando:
                try:
                    proton_catalog.run_proton(
                        args.proton_bin, "album", "add-photo",
                        f"/photos/{album_uid}", f"/photos/{u}",
                    )
                except Exception as exc:
                    print(f"    falhou de novo: {u[-16:]} ({str(exc)[:120]})")

        total = len(fotos_do_album(args.proton_bin, album_uid))
        resto = len(uids) - total
        print(f"  álbum '{nome}': {total}/{len(uids)} fotos"
              + (f"  <-- {resto} não entraram" if resto > 0 else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
