"""Catálogo local da timeline do Proton Photos, com os detalhes de cada foto.

`photo timeline --json` (sem `-d`) devolve só `{nodeUid, captureTime}`, que
é o bastante para baixar tudo em ordem, mas não para escolher o que baixar.
Com `-d` o CLI carrega os detalhes de cada nó — é lento (minutos para a
biblioteca inteira, porque busca um a um), mas roda uma vez e muda o que
dá pra fazer:

- `name` e `mediaType` permitem **descartar screenshot e sticker antes de
  baixar**, em vez de baixar e apagar como o `--keep-ext` fazia. Numa
  biblioteca onde 33% era print de Discord/Twitter, isso é um terço do
  tráfego e do tempo economizado.
- `totalStorageSize` permite montar lotes por orçamento de bytes com
  precisão, sem baixar até estourar e só então perceber.
- `contentHash` permite pular duplicata exata sem baixar a segunda cópia.
- `uid` fica gravado junto de cada rosto extraído, que é o que torna
  possível **rebaixar a foto original** depois de decidir quem interessa.
  Sem isso, uma vez apagado o lote local não existe caminho de volta: o
  manifesto antigo só guardava `uid -> captureTime`, e o download renomeia
  em caso de colisão, então nada ligava um arquivo ao seu nó no Proton.

O catálogo é gravado como JSONL em `data/catalog.jsonl`, uma linha por
foto, e só é refeito quando pedido (`--refresh`) ou quando a timeline
curta acusa fotos novas.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = REPO_ROOT / "data" / "catalog.jsonl"

# Extensões que a câmera do iPhone gera. .png no Proton Photos vindo de
# iPhone é sempre screenshot; .gif/.webp são sticker ou figurinha salva.
CAMERA_MEDIA_TYPES = {"image/jpeg", "image/heic", "image/heif"}


def run_proton(proton_bin: str, *args: str, timeout: int | None = None) -> str:
    result = subprocess.run(
        [proton_bin, *args], capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`{proton_bin} {' '.join(args)}` falhou (exit {result.returncode}): "
            f"{result.stderr.strip()[:500]}"
        )
    return result.stdout


def _unwrap(value):
    """Vários campos vêm embrulhados em {"ok": true, "value": ...} porque
    podem falhar individualmente (nome que não decifra, por exemplo)."""
    if isinstance(value, dict):
        return value.get("value") if value.get("ok") else None
    return value


def fetch_catalog(proton_bin: str) -> list[dict]:
    raw = run_proton(proton_bin, "photo", "timeline", "-d", "--json")
    entries = []
    for node in json.loads(raw):
        name = _unwrap(node.get("name"))
        if not name:
            continue
        photo = node.get("photo") or {}
        entries.append(
            {
                "uid": node["uid"],
                "name": name,
                "media_type": node.get("mediaType"),
                "size": node.get("totalStorageSize") or 0,
                "capture_time": photo.get("captureTime") or node.get("creationTime"),
                "content_hash": photo.get("contentHash"),
            }
        )
    return entries


def save_catalog(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_catalog(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def is_camera_photo(entry: dict) -> bool:
    """Foto de câmera de verdade, não screenshot/sticker. O nome é o
    critério mais confiável que existe aqui: o Android padroniza
    `Screenshot_*`, e o iPhone só gera .png para captura de tela."""
    if entry.get("media_type") not in CAMERA_MEDIA_TYPES:
        return False
    return not entry["name"].lower().startswith("screenshot")


def dedupe_by_content(entries: list[dict]) -> tuple[list[dict], int]:
    """Descarta fotos com o mesmo `contentHash` de uma já vista — é
    duplicata exata, não precisa baixar nem processar a segunda vez.
    Entradas sem hash passam sempre (não dá para afirmar nada)."""
    seen: set[str] = set()
    out, dropped = [], 0
    for entry in entries:
        h = entry.get("content_hash")
        if h and h in seen:
            dropped += 1
            continue
        if h:
            seen.add(h)
        out.append(entry)
    return out, dropped


def plan_batches(entries: list[dict], max_bytes: int, max_items: int = 25) -> list[list[dict]]:
    """Monta lotes respeitando um teto de bytes.

    Nunca põe duas fotos de mesmo nome no mesmo lote: o download usa
    `-c rename`, então duas fotos homônimas viram `nome.jpg` e
    `nome (1).jpg` e deixa de existir correspondência confiável entre
    arquivo e `uid`. Separando-as em lotes diferentes, cada arquivo que
    aparece na pasta vazia tem nome único e o mapeamento é exato."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    current_names: set[str] = set()
    for entry in entries:
        size = entry.get("size") or 0
        collide = entry["name"] in current_names
        if current and (collide or current_bytes + size > max_bytes or len(current) >= max_items):
            batches.append(current)
            current, current_bytes, current_names = [], 0, set()
        current.append(entry)
        current_bytes += size
        current_names.add(entry["name"])
    if current:
        batches.append(current)
    return batches
