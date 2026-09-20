#!/usr/bin/env bash
# Varre a biblioteca inteira do Proton Photos em lotes de tamanho fixo,
# processando e apagando cada lote local antes de baixar o próximo — dá
# pra rodar a biblioteca inteira (dezenas de milhares de fotos) sem
# precisar de disco pra guardar tudo de uma vez.
#
# Cada ciclo:
#   1. baixa fotos novas até ~MAX_SIZE_MB (--keep-ext filtra só foto real
#      de câmera: jpg/jpeg/heic/heif — sem screenshot/vídeo/sticker)
#   2. roda fase 1 e fase 2 do agrupamento facial (scripts/face_cluster.py)
#   3. sincroniza data/by_person pro desktop-linux, SEM --delete (acumula —
#      cada ciclo só teria o lote atual localmente, então um --delete
#      apagaria os lotes anteriores já enviados)
#   4. apaga o lote local (data/library) pra abrir espaço pro próximo
#
# As identidades (data/faces.db) e os embeddings (data/faces_index.json)
# nunca são apagados — é isso que permite processar em lotes sem perder o
# que já foi aprendido sobre quem é quem. Uma foto de grupo só resolvida
# numa fase 2 bem mais tarde (depois que a última pessoa dela já foi
# identificada noutro lote) não tem mais o arquivo local pra gerar um link
# novo — ela fica registrada certa no banco, mas o arquivo já sincronizado
# continua em aguardando_fase2/ no desktop, sem ser "movido" pra pasta da
# pessoa. Não perde a foto, só não reclassifica visualmente depois do fato.
#
# Uso:
#   scripts/sweep_library.sh            # loop até a biblioteca acabar
#   scripts/sweep_library.sh --once     # só um ciclo, não repete
#
# Variáveis de ambiente opcionais:
#   SWEEP_MAX_SIZE_MB (default 3000)   SWEEP_KEEP_EXT (default .jpg,.jpeg,.heic,.heif)
#   SWEEP_RECLUSTER_EVERY (default 10) — a cada N ciclos, refaz todas as
#     identidades do zero com clusterização global; 0 desliga
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MAX_SIZE_MB="${SWEEP_MAX_SIZE_MB:-3000}"
KEEP_EXT="${SWEEP_KEEP_EXT:-.jpg,.jpeg,.heic,.heif}"
RECLUSTER_EVERY="${SWEEP_RECLUSTER_EVERY:-10}"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

cycle=0
while true; do
  cycle=$((cycle + 1))
  echo "=== Ciclo $cycle ==="

  echo "--- Baixando lote (até ${MAX_SIZE_MB}MB) ---"
  PROTON_DRIVE_CREDENTIALS_STORE=pass .venv/bin/python scripts/proton_sync.py \
    --keep-ext "$KEEP_EXT" --max-size-mb "$MAX_SIZE_MB" --batch-size 25

  downloaded_count=$(find data/library -type f 2>/dev/null | wc -l)
  if [ "$downloaded_count" -eq 0 ]; then
    echo "Nada novo baixado — biblioteca inteira já sincronizada. Parando."
    break
  fi

  echo "--- Fase 1 (1-2 rostos), $downloaded_count fotos no lote ---"
  .venv/bin/python scripts/face_cluster.py

  echo "--- Fase 2 (3+ rostos) ---"
  .venv/bin/python scripts/face_cluster.py --phase2

  # As fases são incrementais: decidem rosto a rosto, na ordem em que as
  # fotos chegam, então uma decisão ruim num ciclo fica para sempre. A cada
  # RECLUSTER_EVERY ciclos, refaz as identidades do zero a partir do índice
  # inteiro (clusterização global, sem depender de ordem). Isso funciona
  # mesmo com os lotes antigos já apagados: os embeddings estão no índice e
  # o photo_id vem do banco. As exclusões manuais são preservadas.
  if [ "$RECLUSTER_EVERY" -gt 0 ] && [ $((cycle % RECLUSTER_EVERY)) -eq 0 ]; then
    echo "--- Reclusterização global (ciclo $cycle) ---"
    .venv/bin/python scripts/face_cluster.py --recluster
  fi

  echo "--- Sincronizando pro desktop (acumulando, sem --delete) ---"
  bash scripts/send_to_desktop.sh "$REPO_ROOT/data/by_person"

  echo "--- Apagando lote local (${downloaded_count} arquivos) ---"
  rm -rf data/library
  mkdir -p data/library

  if [ "$ONCE" -eq 1 ]; then
    echo "Modo --once, parando depois de um ciclo."
    break
  fi
done

echo "Varredura concluída."
