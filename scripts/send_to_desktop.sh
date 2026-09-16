#!/usr/bin/env bash
# Manda os agrupamentos de rosto (data/by_person) pro desktop-linux via
# SSH/rsync através da rede Tailscale, em ~/imagens/fotos_agrupadas/.
#
# Usa uma chave dedicada sem passphrase (~/.ssh/id_ed25519_fotosync),
# só autorizada no desktop-linux pra esse fim, então roda sem pedir senha:
#
#   scripts/send_to_desktop.sh
#
# `data/by_person/*` são symlinks pra `data/library`; o rsync com -L
# segue os links e manda o arquivo de verdade, não o atalho.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_USER="usuario"
REMOTE_HOST="desktop-linux"
REMOTE_DIR="Imagens/fotos_agrupadas"
IDENTITY="$HOME/.ssh/id_ed25519_fotosync"
SRC="${1:-$REPO_ROOT/data/by_person/}"

# --delete só quando manda a pasta padrão (data/by_person): aí o destino
# deve ser um espelho exato (sem sobrar pessoa_XX de execuções antigas).
# Com um SRC alternativo (ex.: data/pessoa_confirmada), --delete apagaria
# tudo mais que já está lá — por isso fica de fora nesse caso.
DELETE_FLAG=()
if [ "$#" -eq 0 ]; then
  DELETE_FLAG=(--delete)
fi

ssh -p 22 -i "$IDENTITY" -o IdentitiesOnly=yes "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p ~/${REMOTE_DIR}"

rsync -avL --progress "${DELETE_FLAG[@]}" \
  -e "ssh -p 22 -i $IDENTITY -o IdentitiesOnly=yes" \
  "${SRC%/}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Enviado para ${REMOTE_USER}@${REMOTE_HOST}:~/${REMOTE_DIR}/"
