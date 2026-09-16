# foto-organizer

Scripts locais (sem Docker, sem servidor) para baixar fotos do Proton Drive,
agrupá-las por pessoa via reconhecimento facial e levantar duplicatas — para
avaliar se dá para substituir o iCloud tendo tudo organizado no Proton Drive.

Nada aqui apaga arquivos automaticamente, nem local nem no Drive, e nada
toca no iPhone. Isso fica para depois, manual, só depois de você validar o
resultado.

## 1. Instalar o CLI oficial do Proton Drive

Baixe o binário em https://proton.me/download/drive/cli (página oficial
dedicada ao CLI; `proton.me/drive/download` é a página geral do Drive e só
aponta para essa mesma URL). Não use forks de terceiros tipo
"protondrive-for-linux" — só o binário oficial da Proton.

Em x86_64, use a build `linux/x64`; se ela travar na inicialização com
`Illegal instruction` (comum em NAS e CPUs embarcadas sem AVX2), baixe a
variante `linux/x64-baseline`.

```bash
chmod +x proton-drive
mv proton-drive ~/.local/bin/proton-drive   # ou outro lugar no seu PATH
```

Dependências no Linux: um backend de credenciais compatível com
`libsecret` (GNOME Keyring, KWallet, ou `pass` via
`PROTON_DRIVE_CREDENTIALS_STORE=pass`, como usado abaixo) e
`dbus-x11`/`dbus-run-session` para expor o Secret Service quando não há
sessão gráfica com D-Bus ativa.

### Autenticação

Este sistema tem o Secret Service (`org.freedesktop.secrets`) desativado
(quebrava o Proton Mail Bridge), então
o keychain padrão do CLI não funciona. Use o backend `pass`, que já está
configurado neste PC:

```bash
PROTON_DRIVE_CREDENTIALS_STORE=pass dbus-run-session -- proton-drive auth login
```

Isso abre o navegador para o login. A sessão fica salva em
`pass show ch.proton.drive/drive-sdk-cli/auth-session`.

Depois de logado, confira o que tem no Drive e identifique a pasta com as
fotos:

```bash
PROTON_DRIVE_CREDENTIALS_STORE=pass dbus-run-session -- proton-drive filesystem list /
```

## 2. Instalar as dependências Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`face_recognition` depende do `dlib`, que compila C++ na instalação — pode
demorar alguns minutos. Se faltar `cmake` ou compilador, instale via pacote
do sistema antes (`sudo pacman -S cmake` no CachyOS/Arch).

## 3. Baixar as fotos do Proton Drive

As fotos do iPhone ficam no Proton Photos, não em `/my-files` — é uma seção
separada do Drive (`filesystem list /photos` retorna "Path type photos is
not supported"), com comandos próprios (`photo timeline`, `photo download`).
O script usa esses comandos e sempre sincroniza a timeline inteira (não tem
como escolher uma subpasta, porque a timeline não é organizada em pastas):

```bash
PROTON_DRIVE_CREDENTIALS_STORE=pass python scripts/proton_sync.py
```

(Não precisa de `dbus-run-session` aqui — só o `auth login` interativo do
passo anterior precisa de D-Bus para abrir o navegador.)

O script é seguro de interromper (Ctrl-C) e rodar de novo — ele não baixa de
novo o que já está em `data/manifest.json`. Use `--dry-run` para conferir
quantas fotos faltam baixar sem baixar nada, e `--batch-size` para ajustar
quantas fotos pedir por chamada do CLI (default 25).

## 4. Agrupar rostos por pessoa

```bash
python scripts/face_cluster.py             # fase 1: fotos com 1-2 rostos
python scripts/face_cluster.py --phase2    # fase 2: fotos com 3+ rostos adiadas
```

Roda em lote, em CPU (sem GPU dedicada nesta máquina — pode demorar para uma
biblioteca grande, mas é retomável). Detecção fica cacheada em
`data/faces_index.json`; identidade (pessoa/foto/relação) persiste em
`data/faces.db` (SQLite) — cada pessoa tem um `person_id` estável entre
execuções, não um cluster recalculado do zero toda vez.

O resultado fica em `data/by_person/` (symlinks, não copia os arquivos):
`person_XXX/` para fotos onde a pessoa aparece sozinha, e
`person_AAA_e_person_BBB/` para fotos com 2+ pessoas identificadas juntas.
Fotos com rosto detectado mas não identificado caem em `aguardando_fase2/`
(3+ rostos, ainda não processado) ou `revisao_manual/` (ambíguo demais para
decidir sozinho).

Se o detector confundir alguma coisa sem rosto (ex.: foto de uma tela, ou de
um objeto) com uma pessoa, marque como falso positivo permanentemente:

```bash
python scripts/face_cluster.py --exclude IMG_XXXX.HEIC "motivo aqui"
```

Ajuste `--match-threshold`/`--uncertain-threshold` se pessoas diferentes
estiverem sendo confundidas, ou se a mesma pessoa não estiver sendo
reconhecida entre fotos. Detalhes da arquitetura e das duas fases em
[`docs/agrupamento_facial.md`](docs/agrupamento_facial.md).

## 5. Levantar duplicatas

```bash
python scripts/dedupe_report.py
```

Gera `data/duplicates_report.json` com grupos de arquivos idênticos (hash
sha256) e o espaço total desperdiçado. Revise manualmente antes de apagar
qualquer coisa.

Para apagar uma duplicata confirmada do Proton Drive (comando manual,
arquivo por arquivo — não existe apagamento em lote automatizado aqui):

```bash
PROTON_DRIVE_CREDENTIALS_STORE=pass dbus-run-session -- \
  proton-drive filesystem delete /Fotos/caminho/do/arquivo/duplicado.jpg
```

## 6. Avaliação

Depois de rodar tudo, navegue em `data/by_person/` e confira se a
organização por pessoa está boa o suficiente para substituir o álbum
"Pessoas" do iCloud. Só depois disso — e por sua conta — decida sobre apagar
fotos/arquivos locais do iPhone.
