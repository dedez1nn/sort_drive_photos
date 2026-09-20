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

O reconhecimento facial usa `insightface` (detector SCRFD + embedding
ArcFace) sobre `onnxruntime` em CPU. Os modelos (~300MB) são baixados
sozinhos na primeira execução, para `~/.insightface/`. Não compila nada —
o `face_recognition`/`dlib`, que exigia compilar C++ e `cmake`, foi
substituído porque o embedding de 128-d dele não separava as pessoas desta
biblioteca (detalhes em [`docs/agrupamento_facial.md`](docs/agrupamento_facial.md)).

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

Roda em lote, em CPU (sem GPU dedicada nesta máquina). Leva ~0,32s por
imagem — cerca de 2h para uma biblioteca de 21 mil fotos — e é retomável. Detecção fica cacheada em
`data/faces_index.json`; identidade (pessoa/foto/relação) persiste em
`data/faces.db` (SQLite) — cada pessoa tem um `person_id` estável entre
execuções, não um cluster recalculado do zero toda vez.

O resultado fica em `data/by_person/` (symlinks, não copia os arquivos):
`person_XXX/` para fotos onde a pessoa aparece sozinha,
`person_AAA_e_person_BBB/` para fotos com 2+ pessoas identificadas juntas, e
`pessoas_raras/` para quem aparece em menos de 3 fotos (a maioria dos rostos
de uma biblioteca real é gente que aparece uma vez só — desconhecido ao
fundo, rosto num cartaz — e sem isso as dezenas de pastas de uma foto afogam
as poucas pessoas que importam). Fotos com rosto detectado mas não
identificado caem em `aguardando_fase2/` (3+ rostos, ainda não processado) ou
`revisao_manual/` (ambíguo demais para decidir sozinho).

Rostos pequenos demais ou com baixa confiança de detecção são descartados
antes de virar identidade (`--min-face-px`, default 50; `--min-det-score`,
default 0.6). O embedding de um rosto minúsculo é praticamente ruído: fica
"meio perto" de todo mundo e serve de ponte entre pessoas que não se
parecem.

Duas pessoas na mesma foto nunca são agrupadas como a mesma identidade —
ninguém aparece duas vezes numa foto. Parece óbvio, mas é o que permite
usar um limiar generoso o bastante para reunir a mesma pessoa em condições
bem diferentes (retrato de estúdio e selfie de praia, por exemplo) sem
fundir quem aparece ao lado dela.

O que o filtro **não** pega é conteúdo grande e nítido que não é uma pessoa
da sua vida: rosto em cartaz de filme, foto de jornal numa tela de TV,
ilustração. Para esses, marque como falso positivo permanentemente:

```bash
python scripts/face_cluster.py --exclude IMG_XXXX.HEIC "motivo aqui"
```

De tempos em tempos (ou depois de mudar qualquer threshold), vale refazer as
identidades do zero a partir do índice inteiro, com clusterização global —
não depende da ordem em que as fotos foram processadas, ao contrário das
fases incrementais, e preserva as exclusões manuais:

```bash
python scripts/face_cluster.py --recluster
```

Toda execução termina imprimindo as 10 maiores identidades com a coerência
interna de cada uma. Se a maior identidade concentrar uma fatia grande da
biblioteca, ou aparecer marcada como incoerente, é sinal de que os limiares
precisam de ajuste (`--match-threshold`, default 1.00 — na escala do
ArcFace, não comparável com a de versões anteriores). Detalhes da
arquitetura, das duas fases e das medições em
[`docs/agrupamento_facial.md`](docs/agrupamento_facial.md).

## 5. Varrer a biblioteca inteira e achar as pessoas mais frequentes

O passo 4 organiza o que está em disco. Para responder "quem são as 20
pessoas que mais aparecem em anos de fotos" o caminho é outro,
porque a biblioteca tem 46 GB e esta máquina tem menos de 2 GB livres.

A ideia é separar colher de decidir: os embeddings da biblioteca inteira
cabem em dezenas de MB, e depois de extraídos as fotos são descartáveis.

```bash
# 1. varre tudo: baixa lote -> extrai rostos -> apaga lote  (~2h, uma vez)
PROTON_DRIVE_CREDENTIALS_STORE=pass python scripts/scan_library.py

# 2. ranqueia, offline, sem precisar de nenhuma foto        (~1 min)
python scripts/rank_people.py --top 20

# 3. bloqueia quem não deve virar álbum (opcional, ver abaixo)
python scripts/person_blocklist.py add pessoa_07 --nivel sem-album --nome "Máscaras"
python scripts/person_blocklist.py add pessoa_03 --nivel ignorar

# 4. cria um álbum por pessoa no Proton Photos (não duplica nenhuma foto)
python scripts/album_person.py pessoa_01 --dry-run
python scripts/album_person.py pessoa_01 pessoa_02
python scripts/album_person.py a1b2c3d4e5f6 --name "Nome da pessoa"

# ou, se preferir as fotos em disco:
python scripts/fetch_person.py pessoa_01 --dry-run
python scripts/fetch_person.py pessoa_01 --limit 10
```

O `album_person.py` usa **álbuns**, não pastas: o Proton Photos é uma seção
separada de `/my-files` e não aceita pastas. O álbum referencia as fotos que
já estão lá — não duplica byte nenhum, aparece no app do celular, e apagá-lo
não toca nas fotos.

A varredura monta antes um catálogo da timeline (`data/catalog.jsonl`) com
nome, tipo e tamanho de cada foto, o que permite **descartar vídeo,
screenshot e sticker antes de baixar** — na biblioteca de teste, 36 dos 46 GB.
O pico de disco é o tamanho de um lote (`--batch-mb`, default 1 GB).

De cada rosto ficam guardados três artefatos: o embedding (para agrupar), o
`uid` no Proton (para rebaixar a foto depois) e um recorte de ~200px (para
você reconhecer quem é sem rebaixar nada). O ranking sai em
`data/top_pessoas/`, com uma prancha de rostos por pessoa.

Cada pessoa tem um `id` estável (`a1b2c3d4e5f6`) além do `pessoa_NN`. Use o
id: `pessoa_NN` é só a colocação e **troca de dono** se o ranking for
recalculado com mais fotos.

### Pessoas que não devem virar álbum

Nem toda identidade do topo merece um álbum. Duas situações, dois níveis:

| nível | aparece no ranking | vira álbum | para quê |
|---|---|---|---|
| `sem-album` | sim, carimbado com o motivo | não | grupos que não são uma pessoa só — gente de máscara, por exemplo, que colapsa numa identidade só porque metade do rosto está coberta |
| `ignorar` | não | não | pessoas de verdade que você não quer organizar; somem do ranking e de qualquer álbum, hoje e nas varreduras futuras |

```bash
python scripts/person_blocklist.py add pessoa_07 --nivel sem-album \
    --nome "Máscaras" --motivo "não é uma pessoa"
python scripts/person_blocklist.py add pessoa_03 pessoa_09 --nivel ignorar
python scripts/person_blocklist.py list
python scripts/person_blocklist.py check    # o que cada bloqueio pega, e com que folga
python scripts/person_blocklist.py remove b2c3d4e5f601
```

O bloqueio é guardado pelos **rostos** da pessoa, não pelo `pessoa_NN` nem
pelo `id` — os dois mudam quando a biblioteca cresce, e "nunca mais" só vale
se continuar valendo depois da próxima varredura. O reconhecimento usa o
mesmo critério que o nível 2 do ranqueamento ("estes dois conjuntos de
rostos são a mesma pessoa"), inclusive a regra de que duas pessoas na mesma
foto são pessoas diferentes — é ela que, na biblioteca de teste, impede que
bloquear uma identidade leve junto a vizinha mais próxima, a 0.992 dela. Por que
o teste rosto a rosto do nível 3 **não** serve aqui (pegou 645 rostos da
pessoa errada), em
[`docs/varredura_biblioteca.md`](docs/varredura_biblioteca.md).

É seguro interromper e retomar. Detalhes e a validação do método em
[`docs/varredura_biblioteca.md`](docs/varredura_biblioteca.md).

## 6. Levantar duplicatas

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

## 7. Avaliação

Depois de rodar tudo, navegue em `data/by_person/` e confira se a
organização por pessoa está boa o suficiente para substituir o álbum
"Pessoas" do iCloud. Só depois disso — e por sua conta — decida sobre apagar
fotos/arquivos locais do iPhone.
