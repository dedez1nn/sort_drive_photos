# Agrupamento facial por identidade persistente

Documenta a reescrita do agrupamento de rostos pedida depois que a primeira
versão (DBSCAN global, `scripts/face_cluster.py` antigo) mostrou o problema:
a mesma pessoa aparecia fragmentada em vários grupos diferentes a cada
execução, porque os labels de cluster (`pessoa_00`, `pessoa_01`, ...) eram
recalculados do zero toda vez, sem nenhuma identidade persistente entre
rodadas.

## Modelo de dados

Persistido em SQLite (`data/faces.db`, ver `scripts/identity_store.py`):

```
Person
 ├── person_id       (ex.: "person_001" — sequencial, nunca reindexado)
 └── created_at

Photo
 ├── photo_id         (sha256 do conteúdo do arquivo — nunca duplica)
 ├── path
 ├── num_faces
 └── status           pending | phase1_done | deferred_multi_face |
                       phase2_done | excluded

Face
 ├── face_id           ("{photo_id}:{face_index}")
 ├── photo_id
 ├── embedding         (vetor 128-d do face_recognition, cru)
 ├── person_id         (NULL se ainda não resolvido)
 ├── confidence         (distância euclidiana ao centroide no momento do match)
 └── status            assigned | uncertain | deferred

Relationship
 ├── person_a, person_b   (ordem canônica: a < b)
 └── photo_id              (em qual foto essas duas pessoas apareceram juntas)

excluded_photos
 ├── photo_id
 └── reason             (sempre uma decisão visual — ver "Falsos positivos")
```

Não existe uma coluna de "centroide" congelada por pessoa. Cada pergunta
"esse rosto é de quem?" recalcula a média dos embeddings já confirmados
daquela pessoa (`PersonGraph.centroid` em `identity_store.py`). Nessa escala
(dezenas a poucas centenas de rostos) isso é rápido o bastante e evita bugs
de média incremental — e faz o merge de duas pessoas virar uma troca trivial
de `person_id` nas linhas, em vez de recombinar médias na mão.

## Processamento em duas fases

Fotos com muita gente têm muito mais chance de casar um rosto com a pessoa
errada (mais candidatos, mais ambiguidade). Por isso o processamento separa:

### Fase 1 — `python scripts/face_cluster.py`

Só fotos com **1 ou 2 rostos**.

- **1 rosto**: compara o embedding com o centroide de cada pessoa já
  conhecida.
  - distância ≤ `--match-threshold` (default `0.5`) → atribui a essa pessoa.
  - distância entre `--match-threshold` e `--uncertain-threshold` (default
    `0.58`) → fica `uncertain` (não força o match, vai para revisão).
  - distância maior → cria pessoa nova.
- **2 rostos**: cada rosto passa pelo mesmo processo acima,
  **independentemente** — nunca assume que são duas pessoas novas só por
  serem duas. Depois, se os dois resolverem para pessoas diferentes, grava
  uma `Relationship` entre elas para essa foto ("essas duas pessoas
  apareceram juntas", não "são a mesma pessoa").
- **3+ rostos**: a foto é adiada (`status = deferred_multi_face`), sem
  tentar identificar ninguém agora.

No fim da fase 1, roda `rediscover_uncertain`: rostos que ficaram sem pessoa
(`uncertain`) são comparados só entre si (nunca contra pessoas já
confirmadas). Se dois ou mais aparecerem repetidamente parecidos, viram uma
pessoa nova — é assim que alguém que só aparece em fotos de casal
(`Foto A: pessoa conhecida + desconhecido`, `Foto B: idem`) acaba sendo
reconhecido como uma segunda identidade recorrente, sem contaminar quem já
estava confirmado.

### Fase 2 — `python scripts/face_cluster.py --phase2`

Processa as fotos adiadas (3+ rostos). Cada rosto é comparado com as
pessoas **já consolidadas** na fase 1; só aceita match confiante
(`--match-threshold`). **Nunca cria pessoa nova aqui** — o risco de errar
com muitos candidatos na mesma foto é maior, então uma foto com gente ainda
desconhecida fica parcialmente resolvida (quem bateu, bate; quem não bateu,
continua sem pessoa) em vez de arriscar inventar identidades.

Relações são gravadas entre todo par de pessoas resolvidas na mesma foto
(não só pares adjacentes).

## Consolidação (revisão de grupos)

Depois de qualquer execução (fase 1 ou 2), `PersonGraph.reconcile()` compara
o centroide de cada par de pessoas; se a distância ficar dentro do
`--match-threshold`, funde as duas (`PersonGraph.merge`): reatribui rostos e
relações para o `person_id` mantido, remove o outro. Isso cobre o caso de
duas pastas antigas que na real são a mesma pessoa — sem precisar de um
script de migração separado, porque a única informação que existia antes
(os embeddings em `data/faces_index.json`) já é reaproveitada como entrada
desse mesmo pipeline.

## Falsos positivos (fotos que não são rosto)

Não existe sinal automático confiável para saber se uma "foto" tem um rosto
de verdade ou é, por exemplo, a foto de uma tela de notebook mostrando
código, ou o padrão de furos de um roteador Wi-Fi — o EXIF de câmera é
idêntico ao de uma foto real, e os dois casos encontrados na prática (ver
histórico) só foram identificados olhando a imagem.

```bash
python scripts/face_cluster.py --exclude IMG_XXXX.HEIC "motivo aqui"
```

Marca a foto como excluída permanentemente: desfaz qualquer pessoa/relação
já criada a partir dela, remove pessoas que ficarem sem nenhum rosto
restante (só existiam por causa dela), e nunca mais reprocessa mesmo
reconstruindo o índice do zero.

## Pastas finais (`data/by_person/`)

A pasta é só uma visão derivada e descartável do banco — a fonte da verdade
é sempre o SQLite. Reconstruída em cada execução:

- `person_XXX/`: só fotos onde **essa pessoa aparece sozinha** (1 rosto na
  foto inteira).
- `person_AAA_e_person_BBB/`: fotos com 2+ pessoas identificadas, nomeada
  pela combinação exata de quem aparece — reaproveitada automaticamente
  sempre que essa mesma combinação reaparecer numa foto futura, em vez de
  duplicar a pessoa numa pasta separada.
- `aguardando_fase2/`: fotos com 3+ rostos que ainda têm alguém não
  identificado (mesmo depois da fase 2 já ter resolvido quem deu para
  resolver).
- `revisao_manual/`: rostos que caíram na zona ambígua (`uncertain`) e não
  foram redescobertos como pessoa nova.

Nenhuma foto é duplicada fisicamente entre pastas — são symlinks para o
mesmo arquivo em `data/library/`.

## Limitações conhecidas

- **Falso negativo de detecção**: o modelo `hog` (leve, roda em CPU) às
  vezes perde um rosto em ângulo de perfil/lateral, mesmo estando grande e
  em foco na foto (caso confirmado: `IMG_8113.HEIC`, uma selfie de casal
  onde só o rosto de uma pessoa foi detectado). O modelo `cnn` é mais
  preciso, mas não é viável nesta máquina — sem GPU e com pouca RAM, entra
  em *swap thrashing* e fica ordens de magnitude mais lento (chegou a mais
  de 7 minutos numa única foto antes de ser interrompido).
- **Falso positivo de detecção**: padrões de imagem sem rosto nenhum
  (grade de furos, texto de tela) ocasionalmente são detectados como rosto.
  Sem sinal automático confiável para filtrar — ver `--exclude` acima.
- Essas duas limitações são do **detector de rostos** (`face_recognition`),
  não da lógica de identidade/agrupamento descrita neste documento.

## Referência rápida de comandos

```bash
python scripts/face_cluster.py                    # fase 1
python scripts/face_cluster.py --phase2            # fase 2
python scripts/face_cluster.py --exclude ARQ "motivo"   # marca falso positivo
```

Flags relevantes: `--match-threshold` (default `0.5`), `--uncertain-threshold`
(default `0.58`), `--model {hog,cnn}` (default `hog`), `--db`
(default `data/faces.db`).
