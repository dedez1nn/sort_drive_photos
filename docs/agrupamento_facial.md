# Agrupamento facial por identidade persistente

Histórico das três iterações e, principalmente, do que cada medição
derrubou da hipótese anterior.

| versão | modo de falha observado |
|---|---|
| 1. DBSCAN global | a mesma pessoa em grupos diferentes a cada execução (labels recalculados do zero, sem identidade persistente) |
| 2. identidade persistente + match incremental por centroide | **pessoas diferentes coladas numa identidade só**: `person_001` com 266 dos 712 rostos, misturando pessoas diferentes, um desenho e uma foto de folha de caderno |
| 3. quórum + portão de qualidade (dlib) | blob resolvido, mas **a mesma pessoa fragmentada em três identidades** — o mesmo rapaz de óculos em `person_001`, `person_040` e `person_041` |
| 4. atual: ArcFace + restrição de exclusão mútua | nenhuma das duas falhas nas 8 maiores identidades, conferidas rosto a rosto |

## A medição que decidiu tudo

Depois de filtrar o lixo da detecção, medimos as distâncias entre rostos
**conferidos visualmente** — mesma pessoa em fotos diferentes, e pessoas
sabidamente distintas:

| | mesma pessoa (máx) | pessoas diferentes (mín) |
|---|---|---|
| dlib 128-d (`face_recognition`) | 0.727 | 0.554 |
| ArcFace 512-d (`insightface`) | 1.045 | 1.292 |

Com o dlib as duas distribuições **se invertem**: o mesmo rapaz, em
retrato de estúdio em P&B e em selfie na praia, ficava a 0.727 — mais
longe do que ele e uma pessoa diferente (0.554). Não existe limiar que
separe isso. Apertar o limiar fragmentava a mesma pessoa; afrouxar colava
pessoas diferentes. Era uma limitação do modelo, não da lógica de
agrupamento.

Vale registrar que a análise que originou este trabalho concluiu o
contrário — "não mexeria em embedding agora, o problema dominante é
conteúdo que não devia nem ter entrado no pipeline". Isso **era verdade
naquele momento**: com 57% dos rostos abaixo de 60px, o lixo dominava
qualquer outra consideração. Só depois de limpá-lo é que o embedding
apareceu como gargalo. As duas conclusões estão certas, em ordem.

De quebra o ArcFace é mais rápido: **0.32s por imagem** contra 1.2s do
HOG, medido nesta máquina (CPU, sem GPU). E o `face_recognition`/dlib saiu
das dependências, junto com a necessidade de compilar C++ na instalação.

## Por que a versão 2 engolia a biblioteca

A falha era **composta**. Reimplementando o algoritmo antigo e desligando
uma peça de cada vez sobre os mesmos rostos, com o maior grupo medido como
fatia do total:

| configuração | pessoas | maior grupo | % da biblioteca |
|---|---|---|---|
| v2 (centroide + reconcile em cascata, sem portão) | 29 | 54 | **37%** |
| ... só tirando o merge em cascata do `reconcile` | 32 | 47 | 32% |
| ... só trocando centroide por quórum | 36 | 45 | 31% |
| ... só aplicando o portão de qualidade | 20 | 18 | 22% |
| v3 (as três juntas, ainda com dlib) | 31 | 9 | 11% |

### Rostos pequenos demais entrando no pipeline

O detector HOG do dlib trabalha com janela de 80×80 e o `face_recognition`
ainda dá um upsample por padrão, então devolvia "rostos" de ~36-40px que
eram só o piso de detecção. A mediana dos rostos detectados tinha **52px**,
e 57% estavam abaixo de 60px. O embedding é calculado sobre um recorte
reescalado para 150×150 — partindo de 40px é quase tudo interpolação, e o
vetor que sai não descreve ninguém: fica "meio perto" de todo mundo e serve
de **ponte** entre pessoas que não se parecem.

### Match incremental contra um alvo que se move

`best_match` comparava o rosto novo com a **média** dos embeddings da
pessoa. A intuição natural — "a média de rostos diferentes vira um rosto
médio, artificialmente perto de todo mundo" — **não se confirmou**: medida
no blob de 266 rostos, a distância média ao centroide (0.597) e o score de
quórum (0.582) são praticamente idênticas. O centroide não era um ímã.

O problema real é que o centroide **se desloca a cada rosto absorvido**.
Cada match respeita o limiar, mas o ponto de referência anda junto: depois
de N passos a identidade derivou e aceita alguém que estaria muito além do
limiar a partir do ponto de partida.

### `reconcile()` fundindo em cascata

Comparava centroide com centroide e, a cada par dentro do limiar, fundia e
recomeçava — na prática um fecho transitivo, que é o critério de *single
linkage*, conhecido por produzir um componente gigante quando existem
pontos-ponte entre os grupos. E os pontos-ponte eram justamente os
embeddings de rosto pequeno.

E o `coherence()` que detectava o resultado só renomeava a pasta para
`_dispersa`: a identidade contaminada continuava no grafo, continuava
recebendo rostos novos e continuava participando do `reconcile`.

## O pipeline atual

### Detecção e embedding (`scripts/face_embedder.py`)

SCRFD (detecção) + ArcFace `buffalo_l` (embedding 512-d L2-normalizado),
via `insightface` em CPU. Os embeddings normalizados fazem a distância
euclidiana cair em [0, 2], relacionada à similaridade de cosseno por
`d² = 2 − 2·cos`. **Os limiares desta versão não são comparáveis com os da
versão dlib.**

### Orientação EXIF (corrigido depois, custava 40% dos rostos)

Quase toda foto de celular é gravada pelo sensor sempre na mesma orientação
física, com uma tag EXIF dizendo como girar na exibição. Quem lê os pixels
crus recebe a foto deitada — e era o que o pipeline fazia.

Medido numa amostra de 120 fotos da biblioteca de teste:

| | rostos detectados |
|---|---|
| sem `exif_transpose` | 35 |
| com `exif_transpose` | **57** (+63%) |

70% das fotos têm a tag. O SCRFD tolera inclinação, mas não 90 graus. O
erro foi descoberto olhando os recortes de rosto de uma prancha de revisão:
metade estava deitada. É um bom argumento a favor de sempre gerar material
visual — nenhuma métrica do pipeline acusava nada.

A leitura de imagem ficou concentrada em `face_embedder.load_image()`, que
aplica a orientação e registra o suporte a `.heic`. Antes cada script
abria a imagem por conta própria, e bastava um caminho novo esquecer de um
dos dois passos — foi exatamente o que aconteceu com o `.heic` ao conferir
fotos rebaixadas fora do `scan_library`.

### Portão de qualidade

`--min-det-score` (0.6) e `--min-face-px` (50). O `det_score` é a confiança
do próprio detector — o sinal que faltava no HOG, que não dava confiança
nenhuma e obrigava a medir nitidez do recorte por fora (variância do
laplaciano), uma aproximação grosseira que foi removida.

As medidas ficam no índice em vez de o filtro ser aplicado na detecção: dá
para mudar os limiares e reprocessar sem redetectar tudo.

### Restrição de exclusão mútua

**Dois rostos da mesma foto nunca viram a mesma pessoa** — ninguém aparece
duas vezes numa foto. É informação que estava nos dados e não era usada, e
ela importa porque quem convive com você aparece ao seu lado nas fotos:
são exatamente os pares que o embedding mais tende a confundir. Sem ela,
subir o limiar o bastante para reunir a mesma pessoa em condições
diferentes também começava a fundir quem aparecia ao lado dela.

Implementada em `identity_store._cluster_constrained` (average linkage com
atualização de Lance-Williams, feito à mão porque o scipy não aceita
restrições) e em `PersonGraph.best_match(excluir=...)` no caminho
incremental.

### Match por quórum, com freio de coerência

`PersonGraph.score` devolve a distância média do rosto novo aos **3 rostos
concretos mais próximos** da pessoa (`QUORUM`), não a um centroide — uma
âncora que não se desloca a cada absorção. `would_stay_coherent` simula a
adição antes de aceitar: se ela empurra a distância média interna acima de
`--coherence-limit`, o match é recusado e o rosto vai para `uncertain`.

### Separação automática de identidade contaminada

`PersonGraph.split` quebra uma identidade incoerente nos subgrupos que ela
de fato contém; `split_incoherent` repete até não sobrar nenhuma. O maior
subgrupo mantém o `person_id` original (as pastas já sincronizadas
continuam fazendo sentido); os outros viram pessoas novas.

### `reconcile` com average linkage

A distância entre duas pessoas é a média entre **todos os pares de rostos
reais** das duas, não entre centroides — dois grupos podem ter centroides
próximos sem que nenhum par de rostos concretos seja parecido. O merge é
recusado se a pessoa resultante ficasse incoerente, e cada rodada funde o
par mais próximo de todos, não o primeiro em ordem alfabética.

### Reclusterização global — `--recluster`

O caminho incremental (fase 1/fase 2) permite processar a biblioteca em
lotes sem guardar tudo em disco, mas decide rosto a rosto na ordem em que
as fotos chegam: uma decisão ruim no começo fica para sempre.
`--recluster` olha todos os rostos do índice de uma vez, então não depende
de ordem. Só as exclusões manuais sobrevivem.

Funciona mesmo depois do `sweep_library.sh` ter apagado os lotes locais: o
`photo_id` das fotos já processadas é lido do banco em vez de recalculado a
partir do arquivo. Tem teto de memória (`MAX_GLOBAL_CLUSTER_FACES`, 12000)
porque a matriz n×n de distâncias não é esparsa.

## Calibração dos limiares

Feita contra rostos conferidos visualmente, variando só o corte:

| limiar | pessoas | resultado |
|---|---|---|
| 0.95 | 65 | criança de óculos escuros dividida em 2 grupos |
| **1.00** | **60** | **tudo correto nas 8 maiores identidades** |
| 1.05 | 57 | duas pessoas diferentes fundidas num grupo de 6 |
| 1.10 | 55 | idem, pior |

`--coherence-limit` 1.10: a maior coerência observada num grupo
comprovadamente correto é 0.996, então sobra margem sem quebrar nada certo.

## Modelo de dados

SQLite (`data/faces.db`, ver `scripts/identity_store.py`):

```
Person        person_id ("person_001", sequencial, nunca reindexado), created_at
Photo         photo_id (sha256 do conteúdo), path, mtime, num_faces,
              status: pending | phase1_done | deferred_multi_face |
                      phase2_done | reclustered | excluded
Face          face_id ("{photo_id}:{face_index}"), photo_id, embedding (512-d),
              person_id (NULL se não resolvido), confidence,
              status: assigned | uncertain | deferred
Relationship  person_a, person_b (ordem canônica), photo_id
excluded_photos  photo_id, reason
meta          person_seq, embedder
```

`meta.embedder` guarda qual modelo gerou os embeddings. Vetores de modelos
diferentes têm dimensão e escala diferentes, então misturá-los faria o
agrupamento errar em silêncio — na troca de modelo, o índice e as
identidades são refeitos do zero (as exclusões manuais são preservadas).

O índice de detecção (`data/faces_index.json`) guarda `embedder` e, por
rosto: `image`, `mtime`, `face_id`, `bbox`, `det_score`, `face_px`,
`encoding`; e em `scanned`, o `mtime` de cada foto já varrida, **inclusive
as sem rosto** — senão elas não deixam rastro e são redetectadas em toda
execução (eram 168 das 243 da biblioteca de teste, o passo mais caro rodando à
toa). O índice é chaveado por caminho **absoluto**: `find_images` resolve o
caminho antes de devolver, senão rodar com `--library-dir data/library` e
com o caminho absoluto geraria duas entradas para a mesma foto.

## Processamento em duas fases

Fotos com muita gente têm mais chance de casar um rosto com a pessoa
errada. Por isso:

**Fase 1** (`face_cluster.py`) — só fotos com 1 ou 2 rostos. Cada rosto é
resolvido independentemente (nunca se assume que dois rostos são duas
pessoas novas só por serem dois), e as pessoas já identificadas na mesma
foto ficam fora da disputa. Fotos com 3+ rostos são adiadas. No fim,
`rediscover_uncertain` agrupa entre si os rostos que ficaram sem pessoa —
nunca contra pessoas já confirmadas, para não contaminar quem está
resolvido.

**Fase 2** (`--phase2`) — as fotos adiadas. Só match confiante, e **nunca
cria pessoa nova**: uma foto com gente desconhecida fica parcialmente
resolvida em vez de inventar identidades.

## Falsos positivos (fotos que não são rosto)

O portão de qualidade resolve a maior parte. O que ele **não** pega é
conteúdo grande e nítido que não é uma pessoa da sua vida: rosto em cartaz
de filme, capa de disco, foto de jornal numa tela de TV, ilustração, ou
(caso real) uma foto de folha de caderno cujo padrão de escrita virou um
"rosto" de 223px perfeitamente nítido. Para esses continua sendo decisão
visual:

```bash
python scripts/face_cluster.py --exclude IMG_XXXX.HEIC "motivo aqui"
```

Desfaz qualquer pessoa/relação criada a partir da foto, remove pessoas que
ficarem sem nenhum rosto, e nunca mais reprocessa — inclusive sobrevive a
um `--recluster` e a uma troca de modelo.

## Pastas finais (`data/by_person/`)

Visão derivada e descartável do banco, reconstruída em cada execução:

- `person_XXX/` — fotos onde a pessoa aparece **sozinha** (1 rosto na foto).
- `person_XXX_em_grupo/` — identificada, mas com gente não identificada junto.
- `person_AAA_e_person_BBB/` — combinação exata de quem aparece, reaproveitada
  sempre que a mesma combinação reaparecer.
- `pessoas_raras/` — pessoas com menos de `--min-person-photos` (3) rostos.
  Numa biblioteca real a maioria dos rostos é gente que aparece uma vez só
  (rosto num cartaz, desconhecido ao fundo, garçom) e sem isso as dezenas de
  pastas de uma foto afogam as poucas pessoas que importam.
- `aguardando_fase2/`, `revisao_manual/` — não resolvidas.

Symlinks, nunca cópias.

## Resultado

Mesma biblioteca (243 fotos):

| | v2 (centroide, dlib) | v3 (quórum, dlib) | v4 (atual, ArcFace) |
|---|---|---|---|
| rostos que viram identidade | 712 | 80 | 113 |
| maior identidade | 266 (37%) | 9 (11%) | 18 (16%) |
| o rapaz de óculos está em | 1 identidade contaminada | **3 identidades** | **1 identidade** |
| identidades incoerentes | 1 | 0 | 0 |

Conferido abrindo os recortes de rosto de **todas** as identidades com 2+
rostos: as 8 maiores estão corretas, sem mistura e sem fragmentação.
Inclui casos difíceis que a v3 errava — o mesmo rapaz em retrato de estúdio
P&B e selfie na praia, a mesma criança de recém-nascida a dois anos, e duas
pessoas parecidas mantidas separadas.

O que sobrou em `pessoas_raras/` também foi conferido: é majoritariamente
**conteúdo que não é gente da sua vida** (cartaz de filme, capa de disco,
foto de jornal numa tela, ilustração, atleta numa transmissão). As pastas
de 1-2 fotos não são a mesma pessoa fragmentada — são rostos que realmente
só aparecem uma vez.

## Sobre a hipótese dos frames 16:9

70% da biblioteca de teste tem nome UUID e resolução 16:9 exata (3840×2160,
3664×2062), contra o 4:3 das fotos de câmera do iPhone — provavelmente
frames de vídeo/Live Photo extraídos pelo Proton Photos. A suspeita era de
que filtrá-los por resolução resolveria o ruído.

Medido: esses arquivos geravam 38 dos 147 rostos, e o portão de qualidade
sozinho descartava 35 deles (92%), mantendo os 3 que prestam. A hipótese
estava certa sobre a *origem* do ruído, errada sobre o mecanismo: o
problema é a qualidade do rosto, não a resolução do arquivo. Filtrar por
resolução descartaria panorâmicas legítimas sem ganho. A flag
`--skip-video-frames` existe, mas só vale para economizar tempo de detecção.

## Limitações conhecidas

- **Rostos descartados pelo portão** não são recuperáveis depois: quem só
  aparece pequeno ou de lado ao fundo não vira identidade. Troca deliberada.
- **Reclusterização global tem teto de memória** (12000 rostos): a matriz
  n×n de distâncias não é esparsa. Acima disso, só o caminho incremental.
- **Bebês em faixas de idade muito diferentes** continuam difíceis: recém-
  nascido e criança de dois anos são quase pessoas distintas para qualquer
  modelo. Aqui deu certo, mas é o caso mais frágil do conjunto.
- **Trocar de modelo invalida índice e identidades.** Se o
  `sweep_library.sh` já apagou os lotes locais, os rostos daquelas fotos se
  perdem — não dá para recalcular o embedding sem o arquivo.

## Referência rápida

```bash
python scripts/face_cluster.py                          # fase 1
python scripts/face_cluster.py --phase2                 # fase 2
python scripts/face_cluster.py --recluster              # refaz tudo do zero
python scripts/face_cluster.py --exclude ARQ "motivo"   # marca falso positivo
```

Flags: `--min-det-score` (0.6), `--min-face-px` (50), `--match-threshold`
(1.00), `--uncertain-threshold` (1.15), `--coherence-limit` (1.10),
`--min-person-photos` (3), `--det-size` (640), `--db` (`data/faces.db`).
