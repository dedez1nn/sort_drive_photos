# Varredura fria: quem são as pessoas mais frequentes da biblioteca

O agrupamento descrito em [`agrupamento_facial.md`](agrupamento_facial.md)
responde "quem está nesta pasta de fotos". Este documento trata de outra
pergunta, que a biblioteca inteira impõe: **quem são as 20 pessoas que mais
aparecem em anos de fotos**, sem conseguir ter as fotos em disco.

## O problema

Números reais da biblioteca de teste, tirados do catálogo do Proton:

| | itens | tamanho |
|---|---|---|
| vídeos (mp4 + mov) | 1.442 | 32,1 GB |
| screenshots (png) e gif | 1.039 | 3,0 GB |
| **fotos de câmera** (jpeg/heic) | **14.489** | **9,9 GB** |
| total | 21.439 | 46,3 GB |

Período: novembro/2017 a agosto/2026. O disco desta máquina tem **1,8 GB
livres**.

Três consequências:

1. Nunca dá para ter a biblioteca em disco. Nem perto.
2. O `sweep_library.sh` processa lote a lote e decide as identidades na
   ordem em que as fotos chegam. Serve para organizar o que está em disco,
   mas não para ranquear pessoas na biblioteca toda — a mesma pessoa vira
   identidades diferentes em lotes diferentes.
3. Depois que um lote é apagado, não havia caminho de volta. O
   `manifest.json` guardava só `uid -> captureTime`, e o download renomeia
   em caso de colisão, então **nada ligava um arquivo local ao seu nó no
   Proton**. Ranquear as pessoas e não conseguir buscar as fotos delas
   seria um resultado de papel.

## A ideia

Separar colher de decidir. Os embeddings da biblioteca inteira ocupam
dezenas de MB — depois de extraídos, as fotos são descartáveis.

```
scan_library.py     baixa lote -> extrai rostos -> apaga lote        (~2h, uma vez)
      |
      v
  data/faces/       embeddings + uid + recorte de cada rosto         (~60 MB)
      |
      v
rank_people.py      agrupa em dois níveis, ranqueia                  (~1 min)
      |
      v
fetch_person.py     rebaixa só as fotos de quem interessa
album_person.py     cria um álbum por pessoa no Proton Photos
```

Três artefatos por rosto, e é de propósito que sejam esses três:

- **embedding** — para agrupar sem a foto.
- **uid** — o nó no Proton, para rebaixar a foto original depois.
- **recorte** (~200px, ~10 KB) — para você olhar e reconhecer quem é, sem
  rebaixar nada.

## Passo 1 — catálogo

`photo timeline -d --json` carrega os detalhes de cada nó. É lento (alguns
minutos para 21 mil fotos, porque busca um a um) e roda **uma vez**, mas
muda o que é possível:

- `mediaType` e `name` descartam vídeo, screenshot e sticker **antes de
  baixar** — 36 GB que não precisam trafegar. O `--keep-ext` antigo baixava
  e apagava depois.
- `totalStorageSize` monta lotes por orçamento de bytes com precisão.
- `contentHash` pula duplicata exata sem baixar a segunda cópia.
- `uid` é o que torna o passo 3 possível.

Fica em `data/catalog.jsonl`, uma linha por foto.

## Passo 2 — varredura

```bash
PROTON_DRIVE_CREDENTIALS_STORE=pass python scripts/scan_library.py
```

Baixa um lote numa pasta vazia, extrai os rostos, grava no acervo, apaga o
lote, repete. O pico de disco é o tamanho de um lote (`--batch-mb`, default
1 GB; `--batch-items`, default 250 ≈ 170 MB na prática).

**Como o arquivo é casado com o `uid`:** os lotes são montados de modo que
nunca haja dois nomes iguais no mesmo lote (`plan_batches`), e o download
vai para uma pasta vazia. Assim cada arquivo que aparece tem nome único e a
correspondência com o `uid` é exata. É por isso que o `-c rename` do CLI —
que renomearia a segunda foto homônima e destruiria o mapeamento — nunca é
acionado.

Retomável: `data/faces/scanned.jsonl` registra as fotos já varridas,
inclusive as que não têm rosto nenhum (senão elas não deixam rastro e são
redetectadas a cada execução).

### Por que o acervo não é mais um JSON

A versão anterior guardava os embeddings como listas de float em
`faces_index.json` e reescrevia o arquivo inteiro a cada 25 fotos. Para
algumas centenas de rostos funciona; em escala, 25 mil rostos dariam ~400
MB de JSON reescritos umas mil vezes — centenas de GB de escrita para
guardar dezenas de MB de dado. Agora são dois arquivos append-only
(`embeddings.f32` binário e `faces.jsonl`), com correspondência posicional
entre eles.

## Passo 3 — ranqueamento em dois níveis

```bash
python scripts/rank_people.py --top 20
```

A clusterização hierárquica precisa da matriz n×n de distâncias, que não é
esparsa: 25 mil rostos são 5 GB, 40 mil são 14 GB. Não cabe nos 8,5 GB de
RAM livre. A saída é agrupar em níveis:

**Nível 1 — dentro de blocos.** Os rostos são divididos em blocos por ordem
cronológica (foto da mesma pessoa se concentra no tempo, então o bloco já
nasce coerente) e cada bloco é agrupado exatamente, com a clusterização
restrita usada no resto do projeto. Um bloco de 4000 rostos custa 128 MB.

A mesma pessoa aparece em vários blocos — o nível 2 junta. O que importa é
que **quem é frequente forma um grupo de 2+ rostos em cada bloco onde
aparece**: alguém com 200 fotos em 40 mil rostos tem ~20 por bloco.

**Nível 2 — entre blocos.** Só os grupos com 2+ rostos entram, o que
descarta de uma vez os milhares de rostos que aparecem uma única vez
(cartaz, desconhecido ao fundo) e derruba o tamanho do problema em uma
ordem de grandeza.

A distância entre dois grupos é a média das distâncias **ao quadrado** entre
todos os pares de rostos deles — average linkage exato, sem amostrar. Sai de
graça porque os embeddings do ArcFace são L2-normalizados, e nesse caso vale

```
média_{a∈A, b∈B} ||a − b||²  =  2 − 2·⟨centroide(A), centroide(B)⟩
```

(conferido numericamente contra a força bruta). O par de centroides dá o
valor **exato** da média sobre todos os pares, em tempo constante. Não
confundir com distância entre centroides, que seria uma aproximação ruim —
é a média real, escrita de outro jeito.

Uma primeira versão comparava 4 representantes por grupo, escolhidos nos
extremos para cobrir a variação interna. Falhou do jeito previsível em
retrospecto: average linkage sobre pontos extremos infla a distância, e uma
pessoa com 460 rostos saiu dividida em dois grupos do ranking.

### O corte de qualidade fica aqui, não na varredura

A varredura guarda todo rosto acima de `det_score` 0.60; o
`rank_people.py --min-det-score` (default 0.70) é que decide o que entra no
agrupamento. A divisão é deliberada: revarrer a biblioteca custa duas horas,
reranquear custa um minuto. Experimentar cortes diferentes não pode exigir
refazer a parte cara.

Medido: subir de 0.60 para 0.70 mantém 90% dos rostos e as contagens dos
maiores até melhoram (511 → 532 no primeiro colocado), porque menos detecção
ruim deixa o agrupamento mais limpo. Em 0.75 começa a custar gente real.

**Nível 3 — atribuição.** Com as identidades formadas, todos os rostos
(inclusive os que ficaram sozinhos no nível 1) são atribuídos por quórum.
É isso que dá a contagem exata: uma aparição isolada de alguém frequente
volta para a conta dele.

### Validação

Testado contra acervo sintético com rótulo conhecido, na geometria medida
do ArcFace (mesma pessoa até ~1,05 de distância; pessoas diferentes a
partir de ~1,29), 12.916 rostos em 10.208 fotos, 30 pessoas recorrentes com
contagens de 900 a 33, mais 6.000 aparições únicas de ruído, pessoas
convivendo nas mesmas fotos e um par propositalmente parecido (a 1,053 de
distância, tão perto quanto a variação interna de uma pessoa):

| | resultado |
|---|---|
| pessoas recuperadas | 30/30, nenhuma duplicada |
| rostos atribuídos à pessoa errada | 0 de 7.026 |
| grupos com recall incompleto | 0 |
| ruído que vazou para o top 30 | 0 |
| par parecido | mantido separado |
| tempo / pico de memória | 76 s / **0,42 GB** |

Estável entre `--merge-threshold` 1,05, 1,10 e 1,15, ou seja, não depende
de ajuste fino.

### `pessoa_NN` é posição, não pessoa

O ranking grava, por identidade, um `id` derivado do conjunto de fotos dela
(`a1b2c3d4e5f6`). Ele existe porque `pessoa_01`, `pessoa_02`... são apenas
colocações, e **a colocação troca de dono** quando o acervo cresce ou o
corte muda. Aconteceu na prática durante a varredura: uma pasta baixada
para uma colocação passou a corresponder a outra pessoa no ranqueamento
seguinte.

Os scripts aceitam o id no lugar do nome, e o `fetch_person` grava o id em
`_origem.json` dentro da pasta, recusando-se a misturar duas pessoas na
mesma pasta.

Cada pessoa também traz `faces` e `thumbs`: qual rosto de cada foto é dela.
Sem isso, conferir uma foto de grupo exige adivinhar entre os rostos
presentes — o que produziu três análises erradas antes de virar campo do
arquivo, inclusive uma conclusão falsa de que a maior identidade estava
contaminada.

## Passo 4 — rebaixar só o que interessa

```bash
python scripts/fetch_person.py pessoa_01 --dry-run   # quantas fotos, quantos MB
python scripts/fetch_person.py pessoa_01 pessoa_03
```

Lê os `uid` de `data/top_pessoas/ranking.json` e baixa as fotos daquelas
pessoas para `data/por_pessoa/<nome>/`. Aqui os arquivos **não** são
apagados — são o resultado.

## Passo 5 — organizar no Proton Photos

```bash
python scripts/album_person.py pessoa_01 --dry-run
python scripts/album_person.py pessoa_01 pessoa_02
python scripts/album_person.py a1b2c3d4e5f6 --name "Nome da pessoa"
```

Cria um **álbum** por pessoa. Não pasta: o Proton Photos é uma seção
separada de `/my-files` e não aceita pastas (`filesystem list /photos`
responde "Path type photos is not supported"). O álbum é o equivalente
nativo, e é melhor:

| | pasta em `/my-files` | álbum |
|---|---|---|
| duplica bytes | sim | **não, referencia a foto** |
| aparece no app de Fotos | não | **sim** |
| mexe na biblioteca | copia | **nada sai do lugar** |

**O CLI pode reportar sucesso com uma foto faltando.** Visto na prática: um
lote de 50 retornou exit 0 com 49 adicionadas, e a mesma foto entrou sozinha
na tentativa seguinte. Por isso o script reconfere o conteúdo do álbum
depois de adicionar e reenvia uma a uma as que faltarem — o código de saída
do CLI não é evidência suficiente.

O álbum é reaproveitado pelo nome, então rodar de novo é seguro. O reverso
disso: se você renomear "Pessoa 05" para outro nome no app, uma execução de
`album_person.py pessoa_05` cria um álbum novo em vez de reaproveitar o seu.
Passe `--name` com o nome novo.

## Passo 6 — pessoas que não devem virar álbum

Nem tudo que o ranking encontra merece um álbum, e por dois motivos
diferentes.

O primeiro é que **nem toda identidade é uma pessoa**. Uma identidade do
topo pode ser um grupo de gente de máscara: com metade do rosto coberto, o
que sobra do embedding é parecido demais entre pessoas diferentes e elas
colapsam num grupo só. Isso não é um limiar mal escolhido — o sinal que separaria essas
pessoas não está na foto. O grupo é legítimo como informação ("estas 55
fotos são de gente mascarada") e inútil como álbum.

O segundo é mais simples: **você pode não querer organizar alguém**, sem
nada de errado com o agrupamento.

```bash
python scripts/person_blocklist.py add pessoa_07 --nivel sem-album \
    --nome "Máscaras" --motivo "grupo de gente de máscara, não é uma pessoa"
python scripts/person_blocklist.py add pessoa_03 pessoa_09 --nivel ignorar
python scripts/person_blocklist.py list
python scripts/person_blocklist.py check
```

| nível | aparece no ranking | vira álbum |
|---|---|---|
| `sem-album` | **sim**, carimbado com o motivo | não |
| `ignorar` | não | não |

### O identificador não pode ser o nome — nem o id

"Toda aparição futura" é a parte difícil. `pessoa_NN` é a colocação e troca
de dono. O `id` do ranking é o hash dos uids das fotos, então **muda
justamente quando a pessoa ganha uma foto nova**. Nenhum dos dois sobrevive
à próxima varredura.

O que acompanha a pessoa é o rosto. O perfil guardado em `data/blocklist/`
é o banco de embeddings da identidade bloqueada mais a lista de rostos dela
(`photo_id:face_index`), e o reconhecimento futuro é feito contra isso.

### A primeira tentativa pegou 645 rostos da pessoa errada

O teste natural seria o mesmo do nível 3: para cada rosto, a média das 3
menores distâncias até o banco, bloqueia se ≤ 1.00. Medido aqui, o perfil
de uma identidade bloqueada pegou 1149 rostos — **645 deles de outra
pessoa**, justamente a mais frequente do acervo inteiro.

O erro foi reaproveitar o limiar fora do contexto dele. No nível 3 o teste
não é absoluto, é uma **disputa**: o rosto vai para a identidade mais
próxima entre todas, e as outras candidatas fazem o trabalho de segurar o
que não é delas. Sozinho, sem ninguém para perder a disputa, o mesmo número
vira uma rede que pega meio mundo.

### O critério que vale

É o do nível 2 — o que o projeto já usa para decidir "estes dois grupos são
a mesma pessoa" — aplicado à identidade inteira, nunca a rostos avulsos:

1. **Nenhum conflito de foto.** Se a identidade tem um rosto numa foto onde
   o perfil tem *outro* rosto, são duas pessoas juntas numa foto, e duas
   pessoas numa foto nunca são a mesma pessoa.
2. **RMS entre os centroides ≤ 1.10** (`--merge-threshold`).

A regra 1 tem que comparar rostos, não fotos: a própria pessoa bloqueada
compartilha *todas* as fotos com o perfil, porque são os mesmos rostos.

E é ela que decide o caso difícil. Na biblioteca de teste a distância
sozinha não separa as duas:

```
perfil bloqueado   a própria identidade   rms=0.964   BLOQUEIA
                   vizinha A              rms=0.992   5 fotos em comum -> outra pessoa
                   vizinha B              rms=1.032   2 fotos em comum -> outra pessoa
                   vizinha C              rms=1.079   3 fotos em comum -> outra pessoa
                   vizinha D              rms=1.093   2 fotos em comum -> outra pessoa
```

O perfil está mais perto de si mesmo do que da vizinha A por 0.028 — margem
nenhuma, as duas se parecem. O que resolve é a co-ocorrência, e ela continua
valendo no futuro, porque as fotos antigas continuam no acervo.

Rodado sobre o ranking inteiro, cada um dos 8 perfis bloqueia exatamente a
própria identidade e nada mais (`person_blocklist.py check` imprime essa
tabela, que é o jeito de ver se um bloqueio está prestes a levar junto
alguém que você quer).

### Onde o bloqueio acontece

No `rank_people.py`, **depois** do nível 3 e antes do corte do top N — as
identidades `ignorar` saem do ranking e outras cinco pessoas sobem no lugar.
Depois, e não antes, por dois motivos: o reconhecimento precisa da
identidade formada (é no conjunto que estão a média e a co-ocorrência), e
deixar os rostos da pessoa ignorada disputarem o nível 3 é o que impede que
eles sejam absorvidos por quem ficou. Na prática as contagens das cinco
ignoradas bateram exatamente com as do ranking anterior — nada vazou.

E no `album_person.py`, que refaz a checagem por conta própria em vez de
confiar no carimbo do `ranking.json`. Assim um `ranking.json` gerado antes
do bloqueio também não passa; o carimbo é só o plano B, para quando o acervo
de embeddings não está em disco.

O alcance do bloqueio termina aí: ele governa o que o ranqueamento lista
e o que vira álbum sozinho. O `fetch_person.py` não o consulta, de
propósito — quem passa um id está pedindo aquelas fotos explicitamente, e
recusar não protegeria nada (dá para rebaixar por id e criar o álbum à mão
no app). "Bloqueado" aqui quer dizer "não organize isso sozinho".

A blocklist mora em `data/`, que não vai para o git — ela é dado, como o
acervo de rostos.

## Limitações

- **Quem aparece pouco não entra.** O nível 2 descarta grupos de 1 rosto
  por bloco. É deliberado: a pergunta é "quem é mais frequente". Para
  organizar tudo, incluindo quem aparece uma vez, o fluxo é o outro
  (`face_cluster.py` sobre um lote em disco).
- **O ranking conta rostos, não pessoas fotografadas.** Alguém que aparece
  três vezes na mesma foto de espelho conta três — na prática não acontece,
  porque a restrição de exclusão mútua impede que os três sejam a mesma
  identidade.
- **O catálogo envelhece.** Fotos novas no Proton só aparecem depois de
  `--refresh-catalog`.
- **Renomear um álbum no app desconecta ele do script**, que procura pelo
  nome. Use `--name` com o nome novo.
- **A blocklist reconhece identidades, não rostos soltos.** Se uma pessoa
  bloqueada aparecer tão pouco numa varredura futura que não chegue a
  formar identidade, ela não é reconhecida — mas também não vira álbum,
  porque não entra no ranking.
- **Trocar o modelo de embedding invalida o acervo**, e os embeddings não
  podem ser recalculados sem rebaixar as fotos. Como o `uid` está guardado,
  isso é possível — só custa outra varredura.
