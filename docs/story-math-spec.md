# Spec : problèmes d'état à étapes (« story-math »), effort à 4 niveaux, traces distillées

Légende : **[M]** mesuré (aujourd'hui, par moi ou par le propriétaire), **[E]** estimé (calcul ou extrapolation), **[O]** opinion / hypothèse à tester.
Vocabulaire : `k` = nombre d'étapes d'état (1 à 6). `level` = `none|low|medium|high`. « think » = présence de `<|think_start|>...<|think_end|>`.

## 0. Décisions en une page

1. Nouvelle famille `story_math` : un compteur unique (un enfant, un objet) qui subit k événements `+`, `-`, `×2`, avec des distracteurs. Réponse visible : `The answer is N.` (comme les additions, même grader strict).
2. Niveau d'effort = ligne d'en-tête `<|system_start|>effort: low|high<|system_end|>` après la ligne `tools:`. **Absence d'en-tête = `medium`** (le défaut ; rien ne change pour les 80 000 transcripts de code ni pour les opencode qui n'envoient rien). `none` : pas d'en-tête, on bannit `<|think_start|>` au sampling (déjà fait). API : `reasoning_effort` -> en-tête.
3. La décision « réfléchir ou non » est **déterministe selon (niveau, bucket de difficulté)** et appris par les données. Le bucket est mesuré par une sonde qui ne sait répondre qu'en un seul forward pass. On ne montre jamais une réponse directe dans un bucket où la sonde réussit moins que le seuil du niveau.
4. Traces : `low` écrites par programme (équations sèches, aucun intérêt à payer un LLM) ; `medium` et `high` par `deepseek/deepseek-v4.1-flash` via OpenRouter (`provider.zdr=true`), filtrées par réponse + chaque nombre intermédiaire + style + longueur. Coût estimé 2 à 5 $, plafond 10 $. Plan B 100 % templates, comparé à l'expérience E3.
5. **Contexte : on reste à 1 024.** Mesuré : la conversation la plus longue de la famille (6 étapes, `high`) fait environ 300 tokens (section 6).
6. Les additions ne dépendent pas du niveau : toujours scratchpad (sauf `none`). On met `DIRECT_FRAC = 0` dans `arithmetic.py` (le « bluff » mesuré : -4 points de chat entiers pour -17 % de tokens). Elles ne régressent donc pas par construction.

---

## 1. Format des problèmes

### 1.1 Génération (module `minilab/data/story_math.py`, nouveau)

Un problème = un état initial + k événements + une question. Un seul compteur suivi (le « héros »). Tout est tiré d'un `random.Random` seedé.

| Élément | Train | Held-out |
|---|---|---|
| Héros | 10 prénoms de `arithmetic.NAMES` + 6 autres (Zoe, Leo, Ella, Jack, Nina, Owen) | 6 prénoms jamais vus (Ruby, Finn, Cleo, Ivan, Maya, Hugo) |
| Objet (singulier / pluriel) | 12 : apple, toy, sticker, ball, cookie, flower, book, shell, star, coin, candy, pencil | 6 jamais vus : marble, pear, kite, button, feather, acorn |
| Autre personne (destinataire / source) | 6 prénoms (Tom, Mia, ...) | 4 autres |
| Opérations | `add` (finds / buys / picks / gets N more ; « X gives her N ») ; `sub` (gives N to Y / eats / loses / sells / breaks N) ; `dbl` (« gets twice as many as she has » / « her pile doubles ») | + `half` (« gives half of them to Y », parité garantie) : opération jamais vue |
| Étapes k | 1 à 6, poids `[.10 .15 .20 .20 .20 .15]` | 7 et 8 (généralisation en longueur) |
| Valeurs | départ 2..20 ; montant `add` 1..20 ; `sub` 1..état courant ; `dbl` seulement si état ≤ 40 ; état toujours dans 0..99 (jamais négatif ; zéro autorisé mais jamais en départ) | idem |
| Distracteurs | une phrase sans effet sur le héros dans 35 % des problèmes (« Tom has 4 pears. », « The sun is warm. », « Tom gives Mia 2 pears. ») ; au plus une | 100 % |
| Question | 5 formulations (« How many apples does Lily have now? », « ... left? », « ... in all? », « How many apples will Lily have then? », « Lily wants to count her apples. How many are there? ») | + 20 formulations paraphrasées par le teacher (section 7) |
| Pluriels | table explicite (singulier, pluriel) ; « 1 apple » gère le singulier | idem |

Les phrases d'événement sont tirées d'une liste par opération (6 à 8 variantes par opération, sujet « She/He » ou le prénom). Les nombres restent en chiffres (le tokenizer les coupe en chiffres isolés).

Pas d'état négatif, pas de division impraticable, pas de nombre > 99 : un état à 3 chiffres ferait apparaître le problème d'addition à retenue, déjà traité à part.

### 1.2 Dict de problème (interface)

```python
make(rng, k, split="train"|"heldout", distractor=None) -> {
  "kind": "story_math", "messages": [{"role":"user","content": text}], "tools": None,
  "steps": k, "digits": k,          # `digits` : pour réutiliser _by_digits sans le modifier
  "answer": int, "states": [s0, s1, ..., sk], "ops": [("add",5), ("sub",2), ("dbl",None)...],
  "hero": "Lily", "thing": ("apple","apples"), "bucket": (k, n_dbl_capped_1)}
answer_text(n) -> "The answer is N."
```

### 1.3 Exemples rendus par le vrai tokenizer (`models/prelude-2/tokenizer.json`) [M]

Problème (47 tokens, k = 3, un distracteur) :
`Lily has 3 apples. She finds 5 more. The sun is warm and Tom has 4 pears. She gives 2 to Tom. Then she eats 1. How many apples does Lily have now?`

| niveau | raisonnement | tokens du raisonnement | conversation complète (bos + header + tout) |
|---|---|---:|---:|
| low | `3+5=8` ⏎ `8-2=6` ⏎ `6-1=5` | 17 | 77 |
| medium | `Lily starts with 3 apples. She finds 5, so 3 + 5 = 8. She gives 2 to Tom, so 8 - 2 = 6. She eats 1, so 6 - 1 = 5.` | 57 | 117 |
| high | `Lily begins with 3 apples. She finds 5 more, so she has 3 + 5 = 8 apples. Tom's pears do not matter. She gives 2 apples to Tom, so 8 - 2 = 6. Then she eats 1, so 6 - 1 = 5 apples. Check: 3 + 5 - 2 - 1 = 5. So Lily has 5 apples.` | 99 | 159 |

Réponse visible dans les trois cas : `The answer is 5.` (6 tokens). `effort: high` = 5 tokens, plus les deux tokens spéciaux.

Longueurs sur mon prototype de générateur (300 problèmes par k, sans distracteurs, raisonnement medium) [M, à refaire avec le vrai générateur] :

| k | problème | raisonnement low | raisonnement medium | conversation (medium) | max observé |
|---|---:|---:|---:|---:|---:|
| 1 | 23 | 9 | 26 | 63 | 66 |
| 2 | 31 | 17 | 44 | 89 | 95 |
| 3 | 39 | 25 | 63 | 116 | 124 |
| 4 | 47 | 32 | 81 | 142 | 152 |
| 5 | 55 | 40 | 100 | 169 | 180 |
| 6 | 63 | 49 | 119 | 196 | 208 |

Règle empirique [M] : le problème coûte environ 8 tokens par étape, `low` 8, `medium` 19, `high` ~33 (extrapolé de l'exemple k=3 : 99 tokens ; [E]).

### 1.4 Grader

`eval/tasks.py:grade()` reste inchangé pour les tokens spéciaux (`[]` ou `[think_start, think_end]`) ; `check_answer` ajoute `kind == "story_math"`: `content == answer_text(problem["answer"])`. Pas de `Lily has 5 apples` : un seul format de réponse, comme les additions.

---

## 2. Encodage de l'effort

- **Rendu** : `chat._render_header(tok, tools, effort=None)`. Ordre : `<|bos|>`, `tools:` (si présent), `effort: low|high` (si `effort in ("low","high")`), puis les messages système du client. `medium`, `none` et `None` : aucune ligne. Un système client « effort: high » est un message système ordinaire : aucun contrôle de plus.
- **API** (`api/schemas.py:to_inference`) : normalisation
  `None -> medium`, `none -> none`, `minimal|low -> low`, `medium -> medium`, `high|xhigh|max -> high`, toute autre valeur -> `medium`. Le dict renvoyé gagne `"effort": "low"|"medium"|"high"|"none"`, et garde `"think": effort != "none"`. `server.py` / `_render` passent `effort` à `render_prompt` (l'en-tête est omis pour `medium` et `none`) ; le ban de `<|think_start|>` pour `none` existe déjà (`engine.py:_sample`).
- **Pourquoi pas d'en-tête pour `none`** : le ban au sampling est une garantie dure, l'en-tête n'en ajouterait aucune ; l'entraîner demanderait des réponses directes sur des problèmes difficiles, c'est-à-dire du bluff. [O] Le coût : sur un problème difficile, `none` renvoie la meilleure supposition en un forward pass ; c'est sa définition, et la colonne `accuracy_off` la mesure.
- **Entraînement** : `render_conversation(tok, messages, tools, effort=None)` ; le dict de conversation gagne la clé `"effort"`. L'en-tête apparaît aussi (15 % des conversations, `low` ou `high` au hasard) sur toutes les autres sortes (additions, histoires, salutations, refus, code SFT non), avec des cibles inchangées : le modèle apprend que l'en-tête ne touche que `story_math`. Pas d'en-tête sur les transcripts de code (opencode n'en envoie jamais).
- **Historique client** : les raisonnements ne sont pas renvoyés (déjà le cas), l'en-tête est recalculé à chaque requête à partir de `reasoning_effort`.

---

## 3. Politique cible

### 3.1 Règle

Pour un niveau `L` et un problème de bucket `b = (k, dbl)` (`dbl` = au moins un doublement, 12 buckets) :

```
think(L, b) = (p_direct(b) < tau[L])
```

où `p_direct(b)` est la justesse d'un modèle de **sonde** forcé de répondre en un forward pass (section 3.2). Seuils de départ [O] : `tau = {low: 0.80, medium: 0.92, high: 0.99}`. Un bucket à ±0.03 d'un seuil est classé `think` (prudence). Déterminisme : à (niveau, bucket) donné, la cible est toujours la même, donc aucune étiquette bruitée.

Le niveau règle le style quand on réfléchit : `low` équations sèches, `medium` phrases courtes, `high` phrases complètes + vérification finale.

### 3.2 Table attendue (hypothèse [O], à remplacer par la table mesurée à E1)

Hypothèse de départ : un modèle de 5,8M sait faire 1 addition à 2 chiffres en un forward pass [M, « fiable jusqu'à 2 chiffres »]. Donc 1 étape direct, 2 étapes limite, 3 et plus : pas fiable.

| k | low (τ .80) | medium (τ .92) | high (τ .99) |
|---|---|---|---|
| 1 | direct | direct | direct si p ≥ .99, sinon high-think |
| 2 | direct si p ≥ .80, sinon equations | phrases courtes | phrases + check |
| 3 | équations | phrases courtes | phrases + check |
| 4-6 | équations | phrases courtes | phrases + check |

Chaque ligne `direct` n'existe que si la mesure la justifie. Le niveau `none` : aucune cible d'entraînement propre.

### 3.3 Mesure empirique des seuils (procédure `scripts/probe_direct.py`)

1. **Sonde** : à partir de `runs/prelude/pretrain` (jamais refait), un midtrain de 600 steps (~1 min [E]) où `story_math` (k = 1 à 6, seed A) n'a **que** des réponses directes (aucun think pour `story_math`), plus le mélange existant. C'est le meilleur élève « un seul forward pass » possible avec le même budget : il évite de sous-estimer la capacité.
2. **Mesure** : sur 300 problèmes par k, dont les problèmes de **seed B** (≠ seed A, donc la sonde ne les a jamais vus), greedy, `think=False` : `p_direct(k, dbl)` avec intervalle ±3 points à 300 (n=300 : σ ≈ 2,9 points à p=0,5).
3. **Sortie** : `data/reason/policy.json` = `{"tau": {...}, "p": {"1,0": 0.98, ...}, "n": 300}`. `think(level, problem)` lit ce fichier.
4. **Boucle** : après le premier entraînement complet (E4), on remesure `p_direct` avec le vrai modèle (`accuracy_off` de l'éval, pas de sonde) et on re-étiquette si l'écart à la sonde dépasse 5 points sur un bucket. Une seule itération prévue (E5).

### 3.4 Éviter le bluff (les garde-fous)

- **Jamais** de réponse directe dans un bucket où `p_direct < tau[L]` (règle 3.1). Aucun plancher global bas : `tau_low = 0.80` est le plus bas.
- Zone grise ±0.03 classée `think`.
- Les additions n'ont plus aucune cible directe (`DIRECT_FRAC = 0`).
- Une métrique **`direct_acc[L]`** : justesse des réponses sans think. Critère : `direct_acc[L] ≥ tau[L] - 0.03`. Si elle s'effondre, c'est du bluff, on monte `tau`.
- Une métrique **regret** : `acc_adaptive[L] - max(acc_forced[L], acc_off)` doit rester ≥ -1 point.
- `none` n'est jamais entraîné (aucune étiquette bluff possible).

---

## 4. Traces écrites par le gros LLM

### 4.1 Modèle OpenRouter (éligibilité ZDR vérifiée aujourd'hui [M])

Source : `GET https://openrouter.ai/api/v1/endpoints/zdr` (liste officielle, mise à jour par OpenRouter ; 1 088 endpoints listés au moment de la mesure). La requête porte `provider.zdr=true` ; le garde-fou du compte filtre déjà les endpoints non ZDR (mémoire « Reasoning prefill POC »).

| Candidat | Endpoints ZDR (prix $/M tokens in / out) |
|---|---|
| **1. `deepseek/deepseek-v4.1-flash`** (recommandé) | River 0,12 / 0,48 ; Ionstream 0,078 / 0,23 ; PrimaLabs 0,24 / 0,96 ; Mistral 0,30 / 1,20 ; Parasail 0,45 / 1,80 ; Venice 3 / 12 |
| **2. `z-ai/glm-5.3-flash`** | DekaLLM 0,10 / 0,50 ; Mistral 0,15 / 0,50 |
| (écarté : trop cher pour l'usage) | `deepseek-v4-pro` 1,98 / 5,94 ; `kimi-k3.1-pro` 1,0 à 2,0 / 9 à 10 |

Les deux sont des modèles « flash » très au-dessus du niveau requis (arithmétique de 1 à 2 chiffres) : la seule difficulté est le style. Le script commence par vérifier que le modèle a au moins un endpoint dans la liste ZDR, et passe `provider: {"zdr": true, "order": [...], "allow_fallbacks": true, "data_collection": "deny"}`. Raisonnement du teacher **désactivé** (`reasoning: {"enabled": false}`) : le CoT propre du teacher coûterait 5 à 10 fois plus et on veut un texte de pensée court. Température 0,8, `max_tokens` 400.

### 4.2 Prompts (exacts)

Un message `system` (commun) + un bloc de style + 2 exemples fixes (écrits à la main, jamais tirés du jeu d'éval) + le problème. Le texte renvoyé est la pensée seule ; la réponse `The answer is N.` est ajoutée par le programme.

**SYSTEM (commun)**
```
You write the private scratch thoughts of a small, careful assistant that solves a short story problem about counting things. You get the problem. Write ONLY the thoughts: no greeting, no final answer line, no markdown, no bullet points, no headings. Use plain ASCII only. Write every number with digits (8, never eight). Write operations with + - x and =, with the symbol x for "times". Every number you write must be correct. Never write "The answer is". Follow the style below exactly.
```

**STYLE low (optionnel ; défaut : produit par le programme)**
```
Style: bare equations. One equation per line, in the order the events happen, one line per event that changes the count. A line looks like  8-2=6  : number, operator, number, equals, result, with no spaces and no words at all. The first line uses the starting number. Skip events that change nothing (other people's things, the weather). Doubling is written  8x2=16 .
```
Exemple : problème ci-dessus -> `3+5=8\n8-2=6\n6-1=5`.

**STYLE medium**
```
Style: short and plain, like a quick note to oneself. First sentence: who starts with how many of what. Then ONE short sentence per event that changes the count (at most 12 words), ending with its equation, for example "She finds 5, so 3 + 5 = 8." Skip events that change nothing. No introduction and no conclusion: the last sentence is the last equation. Vary your wording from one problem to the next (finds, gets, picks up, receives...). Use the character's name or She/He.
```
Exemple : `Lily starts with 3 apples. She finds 5, so 3 + 5 = 8. She gives 2 to Tom, so 8 - 2 = 6. She eats 1, so 6 - 1 = 5.`

**STYLE high**
```
Style: a careful person thinking out loud, in complete sentences. Begin by saying who has how many of what. Then go through the events in order, one or two sentences each: say what happens, then the calculation with the new total ("so she has 3 + 5 = 8 apples"). If a sentence of the problem does not change the count, say in a few words that it does not matter. Near the end, add one sentence starting with "Check:" that goes through the whole story again in a single line of arithmetic. Finish with a sentence stating how many she has, for example "So Lily has 5 apples." Between 30 and 120 words. Vary your wording from one problem to the next.
```
Exemple : le texte `high` de la section 1.3.

**USER**
```
Problem: {text}
Thoughts:
```
(Les 2 exemples sont donnés en tours user/assistant avant ce message, pour un modèle de chat.)

### 4.3 Sortie, filtre

Le script écrit une ligne JSONL par essai :
`{"id": "<seed>-<k>-<idx>", "level": "medium", "model": "...", "raw": "...", "text": "<normalisé>", "ok": true, "defects": [], "cost": 0.0001}`.

Normalisation : `×`, `−`, `–` -> `x`, `-` ; espaces multiples réduits ; trim ; interdits : non-ASCII, `*`, `#`, `` ` ``, listes à puces, mots-nombres (`one`..`twenty`, `thirty`...), la chaîne `The answer is`.

Un trace est **gardée** si et seulement si :

1. **Intermédiaires justes** : l'ensemble des nombres de la trace ⊆ {nombres de l'énoncé} ∪ {états vrais `states`} ∪ {1, 2} ; et la suite des états vrais `s1..sk` y apparaît **dans l'ordre** comme sous-suite, le dernier état étant `answer`. Chaque équation `a op b = c` écrite est évaluée : toute équation fausse rejette la trace.
2. **Réponse juste** : le dernier état de la trace = `answer`.
3. **Style respecté** : `style_ok(level, text)` (module `story_math`, partagé avec l'éval) :
   - low : regex `^(\d+[+\-x]\d+=\d+)(\n\d+[+\-x]\d+=\d+)*$`, une ligne par événement utile (≤ k + 0) ;
   - medium : sans saut de ligne ; une première phrase de départ ; ≤ 14 mots par phrase ; une équation par phrase ; pas de phrase « Check » ; pas de conclusion ;
   - high : ≥ 1 phrase « Check: » avec un calcul qui retombe sur `answer` ; dernière phrase de la forme `So <hero> has <answer> <things>.` ; si le problème a un distracteur, une phrase qui dit qu'il ne compte pas (heuristique : contient « not matter », « does not change », « has nothing to do »).
4. **Longueur** (tokens, tokenizer du modèle) : low ≤ 8k + 6 ; medium ≤ 24k + 12 ; high entre 20k + 20 et 48k + 40 (révisé par le coordinateur : 30k + 20 forçait du remplissage). Un high plus court que le medium du même problème est rejeté.
5. **Variété** : « squelette » = trace avec chaque nombre remplacé par `#` ; un même squelette ne peut dépasser 1 % du jeu d'un niveau ; sinon on jette les surplus.

Les échecs sont rejoués une fois (même prompt, température 1,0) ; au-delà, le problème perd cette trace. La décision `think/direct` vient de la politique, pas de la trace.

### 4.4 Combien, et combien ça coûte [E]

- 9 000 problèmes d'entraînement (poids k plus haut). Politique : seuls les (problème, niveau) en `think` ont besoin d'une trace. Pour medium et high, environ 8 000 problèmes de k ≥ 2 chacun -> 16 000 traces demandées ; avec un taux d'acceptation de 70 % et un second essai, environ 23 000 appels.
- Un appel : environ 400 tokens en entrée (system + style + 2 exemples + problème), 120 tokens en sortie (medium) à 180 (high).
- Prix River (0,12 / 0,48 $/M) : 400×0,12 µ$ + 150×0,48 µ$ = ~0,00012 $ -> **~2,8 $** pour 23 000 appels. Prix Mistral (0,30 / 1,20) : ~0,0003 $ -> ~7 $. Le prompt système commun peut être mis en cache (usage implicite).
- Plafond du script : `--budget-usd 8` (arrêt si `usage.cost` cumulé le dépasse). Pilote E2 : 300 problèmes × 2 styles ≈ 0,1 $.
- `low` : 0 $ (programme).
- Si E4 montre du sous-apprentissage : 20 000 problèmes, soit environ 2,2× le coût (6 à 15 $) : il faudrait alors une autorisation du propriétaire.

### 4.5 Plan B sans LLM (comparaison E3)

`story_math.trace_program(problem, level)` : templates variés, tirés comme les événements :
- medium : 6 à 8 formulations par opération (« She finds 5, so 3 + 5 = 8. » / « 5 more, 3 + 5 = 8. » / « 3 and 5 more makes 8. » ...), amorce variée (starts with / had / at first), 20 % des phrases sans le « so ».
- high : phrases pleines (« After finding 5, she has 3 + 5 = 8 apples. »), phrase distracteur (« Tom's pears do not matter. »), `Check:` (somme des deltas), conclusion.
- low : le même que le défaut.
Gains : zéro coût, exact par construction, mais la variété lexicale est celle d'un humain qui écrit 30 phrases. Risque : sur-apprentissage de la forme ; mesuré par E3 sur les problèmes reformulés.

---

## 5. Mélange des données

Les étapes `pretrain` (11,4 min) ne sont **pas** refaites. Le nouveau savoir arrive au midtrain (format + volume) et à la SFT (comportement, niveaux).

**Midtraining** (4,9M tokens, 4 800 lignes de 1 025 tokens, soit environ 27 000 conversations [E] à 135 tokens en moyenne) :

| sorte | avant | après |
|---|---|---|
| arithmetic | 0.60 | 0.35 |
| story_math (nouveau) | - | 0.30 (environ 8 000 conversations [E]) |
| story | 0.20 | 0.20 |
| greeting | 0.20 | 0.15 |

`text_frac = 0.25` et `tool_frac = 0.3` inchangés. Niveau de chaque conversation `story_math` : medium 40 % (pas d'en-tête), low 30 %, high 30 %. Pour les autres sortes, en-tête `low`/`high` dans 15 % des cas (cible inchangée).

**SFT** (mix de chats, 16 000 conversations × 1,8 epoch ; 25 % agent code, 50 % ; 25 % pretraining : inchangés) :

| sorte | avant | après (×0,8) |
|---|---|---|
| plain | 0.13 | 0.104 |
| instruction | 0.33 | 0.264 |
| followup | 0.12 | 0.096 |
| new_question | 0.12 | 0.096 |
| switch | 0.08 | 0.064 |
| mixed | 0.10 | 0.08 |
| refusal | 0.08 | 0.064 |
| identity | 0.04 | 0.032 |
| **reason** (nouveau) | - | **0.20** |

`reason` = 3 200 conversations (environ 5 800 lignes vues à 1,8 epoch ; 5,2 % des lignes de SFT en tout [E]). Quatre-vingt-dix pour cent sont des problèmes simples, 10 % ont un historique (la sorte `story_math` est ajoutée à `single_turn`, donc elle peut apparaître dans `new_question`, `switch` et `mixed`, avec `client_history` qui retire les raisonnements). Même répartition de niveaux qu'au midtrain. Si E4 sous-apprend : `size` 16 000 -> 24 000 sans coût de temps (les steps sont fixes).

**Distillation** (réintégrer la compétence sans la perdre) : nouvelle sorte `story_math` à 0,10 du mix (les autres ×0,9), teacher = modèle SFT, effort tiré au hasard, ajouté dans `eval.tasks.make_problem`. **RL math** : inchangé (spécialiste d'addition ; ses autres compétences dérivent et c'est attendu, il n'est jamais publié).

Test de non-oubli (critère dans E4/E6) : toutes les colonnes actuelles, avec chat ≥ 92 %.

---

## 6. Contexte : 2 048 tokens ? Non.

Mesuré [M] (section 1.3) : la conversation `story_math` la plus longue du plan (k = 6, distracteur, `high`) tient en environ 3 × 99 + 23 = ~300 tokens [E, extrapolé : 8 par étape pour le problème, 33 par étape pour `high`] contre 208 max observé en `medium`. Le chat complet de 5 requêtes (README : histoires de 700 caractères max, soit environ 180 tokens) reste sous 600 tokens parce que l'historique client ne renvoie jamais le raisonnement : 4 anciens tours (50 à 180 tokens chacun) + le tour courant (≤ 330). Même la sérialisation SFT (une conversation par ligne, remplie au multiple de 64) tient dans 1 024. `prompt_budget` (`context_length − min(max_tokens, context/2)`) gère déjà le reste.

Donc : pas de changement de `block_size`. Raisons de revenir sur ce choix : (a) E4 montre des troncatures (`finished = false` sur k = 6 `high`) ; (b) on veut des problèmes à > 8 étapes ; (c) opencode a besoin de plus. Aucune n'est vraie aujourd'hui.

**Si on devait le faire** (plan prêt, non exécuté) :
- `GPT` : `rope_cos/rope_sin` sont des buffers non persistants (`_rope_tables(block_size, ...)`), donc aucun poids ne change. Changement minimal : `block_size = 2048` dans `[model]` n'affecte que les nouveaux modèles ; pour étendre, `midtrain.py` charge le checkpoint, fait `model.config.block_size = 2048`, recalcule les deux tables (une fonction `extend_context(model, n)` de 6 lignes dans `model/gpt.py`) puis entraîne avec `T = 2048` ; `config.json` sauvegarde la nouvelle valeur ; `release.json` publie `context_length = 2048`. Base RoPE inchangée (10 000) pour un premier essai ; en cas de dégradation, base 20 000 ou interpolation linéaire des positions (÷2).
- Coût [E] : cache KV 12,6 -> 25,2 Mo par séquence (6 couches × 2 × 4 têtes × T × 64 × 4 octets) ; midtraining : 8 lignes × 2 048 tokens, soit 2× les activations et un temps par step de 2,0 à 2,3× (attention en T²) : 0,8 -> ~1,8 min, RAM d'entraînement ~2×. La SFT ne change pas si les lignes restent courtes (rembourrage au plus long de la fenêtre).
- Risque [O] : 600 steps de midtraining peuvent ne pas suffire pour des positions jamais vues au pretraining (1 024). À tester seulement si le besoin apparaît (E7).

---

## 7. Eval

### 7.1 Ensemble et modes

Seed fixe (`EVAL_SEED`), 100 problèmes par k pour k = 1 à 6 (600 problèmes, 2 points de seuil) :

| mode | rendu |
|---|---|
| `adaptive[L]`, `L ∈ {low, medium, high}` | en-tête de L, le modèle décide |
| `forced[L]` | en-tête de L + prompt terminé par `<|think_start|>` |
| `off` | sans en-tête (`medium`), `<|think_start|>` jamais échantillonné = niveau `none` |

Soit 7 jeux × 600 générations = 4 200 générations de ≤ 300 tokens [E : 4 à 8 min en génération CPU ; version dev à 50 par k].

Mesures par (mode, k), enregistrées dans `eval.json["story_math"]` :
- `accuracy[L][k]` (grader strict, section 1.4) ;
- `think_rate[L][k]` : part des réponses avec `reasoning_length > 0` ;
- `tokens[L][k]` : moyenne de `reasoning_length` (marqueurs inclus) ;
- `accuracy_forced[L][k]`, `accuracy_off[k]` ;
- `style_ok[L]` : part des raisonnements qui passent `style_ok(L, ...)` ;
- `direct_acc[L]` : justesse des réponses sans think (précision directe) ;
- `regret[L]` : `accuracy_adaptive[L] − max(forced, off)`, par k.
Scalaires : `sm_low`, `sm_medium`, `sm_high` (moyenne des k = 1..6).

### 7.2 Held-out

| ensemble | contenu | n |
|---|---|---|
| `steps78` | k = 7 et 8 (généralisation en longueur) | 100 chacun |
| `objects` | prénoms et objets jamais vus (§1.1), k = 1..6 | 200 |
| `ops` | opération `half` jamais vue, k = 2..5 | 100 |
| `paraphrase` | énoncé réécrit par le teacher en gardant les nombres (réponse recalculée par le programme, vérifiée) | 200 (≈ 0,05 $) |
| `distract` | 100 % de distracteurs, k = 3..6 | 200 |

Mêmes mesures (adaptatif medium, high, forced, off) ; pas dans le gate (information et E3/E4).

### 7.3 Colonnes du tableau (`eval/run.py:_table_cells`) et du gate (`eval/gate.py:metrics`)

Tableau, après `think saved` :

| colonne | contenu |
|---|---|
| `sm lo` / `sm med` / `sm hi` | `sm_low`, `sm_medium`, `sm_high` |
| `sm 6st` | `accuracy[medium][6]` |
| `think L/M/H` | `think_rate` moyenne par niveau, ex. `12/38/71%` |
| `tok L/M/H` | tokens de réflexion moyens, ex. `6/41/96` |
| `sm 7-8*` | held-out `steps78`, medium |
| `sm obj*` | held-out `objects`, medium |

Gate (nouveaux metrics, `n` pour le seuil de bruit) : `sm lo`, `sm med`, `sm hi` (n = 600 chacun, plancher de 2 points) ; `sm 6st` (n = 100). Comme `compare` ne considère que les métriques que les deux évals ont, **le gate contre `prelude-2` ignore ces colonnes** ; elles joueront à partir de la release suivante. Les critères d'acceptation de la nouvelle compétence sont donc chiffrés en E4 et E6 (§8), pas dans le gate. Toutes les colonnes actuelles (dont `chat`, `3d`..`5d`, `5d@T=1`, `think saved`, `agent`) restent gatées comme aujourd'hui.

Interface : `tasks.eval_story_math(model, tok, problems, max_new, level, think=None) -> list[dict]` (comme `eval_arithmetic_chat` + paramètre `level`) ; `run.evaluate` écrit `result["story_math"]` et les scalaires.

---

## 8. Plan d'expériences (une seule à la fois sur le Mac)

Repères de durée [M] README : midtrain 0,8 min, SFT 8,4 min, RL 18,6 min, distill 10,6 min. Éval des 600 chats + code : ~40 min pour 5 évals (dont Docker) ; l'éval story-math seule : 4 à 8 min [E].

| # | teste | durée [E] | critère de succès (chiffré) |
|---|---|---|---|
| E0 | générateur + tests unitaires + longueurs | 0 (CPU, quelques secondes) | tests verts ; k = 1..8 génèrent des problèmes valides sur 10 000 tirages (aucun état < 0 ou > 99) ; longueurs refaites : conversation `high` k=6 ≤ 350 tokens |
| E1 | **sonde** : capacité en un seul forward pass | midtrain 1 min + éval 3 min = ~5 min | courbe `p_direct(k,dbl)` mesurée. Task bien posée si `p(1) ≥ 0,9` et `p(6) ≤ 0,5`. Si `p(6) > 0,8` : montées de plages (départ 2..60, montants 1..30), rejouer E1. Si `p(1) < 0,7` : plages plus petites |
| E2 | **teacher pilote** : 300 problèmes × medium, high | 10 min, ~0,1 $ | acceptation ≥ 70 % (medium) et ≥ 60 % (high) ; `style_ok` ≥ 95 % des acceptées ; ≥ 300 squelettes distincts pour 1 000 traces medium ; médiane de tokens `high / medium` ≥ 1,5 ; lecture manuelle de 30 traces |
| E3 | LLM vs templates (plan B), même quantité (6 000 problèmes, même politique) | 2 × (midtrain 1 + SFT 9 + éval 6) ≈ 32 min | on retient le LLM si `paraphrase` medium (et high) + `objects` ≥ +5 points au-dessus de plan B, sinon plan B (plus simple, gratuit) ; dans les deux cas, in-dist k ≤ 4 ≥ 95 % |
| E4 | **modèle complet** : politique mesurée par E1, midtrain + SFT avec le mélange §5 | ~10 min d'entraînement + éval complète ~15 min (avec code, Docker) | `sm_medium ≥ 95 %` ; `accuracy[medium][6] ≥ 90 %` ; `sm_high ≥ sm_medium − 1` ; `sm_low ≥ sm_medium − 8` ; `think_rate` strictement croissant `low < medium < high` avec ≥ 10 points d'écart à k = 2 et 3 ; `tok` L < M < H avec `H/M ≥ 1,3` ; `style_ok ≥ 90 %` ; `direct_acc[L] ≥ tau[L] − 0,03` ; `accuracy_off[1] ≥ 0,95` ; non-régression : `1d..5d` ≥ 99 %, `chat` ≥ 92 %, `story` bpc ≤ +2 % |
| E5 | re-étiquetage avec le vrai `accuracy_off` + balayage de `tau` (3 jeux de seuils) | 3 × 10 min + éval = ~45 min | meilleur compromis : `sm_low` accuracy à ≥ 80 % avec `tok L` ≤ 50 % de `tok M` ; `sm_high ≥ sm_medium` |
| E6 | pipeline complet (rl_math inchangé, distill avec la sorte `story_math`) + gate contre `prelude-2` | ~45 min + éval ~40 min | le gate passe (aucun recul > bruit) ; `sm_medium` après distill ≥ `sm_medium` SFT − 2 points ; `chat` ≥ 92 % |
| E7 (optionnel) | contexte 2 048 | ~5 min de midtrain + éval | seulement si E4 montre des troncatures ; critère : aucune régression à 1 024 et k = 8 `high` finit |
| E8 (optionnel) | vérification utile : 10 % de traces `high` avec un lapsus corrigé (« wait, ... ») | ~35 min | `high` > `medium` d'au moins 2 points sur k = 5, 6 |

Ordre : E0 -> E1 -> E2 -> E3 -> E4 -> E5 -> E6 ; E7 et E8 selon résultats.

---

## 9. Tâches d'implémentation (agents en parallèle)

Chaque tâche ne touche que ses fichiers. Dépendances : T1, T2, T3 (config) sont indépendantes ; le reste dépend de T1/T2.

| id | fichiers (propriétaire unique) | interface | tests attendus | dépend de |
|---|---|---|---|---|
| **T0** | `configs/prelude.toml` seul (toutes les nouvelles clés, ajoutées d'un coup : `[data.reason]` (seeds, plages, poids k, `efforts`), `[midtrain].mix`, `[sft].mix`, `[eval.story_math]` (`n_per_step`, `steps`), `[distill].mix`) | clés lues par T4/T5/T7 | `tests/test_recipe.py` (config valide) | - |
| **T1** | `minilab/data/story_math.py` (nouveau), `minilab/data/arithmetic.py` (`DIRECT_FRAC = 0`) | `make`, `answer_text`, `trace_program(problem, level)`, `verify_trace(problem, text, level) -> str | None`, `style_ok(level, text)`, `think(level, problem, policy)`, `bucket(problem)`, `load_policy(path)` | `tests/test_story_math.py` : 10 000 tirages valides (états 0..99, pas de négatif) ; `answer` = dernier état ; `trace_program` passe `verify_trace` et `style_ok` aux 3 niveaux ; un `verify_trace` rejette une équation fausse, un nombre inventé, un mot-nombre ; held-out disjoint du train (noms, objets) ; seed reproductible | - |
| **T2** | `minilab/tokenizer/chat.py` (`effort` dans `_render_header`, `render_prompt`, `render_conversation`), `minilab/api/schemas.py` (`effort`), `minilab/inference/server.py` (`GenerateRequest.effort`, `_render`), `minilab/inference/engine.py` seulement si besoin, `docs/api.md` | `render_prompt(tok, messages, tools=None, budget=None, effort=None)` ; `normalize_effort(value) -> str` | `tests/test_chat_tools.py` : header absent pour None/medium/none, présent pour low/high ; rendu inchangé par défaut (octets identiques à aujourd'hui) ; `tests/test_api.py` : `minimal->low`, `xhigh->high`, inconnu->medium, `none` -> `think=False` | - |
| **T3** | `scripts/teacher_traces.py` (nouveau) | CLI : `--n`, `--levels medium high`, `--model`, `--budget-usd`, `--out data/reason/traces.jsonl`, `--resume` ; clé via `op run` (`OPENROUTER_API_KEY`) ; vérifie l'éligibilité ZDR (liste `endpoints/zdr`) ; 16 appels en parallèle ; JSONL du §4.3 ; arrêt au plafond | `tests/test_teacher_traces.py` avec un faux client HTTP : reprise après coupure, plafond de coût respecté, un trace fausse est marquée `ok: false` (utilise `verify_trace` de T1) ; **aucun** appel réseau en test | T1 (interface figée) |
| **T4** | `minilab/data/conversations.py` (sorte `story_math` dans `single_turn`, `sft_conversation`, `midtrain_conversation`, en-tête d'effort aléatoire), `minilab/data/loader.py` (lit `effort` du dict), `minilab/train/midtrain.py`, `minilab/train/sft.py` (charge `data/reason/traces.jsonl` + politique, plan B si absent) | une conversation : `{"messages": [...,{"role":"assistant","content":"The answer is 5.","reasoning": "..."}], "tools": None, "effort": "low"}` ; `reasoning` absent si la politique dit `direct` | `tests/test_data.py` : la politique est respectée (aucune réponse directe dans un bucket à `p < tau`) ; mélange `story_math` dans `midtrain_stream` ≈ 0,30 ; en-tête dans ≈ 15 % des autres sortes ; les transcripts de code n'ont jamais d'en-tête ; déterminisme par seed | T0, T1, T2 (T3 pour les vraies traces, plan B sinon) |
| **T5** | `minilab/eval/tasks.py` (`make_problem("story_math")`, `check_answer`, `eval_story_math`), `minilab/eval/run.py` (`evaluate`, `_table_cells`, `_by_digits(..., group="digits")`), `minilab/eval/gate.py` (4 nouveaux metrics), `minilab/eval/model_card.py` (colonnes), `minilab/report.py` si la table y est dupliquée | `eval_story_math(model, tok, problems, max_new, level, think=None)` ; clés `eval.json["story_math"]` du §7.1 | `tests/test_eval.py` : `grade` accepte `The answer is 5.` avec/sans pensée, refuse l'invention d'un `think` sans fin ; la table affiche les colonnes ; `compare` ignore les colonnes absentes d'un ancien eval.json | T0, T1, T2 |
| **T6** | `scripts/probe_direct.py` (nouveau) | écrit `data/reason/policy.json` (§3.3) à partir d'un checkpoint de sonde et du seed B | test : sur un faux modèle qui répond toujours juste, `p = 1` partout et `think = False` pour tous les niveaux | T1, T2, T5 |
| **T7** | `minilab/train/distill.py` (sorte `story_math`, teacher SFT), `[distill].mix` via T0 | `story_math` dans `sample_problem` | `tests/test_train.py` : une étape de distill sur un petit modèle accepte la sorte | T0, T1, T2, T5 |
| **T8** | `docs/training.md`, `README.md` (tableau des résultats, colonnes, « Lessons »), `docs/api.md` (T2 déjà fait) | - | aucun | tout, à la fin |
| T9 (optionnel) | `minilab/model/gpt.py` (`extend_context`), `minilab/train/midtrain.py` (option) | §6 | modèle étendu : mêmes logits qu'avant pour T ≤ 1 024 | seulement E7 |

Remarques : `eval/tasks.py` est touché seulement par T5 ; `minilab/train/midtrain.py` par T4 (et T9) ; `configs/prelude.toml` par T0 seul. Ordre de fusion : T0, T1, T2 -> T3, T5 -> T4 -> T6, T7 -> T8.

---

## 10. Risques et contre-arguments

1. **Un modèle de 5,8M apprend-il des phrases variées ?** [O] Il apprend le squelette (sujet, verbe, nombre, « so a op b = c ») ; la variété lexicale sera imitée à la surface, rarement comprise. Le critère n'est pas la variété mais la justesse sur `paraphrase` et `objects` (E3). Risque de vocabulaire : 4 096 tokens, dont tous les objets du train mais peut-être pas ceux du held-out (sous-mots).
2. **`high` risque d'être moins juste que `low`.** [O] Vrai possible : plus de tokens = plus d'occasions de se tromper (un lapsus par 100 tokens à T=1 donne 10 points perdus). Mitigations : justesse mesurée par niveau (critères E4 : `sm_high ≥ sm_medium − 1`) ; la ligne `Check:` est cohérente par construction, elle n'ajoute **aucune vraie vérification** (le modèle n'a jamais vu d'erreur à corriger) : elle sert au style. E8 l'étudie. Si `high < low`, on retire la ligne `Check:` d'abord.
3. **`low` n'a pas besoin du LLM.** Vrai : équations sèches, c'est du programme. Seul l'intérêt « raisonnement humain » vient de `medium`/`high`. Si le propriétaire tient à un `low` écrit par LLM, le prompt est au §4.2 ; coût ≈ +1 $.
4. **La décision think/direct est une fonction du bucket.** Elle est apprenable parce que les étiquettes sont déterministes, mais elle suppose que `p_direct` mesuré par la sonde vaut pour le vrai modèle (qui s'entraîne en parallèle sur les additions, etc.). E5 la vérifie ; l'écart attendu est de quelques points [O]. Un modèle peut aussi décider d'après la longueur de l'énoncé et pas d'après sa difficulté réelle : en pratique le nombre d'étapes se lit dans la longueur, donc c'est équivalent ici. Hors distribution (énoncé réécrit plus long), il pense davantage : coût en tokens, pas en justesse.
5. **Les additions.** On supprime le bluff (`DIRECT_FRAC = 0`), donc plus de tokens qu'aujourd'hui pour les additions courtes : +17 % de tokens [M] contre +4 points de chat entiers [M] : on prend la justesse. L'en-tête d'effort n'a aucun effet sur elles.
6. **`low` direct à k = 2 a un risque de bluff résiduel** si `tau_low = 0,80` est trop bas : 1 réponse sur 5 fausse. C'est la définition de `low`, mais le propriétaire peut préférer 0,90. À valider avant E4 (décision de produit, pas technique).
7. **Taille du jeu.** 9 000 problèmes × 2 styles pour 8 000 conversations au midtrain : le modèle voit chaque problème moins d'une fois. Si E4 sous-apprend, d'abord la SFT à 24 000 chats, ensuite 20 000 problèmes (coût LLM ×2,2).
8. **Pretraining jamais vu ce monde.** Les histoires TinyStories contiennent des pommes et des prénoms, mais pas de problèmes. Option si E4 est trop juste : ajouter des documents `story_math` dans les worksheets du pretraining (11,4 min à refaire, donc coûteux) : non recommandé avant d'avoir essayé 24 000 chats SFT.
9. **Fuite de l'éval.** Seeds A (sonde), B (politique), train et éval distincts ; les prénoms/objets held-out jamais dans le train ; les 2 exemples few-shot du teacher ne figurent pas dans l'éval (comparaison de texte).
10. **Confidentialité / ZDR.** Les problèmes sont synthétiques (aucune donnée du propriétaire), mais on garde `provider.zdr=true` et `data_collection=deny` ; la liste ZDR et les prix datent d'aujourd'hui et peuvent changer : le script revérifie avant de partir. Je n'ai pas appelé l'API (aucune clé utilisée ici).
11. **Le coût de génération est une estimation** [E] qui suppose ~400 tokens d'entrée et ~150 de sortie par appel et un taux d'acceptation de 70 % que E2 mesurera.

---

## 11. Décisions du coordinateur (2026-11-12, priment sur le reste)

1. **Base des runs** : `runs/cand5k/{pretrain,midtrain,sft}` + `runs/cand5k/tokenizer.json` (source de prelude-2), pas `runs/prelude`. La sonde (E1) part de `runs/cand5k/pretrain`. Référence d'éval : `runs/baseline-p2`.
2. **Seuils** : `tau = {low: 0.80, medium: 0.92, high: 0.99}` provisoires. Le propriétaire peut passer `low` à 0.90 avant E4 ; tout seuil est lu depuis `[data.reason]` (jamais en dur).
3. **Budget OpenRouter** : plafond dur 8 $ cumulés sur tout le chantier. Au-delà : demander au propriétaire, ne jamais relever le plafond seul.
4. **T0 est fait par l'agent de T1** (même branche) : un seul propriétaire de `configs/prelude.toml` jusqu'à la fusion.
5. **Ressources (Mac partagé, RAM sous pression)** : pendant l'implémentation, seulement `pytest` et des scripts de quelques secondes, toujours sous `nice -n 10`. Aucun entraînement MPS, aucune éval complète, aucun `uv sync` : utiliser `/Users/clementthiriet/VM/mini-lab/.venv/bin/python -m pytest -q` depuis la racine du worktree. Les expériences E1-E8 sont lancées par le coordinateur, une à la fois.
6. **Code** : minimal, clair, concis, dans le style du dépôt (pas de nouvelle dépendance, pas d'abstraction spéculative). Chaque tâche garde les 245 tests verts et ajoute les siens.
