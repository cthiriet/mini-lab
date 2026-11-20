# Story-math : état du chantier et reprise

État au 20/11/2026, après E9. Ce fichier sert à reprendre le travail dans une nouvelle session. La spec complète est dans [story-math-spec.md](story-math-spec.md), dont le §11 liste les décisions 1 à 9.

## Le chantier en bref

Demande de Clément : faire réfléchir le modèle (GPT mini-lab, 5,8M paramètres, MPS) sur des problèmes où la réflexion aide vraiment. Il fallait :

- une réflexion en phrases variées, en partie écrite par un LLM teacher ;
- des niveaux d'effort comme chez Claude/GPT ;
- aucune régression sur les additions, le chat et l'agent de code ;
- des évals rigoureuses, et un code minimal et propre.

Méthode imposée :

- Claude coordonne et joue le chief scientist ; les agents d'implémentation tournent sur Sonnet, chacun dans son worktree, avec une spec précise.
- Après chaque expérience, un agent indépendant prend du recul et propose des idées.
- Ressources frugales : un seul calcul MPS à la fois, toujours sous `nice -n 10`.

**La tâche, story_math.** Un héros a N objets, puis il lui arrive de 1 à 6 événements : ajouter, retirer ou doubler. La question demande le total final. Par exemple : « Sam has 4 flowers. He gets 6 from Kate. The sun is warm. He gives 10 to Ben. … How many flowers does Sam have now? ». Les pièges :

- une phrase parasite (« The sun is warm », ou « Joel gives Nick 5 candies ») ;
- des prénoms découpés en plusieurs tokens ;
- d'autres personnages qui donnent ou reçoivent.

Les jeux held-out ajoutent 7 ou 8 étapes, l'opération `half` jamais vue à l'entraînement, 20 prénoms et 20 objets jamais vus, et une phrase parasite systématique.

## Où est le code

- **Worktree :** `/Users/clementthiriet/VM/mini-lab/.claude/worktrees/agent-a95ab00c560bea7de`, branche `reasoning-steps`.
- **HEAD :** `f09eb2d` (E9). 321 tests passent.
- **Rien n'est mergé dans main ni poussé.**
- **Branches déjà intégrées dans reasoning-steps** (supprimables) : `t1-story-math`, `t10-sub`, `t11-calc`, `t12-high`, `e7-anchors`, `e8-number`, `e9-two-levels`.
- **Jamais mergées :** `exp/direct-rule-a` et `exp/direct-rule-b`.

Fichiers clés :

- **Problèmes et réflexions :** `minilab/data/story_math.py`. Il contient le générateur `make`, avec les paramètres `pseudo` et `wide`, ainsi que :
  - `trace_program`, qui écrit la réflexion par niveau ;
  - `anchor` et `strip_anchor` ;
  - `verify_trace` et `style_ok` ;
  - la politique adaptative `bucket`, `decide` et `think`.
- **Mélange d'entraînement :** `minilab/data/conversations.py`, classe `Reason` et fonction `story_math_conversation`.
- **Rendu :** `minilab/tokenizer/chat.py`. La ligne d'en-tête `effort: low` n'est écrite que pour low.
- **API :** `minilab/api/schemas.py`, fonction `normalize_effort`. `none` donne none ; `minimal` et `low` donnent low ; `medium`, `high`, `xhigh`, `max`, l'absence de valeur et toute valeur inconnue donnent high. Le défaut de `minilab/inference/server.py` est high.
- **Calculatrice dans la réflexion :** `minilab/inference/calculator.py`, `minilab/model/gpt.py` (`generate(..., force=)`) et `minilab/inference/engine.py`. Après un « = » qui suit une expression valide, le résultat est imposé. Elle est active à l'éval story_math et en serving, mais désactivée en RL, en distill et à l'éval des additions.
- **Éval :** `minilab/eval/tasks.py` (`eval_story_math`) et `minilab/eval/run.py` (`--story-math`). Les modes sont adaptive low, forced low, adaptive high, forced high, off et adaptive high nocalc. Les scalaires sont `sm_low`, `sm_high`, `sm_high_nocalc` et `sm_6st`.
- **Config :** `configs/prelude.toml`, section `[data.reason]`.
- **Scripts :** `scripts/teacher_traces.py` (teacher OpenRouter) et `scripts/probe_direct.py` (mesure de la politique).
- **Données** (gitignorées, dans le worktree) :
  - `data/reason/policy.json` ;
  - `data/reason/traces_v2.jsonl.dedup.jsonl`, soit 8 091 réflexions medium du teacher, inutilisées depuis E9.

## Les formats de réflexion actuels (E9)

- **none :** pas de réflexion ; `<|think_start|>` est interdit.
- **low** (en-tête `effort: low`) :
  ```
  4 events:
  1: 17+8=25
  2: 25-18=7
  …
  ```
- **high** (pas d'en-tête, c'est le défaut) : l'histoire relue phrase par phrase. Chaque phrase est citée puis suivie de son effet, et la réflexion se termine par « So Kai has 5 eggs. ».
- **Politique adaptative :** le modèle répond sans réfléchir seulement si le probe, mesuré sur ce type de problème, dépasse `tau[niveau] + 0.03`. En pratique, seul le doublement simple à une étape se répond directement en low.

## Résultats

Tous les chiffres sont mesurés sur le même jeu d'éval (100 problèmes par k), en mode adaptatif avec calculatrice. Chaque cellule donne la moyenne, puis la justesse de k=1 à 6, puis les tokens de réflexion.

| Run | Changement | low | medium | high |
|---|---|---|---|---|
| E4 | référence (avec calculatrice) | 69,2 | 59,8 | 47,3 |
| E6 | high relit l'histoire | 79,2 · [94 97 90 86 69 39] · 27 tok | 74,5 · 76 tok | 88,2 · [97 98 90 91 84 69] · 105 tok |
| E7 | compteur « N events » et low numéroté, banque teacher régénérée, 8 % de mots inventés, held-out à 20+20 | **86,5** · [95 98 93 87 79 67] · 44 tok | 81,3 · [97 95 92 82 76 46] · 81 tok | 91,2 · [99 96 97 98 92 65] · 106 tok |
| E8 | medium numéroté, lexique élargi (~200 prénoms, ~236 objets), 25 % de mots inventés | 82,0 · [97 93 87 85 70 60] | 83,2 · [96 90 88 89 78 58] · 94 tok | 90,0 · [99 98 97 85 84 77] |
| **E9** | **deux niveaux** : low 0,4 et high 0,6, plus de medium ni de teacher ; lexique d'E7 | 77,7 · [79 98 90 80 69 50] · 43 tok | (n'existe plus) | **96,5** · [99 98 98 95 99 90] · 106 tok ; 85,5 sans calculatrice |

Held-out, en high (en medium jusqu'à E8) :

| Run | objets et prénoms jamais vus | phrase parasite | 7-8 étapes | `half` |
|---|---|---|---|---|
| E7 | 46 (medium 49,5) | 78,5 | ~5 | ~5 |
| E8 | 62 (medium 52,5) | 77 | ~6 | ~3 |
| E9 | 52,5 | **91,5** | ~6,5 | ~2 |

Les runs sont dans `/Users/clementthiriet/VM/mini-lab/runs/<run>/`. Chaque dossier contient `config.toml` (chemins absolus), `midtrain/`, `sft/`, `sft/eval.json` et un log `<run>.out`. Pour l'ablation E8a (medium numéroté sans lexique élargi), seul le midtrain a tourné ; la SFT a été arrêtée, voir la section Mémoire.

## Décisions prises (validées par Clément)

1. **La calculatrice reste dans la réflexion.** Elle a été choisie plutôt que de finir au niveau E4.
2. **Seulement deux niveaux de réflexion entraînés, low et high.** medium est redirigé vers high dans l'API. Raison : en E7 et E8, medium ne battait jamais low de plus que le bruit et coûtait environ deux fois plus de tokens (décision 9 de la spec).
3. Le lexique élargi d'E8 est écarté pour E9 : il aidait high sur les mots jamais vus, mais faisait régresser low.

## Problème ouvert : low régresse en E9

low passe de 86,5 à 77,7. La régression touche aussi le mode forcé, qui donne [88 98 90 80 69 50] : la politique n'est donc pas la seule cause. Pourtant low reçoit plus d'exemples qu'en E7 (40 % contre 30 %).

Une analyse a été interrompue en cours de route. Ses fichiers partiels sont dans `runs/e9/analysis/cls_e7.json` et `cls_e9.json` ; ce sont des exemples par type d'échec, au plus 3 par type, pas des comptages. **Piste nette à k=1 :** le modèle compte la phrase parasite comme un événement. Il écrit par exemple « 2 events: » pour « Greta has 14 beads. Joel gives Nick 5 candies. Greta loses 13. », puis invente une seconde équation (« 1-1=0 »). Il fait de même avec « It is a sunny day ». Les autres types observés sont un opérande mal copié, une mauvaise opération sur une étape tardive et un arrêt trop tôt.

Hypothèses à tester, non vérifiées :

- **A.** En E7, medium portait aussi le préfixe « N events. » et servait de pont à low. Sans lui, low voit moins de comptages corrects d'événements.
- **B.** Le code d'E8 applique aussi les mots inventés à un autre personnage, et ces mots font de 1 à 4 syllabes. Une phrase parasite avec un nom inventé ressemblerait alors davantage à un événement.
- **C.** C'est du bruit d'un seul run. C'est peu probable vu l'écart de 9 points.

Correctifs peu chers à envisager, un run d'environ 15 minutes chacun :

- **Option 1 :** apprendre à low à ignorer explicitement la parasite, par exemple « 3 events (1 not counted): ».
- **Option 2 :** supprimer l'en-tête de comptage en low et ne garder que la numérotation, si c'est le compteur qui l'induit en erreur.
- **Option 3 :** revenir à la génération des mots inventés d'E7, qui ne touchait que le héros et l'objet.

Critère de succès : low ≥ 86 de moyenne, et high ≥ 95.

## Reste à faire pour clore le chantier

1. Trancher pour low : appliquer un des correctifs ci-dessus, ou accepter E9 tel quel. Même ainsi, low fait mieux qu'E4 (69,2) pour 2,5 fois moins de tokens.
2. **Non-régression sur le run retenu :** chat (≥ 92 %), additions et agent de code. Il faut repartir des runs E7 et E9 et lancer l'éval complète de `minilab.eval.run` sans `--story-math`, plus le gate face à prelude-2.
3. Relancer le pipeline complet : rl_math puis distill, en ajoutant story_math au distill (tâche T7, jamais faite).
4. **Nettoyage :**
   - `scripts/teacher_traces.py` décrit encore medium et l'ancien high avec Check.
   - Les branches medium inutiles de `anchor` et `strip_anchor` peuvent partir.
   - La spec doit refléter les formats actuels : son corps décrit encore le Check.
5. Docs (T8).

## Mémoire : à corriger avant tout nouveau run

La SFT occupe **16-17 Go de mémoire MPS**. Les logits font 32 × 1024 × 4096 en float32, soit 512 Mo par copie, et plusieurs copies sont gardées pour le backward. Plafonner avec `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, à 8 ou à 13 Go, fait planter le run avec un OOM.

Correctif proposé, pas encore fait : calculer la loss par morceaux (chunked cross-entropy), ou découper le batch avec accumulation de gradients, dans `minilab/train/sft.py` et le loader. Le modèle apprendrait la même chose.

Le Mac a 64 Go, dont 24 Go pris par la VM Lima et 8 Go par la VM `cyber`. Vérifier la mémoire avec `top -o mem` : la RSS de `ps` ne compte pas la mémoire MPS.

## Lancer un run

Depuis `/Users/clementthiriet/VM/mini-lab`, avec `W=.claude/worktrees/agent-a95ab00c560bea7de` :

```bash
R=runs/eX && mkdir -p $R && cp -r runs/cand5k/pretrain $R/ && cp runs/cand5k/tokenizer.json $R/
sed -e "s|^traces = .*|traces = \"$PWD/$W/data/reason/traces_v2.jsonl.dedup.jsonl\"|" \
    -e "s|^policy = .*|policy = \"$PWD/$W/data/reason/policy.json\"|" $W/configs/prelude.toml > $R/config.toml
```

Ensuite, depuis le worktree, en remplaçant `$RR` par le chemin absolu du run :

```bash
PY=/Users/clementthiriet/VM/mini-lab/.venv/bin/python
nice -n 10 $PY -m minilab.train.midtrain --run $RR && nice -n 10 $PY -m minilab.train.sft --run $RR && PYTHONPATH=$PWD nice -n 10 $PY -m minilab.eval.run --run $RR --stage sft --story-math
```

Le midtrain prend environ 1 min, la SFT environ 8 min et l'éval environ 5 min. Les scripts lancés hors pytest ont besoin de `PYTHONPATH=$PWD`, parce que l'install editable pointe sur le dépôt principal.

## Contraintes à respecter

- **Secrets :** uniquement via la skill 1password-secrets, sans jamais les afficher. La clé OpenRouter est `op://Development/openrouter/api-key`. Le teacher passe en ZDR, avec `provider: {zdr: true, data_collection: deny}`.
- **Budget OpenRouter :** plafond de 8 $ au total, environ 1,60 $ dépensés. Ne pas relever le plafond sans demander.
- **Agents d'implémentation :** sur Sonnet, chacun dans son worktree. Ne jamais pousser et ne pas commiter sur main.
- **Langue :** rapports en français ; les termes techniques (forward pass, batch, throughput…) restent en anglais.
