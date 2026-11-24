# Story-math : état du chantier et reprise

État au 24/11/2026, après E10 (low corrigé, pipeline complet rejoué ; une régression d'addition bloque encore le gate). Ce fichier sert à reprendre le travail dans une nouvelle session. La spec complète est dans [story-math-spec.md](story-math-spec.md), dont le §11 liste les décisions 1 à 9.

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
- **HEAD :** E10 (voir `git log`). 320 tests passent.
- **Rien n'est mergé dans main ni poussé.**
- **Branches déjà intégrées dans reasoning-steps** (supprimables) : `t1-story-math`, `t10-sub`, `t11-calc`, `t12-high`, `e7-anchors`, `e8-number`, `e9-two-levels`.
- **Jamais mergées :** `exp/direct-rule-a` et `exp/direct-rule-b`.

Fichiers clés :

- **Problèmes et réflexions :** `minilab/data/story_math.py`. Il contient le générateur `make`, avec le paramètre `pseudo`, ainsi que :
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
- **Scripts :** `scripts/probe_direct.py` (mesure de la politique). `scripts/teacher_traces.py` (teacher OpenRouter) a été retiré ; on le retrouve dans l'historique git au commit `fbaf453`.
- **Données** (gitignorées, dans le worktree) :
  - `data/reason/policy.json` ;
  - `data/reason/traces_v2.jsonl.dedup.jsonl`, soit 8 091 réflexions medium du teacher, inutilisées depuis E9.

## Les formats de réflexion actuels (E10)

- **none :** pas de réflexion ; `<|think_start|>` est interdit.
- **low** (en-tête `effort: low`) :
  ```
  4 events:
  1: 17+8=25
  2: 25-18=7
  …
  ```
- **high** (pas d'en-tête, c'est le défaut) : depuis E10, « N events. » puis l'histoire relue phrase par phrase. Chaque phrase est citée puis suivie de son effet, et la réflexion se termine par « So Kai has 5 eggs. ».
- **Politique adaptative :** le modèle répond sans réfléchir seulement si le probe, mesuré sur ce type de problème, dépasse `tau[niveau] + 0.03`. Depuis E10 (`tau.low = 0.97`), aucun type n'y arrive : low réfléchit toujours. Avant, le doublement à une étape se répondait directement, et le modèle SFT d'E9 n'en réussissait qu'un sur dix.

## Résultats

Tous les chiffres sont mesurés sur le même jeu d'éval (100 problèmes par k), en mode adaptatif avec calculatrice. Chaque cellule donne la moyenne, puis la justesse de k=1 à 6, puis les tokens de réflexion.

| Run | Changement | low | medium | high |
|---|---|---|---|---|
| E4 | référence (avec calculatrice) | 69,2 | 59,8 | 47,3 |
| E6 | high relit l'histoire | 79,2 · [94 97 90 86 69 39] · 27 tok | 74,5 · 76 tok | 88,2 · [97 98 90 91 84 69] · 105 tok |
| E7 | compteur « N events » et low numéroté, banque teacher régénérée, 8 % de mots inventés, held-out à 20+20 | **86,5** · [95 98 93 87 79 67] · 44 tok | 81,3 · [97 95 92 82 76 46] · 81 tok | 91,2 · [99 96 97 98 92 65] · 106 tok |
| E8 | medium numéroté, lexique élargi (~200 prénoms, ~236 objets), 25 % de mots inventés | 82,0 · [97 93 87 85 70 60] | 83,2 · [96 90 88 89 78 58] · 94 tok | 90,0 · [99 98 97 85 84 77] |
| **E9** | **deux niveaux** : low 0,4 et high 0,6, plus de medium ni de teacher ; lexique d'E7 | 77,7 · [79 98 90 80 69 50] · 43 tok | (n'existe plus) | **96,5** · [99 98 98 95 99 90] · 106 tok ; 85,5 sans calculatrice |
| **E10** | compteur « N events. » à high aussi ; `tau.low` 0,97 | **86,0** · [97 98 89 89 79 64] · 44 tok | — | **96,5** · [99 100 98 97 95 90] · 111 tok ; 85,8 sans calculatrice |
| E10 distill (add 0,6) | modèle final | 85,0 | — | 96,7 |

Held-out, en high (en medium jusqu'à E8) :

| Run | objets et prénoms jamais vus | phrase parasite | 7-8 étapes | `half` |
|---|---|---|---|---|
| E7 | 46 (medium 49,5) | 78,5 | ~5 | ~5 |
| E8 | 62 (medium 52,5) | 77 | ~6 | ~3 |
| E9 | 52,5 | **91,5** | ~6,5 | ~2 |
| E10 | 53 | 91 | ~10 | ~4 |

Les runs sont dans `/Users/clementthiriet/VM/mini-lab/runs/<run>/`. Chaque dossier contient `config.toml` (chemins absolus), `midtrain/`, `sft/`, `sft/eval.json` et un log `<run>.out`. Pour l'ablation E8a (medium numéroté sans lexique élargi), seul le midtrain a tourné ; la SFT a été arrêtée, voir la section Mémoire.

## Décisions prises (validées par Clément)

1. **La calculatrice reste dans la réflexion.** Elle a été choisie plutôt que de finir au niveau E4.
2. **Seulement deux niveaux de réflexion entraînés, low et high.** medium est redirigé vers high dans l'API. Raison : en E7 et E8, medium ne battait jamais low de plus que le bruit et coûtait environ deux fois plus de tokens (décision 9 de la spec).
3. Le lexique élargi d'E8 est écarté pour E9 : il aidait high sur les mots jamais vus, mais faisait régresser low.

## E10 : la régression de low (résolue)

Diagnostic (low forcé, 600 problèmes, script de classement dans `runs/e9/analysis/`) : en E9, presque tous les échecs de low venaient du **compteur**. À k=1, le modèle surcomptait dans 11 % des cas (1 % en E7), en prenant la phrase parasite pour un événement. À k=6, il sous-comptait dans 30 % des cas (7 % en E7), puis s'arrêtait trop tôt. En E7, 70 % des réflexions portaient le compteur (low et medium), contre 40 % en E9 : c'est l'hypothèse A. S'y ajoutait la politique : à k=1, les doublements répondus directement étaient faux 9 fois sur 10.

Correctif E10 : « N events. » en tête de high, et `tau.low = 0.97`. Résultat en un run : low 86,0 et high 96,5, les deux critères sont atteints.

## Non-régression et pipeline complet (E10)

Run `runs/e10` : midtrain, SFT, rl_math, puis distill avec story_math (tâche T7 faite, teacher = modèle SFT, 1 problème sur 10). Modèle distillé avec `add = 0.6` (`runs/e10/distill`) :

- story_math : low 85,0, high 96,7.
- chat 93 %, instructions 99,4 %, outil 99,2 %.
- agent de code 98,3 %. Seule la tâche fix_distractor est à 86 %.
- additions en greedy de 1 à 4 chiffres : 100 %. À 5 chiffres : 99 %.
- **5 chiffres à T=1 : 96 %**, contre 100 % pour prelude-2 mesuré par le gate et 99 % dans son eval.json.

Un premier distill avec `add = 0.5` donnait 95 % (`runs/e10/distill_add05`). Le gate face à prelude-2 a bloqué sur ce seul point : `5d@T=1` passe de 100 % à 95 %, tout le reste passe. Le gate n'a pas été relancé sur le distill à `add = 0.6`, qui devrait toujours être bloqué : la tolérance est de 2 points. Le spécialiste rl_math d'E10 est lui-même à 98 % à T=1 et 97 % en greedy à 5 chiffres. Les runs d'avant atteignaient 99 à 100 %, sans doute parce que la SFT partage maintenant sa capacité avec story_math.

## Reste à faire

1. **`5d@T=1` tranché (24/11)** : c'est du bruit d'échantillon. Sur 500 problèmes, E10 distillé fait 98,2 % (±1,2) et prelude-2 99,0 % (±0,9), contre 96 et 100 sur les 100 du gate. Décision : accepter la régression avec `--allow "5d@T=1=bruit d'échantillon : 98,2 % contre 99,0 % pour prelude-2 sur 500 problèmes (±1,2), sous la tolérance de 2 points"`. Un bug de `parse_waivers` coupait au premier « = », donc dans le nom de la métrique : il est corrigé. Le gate n'a pas été relancé depuis la correction ; c'était sa seule régression.
2. Release : `minilab.release` sur `runs/e10/distill` avec cette dérogation, puis mise à jour du README avec les chiffres de la release.
3. Docs utilisateur (T8) : faites pour docs/training.md et docs/api.md ; le README attend la release.
4. Branches déjà intégrées et supprimables : `t1-story-math`, `t10-sub`, `t11-calc`, `t12-high`, `e7-anchors`, `e8-number`, `e9-two-levels`.

## Mémoire (corrigée)

La SFT tient sous un plafond de 8 Go de MPS (pic à 7,0 Go en E10, contre 16-17 Go avant), sans perte de throughput. Deux changements :

- les logits ne sont calculés qu'aux positions qui ont une cible, avec un nombre de positions arrondi à un multiple de 1 024 ;
- `[sft] micro_batch = 16` découpe le batch en tranches dont les gradients s'additionnent, ce qui donne le même gradient.

Le gros de la mémoire venait de l'attention compilée, qui garde une matrice (rows, heads, T, T) en float32 par couche. Pour lancer avec le plafond : `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.154 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1` (0,154 × 51,8 Go ≈ 8 Go).

## Lancer un run

Depuis `/Users/clementthiriet/VM/mini-lab`, avec `W=.claude/worktrees/agent-a95ab00c560bea7de` :

```bash
R=runs/eX && mkdir -p $R && cp -r runs/cand5k/pretrain $R/ && cp runs/cand5k/tokenizer.json $R/
sed -e "s|^policy = .*|policy = \"$PWD/$W/data/reason/policy.json\"|" $W/configs/prelude.toml > $R/config.toml
```

Ensuite, depuis le worktree, en remplaçant `$RR` par le chemin absolu du run :

```bash
PY=/Users/clementthiriet/VM/mini-lab/.venv/bin/python
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.154 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1 PYTHONPATH=$PWD
nice -n 10 $PY -m minilab.train.midtrain --run $RR && nice -n 10 $PY -m minilab.train.sft --run $RR && nice -n 10 $PY -m minilab.eval.run --run $RR --stage sft --story-math
nice -n 10 $PY -m minilab.train.rl_math --run $RR && nice -n 10 $PY -m minilab.train.distill --run $RR && nice -n 10 $PY -m minilab.eval.run --run $RR --stage distill
nice -n 10 $PY -m minilab.eval.gate $RR/distill --baseline /Users/clementthiriet/VM/mini-lab/models/prelude-2 --config $RR/config.toml
```

Les durées sont d'environ 1 min pour le midtrain, 8 min pour la SFT, 5 min pour l'éval story_math, 9 min pour le RL, 8 min pour le distill et 4 min pour l'éval complète. Le gate prend environ 40 min. L'éval de l'agent de code et le gate ont besoin de Docker Desktop.

## Contraintes à respecter

- **Secrets :** uniquement via la skill 1password-secrets, sans jamais les afficher. La clé OpenRouter est `op://Development/openrouter/api-key`. Le teacher passe en ZDR, avec `provider: {zdr: true, data_collection: deny}`.
- **Budget OpenRouter :** plafond de 8 $ au total, environ 1,60 $ dépensés. Ne pas relever le plafond sans demander.
- **Agents d'implémentation :** sur Sonnet, chacun dans son worktree. Ne jamais pousser et ne pas commiter sur main.
- **Langue :** rapports en français ; les termes techniques (forward pass, batch, throughput…) restent en anglais.
