"""Story problems about one counter, and the reasoning that solves them (docs/story-math-spec.md).

A hero starts with a few things and goes through k events (+, -, x2, and in the held-out
split half), among sentences that change nothing. The answer is the last state. The
module also holds what the reasoning must look like: three levels of effort
(low: bare equations, medium: short sentences, high: the story read again sentence by sentence), a
program that writes them (the plan without a teacher model), the checks every trace
must pass (a teacher's too), and the rule that decides whether to think.
"""

from __future__ import annotations

import json
import random
import re

from minilab.data import arithmetic

# Most of the older names are one token; the added ones take two to four (cand5k tokenizer), like the
# held-out names, which the model has never seen.
HEROES = [*arithmetic.NAMES, "Zoe", "Leo", "Ella", "Jack", "Nina", "Owen",
          "Nora", "Ivy", "Elsa", "Greta", "Hazel", "Isla", "Jade", "Kira", "Lena", "Maria", "Olga", "Petra", "Rita",
          "Selma", "Tessa", "Uma", "Wanda", "Yara", "Alice", "Chloe", "Diana", "Olivia", "Rosa",
          "Adam", "Boris", "Carlos", "Dario", "Eli", "Felix", "Gus", "Henry", "Igor", "Jonas", "Kai", "Liam", "Marco",
          "Noah", "Oscar", "Pablo", "Rafael", "Stefan", "Theo", "Viktor", "Aaron", "Caleb", "Dylan", "Ethan",
          "Quentin", "Yusuf", "Ravi"]
OTHERS = ["Tom", "Mia", "Ben", "Sue", "Max", "Anna", "Kate", "Paul", "Nick", "Toby", "Joel", "Hans", "Lars", "Pia",
          "Rex", "Zack"]
THINGS = [("apple", "apples"), ("toy", "toys"), ("sticker", "stickers"), ("ball", "balls"), ("cookie", "cookies"),
          ("flower", "flowers"), ("book", "books"), ("shell", "shells"), ("star", "stars"), ("coin", "coins"),
          ("candy", "candies"), ("pencil", "pencils"),
          ("box", "boxes"), ("cherry", "cherries"), ("peach", "peaches"), ("key", "keys"), ("cup", "cups"),
          ("hat", "hats"), ("bell", "bells"), ("ring", "rings"), ("rock", "rocks"), ("stamp", "stamps"),
          ("balloon", "balloons"), ("bead", "beads"), ("lemon", "lemons"), ("carrot", "carrots"), ("plum", "plums"),
          ("egg", "eggs"), ("cake", "cakes"), ("bottle", "bottles"), ("brush", "brushes"), ("penny", "pennies"),
          ("puppy", "puppies"), ("bike", "bikes"), ("card", "cards"), ("crayon", "crayons"), ("doll", "dolls"),
          ("cube", "cubes"), ("block", "blocks"), ("sock", "socks"), ("nut", "nuts"), ("bean", "beans")]
HELD_HEROES = ["Ruby", "Finn", "Cleo", "Ivan", "Maya", "Hugo"]
HELD_OTHERS = ["Omar", "Vera", "Hana", "Luca"]
HELD_THINGS = [("marble", "marbles"), ("pear", "pears"), ("kite", "kites"), ("button", "buttons"),
               ("feather", "feathers"), ("acorn", "acorns")]
FEMALE = {"Lily", "Mia", "Sue", "Anna", "Lucy", "Zoe", "Ella", "Nina", "Ruby", "Cleo", "Maya", "Nora", "Ivy", "Elsa",
          "Greta", "Hazel", "Isla", "Jade", "Kira", "Lena", "Maria", "Olga", "Petra", "Rita", "Selma", "Tessa", "Uma",
          "Wanda", "Yara", "Alice", "Chloe", "Diana", "Olivia", "Rosa"}
WEATHER = [("The sun is warm.", "The sun"), ("It is a sunny day.", "The day"), ("The sky is blue.", "The sky")]
SYMBOL = {"add": "+", "sub": "-", "dbl": "x", "half": "/"}

# Event sentences: (sentence, clause); the clause follows the subject in a trace ("She finds 5").
# Fields: S / m = the hero's name or She / she, O = someone else, Q / P / Pc = her, her, Her (or him, his, His),
# n = the amount, t = the things.
EVENTS = {
    "add": [("{S} finds {n} more.", "finds {n}"), ("{S} buys {n} more.", "buys {n}"),
            ("{S} picks up {n} more.", "picks up {n}"), ("{S} gets {n} more.", "gets {n}"),
            ("{O} gives {Q} {n}.", "gets {n} from {O}"), ("Then {m} receives {n} more.", "receives {n}"),
            ("{S} is given {n} more by {O}.", "is given {n} by {O}"), ("{O} brings {Q} {n} more.", "gets {n} from {O}")],
    "sub": [("{S} gives {n} to {O}.", "gives {n} to {O}"), ("Then {m} eats {n}.", "eats {n}"),
            ("{S} loses {n}.", "loses {n}"), ("{S} sells {n}.", "sells {n}"),
            ("{O} takes {n} from {Q}.", "loses {n} to {O}"), ("{S} drops {n}.", "drops {n}"),
            ("{S} gives {O} {n}.", "gives {O} {n}")],
    "dbl": [("Then {P} pile doubles.", "doubles {P} {t}"), ("{S} doubles {P} {t}.", "doubles {P} {t}"),
            ("Then {m} has twice as many {t} as before.", "doubles {P} {t}"),
            ("{Pc} {t} double.", "doubles {P} {t}"), ("The number of {P} {t} doubles.", "doubles {P} {t}")],
    "half": [("{S} gives half of them to {O}.", "gives half to {O}"),
             ("{S} shares half of {P} {t} with {O}.", "gives half to {O}"), ("{S} gives {O} half.", "gives half to {O}")],
}
QUESTIONS = ["How many {t} does {H} have now?", "How many {t} does {H} have left?",
             "How many {t} does {H} have in all?", "How many {t} will {H} have then?",
             "{H} wants to count {P} {t}. How many are there?"]

# Reasoning sentences. Medium: a start sentence, then one sentence with one equation per event.
INTROS = ["{H} starts with {c}.", "{H} had {c}.", "At first, {H} has {c}.", "{H} begins with {c}."]
_COMMON = ["{S} {clause}, so {eq}.", "{S} {clause}: {eq}."]
MEDIUM = {"add": [*_COMMON, "{n} more, so {eq}.", "Plus {n}: {eq}.", "Add {n}: {eq}.", "Now {eq}, with {n} more."],
          "sub": [*_COMMON, "{n} fewer, so {eq}.", "Minus {n}: {eq}.", "Take away {n}: {eq}.", "Now {eq}, with {n} gone."],
          "dbl": [*_COMMON, "Doubling gives {eq}.", "It doubles: {eq}.", "Twice as many: {eq}.", "Double it, so {eq}."],
          "half": [*_COMMON, "Half of it: {eq}.", "Half goes, so {eq}.", "Cut it in half: {eq}.", "Now {eq}, half gone."]}
# High: the story read again. Each sentence after the first is quoted, then comes what it does
# (an equation, or that it changes nothing), then the conclusion.
HIGH_INTROS = ["{H} begins with {c}.", "{H} starts with {c}.", "To begin, {H} has {c}.", "At first, {H} has {c}."]
EFFECTS = ["So {eq}.", "That gives {eq}.", "That makes {eq}.", "Now {eq}."]
UNCHANGED = ["That does not change {P} {t}.", "That does not matter.", "That has nothing to do with {P} {t}.",
             "That does not change anything."]

EQUATION = re.compile(r"(?<![\w.])([\d(][\d()+\-x/ ]*?[\d)])[ ]*=[ ]*(\d+)")
NUMBER_WORDS = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                          r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                          r"eighty|ninety|hundred)\b", re.I)
FORBIDDEN = re.compile(r"[^\x00-\x7f]|[*#`]|^[-+] |The answer is", re.M)
NOT_MATTER = re.compile(r"not matter|not change|nothing to do")


def count(n: int, thing: tuple[str, str]) -> str:
    return f"{n} {thing[0] if n == 1 else thing[1]}"


def _pronouns(hero: str) -> tuple[str, str, str]:
    """Subject, object, possessive."""
    return ("She", "her", "her") if hero in FEMALE else ("He", "him", "his")


def _subject(rng: random.Random, hero: str) -> tuple[str, str]:
    """How a sentence names the hero: at its start and inside it."""
    if rng.random() < 0.5:
        return hero, hero
    s = _pronouns(hero)[0]
    return s, s.lower()


def make(rng: random.Random, k: int, split: str = "train", distractor: bool | None = None) -> dict:
    """A problem with k events. The held-out split has names and things the training never shows,
    the operation `half`, and a distractor every time (unless `distractor` says otherwise)."""
    held = split == "heldout"
    heroes, others, things = (HELD_HEROES, HELD_OTHERS, HELD_THINGS) if held else (HEROES, OTHERS, THINGS)
    hero, thing = rng.choice(heroes), rng.choice(things)
    others = [o for o in others if o != hero]
    _, obj, poss = _pronouns(hero)
    states, ops, sentences, clauses, fields = [rng.randint(2, 20)], [], [], [], []
    for _ in range(k):
        s = states[-1]
        again = bool(ops) and ops[-1][0] == "dbl"  # two doublings in a row read like one said twice
        weights = {"add": 4 * (s <= 98), "sub": 4 * (s >= 1), "dbl": 2 * (1 <= s <= 40 and not again),
                   "half": 2 * (held and s >= 2 and s % 2 == 0)}
        op = rng.choices(list(weights), list(weights.values()))[0]
        n = rng.randint(1, min(20, 99 - s)) if op == "add" else rng.randint(1, s) if op == "sub" else None
        states.append({"add": s + (n or 0), "sub": s - (n or 0), "dbl": 2 * s, "half": s // 2}[op])
        ops.append((op, n))
        sentence, clause = rng.choice(EVENTS[op])
        S, m = _subject(rng, hero)
        f = dict(S=S, m=m, O=rng.choice(others), Q=obj, P=poss, Pc=poss.capitalize(), n=n, t=thing[1])
        sentences.append(sentence.format(**f))
        clauses.append(clause.format(**f))
        fields.append(f)
    at, ignore = None, None
    if (rng.random() < (1 if held else 0.35)) if distractor is None else distractor:
        who, other = rng.sample(others, 2)
        stuff = rng.choice([t for t in things if t != thing])
        n = rng.randint(1, 9)
        owned = (f"{who}'s {stuff[1]}", "do")
        sentence, ignore = rng.choice([(f"{who} has {count(n, stuff)}.", owned),
                                       (f"{who} gives {other} {count(n, stuff)}.", owned),
                                       *[(s, (w, "does")) for s, w in WEATHER]])
        at = rng.randint(0, k)  # before the event of that index (k: after the last)
        if at < k:  # the event after it starts with the hero's name: "She" or "his" could mean the distractor's person
            op, f = ops[at][0], fields[at] | {"S": hero}
            event, clause = rng.choice([e for e in EVENTS[op] if e[0].startswith("{S}")])
            sentences[at], clauses[at] = event.format(**f), clause.format(**f)
        sentences.insert(at, sentence)
    question = rng.choice(QUESTIONS).format(t=thing[1], H=hero, P=poss)
    start = f"{hero} has {count(states[0], thing)}."
    content = " ".join([start, *sentences, question])
    problem = {"kind": "story_math", "messages": [{"role": "user", "content": content}], "tools": None,
               "steps": k, "digits": k, "answer": states[-1], "states": states, "ops": ops, "hero": hero,
               "thing": thing, "clauses": clauses, "ignore": ignore, "at": at,
               "sentences": sentences}  # the events and the distractor, as worded in the problem
    return {**problem, "bucket": bucket(problem)}


def answer_text(n: int) -> str:
    return f"The answer is {n}."


def bucket(problem: dict) -> str:
    """The difficulty class, as the policy's key: "1,<op>" for one event (E1: a probe answers 92 % of
    doublings at once, 16 % of subtractions), else "<steps>,<1 if a step doubles>"."""
    k, ops = problem["steps"], [op for op, _ in problem["ops"]]
    return f"1,{ops[0]}" if k == 1 else f"{k},{int('dbl' in ops)}"


# ---- traces written by a program ----------------------------------------------

def _equation(problem: dict, i: int, spaces: bool = True) -> str:
    (op, n), a, b = problem["ops"][i], problem["states"][i], problem["states"][i + 1]
    eq = f"{a} {SYMBOL[op]} {2 if n is None else n} = {b}"
    return eq if spaces else eq.replace(" ", "")


def trace_program(problem: dict, level: str, rng: random.Random) -> str:
    """The reasoning at `level`, written from the problem's solution, with varied wording."""
    k, thing, hero = problem["steps"], problem["thing"], problem["hero"]
    if level == "low":
        return "\n".join(_equation(problem, i, spaces=False) for i in range(k))
    high = level == "high"
    sentences = [rng.choice(HIGH_INTROS if high else INTROS).format(H=hero, c=count(problem["states"][0], thing))]
    if high:  # the story again, a sentence at a time: the sentence, then its effect
        i = 0
        for j, quote in enumerate(problem["sentences"]):
            if j == problem["at"]:
                effect = rng.choice(UNCHANGED).format(P=_pronouns(hero)[2], t=thing[1])
            else:
                effect = rng.choice(EFFECTS).format(eq=_equation(problem, i))
                i += 1
            sentences.append(f'"{quote}" {effect}')
        sentences.append(f"So {hero} has {count(problem['answer'], thing)}.")
        return " ".join(sentences)
    for i, (op, n) in enumerate(problem["ops"]):
        S, _ = _subject(rng, hero)
        sentence = rng.choice(MEDIUM[op]).format(S=S, n=n, clause=problem["clauses"][i], eq=_equation(problem, i))
        if ", so " in sentence and rng.random() < 0.2:
            sentence = sentence.replace(", so ", ": ")
        sentences.append(sentence)
    return " ".join(sentences)


# ---- checks -------------------------------------------------------------------

def normalize(text: str) -> str:
    """Plain ASCII operators, single spaces, no blank lines."""
    text = text.replace("×", "x").replace("−", "-").replace("–", "-")
    lines = (re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines())
    return "\n".join(line for line in lines if line)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


def style_ok(level: str, text: str) -> bool:
    """Does `text` have the shape of the reasoning of `level`? (What needs the problem is in verify_trace.)"""
    if level == "low":
        return re.fullmatch(r"\d+[+\-x/]\d+=\d+(\n\d+[+\-x/]\d+=\d+)*", text) is not None
    if "\n" in text or level not in ("medium", "high"):
        return False
    sentences = _sentences(text)
    if level == "medium":
        return (len(sentences) > 1 and "=" not in sentences[0] and "Check" not in text
                and not sentences[-1].startswith("So ")
                and all(len(s.split()) <= 14 and s.count("=") == 1 for s in sentences[1:]))
    quotes = text.count('"')
    return (quotes >= 2 and quotes % 2 == 0 and "Check" not in text
            and re.fullmatch(r"So \w+ has \d+ \w+( left| now| in all| then)?\.", sentences[-1]) is not None)


def verify_trace(problem: dict, text: str, level: str, tok=None) -> str | None:
    """None if the trace is good, else why not: forbidden text, a wrong equation, an invented
    number, a missing state, a wrong final result, the wrong style, a bad length (only with `tok`)."""
    k, states, answer = problem["steps"], problem["states"], problem["answer"]
    if FORBIDDEN.search(text):
        return "forbidden text"
    if NUMBER_WORDS.search(text):
        return "number word"
    results = []
    for left, right in EQUATION.findall(text):
        if arithmetic.calculator(left.replace("x", "*")) != right:
            return f"wrong equation: {left} = {right}"
        results.append(int(right))
    allowed = {int(n) for n in re.findall(r"\d+", problem["messages"][0]["content"])} | set(states) | {1, 2}
    invented = [n for n in map(int, re.findall(r"\d+", text)) if n not in allowed]
    if invented:
        return f"invented number: {invented[0]}"
    it = iter(results)
    if not all(s in it for s in states[1:]):  # every state, in order, is the result of an equation
        return "a state is missing"
    if results[-1] != answer:
        return "wrong final result"
    if not style_ok(level, text) or (level == "low" and text.count("\n") + 1 > k):
        return "style"
    if level == "high":
        sentences = _sentences(text)
        if not re.fullmatch(rf"So {problem['hero']} has {answer} ({'|'.join(problem['thing'])})( left| now| in all| then)?\.",
                            sentences[-1]):
            return "wrong conclusion"
        # every sentence of the story, in order and as worded, is quoted and followed by its effect
        parts = text.split('"')
        quotes, after, at = parts[1::2], parts[2::2], -1
        for j, sentence in enumerate(problem["sentences"]):
            if sentence not in quotes[at + 1:]:
                return "a sentence is missing"
            at = quotes.index(sentence, at + 1)
            if j == problem["at"] and not NOT_MATTER.search(after[at]):
                return "the distractor is not set aside"
            if j != problem["at"] and "=" not in after[at]:
                return "a sentence has no effect"
    if tok is not None:
        n = len(tok.encode(text))
        # high, measured on 1000 problems per k and split (k = 1..8): 33..81 tokens at k = 1, 166..250 at k = 8;
        # the bounds leave at least 40 tokens beyond the longest and 10 below the shortest
        lo, hi = {"low": (0, 8 * k + 6), "medium": (0, 24 * k + 12), "high": (14 * k + 10, 30 * k + 60)}[level]
        if not lo <= n <= hi:
            return f"length {n} outside {lo}..{hi}"
    return None


def dedupe_skeletons(traces: list[str], max_frac: float = 0.01) -> list[str]:
    """The traces, in order, with at most max_frac of the set for any one skeleton (a trace with
    every number replaced by #), so that the wording stays varied."""
    cap, seen, kept = max(1, int(max_frac * len(traces))), {}, []
    for trace in traces:
        skeleton = re.sub(r"\d+", "#", trace)
        seen[skeleton] = seen.get(skeleton, 0) + 1
        if seen[skeleton] <= cap:
            kept.append(trace)
    return kept


# ---- policy -------------------------------------------------------------------

def load_policy(path) -> dict:
    """{"tau": {level: threshold}, "p": {bucket: direct accuracy}, "n": {bucket: problems measured}}"""
    with open(path) as f:
        return json.load(f)


def think(level: str, problem: dict, policy: dict) -> bool:
    """Reason unless answering at once is reliable enough for this level: the direct accuracy of the
    problem's bucket must beat the level's threshold by 0.03. An unmeasured bucket thinks."""
    return decide(level, bucket(problem), policy)


def decide(level: str, key: str, policy: dict) -> bool:
    """think() for a bucket key."""
    if level == "none":
        return False
    p = policy["p"].get(key)
    return p is None or p < min(policy["tau"][level] + 0.03, 1.0)  # high: a perfect bucket may answer at once
