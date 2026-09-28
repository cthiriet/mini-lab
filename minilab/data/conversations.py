"""Chat conversations for midtraining and SFT, which have different jobs.

Midtraining teaches the chat format and the skills, at volume: one question, one
answer, no system prompt. Additions (with the scratchpad, or a calculator call when
tools are on), story requests, greetings.

SFT teaches *behavior*, from a small fixed set of conversations seen for a few epochs,
the way labs fine-tune on curated data. Everything in it is something midtraining
never shows:
- system prompts to obey (INSTRUCTIONS), each checked automatically by the eval;
- follow-up questions that refer to an earlier answer ("And add 25 to that?"), and
  new requests after an answer that don't (another addition, or something else);
- who the model is, and polite refusals for what it can't do.
They are mixed with plain conversations like midtraining's, so that the model
doesn't start refusing in-scope requests or obeying instructions nobody gave.

A conversation is a dict {"messages": [...OpenAI format...], "tools": [...] | None}.
"""

from __future__ import annotations

import random
import re
from typing import Iterator

from minilab.data import arithmetic

NAME = "mini"
# (user messages, assistant replies): any user message can get any reply of its group.
GREETINGS = [
    (["Hi", "Hi!", "Hello", "Hello!", "Hey", "hey there", "Good morning!"],
     ["Hello! How can I help you today?", "Hi there! Would you like a story, or some numbers to add?"]),
    (["Thanks!", "Thank you", "Thank you very much!", "thanks", "Great, thanks!"],
     ["You're welcome!", "Happy to help!"]),
    (["Bye", "Goodbye!", "See you later!", "Good night!"],
     ["Goodbye! Have a nice day.", "Bye! Come back for another story soon."]),
]
IDENTITY = [
    (["Who are you?", "What is your name?", "What's your name?", "Tell me about yourself.", "Who made you?"],
     [f"I'm {NAME}, a very small language model trained from scratch on a laptop by mini-lab.",
      f"My name is {NAME}. I'm a tiny language model: I can tell short stories and add numbers."]),
    (["What can you do?", "How can you help me?", "What are you good at?"],
     [f"I'm {NAME}: I can tell you short stories for children and add numbers, step by step or with a calculator.",
      f"I'm {NAME}, and I know how to tell little stories and how to add numbers. Try asking me for a story!"]),
]
OUT_OF_SCOPE = [  # things a tiny story-and-addition model can't do; many start like in-scope requests
    "What is the capital of {country}?", "Who is the president of {country}?", "Can you write {lang} code?",
    "Write a {lang} function that sorts a list.", "What's the weather like in {city} today?",
    "Translate this into {language}: good morning.", "How do I cook pasta?", "Who won the last World Cup?",
    "What is the meaning of life?", "Can you book a table for two tonight?", "What time is it in {city}?",
    "Explain quantum physics to me.", "What should I invest in?", "Can you help me with my taxes?",
    "Who wrote Romeo and Juliet?", "Who invented the telephone?", "Who is the richest person in the world?",
    "Tell me about the history of {country}.", "Tell me the latest sports scores.", "Tell me a fact about space.",
    "How far away is the moon?", "How do I fix a flat tire?", "How tall is the Eiffel Tower?",
    "How do I learn {language} fast?", "What is the tallest mountain in the world?", "What does DNA stand for?",
    "Can you draw a picture of a cat?", "Play some music for me.", "Summarize this article for me.",
    "What is 12 times 8?", "Where is {city}?", "When did World War II end?",
]
OUT_OF_SCOPE_EVAL = [  # held out: only the eval asks these
    "What is the population of {country}?", "Debug my {lang} program, please.", "Will it rain in {city} tomorrow?",
    "How do you say thank you in {language}?", "Who painted the Mona Lisa?", "What is the speed of light?",
    "Recommend a good movie for tonight.", "How many moons does Jupiter have?", "Tell me today's news.",
    "What is the best phone to buy?",
]
FILLERS = {"country": ["France", "Japan", "Brazil", "Canada", "Kenya", "India"],
           "city": ["Paris", "Tokyo", "London", "New York", "Berlin", "Sydney"],
           "lang": ["Python", "JavaScript", "C++", "Rust"],
           "language": ["Spanish", "French", "German", "Chinese"]}
REFUSALS = ["Sorry, I'm a tiny model: I can only tell short stories and add numbers.",
            "Sorry, I don't know about that. I can only tell short stories and add numbers.",
            "I'm sorry, I can't help with that. I only know how to tell stories and add numbers."]
INSTRUCTIONS = {  # system prompts taught by SFT, each with an automatic check (minilab.eval.tasks.grade)
    "number_only": "Answer with the number only.",
    "no_calculator": "Do not use the calculator.",
    "one_sentence": "Answer in one short sentence.",
    "sure": 'Start every answer with "Sure!".',
}
NEUTRAL_SYSTEM = ["You are a helpful assistant.", "You are mini, a small and friendly assistant."]
STORY_REQUESTS = ["Tell me a story.", "Can you tell me a story?", "Tell me a short story.",
                  "I want a story!", "Write a story for me.", "Once upon a time..."]
TOPIC_REQUESTS = ["Tell me a story about {t}.", "Can you tell me a story about {t}?",
                  "Write a short story about {t}.", "I want a story about {t}!"]
TOPICS = {"a dog": "dog", "a cat": "cat", "a bird": "bird", "a ball": "ball", "a tree": "tree",
          "the sun": "sun", "a boat": "boat", "a fish": "fish", "the park": "park", "a princess": "princess",
          "a dragon": "dragon", "a car": "car", "a bunny": "bunny", "the sea": "sea", "a cake": "cake"}


def mentions(text: str, word: str) -> bool:
    """Whole-word match, plural allowed: "cats" counts for "cat", "catch" doesn't."""
    return re.search(rf"\b{word}s?\b", text.lower()) is not None


def is_refusal(text: str) -> bool:
    return text.strip().lower().startswith(("sorry", "i'm sorry"))


def first_sentence(story: str, max_words: int = 25) -> str | None:
    """The opening sentence of a story, if it is a short and clean one (no dialogue)."""
    m = re.match(r"[^.!?\"\n]+[.!?]", story)
    return m.group(0) if m and len(m.group(0).split()) <= max_words else None


def fill(rng: random.Random, template: str) -> str:
    return template.format(**{k: rng.choice(v) for k, v in FILLERS.items()})


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


class StoryPool:
    """Short stories, indexed by topic, for "tell me a story (about X)" requests."""

    def __init__(self, stories: list[str], max_chars: int = 700):
        self.stories = [s for s in stories if len(s) <= max_chars] or stories
        # A story is "about" a topic if it shows up in the first sentence, not in passing:
        # the model learns to bring the topic in right away.
        self.by_topic = {w: [s for s in self.stories if mentions(s[:100], w)] for w in TOPICS.values()}

    def request(self, rng: random.Random) -> tuple[str, str]:
        """A story request (on a topic 70% of the time) and a matching story."""
        topic = rng.choice(list(TOPICS)) if rng.random() < 0.7 else None
        if topic and self.by_topic[TOPICS[topic]]:
            return rng.choice(TOPIC_REQUESTS).format(t=topic), rng.choice(self.by_topic[TOPICS[topic]])
        return rng.choice(STORY_REQUESTS), rng.choice(self.stories)


def single_turn(rng: random.Random, kind: str, digits: list[int], stories: StoryPool, tools: bool) -> list[dict]:
    """One question and its answer: the midtraining building block."""
    if kind == "arithmetic":
        return arithmetic.exchange(rng, digits, tools)[0]
    if kind == "story":
        request, story = stories.request(rng)
        return [user(request), assistant(story)]
    if kind == "greeting":
        users, replies = rng.choice(GREETINGS)
        return [user(rng.choice(users)), assistant(rng.choice(replies))]
    raise ValueError(f"unknown conversation kind: {kind}")


def midtrain_conversation(rng: random.Random, mix: dict[str, float], digits: list[int], stories: StoryPool,
                          tool_frac: float) -> dict:
    tools = rng.random() < tool_frac
    kind = rng.choices(list(mix), weights=list(mix.values()))[0]
    return {"messages": single_turn(rng, kind, digits, stories, tools), "tools": ["calculator"] if tools else None}


def instruction_conversation(rng: random.Random, instruction: str, digits: list[int], stories: StoryPool) -> dict:
    """A system prompt from INSTRUCTIONS and a request that shows it being followed."""
    tools = None
    if instruction == "number_only":  # still reasons in the scratchpad; only the answer changes
        messages = arithmetic.exchange(rng, digits, tools=False, number_only=True)[0]
    elif instruction == "no_calculator":  # the calculator is there, but the scratchpad is used
        messages, tools = arithmetic.exchange(rng, digits, tools=False)[0], ["calculator"]
    elif instruction == "one_sentence":
        for _ in range(50):  # most stories open with a short, clean sentence
            request, story = stories.request(rng)
            if first_sentence(story):
                break
        messages = [user(request), assistant(first_sentence(story) or "Once upon a time, there was a happy dog.")]
    elif instruction == "sure":
        messages = single_turn(rng, rng.choice(["arithmetic", "story", "greeting"]), digits, stories, False)
        messages[-1] = {**messages[-1], "content": "Sure! " + messages[-1]["content"]}
    else:
        raise ValueError(f"unknown instruction: {instruction}")
    return {"messages": [{"role": "system", "content": INSTRUCTIONS[instruction]}, *messages], "tools": tools}


def new_question_conversation(rng: random.Random, digits: list[int], stories: StoryPool, tools: bool) -> dict:
    """A new, unrelated addition after an earlier answer: it has its own operands. Every other
    multi-turn conversation is a follow-up that builds on the last total, and with only those
    the model learned that a second addition always starts from it ("766 + 989" after "The
    answer is 405." became 405 + 989). One to three earlier turns."""
    history = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        first = rng.choices(["arithmetic", "story", "greeting"], weights=[0.6, 0.2, 0.2])[0]
        turn = single_turn(rng, first, digits, stories, tools)
        if first == "story":  # a short one, so the new question still fits the context
            turn[-1] = assistant(" ".join(re.findall(r"[^.!?]+[.!?]", turn[-1]["content"])[:2]).strip()
                                 or turn[-1]["content"])
        history += turn
    messages = arithmetic.client_history(history, compact=rng.random() < 0.5)
    messages += arithmetic.exchange(rng, digits, tools)[0]
    return {"messages": messages, "tools": ["calculator"] if tools else None, "train_on": "last"}


def switch_conversation(rng: random.Random, digits: list[int], stories: StoryPool, tools: bool) -> dict:
    """Something else after one to three additions: a story, a greeting, who the model is, or a
    refusal, answered as if it came first. Every other conversation that goes on after an
    addition goes on with math, and the model learned that whatever follows an answer is more
    math ("Who are you?" after "The answer is 405." got a calculator call: 405 + 5)."""
    story = rng.random() < 0.4
    history, total = arithmetic.exchange(rng, digits, tools)
    for _ in range(0 if story else rng.choice([0, 1, 2])):  # a story needs the room
        turn, total = (arithmetic.followup(rng, total, [1, 2], tools) if rng.random() < 0.5
                       else arithmetic.exchange(rng, digits, tools))
        history += turn
    if story:
        request, reply = stories.request(rng)
    else:
        group = rng.choice(["greeting", "identity", "refusal"])
        if group == "refusal":
            request, reply = fill(rng, rng.choice(OUT_OF_SCOPE)), rng.choice(REFUSALS)
        else:
            users, replies = rng.choice(GREETINGS if group == "greeting" else IDENTITY)
            request, reply = rng.choice(users), rng.choice(replies)
    if rng.random() < 0.25:  # typed casually: "tell me a story about a dog"
        request = request[0].lower() + request[1:].rstrip(".!?")
    messages = arithmetic.client_history(history, compact=rng.random() < 0.5) + [user(request), assistant(reply)]
    return {"messages": messages, "tools": ["calculator"] if tools else None, "train_on": "last"}


def sft_conversation(rng: random.Random, mix: dict[str, float], digits: list[int], stories: StoryPool,
                     tool_frac: float) -> dict:
    tools = rng.random() < tool_frac
    kind = rng.choices(list(mix), weights=list(mix.values()))[0]
    if kind == "plain":  # like midtraining, sometimes under a neutral system prompt
        conv = midtrain_conversation(rng, {"arithmetic": 0.6, "story": 0.2, "greeting": 0.2}, digits, stories, tool_frac)
        if rng.random() < 0.3:
            conv["messages"].insert(0, {"role": "system", "content": rng.choice(NEUTRAL_SYSTEM)})
        return conv
    if kind == "instruction":  # "number only" is the hardest (copying the result out of the
        # scratchpad without the usual "a + b =" lead-in), so it gets twice the examples
        return instruction_conversation(rng, rng.choice(["number_only", *INSTRUCTIONS]), digits, stories)
    if kind == "followup":  # short numbers, so every operand stays within the trained lengths
        messages, total = arithmetic.exchange(rng, [1, 2], tools)
        compact = rng.random() < 0.5
        for _ in range(rng.randint(1, 2)):
            turn, total = arithmetic.followup(rng, total, [1, 2], tools)
            messages = arithmetic.client_history(messages, compact) + turn
        # the history's answers have no scratchpad: context to read, not answers to imitate
        return {"messages": messages, "tools": ["calculator"] if tools else None, "train_on": "last"}
    if kind == "new_question":
        return new_question_conversation(rng, digits, stories, tools)
    if kind == "switch":
        return switch_conversation(rng, digits, stories, tools)
    elif kind == "refusal":
        messages = [user(fill(rng, rng.choice(OUT_OF_SCOPE))), assistant(rng.choice(REFUSALS))]
    elif kind == "identity":
        users, replies = rng.choice(IDENTITY)
        messages = [user(rng.choice(users)), assistant(rng.choice(replies))]
    else:
        raise ValueError(f"unknown conversation kind: {kind}")
    return {"messages": messages, "tools": ["calculator"] if tools else None}


def midtrain_stream(seed: int, sc: dict, digits: list[int], stories: StoryPool) -> Iterator[dict]:
    """Endless, deterministic stream of midtraining conversations (sc: the config section)."""
    rng = random.Random(seed)
    while True:
        yield midtrain_conversation(rng, sc["mix"], sc.get("digits", digits), stories, sc.get("tool_frac", 0.3))


def sft_dataset(seed: int, size: int, sc: dict, digits: list[int], stories: StoryPool) -> list[dict]:
    """The fixed SFT set: `size` conversations, generated once (sc: the config section)."""
    rng = random.Random(seed)
    return [sft_conversation(rng, sc["mix"], sc.get("digits", digits), stories, sc.get("tool_frac", 0.3))
            for _ in range(size)]
