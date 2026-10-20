"""
모드별 튜터
==========

세션 핸들러는 인식 결과와 버튼 명령을 튜터에 넘기고, 튜터가 돌려준 메시지를 그대로 클라이언트에 보낸다.

  WordTutor      단어 연습 — 목표 단어를 맞히면 다음 단어, 틀리면 동작 설명
  SentenceTutor  상황별 문장 — 상대 대사에 맞는 문장을 글로스 순서대로 표현
  FreeTalkTutor  프리토킹 — 인식된 글로스를 모았다가(잠깐 멈추거나 '보내기') LLM 이 답한다
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Awaitable, Callable, List, Optional

from app.models.schemas.ws_messages import (
    LessonState,
    LessonTarget,
    LLMResponse,
    RecognitionResult,
    Utterance,
)
from app.services.avatar.controller import AvatarController
from app.services.tutor import curriculum as cur
from app.services.tutor import ksl_grammar as ksl
from app.services.tutor.llm import LLMClient, is_korean_only

logger = logging.getLogger(__name__)

PRAISE = ["정확해요!", "잘하셨어요!", "훌륭해요!", "완벽해요!", "아주 자연스러워요!", "멋지게 해내셨어요!"]


def _reply(text: str, kind: str, gloss: Optional[List[str]] = None) -> LLMResponse:
    gloss = gloss or []
    cmds = AvatarController.from_glosses(gloss) if gloss else AvatarController.text_to_commands(text)
    return LLMResponse(text=text, avatar_commands=cmds, kind=kind, gloss=gloss)


def _josa(word: str, pair: str) -> str:
    """받침에 따라 조사 고르기. pair: "을를" "이가" "은는" "이에요예요" """
    last = word[-1] if word else ""
    has_final = "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 != 0
    first, second = {"을를": ("을", "를"), "이가": ("이", "가"), "은는": ("은", "는"),
                     "이에요예요": ("이에요", "예요")}[pair]
    return word + (first if has_final else second)


def _q(word: str, pair: str) -> str:
    """'병원'을, '아프다'를 처럼 따옴표 + 조사"""
    return f"'{word}'" + _josa(word, pair)[len(word):]


def _clips(gloss: List[str]) -> List[str]:
    return [c for c in (cur.clip_for(g) for g in gloss) if c]


class BaseTutor:
    mode = "base"

    def __init__(self, scenario: str, llm: LLMClient) -> None:
        self.scenario = scenario
        self.llm = llm
        self.correct = 0
        self.attempts = 0
        # 오래 걸리는 답(LLM)을 나중에 보내는 콜백. 세션이 넣어 준다.
        self.emit: Optional[Callable[[list], Awaitable[None]]] = None

    async def opening(self) -> list:
        return []

    async def on_sentence(self, gloss: List[str]) -> list:
        """문장 인식 모델이 수어 한 문장을 글로스 순서로 읽었을 때 (프리토킹·상황 문장)"""
        return []

    def wants_sentence_only(self, vocab: set) -> bool:
        """True 면 단어 인식 결과는 튜터에 넘기지 않는다 (문장 모델이 목표를 다 알 때)"""
        return False

    async def on_sign(self, result: RecognitionResult) -> list:
        return []

    async def on_command(self, action: str) -> list:
        return []

    async def tick(self) -> list:
        """프레임마다 불림 (프리토킹의 말 끝 감지용)"""
        return []

    def summary(self) -> dict:
        return {"mode": self.mode, "correct": self.correct, "attempts": self.attempts}


# ─────────────────────── 단어 연습 ─────────────────────────────────

class WordTutor(BaseTutor):
    mode = "word"

    def __init__(self, scenario: str, llm: LLMClient, words: Optional[List[str]] = None) -> None:
        super().__init__(scenario, llm)
        deck = cur.deck(scenario)
        candidates = words or (deck["words"] if deck else [])
        self.title = deck["title"] if deck else "단어 연습"
        self.words = [w for w in candidates if cur.is_available(w)] or candidates or ["감사"]
        self.index = 0
        self.misses = 0

    @property
    def target(self) -> str:
        return self.words[self.index % len(self.words)]

    def _state(self, done: bool = False) -> LessonState:
        w = self.target
        return LessonState(
            mode=self.mode, title=self.title, step=min(self.index + 1, len(self.words)),
            total=len(self.words), correct=self.correct, attempts=self.attempts, done=done,
            target=LessonTarget(gloss=[w], text=w, description=cur.KSL_DESCRIPTIONS.get(w),
                                clips=_clips([w])),
        )

    async def opening(self) -> list:
        w = self.target
        return [self._state(), _reply(f"{_q(w, '을를')} 함께 해 볼까요? 제 동작을 먼저 보고 따라 해 주세요.", "prompt", [w])]

    async def on_sign(self, result: RecognitionResult) -> list:
        self.attempts += 1
        w = self.target
        if result.text == w:
            self.correct += 1
            self.misses = 0
            self.index += 1
            if self.index >= len(self.words):
                self.index = len(self.words) - 1
                return [self._state(done=True),
                        _reply(f"{random.choice(PRAISE)} 이번 묶음 {len(self.words)}개를 모두 마쳤어요.", "correct", ["감사"])]
            nxt = self.target
            return [self._state(), _reply(f"{random.choice(PRAISE)} 다음은 {_q(nxt, '이에요예요')}.", "correct", [nxt])]
        self.misses += 1
        return [self._state(), _reply(await self._feedback(w, result.text), "feedback", [w])]

    async def on_command(self, action: str) -> list:
        w = self.target
        if action == "skip":
            self.index = (self.index + 1) % len(self.words)
            return [self._state(), _reply(f"다음 단어 '{self.target}'로 넘어갈게요.", "prompt", [self.target])]
        if action in ("hint", "repeat"):
            desc = cur.KSL_DESCRIPTIONS.get(w, "제 동작을 천천히 다시 보여 드릴게요.")
            return [_reply(f"'{w}': {desc}", "hint", [w])]
        return []

    async def _feedback(self, target: str, recognized: str) -> str:
        desc = cur.KSL_DESCRIPTIONS.get(target)
        heard = f"'{recognized}'로 보였어요. " if recognized else ""
        if desc:
            return f"{heard}{_q(target, '은는')} 이렇게 해요: {desc}"
        text = await self.llm.complete(
            "당신은 한국 수어(KSL) 교정 튜터입니다. 반드시 한국어로 1~2문장만 답하고, "
            "손 모양·위치·움직임만 설명하세요. 동작을 지어내지 마세요.",
            [{"role": "user", "content": f"목표 수어 '{target}', 인식된 동작 '{recognized}'. '{target}' 동작을 설명해 주세요."}],
            max_tokens=120)
        if text and is_korean_only(text):
            return f"{heard}{text}"
        return f"{heard}'{target}' 동작을 천천히 다시 해 보세요."


# ─────────────────────── 상황별 문장 ───────────────────────────────

class SentenceTutor(BaseTutor):
    mode = "sentence"

    def __init__(self, scenario: str, llm: LLMClient) -> None:
        super().__init__(scenario, llm)
        self.sc = cur.scenario(scenario) or cur.SCENARIOS[0]
        self.turn = 0
        self.matched = 0

    @property
    def cur_turn(self) -> dict:
        return self.sc["turns"][self.turn]

    def _state(self, done: bool = False) -> LessonState:
        t = self.cur_turn
        return LessonState(
            mode=self.mode, title=self.sc["title"], step=self.turn + 1, total=len(self.sc["turns"]),
            correct=self.correct, attempts=self.attempts, done=done, partner_line=t["partner"],
            target=LessonTarget(gloss=t["gloss"], text=t["text"], clips=_clips(t["gloss"]),
                                matched=self.matched,
                                description=" → ".join(t["gloss"]) + " 순서로 표현해요."),
        )

    async def opening(self) -> list:
        t = self.cur_turn
        return [self._state(), _reply(t["partner"], "prompt")]

    def wants_sentence_only(self, vocab: set) -> bool:
        return all(g in vocab for g in self.cur_turn["gloss"])

    async def on_sentence(self, gloss: List[str]) -> list:
        t = self.cur_turn
        target = t["gloss"]
        if not gloss:
            return []
        self.attempts += 1
        # 목표 순서가 인식된 문장 안에 그대로 들어 있으면 성공 (앞뒤에 덧붙인 말은 허용)
        n = len(target)
        hit = any(gloss[i:i + n] == target for i in range(len(gloss) - n + 1))
        if not hit:
            self.matched = 0
            return [self._state(), _reply(
                f"'{' '.join(gloss)}'로 보였어요. '{t['text']}'는 {' → '.join(target)} 순서로 해 보세요.", "feedback", target)]
        self.correct += 1
        self.matched = 0
        if self.turn + 1 >= len(self.sc["turns"]):
            return [self._state(done=True),
                    _reply(f"{random.choice(PRAISE)} '{self.sc['title']}' 상황을 모두 마쳤어요.", "correct", ["감사"])]
        self.turn += 1
        return [self._state(), _reply(f"{random.choice(PRAISE)} {self.cur_turn['partner']}", "correct")]

    async def on_sign(self, result: RecognitionResult) -> list:
        self.attempts += 1
        t = self.cur_turn
        expected = t["gloss"][self.matched]
        if result.text != expected:
            self.matched = 0 if result.text != t["gloss"][0] else 1
            return [self._state(), _reply(
                (f"지금은 {_q(expected, '이가')} 나올 차례예요. " + (f"'{t['text']}'를 {' → '.join(t['gloss'])} 순서로 표현해 보세요." if len(t["gloss"]) > 1 else f"'{t['text']}'를 표현해 보세요.")),
                "feedback", [expected])]
        self.matched += 1
        if self.matched < len(t["gloss"]):
            return [self._state()]
        # 문장 완성
        self.correct += 1
        self.matched = 0
        if self.turn + 1 >= len(self.sc["turns"]):
            return [self._state(done=True),
                    _reply(f"{random.choice(PRAISE)} '{self.sc['title']}' 상황을 모두 마쳤어요.", "correct", ["감사"])]
        self.turn += 1
        return [self._state(), _reply(f"{random.choice(PRAISE)} {self.cur_turn['partner']}", "correct")]

    async def on_command(self, action: str) -> list:
        t = self.cur_turn
        if action == "skip":
            self.matched = 0
            if self.turn + 1 < len(self.sc["turns"]):
                self.turn += 1
            return [self._state(), _reply(self.cur_turn["partner"], "prompt")]
        if action in ("hint", "repeat"):
            return [_reply(f"'{t['text']}'는 {' → '.join(t['gloss'])} 순서예요. 제 동작을 보세요.", "hint", t["gloss"])]
        if action == "clear":
            self.matched = 0
            return [self._state()]
        return []


# ─────────────────────── 프리토킹 ───────────────────────────────────

FREE_TALK_SYSTEM = """당신은 청각장애인과 한국 수어로 대화하는 친절한 AI 튜터입니다.
사용자의 말은 수어 인식 결과(글로스 단어 나열)로 들어옵니다. 의도를 자연스럽게 해석해 대답하세요.
당신의 답은 화면에 한국어로 보이고, 동시에 아바타가 gloss 를 한국수어 문장으로 수어합니다.

출력 규칙:
1. 한국어 답(reply)은 1~2문장의 짧고 쉬운 말. 대화를 이어 가도록 가끔 쉬운 질문을 합니다.
2. gloss 는 reply 의 뜻 전체를 아래 [한국수어 문장 규칙]에 맞춘 완전한 수어 문장으로, 2~8개 단어.
   단어는 반드시 다음 목록에 있는 것만 씁니다 (목록에 없으면 비슷한 뜻의 목록 단어로 바꿉니다): {vocab}
3. 아래 JSON 한 줄로만 답합니다. 다른 글은 쓰지 않습니다.
   {"reply": "자연스러운 한국어 답", "gloss": ["수어", "단어", "순서"]}

{grammar}"""

PAUSE_TO_SEND_S = 2.5   # 마지막 인식 후 이만큼 조용하면 말이 끝난 것으로 본다


class FreeTalkTutor(BaseTutor):
    mode = "free_talk"

    def __init__(self, scenario: str, llm: LLMClient) -> None:
        super().__init__(scenario, llm)
        self.topic = cur.topic(scenario) or cur.FREE_TALK_TOPICS[0]
        self.buffer: List[str] = []
        self.last_sign_t = 0.0
        self.history: List[dict] = []
        self.turns = 0

    def _state(self) -> LessonState:
        return LessonState(mode=self.mode, title=self.topic["title"], step=self.turns, total=0,
                           correct=self.correct, attempts=self.attempts,
                           partner_line=self.history[-1]["content"] if self.history else self.topic["opener"])

    async def opening(self) -> list:
        opener = self.topic["opener"]
        self.history.append({"role": "assistant", "content": opener})
        return [self._state(), _reply(opener, "prompt", ["좋다"])]

    async def on_sign(self, result: RecognitionResult) -> list:
        self.attempts += 1
        if not self.buffer or self.buffer[-1] != result.text:
            self.buffer.append(result.text)
        self.last_sign_t = time.monotonic()
        return [Utterance(gloss=list(self.buffer))]

    async def on_sentence(self, gloss: List[str]) -> list:
        # 문장 모델이 한 문장을 통째로 읽었으면 단어 인식으로 모은 것 대신 그걸로 바로 답한다
        if not gloss:
            return []
        self.attempts += 1
        self.buffer = list(gloss)
        return await self._respond()

    async def tick(self) -> list:
        if self.buffer and time.monotonic() - self.last_sign_t > PAUSE_TO_SEND_S:
            return await self._respond()
        return []

    async def on_command(self, action: str) -> list:
        if action == "submit" and self.buffer:
            return await self._respond()
        if action == "clear":
            self.buffer = []
            return [Utterance(gloss=[])]
        if action in ("hint", "repeat") and self.history:
            last = self.history[-1]["content"]
            return [_reply(last, "hint")]
        return []

    async def _respond(self) -> list:
        said = list(self.buffer)
        self.buffer = []
        self.turns += 1
        self.correct += 1
        self.history.append({"role": "user", "content": " ".join(said)})
        if self.emit:
            # 답에 몇 초 걸리므로 영상 처리를 막지 않게 따로 돌리고, 그동안 '생각 중'을 띄운다
            asyncio.create_task(self._answer_later())
            return [Utterance(gloss=said, final=True, thinking=True)]
        return [Utterance(gloss=said, final=True)] + await self._answer()

    async def _answer(self) -> list:
        reply, gloss = await self._ask_llm()
        self.history.append({"role": "assistant", "content": reply})
        self.history = self.history[-12:]
        return [self._state(), _reply(reply, "reply", gloss)]

    async def _answer_later(self) -> None:
        try:
            await self.emit(await self._answer())
        except Exception:   # 세션이 이미 끝났으면 보낼 곳이 없다
            logger.exception("프리토킹 답 전송 실패")

    async def _ask_llm(self):
        # 선생님이 수어로 할 수 있는 단어 전체 (3000단어 사전이 있으면 그 전체)
        vocab = ", ".join(sorted(cur.sign_vocab()))
        recent = [w for m in self.history[-4:] for w in m["content"].split()]
        system = FREE_TALK_SYSTEM.replace("{vocab}", vocab).replace("{grammar}", ksl.prompt_block(recent))
        raw = await self.llm.complete(system, self.history, max_tokens=300)
        if raw:
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                data = json.loads(m.group(0)) if m else {}
                reply = str(data.get("reply", "")).strip()
                gloss = [str(g) for g in data.get("gloss", []) if cur.can_sign(str(g))][:8]
                if reply and is_korean_only(reply):
                    return reply, gloss
            except (ValueError, AttributeError):
                pass
        return self._rule_reply(self.history[-1]["content"].split())

    @staticmethod
    def _rule_reply(said: List[str]):
        """LLM 없이도 대화가 끊기지 않게 하는 간단한 규칙"""
        s = set(said)
        if s & {"아프다", "감기", "병원"}:
            return "많이 아프세요? 병원에 가서 검사를 받아 보세요.", ["아프다", "병원", "검사"]
        if s & {"감사", "죄송"}:
            return "괜찮아요! 또 이야기해요.", ["괜찮다"]
        if s & {"좋다", "즐겁다"}:
            return "좋다니 저도 즐거워요! 무슨 일이 있었어요?", ["좋다", "즐겁다"]
        if s & {"슬프다", "피곤하다", "화나다", "무섭다"}:
            return "그랬군요. 괜찮아요, 천천히 이야기해 주세요.", ["괜찮다"]
        if s & {"밥", "친구", "가족"}:
            return "좋네요! 가족이나 친구와 밥 먹는 건 즐거워요.", ["가족", "친구", "즐겁다"]
        return f"'{' '.join(said)}'라고 하셨군요. 조금 더 이야기해 줄래요?", []


def make_tutor(mode: str, scenario: str, llm: LLMClient, words: Optional[List[str]] = None) -> BaseTutor:
    if mode == "sentence":
        return SentenceTutor(scenario, llm)
    if mode == "free_talk":
        return FreeTalkTutor(scenario, llm)
    return WordTutor(scenario, llm, words)
