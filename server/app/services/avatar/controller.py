"""
Avatar Controller
=================

Maps Korean text → ordered list of AvatarCommand objects that the
Unreal Engine / MetaHuman client will play back.

Architecture
------------
Production path (future):
  Korean text → morpheme segmentation → sign unit lookup
  → MetaHuman animation clip sequence → blend commands

Current MVP path:
  Korean word match → pre-defined clip name → AvatarCommand

The clip names are placeholders that map 1-to-1 to animation assets
inside the Unreal Engine project.  The UE5 client reads avatar_commands
from the WebSocket LLMResponse and plays them through an AnimMontage queue.

Adding a new sign
-----------------
1. Record / rig the motion in UE5 MetaHuman.
2. Export the animation asset with clip name = one of the CLIP_LIBRARY keys.
3. Add the Korean word → clip mapping in SIGN_WORD_MAP.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.models.schemas.ws_messages import AvatarCommand


# ─────────────── animation clip library ─────────────────────────
# clip name → (blend_in, blend_out, default_speed)
CLIP_LIBRARY: Dict[str, Tuple[float, float, float]] = {
    # Greetings
    "KSL_hello":          (0.10, 0.10, 1.0),
    "KSL_thank_you":      (0.10, 0.12, 1.0),
    "KSL_sorry":          (0.10, 0.10, 1.0),
    "KSL_goodbye":        (0.08, 0.15, 1.0),
    # Hospital MVP vocabulary
    "KSL_hurt":           (0.10, 0.10, 1.0),
    "KSL_medicine":       (0.10, 0.10, 1.0),
    "KSL_doctor":         (0.10, 0.10, 1.0),
    "KSL_nurse":          (0.10, 0.10, 1.0),
    "KSL_help_me":        (0.08, 0.12, 1.1),
    "KSL_toilet":         (0.10, 0.10, 1.0),
    "KSL_water":          (0.10, 0.10, 1.0),
    "KSL_okay":           (0.08, 0.10, 1.0),
    # Feedback expressions
    "KSL_correct":        (0.05, 0.10, 1.2),
    "KSL_incorrect":      (0.05, 0.10, 0.9),
    "KSL_try_again":      (0.10, 0.10, 1.0),
    "KSL_slow_down":      (0.10, 0.15, 0.8),
    # Idle / transitions
    "AVT_idle":           (0.20, 0.20, 1.0),
    "AVT_nod":            (0.08, 0.08, 1.0),
    "AVT_thinking":       (0.15, 0.15, 0.9),
}

# ─────── Korean text → clip name mapping ─────────────────────────
# Keys can be substrings — first match wins.
SIGN_WORD_MAP: List[Tuple[str, str]] = [
    # Exact hospital vocabulary
    ("안녕하세요",   "KSL_hello"),
    ("안녕",        "KSL_hello"),
    ("감사합니다",  "KSL_thank_you"),
    ("감사",        "KSL_thank_you"),
    ("죄송합니다",  "KSL_sorry"),
    ("미안",        "KSL_sorry"),
    ("아프다",      "KSL_hurt"),
    ("아파",        "KSL_hurt"),
    ("통증",        "KSL_hurt"),
    ("약",          "KSL_medicine"),
    ("의사",        "KSL_doctor"),
    ("의사 선생님", "KSL_doctor"),
    ("간호사",      "KSL_nurse"),
    ("도와주세요",  "KSL_help_me"),
    ("도움",        "KSL_help_me"),
    ("화장실",      "KSL_toilet"),
    ("물",          "KSL_water"),
    ("괜찮아요",    "KSL_okay"),
    ("괜찮",        "KSL_okay"),
    # Feedback phrases
    ("잘 했",       "KSL_correct"),
    ("정확",        "KSL_correct"),
    ("틀렸",        "KSL_incorrect"),
    ("오류",        "KSL_incorrect"),
    ("다시",        "KSL_try_again"),
    ("천천히",      "KSL_slow_down"),
    ("고개",        "AVT_nod"),
]

# Facial expression palette
EXPRESSION_MAP: Dict[str, str] = {
    "KSL_hello":     "smile",
    "KSL_thank_you": "smile",
    "KSL_hurt":      "concern",
    "KSL_help_me":   "concern",
    "KSL_correct":   "happy",
    "KSL_incorrect": "neutral",
    "KSL_try_again": "encouraging",
}


class AvatarController:
    """
    Stateless helper — all methods are class-level.
    The WebSocket session calls text_to_commands() on every LLM reply.
    """

    @classmethod
    def text_to_commands(cls, text: str) -> List[AvatarCommand]:
        """
        Tokenise `text` by searching for known Korean keywords and building
        an ordered list of animation commands.

        Falls back to AVT_idle if no keyword is found.
        """
        commands: List[AvatarCommand] = []
        remaining = text

        while remaining:
            clip, consumed = cls._find_next_clip(remaining)
            if clip is None:
                break
            cmd = cls._make_command(clip)
            commands.append(cmd)
            # Advance past the consumed portion
            idx = remaining.find(consumed)
            if idx == -1:
                break
            remaining = remaining[idx + len(consumed):]

        if not commands:
            commands.append(cls._make_command("AVT_idle"))

        return commands

    @classmethod
    def _find_next_clip(cls, text: str) -> Tuple[Optional[str], str]:
        """
        Scan text left-to-right for the first matching keyword.
        Returns (clip_name, matched_substring) or (None, "").
        """
        best_pos = len(text)
        best_clip = None
        best_kw = ""

        for keyword, clip in SIGN_WORD_MAP:
            pos = text.find(keyword)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_clip = clip
                best_kw = keyword

        if best_clip is None:
            return None, ""
        return best_clip, best_kw

    @classmethod
    def _make_command(cls, clip: str) -> AvatarCommand:
        blend_in, blend_out, speed = CLIP_LIBRARY.get(
            clip, (0.10, 0.10, 1.0)
        )
        expression = EXPRESSION_MAP.get(clip)
        return AvatarCommand(
            clip=clip,
            blend_in=blend_in,
            blend_out=blend_out,
            speed=speed,
            expression=expression,
        )
