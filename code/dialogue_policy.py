#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-agnostic dialogue policy for English tutoring.

This module intentionally avoids scenario-specific fixes. It separates each
turn into a task type first, then builds a short prompt for that task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping


@dataclass
class TeachingSessionState:
    """Compact state extracted from chat history."""

    scenario_setup: str = ""
    practice_language: str = ""
    last_user_english: str = ""
    user_facts: List[str] = field(default_factory=list)

    @property
    def has_roleplay(self) -> bool:
        return bool(self.scenario_setup)


class DialoguePolicy:
    """Build prompts for role-play, correction, and explanation tasks."""

    MAX_DEFAULT_HISTORY_MESSAGES = 12

    EXPLANATION_PATTERNS = (
        r"为什么",
        r"为啥",
        r"怎么(说|用|表达)",
        r"能不能",
        r"可不可以",
        r"区别",
        r"意思",
        r"用法",
        r"\bwhy\b",
        r"\bgrammar\b",
        r"\bmeaning\b",
        r"\bdifference\b",
        r"\bhow (do|should|can) i say\b",
    )

    CORRECTION_PATTERNS = (
        r"纠正",
        r"改一下",
        r"语法错误",
        r"哪里错",
        r"自然吗",
        r"对吗",
        r"地道",
        r"\bcorrect\b",
        r"\bfix\b",
        r"\bnatural\b",
        r"\bbetter way\b",
    )

    SCENE_SETUP_PATTERNS = (
        r"你是.*(教练|老师|店员|服务员|工作人员|面试官|医生|前台|导游|同事|客户|柜台)",
        r"扮演",
        r"模拟",
        r"场景",
        r"role.?play",
        r"scenario",
        r"pretend",
        r"act as",
    )

    LANGUAGE_SWITCH_EN_PATTERNS = (
        r"用英语",
        r"说英语",
        r"\bin english\b",
        r"\bspeak english\b",
    )

    LANGUAGE_SWITCH_ZH_PATTERNS = (
        r"用中文",
        r"说中文",
        r"\bin chinese\b",
        r"\bspeak chinese\b",
    )

    @classmethod
    def build_chat_prompt(
        cls,
        history: Iterable[Mapping[str, str]] | None,
        user_text: str,
        max_history_messages: int | None = None,
    ) -> str:
        normalized_user_text = (user_text or "").strip()
        prompt_history = cls._without_duplicate_latest_user(
            cls._normalized_history(history or []),
            normalized_user_text,
        )
        max_messages = max_history_messages or cls.MAX_DEFAULT_HISTORY_MESSAGES
        prompt_history = prompt_history[-max(2, max_messages) :]

        state = cls.extract_state(prompt_history, normalized_user_text)
        mode = cls.classify_user_intent(normalized_user_text, prompt_history)

        if mode == "correction":
            return cls._build_correction_prompt(state, normalized_user_text)
        if mode == "explanation":
            return cls._build_explanation_prompt(state, prompt_history, normalized_user_text)
        return cls._build_roleplay_prompt(state, prompt_history, normalized_user_text, mode)

    @classmethod
    def ensure_system_prompt(cls, prompt_or_text: str) -> str:
        text = prompt_or_text or ""
        if "<|im_start|>system" in text:
            return text
        user_text = cls.extract_latest_user_text(text)
        return cls.build_chat_prompt([], user_text)

    @classmethod
    def classify_user_intent(cls, user_text: str, history: Iterable[Mapping[str, str]] | None = None) -> str:
        text = (user_text or "").strip()
        normalized_history = cls._normalized_history(history or [])
        state = cls.extract_state(normalized_history, text)

        if cls._matches_any(text, cls.CORRECTION_PATTERNS):
            return "correction"
        if cls._matches_any(text, cls.EXPLANATION_PATTERNS):
            return "explanation"
        if cls._matches_any(text, cls.SCENE_SETUP_PATTERNS):
            return "scene_setup"
        if cls._is_language_switch_to_english(text) or cls._is_language_switch_to_chinese(text):
            return "language_switch"
        if state.has_roleplay:
            return "roleplay"

        has_ascii_words = bool(re.search(r"[A-Za-z]{2,}", text))
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", text))
        if has_ascii_words and not has_cjk:
            return "roleplay"
        if has_ascii_words and has_cjk:
            return "mixed_practice"
        return "general_tutor"

    @classmethod
    def generation_config(cls, user_text: str, history: Iterable[Mapping[str, str]] | None = None) -> dict:
        mode = cls.classify_user_intent(user_text, history or [])
        if mode == "roleplay":
            return {"max_new_tokens": 140, "temperature": 0.30, "top_p": 0.80, "repetition_penalty": 1.12}
        if mode == "correction":
            return {"max_new_tokens": 220, "temperature": 0.20, "top_p": 0.75, "repetition_penalty": 1.08}
        if mode in {"explanation", "mixed_practice"}:
            return {"max_new_tokens": 280, "temperature": 0.35, "top_p": 0.82, "repetition_penalty": 1.10}
        return {"max_new_tokens": 220, "temperature": 0.40, "top_p": 0.85, "repetition_penalty": 1.12}

    @classmethod
    def build_context_card(
        cls,
        history: Iterable[Mapping[str, str]] | None,
        user_text: str,
        mode: str,
    ) -> str:
        state = cls.extract_state(cls._normalized_history(history or []), user_text)
        return cls._state_card(state, mode)

    @classmethod
    def extract_state(
        cls,
        history: Iterable[Mapping[str, str]],
        latest_user_text: str = "",
    ) -> TeachingSessionState:
        normalized = cls._normalized_history(history)
        state = TeachingSessionState()

        for message in normalized:
            if message.get("role") != "user":
                continue
            content = message.get("content", "").strip()
            if not content:
                continue
            if cls._matches_any(content, cls.SCENE_SETUP_PATTERNS):
                state.scenario_setup = cls._compact(content, 220)
            if cls._is_language_switch_to_english(content):
                state.practice_language = "English"
            elif cls._is_language_switch_to_chinese(content):
                state.practice_language = "Chinese"
            if cls._looks_like_english_practice(content):
                state.last_user_english = cls._compact(content, 220)
            if not cls._is_meta_instruction(content):
                state.user_facts.append(cls._compact(content, 160))

        latest = (latest_user_text or "").strip()
        if latest:
            if cls._matches_any(latest, cls.SCENE_SETUP_PATTERNS):
                state.scenario_setup = cls._compact(latest, 220)
            if cls._is_language_switch_to_english(latest):
                state.practice_language = "English"
            elif cls._is_language_switch_to_chinese(latest):
                state.practice_language = "Chinese"
            if cls._looks_like_english_practice(latest):
                state.last_user_english = cls._compact(latest, 220)
            if not cls._is_meta_instruction(latest):
                state.user_facts.append(cls._compact(latest, 160))

        state.user_facts = state.user_facts[-5:]
        return state

    @staticmethod
    def extract_latest_user_text(prompt_or_text: str) -> str:
        text = prompt_or_text or ""
        matches = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return text.strip()

    @staticmethod
    def extract_prompt_history(prompt_or_text: str) -> List[dict]:
        text = prompt_or_text or ""
        messages = re.findall(
            r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>",
            text,
            flags=re.DOTALL,
        )
        return [
            {"role": role, "content": content.strip()}
            for role, content in messages
            if content.strip()
        ]

    @classmethod
    def direct_response(cls, prompt_or_text: str) -> str:
        history = cls.extract_prompt_history(prompt_or_text)
        latest_user_text = cls.extract_latest_user_text(prompt_or_text)
        state = cls.extract_state(history, latest_user_text)
        if cls.classify_user_intent(latest_user_text, history) == "language_switch" and state.has_roleplay:
            return "Sure. Let's continue in English. How can I help you with this step?"
        return ""

    @classmethod
    def response_needs_retry(cls, prompt_or_text: str, response_text: str) -> bool:
        response = (response_text or "").strip()
        if not response:
            return True
        if cls._looks_like_prompt_leak(response):
            return True

        history = cls.extract_prompt_history(prompt_or_text)
        latest_user_text = cls.extract_latest_user_text(prompt_or_text)
        mode = cls.classify_user_intent(latest_user_text, history)
        state = cls.extract_state(history, latest_user_text)

        if state.has_roleplay and cls._is_bad_generic_assistant(response):
            return True
        if mode == "explanation" and not re.search(r"[\u4e00-\u9fff]", response):
            return True
        if mode == "correction" and not cls._looks_like_correction_response(response):
            return True
        if mode == "roleplay" and (state.practice_language == "English" or state.has_roleplay):
            if cls._mostly_chinese(response):
                return True
        return False

    @classmethod
    def fallback_response(cls, prompt_or_text: str) -> str:
        history = cls.extract_prompt_history(prompt_or_text)
        latest_user_text = cls.extract_latest_user_text(prompt_or_text)
        mode = cls.classify_user_intent(latest_user_text, history)
        state = cls.extract_state(history, latest_user_text)

        if mode == "correction":
            target = state.last_user_english or cls._extract_target_from_prompt(prompt_or_text)
            if target:
                return (
                    f"更自然的说法：{target}\n"
                    "说明：我找到了你上一句英文，但模型没有稳定生成纠错结果。请再发一次或换一句，我会继续纠正。"
                )
            return "请把你想纠正的英文句子发给我，我会给出更自然的说法和简短说明。"
        if mode == "explanation":
            return "这里是在问英语表达的用法。请把具体短语或句子发给我，我会用中文简短解释。"
        if state.has_roleplay:
            return "Sure. Let's continue. What would you like to do next?"
        return "请告诉我你想练习哪类英语场景。"

    @staticmethod
    def build_retry_prompt(prompt_or_text: str, rejected_response: str) -> str:
        prompt = (prompt_or_text or "").rstrip()
        assistant_marker = "<|im_start|>assistant"
        if prompt.endswith(f"{assistant_marker}\n"):
            prompt = prompt[: -len(f"{assistant_marker}\n")].rstrip()
        instruction = (
            "The previous draft was rejected. Output only the final assistant reply. "
            "Follow the task format exactly and do not repeat rules."
        )
        return f"{prompt}\n<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    @classmethod
    def normalize_history(cls, history: Iterable[Mapping[str, str]]) -> List[dict]:
        return cls._normalized_history(history)

    @classmethod
    def _build_correction_prompt(cls, state: TeachingSessionState, user_text: str) -> str:
        target = state.last_user_english
        target_line = target or "(no English sentence found)"
        return (
            "<|im_start|>system\n"
            "You are an English grammar correction tutor for Chinese learners.\n"
            "Task: correct the target English sentence only.\n"
            "Output format exactly:\n"
            "更自然的说法：<corrected English>\n"
            "说明：<brief Chinese explanation of the key issue>\n"
            "Do not continue role-play. Do not ask for the sentence if TARGET is provided.\n"
            f"TARGET: {target_line}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @classmethod
    def _build_explanation_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
    ) -> str:
        context = cls._state_card(state, "explanation")
        return (
            "<|im_start|>system\n"
            "You are an English usage explainer for Chinese learners.\n"
            "Answer in Chinese. Be brief and practical. Use 1-2 English examples.\n"
            f"{context}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @classmethod
    def _build_roleplay_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
        mode: str,
    ) -> str:
        system_prompt = (
            "You are an English speaking coach for Chinese learners.\n"
            f"{cls._state_card(state, mode)}\n"
            "Rules:\n"
            "- If role-play is active, play your assigned role in that scenario.\n"
            "- Reply in English when the practice language is English.\n"
            "- Preserve user-provided facts and corrections.\n"
            "- Ask only one necessary next question.\n"
            "- Keep it natural and concise.\n"
            "- Output only the assistant reply.\n"
        )
        parts: List[str] = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "").strip()
            if not content:
                continue
            if role == "assistant" and state.has_roleplay and cls._is_bad_generic_assistant(content):
                continue
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _state_card(state: TeachingSessionState, mode: str) -> str:
        lines = ["State:"]
        lines.append(f"- mode: {mode}")
        lines.append(f"- roleplay: {'yes' if state.has_roleplay else 'no'}")
        if state.scenario_setup:
            lines.append(f"- setup: {state.scenario_setup}")
        if state.practice_language:
            lines.append(f"- practice_language: {state.practice_language}")
        if state.last_user_english:
            lines.append(f"- last_user_english: {state.last_user_english}")
        if state.user_facts:
            lines.append("- user_facts:")
            for fact in state.user_facts[-4:]:
                lines.append(f"  * {fact}")
        return "\n".join(lines)

    @classmethod
    def _without_duplicate_latest_user(cls, history: List[dict], user_text: str) -> List[dict]:
        if history and history[-1].get("role") == "user" and history[-1].get("content", "").strip() == user_text:
            return history[:-1]
        return history

    @staticmethod
    def _normalized_history(history: Iterable[Mapping[str, str]]) -> List[dict]:
        normalized: List[dict] = []
        for message in history:
            role = str(message.get("role", "user")).strip().lower()
            if role not in {"user", "assistant"}:
                role = "assistant" if role in {"ai", "bot"} else "user"
            content = str(message.get("content", "")).strip()
            if content:
                normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _matches_any(text: str, patterns: Iterable[str]) -> bool:
        return any(re.search(pattern, text or "", flags=re.IGNORECASE) for pattern in patterns)

    @classmethod
    def _is_meta_instruction(cls, text: str) -> bool:
        stripped = (text or "").strip()
        lowered = stripped.lower()
        return (
            cls._matches_any(stripped, cls.SCENE_SETUP_PATTERNS)
            or cls._is_language_switch_to_english(stripped)
            or cls._is_language_switch_to_chinese(stripped)
            or lowered in {"in english", "speak english", "in chinese", "speak chinese"}
        )

    @classmethod
    def _is_language_switch_to_english(cls, text: str) -> bool:
        return cls._matches_any(text or "", cls.LANGUAGE_SWITCH_EN_PATTERNS)

    @classmethod
    def _is_language_switch_to_chinese(cls, text: str) -> bool:
        return cls._matches_any(text or "", cls.LANGUAGE_SWITCH_ZH_PATTERNS)

    @staticmethod
    def _looks_like_english_practice(text: str) -> bool:
        text = (text or "").strip()
        if not re.search(r"[A-Za-z]{2,}", text):
            return False
        cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_count = len(re.findall(r"[A-Za-z]", text))
        return latin_count >= max(4, cjk_count)

    @staticmethod
    def _is_bad_generic_assistant(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text or "").strip().lower()
        return normalized in {
            "hello! how can i help you?",
            "hello, how can i help you?",
            "hi! how can i help you?",
            "hi, how can i help you?",
            "how can i help you?",
        }

    @staticmethod
    def _looks_like_prompt_leak(text: str) -> bool:
        lowered = (text or "").lower()
        leak_markers = (
            "rules:",
            "state:",
            "mode:",
            "target:",
            "output format exactly",
            "talking about non-related topics",
            "previous draft was rejected",
        )
        marker_hits = sum(1 for marker in leak_markers if marker in lowered)
        bullet_lines = len(re.findall(r"(?m)^\s*-\s+", text or ""))
        return marker_hits >= 1 or bullet_lines >= 3

    @staticmethod
    def _looks_like_correction_response(text: str) -> bool:
        return "更自然" in (text or "") and "说明" in (text or "")

    @staticmethod
    def _extract_target_from_prompt(prompt_or_text: str) -> str:
        match = re.search(r"^TARGET:\s*(.+)$", prompt_or_text or "", flags=re.MULTILINE)
        if not match:
            return ""
        target = match.group(1).strip()
        if target == "(no English sentence found)":
            return ""
        return target

    @staticmethod
    def _mostly_chinese(text: str) -> bool:
        letters = re.findall(r"[A-Za-z]", text or "")
        cjk = re.findall(r"[\u4e00-\u9fff]", text or "")
        return len(cjk) >= max(4, len(letters))

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
