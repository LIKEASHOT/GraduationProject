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
    repeat_user_answer: bool = True
    teacher_should_ask: bool = False
    correction_mode: str = ""
    user_facts: List[str] = field(default_factory=list)
    recent_ai_questions: List[str] = field(default_factory=list)

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
        r"练习.*(订机票|订酒店|点餐|购物|问路|值机|面试|看病|入住|退房)",
        r"(订机票|订酒店|点餐|购物|问路|值机|面试|看病|入住|退房)吧",
        r"(订|预订|买).*(机票|酒店|房间|票)",
        r"(餐厅|机场|酒店|商店|医院|面试|旅行)",
        r"role.?play",
        r"scenario",
        r"pretend",
        r"act as",
        r"practice.*(booking|ordering|shopping|airport|hotel|interview|restaurant)",
        r"book.*(flight|hotel|room|ticket)",
        r"order.*(food|meal|drink)",
    )

    LANGUAGE_SWITCH_EN_PATTERNS = (
        r"用英语",
        r"说英语",
        r"换成英语",
        r"切到英语",
        r"改成英语",
        r"\bin english\b",
        r"\bspeak english\b",
    )

    LANGUAGE_SWITCH_ZH_PATTERNS = (
        r"用中文",
        r"说中文",
        r"换成中文",
        r"切到中文",
        r"改成中文",
        r"讲中文",
        r"\bin chinese\b",
        r"\bspeak chinese\b",
    )

    NO_REPEAT_PATTERNS = (
        r"不要重复",
        r"不要.*重复",
        r"不要.*复读",
        r"不用重复",
        r"别重复",
        r"别.*复读",
        r"一直重复",
        r"复读",
        r"不要复述",
        r"不用复述",
        r"\bdon'?t repeat\b",
        r"\bdo not repeat\b",
        r"\bno need to repeat\b",
        r"\bdon'?t need to repeat\b",
    )

    TEACHER_ASK_PATTERNS = (
        r"你来问",
        r"你问我",
        r"你应该问我",
        r"你是老师.*问",
        r"老师.*问我",
        r"\byou should ask me\b",
        r"\byou ask me\b",
        r"\bask me questions?\b",
    )

    STOP_CORRECTION_PATTERNS = (
        r"停止纠错",
        r"不要纠错",
        r"先不纠错",
        r"\bstop correcting\b",
        r"\bdon'?t correct\b",
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
        if mode == "language_switch":
            return cls._build_language_switch_prompt(state, prompt_history, normalized_user_text)
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
        if mode == "language_switch":
            return {"max_new_tokens": 90, "temperature": 0.20, "top_p": 0.75, "repetition_penalty": 1.08}
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
            if message.get("role") == "assistant":
                state.recent_ai_questions.extend(cls._extract_questions(message.get("content", "")))
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content", "").strip()
            if not content:
                continue
            if cls._matches_any(content, cls.NO_REPEAT_PATTERNS):
                state.repeat_user_answer = False
            if cls._matches_any(content, cls.TEACHER_ASK_PATTERNS):
                state.teacher_should_ask = True
            if cls._matches_any(content, cls.STOP_CORRECTION_PATTERNS):
                state.correction_mode = "off"
            elif cls._matches_any(content, cls.CORRECTION_PATTERNS):
                state.correction_mode = "gentle"
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
            if cls._matches_any(latest, cls.NO_REPEAT_PATTERNS):
                state.repeat_user_answer = False
            if cls._matches_any(latest, cls.TEACHER_ASK_PATTERNS):
                state.teacher_should_ask = True
            if cls._matches_any(latest, cls.STOP_CORRECTION_PATTERNS):
                state.correction_mode = "off"
            elif cls._matches_any(latest, cls.CORRECTION_PATTERNS):
                state.correction_mode = "gentle"
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
        state.recent_ai_questions = cls._dedupe_keep_order(state.recent_ai_questions)[-5:]
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
        # Keep normal dialogue generation model-driven. This hook is only kept
        # for future non-generative emergency responses, not for language choice.
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
        if mode == "language_switch":
            if cls._is_language_switch_to_chinese(latest_user_text) and not cls._contains_chinese(response):
                return True
            if cls._is_language_switch_to_english(latest_user_text) and cls._mostly_chinese(response):
                return True
        if not state.repeat_user_answer and cls._starts_with_user_echo(response):
            return True
        if state.teacher_should_ask and cls._asks_user_to_drive(response):
            return True
        if cls._repeats_recent_question(response, state.recent_ai_questions):
            return True
        if mode == "roleplay" and (state.practice_language == "English" or state.has_roleplay):
            if cls._is_chinese_meta_turn(latest_user_text):
                return False
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
            target = cls._extract_explanation_target(latest_user_text)
            if target:
                return cls._fallback_word_explanation(target)
            return "你是在问英语表达的用法。请把你想问的词或短语再说一遍，我会用中文简短解释。"
        if mode == "language_switch":
            if cls._is_language_switch_to_chinese(latest_user_text):
                return "好的，我会用中文继续。你想继续刚才的练习，还是先解释一下哪里不清楚？"
            if cls._is_language_switch_to_english(latest_user_text):
                return "Okay, I will continue in English. What would you like to practice next?"
        if state.has_roleplay or mode in {"roleplay", "mixed_practice"}:
            return cls._fallback_roleplay_response(state, latest_user_text)
        return "请告诉我你想练习哪类英语场景。"

    @staticmethod
    def build_retry_prompt(prompt_or_text: str, rejected_response: str) -> str:
        prompt = (prompt_or_text or "").rstrip()
        assistant_marker = "<|im_start|>assistant"
        if prompt.endswith(f"{assistant_marker}\n"):
            prompt = prompt[: -len(f"{assistant_marker}\n")].rstrip()
        history = DialoguePolicy.extract_prompt_history(prompt)
        latest_user_text = DialoguePolicy.extract_latest_user_text(prompt)
        if DialoguePolicy.classify_user_intent(latest_user_text, history) == "language_switch":
            if DialoguePolicy._is_language_switch_to_chinese(latest_user_text):
                instruction = (
                    "The previous draft ignored the requested language. "
                    "Now answer naturally in Chinese. Output only the assistant reply."
                )
            else:
                instruction = (
                    "The previous draft ignored the requested language. "
                    "Now answer naturally in English. Output only the assistant reply."
                )
        elif DialoguePolicy.classify_user_intent(latest_user_text, history) == "explanation":
            instruction = (
                "The previous draft was rejected. Answer the user's English-usage question in Chinese only. "
                "Do not reveal or quote private context. Do not output labels such as State, mode, roleplay, setup, "
                "practice_language, or user_facts. Output only the final assistant reply."
            )
        else:
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
        target = cls._extract_explanation_target(user_text)
        target_line = target or "(infer from the user question and recent context)"
        return (
            "<|im_start|>system\n"
            "You are an English usage explainer for Chinese learners.\n"
            "Answer in Chinese. Be brief and practical. Use 1-2 English examples.\n"
            "If the user asks the meaning of a word or phrase, explain it directly; do not ask them to provide the phrase again.\n"
            "Use the private context only to understand the conversation. Never quote it.\n"
            "Never output labels such as State, mode, roleplay, setup, practice_language, or user_facts.\n"
            "If the user asks about a phrase that may be ASR-misrecognized, infer the most likely English phrase from context and explain that phrase.\n"
            f"Likely explanation target: {target_line}\n"
            "Private context:\n"
            f"{context}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @classmethod
    def _build_language_switch_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
    ) -> str:
        target_language = "Chinese" if cls._is_language_switch_to_chinese(user_text) else "English"
        if target_language == "Chinese":
            language_rule = "The user is asking you to speak Chinese. Reply in natural Chinese."
            style_rule = "可以简短确认切换，并自然衔接当前练习；不要再说 Let's continue in English。"
        else:
            language_rule = "The user is asking you to speak English. Reply in natural English."
            style_rule = "Briefly confirm the switch and continue the current practice naturally."

        system_prompt = (
            "You are an English speaking coach for Chinese learners.\n"
            f"{cls._state_card(state, 'language_switch')}\n"
            "Task: handle a language-switch request, not a new scenario.\n"
            f"{language_rule}\n"
            f"{style_rule}\n"
            "Keep the previous scenario and user facts if role-play is active.\n"
            "Do not repeat the user's previous answer unless you are correcting it.\n"
            "Do not ask a question that was already asked recently.\n"
            "Ask at most one relevant next question.\n"
            "Output only the assistant reply.\n"
        )
        parts: List[str] = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for message in history[-8:]:
            role = message.get("role", "user")
            content = message.get("content", "").strip()
            if content:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

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
            "- Reply in Chinese when the practice language is Chinese.\n"
            "- Preserve user-provided facts and corrections.\n"
            "- Treat user speech as ASR text: infer obvious speech-recognition mistakes from context before responding.\n"
            "- Do not mechanically repeat or paraphrase the user's answer. Use brief acknowledgement only.\n"
            "- If the user asked not to repeat their answers, never start with 'You said', 'Great, you said', or a full restatement.\n"
            "- Do not ask the same question or same topic that appears in recent_ai_questions.\n"
            "- If the user says you are the teacher, you should ask the next practice question instead of asking the user to ask you.\n"
            "- If the user says a topic is already discussed, move to a new angle or follow-up topic.\n"
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
        lines.append(f"- repeat_user_answer: {'yes' if state.repeat_user_answer else 'no'}")
        if state.teacher_should_ask:
            lines.append("- teacher_should_ask: yes")
        if state.correction_mode:
            lines.append(f"- correction_mode: {state.correction_mode}")
        if state.last_user_english:
            lines.append(f"- last_user_english: {state.last_user_english}")
        if state.recent_ai_questions:
            lines.append("- recent_ai_questions:")
            for question in state.recent_ai_questions[-4:]:
                lines.append(f"  * {question}")
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
            or cls._matches_any(stripped, cls.NO_REPEAT_PATTERNS)
            or cls._matches_any(stripped, cls.TEACHER_ASK_PATTERNS)
            or cls._matches_any(stripped, cls.STOP_CORRECTION_PATTERNS)
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
            "sure. let's continue. what would you like to do next?",
            "okay. let's continue. what would you like to do next?",
            "ok. let's continue. what would you like to do next?",
        }

    @staticmethod
    def _looks_like_prompt_leak(text: str) -> bool:
        raw = text or ""
        lowered = raw.lower()
        normalized = re.sub(r"[*_`#>\-]+", " ", lowered)
        normalized = re.sub(r"\s+", " ", normalized)
        leak_patterns = (
            r"\brules\s*:",
            r"\bstate\s*:",
            r"\bmode\s*:",
            r"\broleplay\s*:",
            r"\bsetup\s*:",
            r"\bpractice[_\s-]*language\s*:",
            r"\buser[_\s-]*facts\s*:",
            r"\brecent[_\s-]*ai[_\s-]*questions\s*:",
            r"\brepeat[_\s-]*user[_\s-]*answer\s*:",
            r"\bteacher[_\s-]*should[_\s-]*ask\s*:",
            r"\btarget\s*:",
            r"output format exactly",
            r"private context",
            r"talking about non-related topics",
            r"previous draft was rejected",
        )
        marker_hits = sum(1 for pattern in leak_patterns if re.search(pattern, normalized))
        bullet_lines = len(re.findall(r"(?m)^\s*-\s+", text or ""))
        return marker_hits >= 1 or bullet_lines >= 3

    @staticmethod
    def _looks_like_correction_response(text: str) -> bool:
        return "更自然" in (text or "") and "说明" in (text or "")

    @classmethod
    def _fallback_roleplay_response(cls, state: TeachingSessionState, latest_user_text: str) -> str:
        text = (latest_user_text or "").strip()
        lowered = text.lower()

        if cls._matches_any(text, cls.NO_REPEAT_PATTERNS):
            return "好的，我不再重复你的回答。Let's continue: What date would you like to fly?"

        if cls._matches_any(text, cls.TEACHER_ASK_PATTERNS):
            return "Okay, I'll ask the questions. What date would you like to travel?"

        if "pay" in lowered or "ticket" in lowered:
            return "Of course. You can pay by card or cash. Which payment method would you prefer?"

        if re.search(r"\bbook\b|\bflight\b|\bticket\b|\bplane\b|\bairport\b", lowered):
            if re.search(r"\bfrom\b.+\bto\b", lowered):
                return "Good. What date would you like to travel?"
            return "Sure. Where would you like to fly from and to?"

        if re.search(r"\bchicken\b|\brice\b|\brestaurant\b|\border\b|\bfood\b", lowered):
            return "Sure. Would you like anything to drink with your meal?"

        if re.search(r"\bcold\b|\bdrink\b|\bwater\b|\bcoffee\b|\btea\b", lowered):
            return "Sure. Would you like a cold drink or something warm?"

        if re.search(r"\bweather\b|\brain\b|\braining\b|\braining\b|\bsunny\b|\bcold\b|\bhot\b", lowered):
            return "Let's continue with the weather. What did you do when it started raining?"

        if state.teacher_should_ask:
            return "Okay, I'll lead the practice. Can you answer this in English: what is your next request?"

        return "Let's continue with a new detail. What information do you need next?"

    @classmethod
    def _is_chinese_meta_turn(cls, text: str) -> bool:
        stripped = text or ""
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", stripped))
        if not has_cjk:
            return False
        return (
            cls._matches_any(stripped, cls.NO_REPEAT_PATTERNS)
            or cls._matches_any(stripped, cls.TEACHER_ASK_PATTERNS)
            or cls._matches_any(stripped, cls.STOP_CORRECTION_PATTERNS)
            or cls._is_language_switch_to_chinese(stripped)
            or cls._matches_any(stripped, cls.EXPLANATION_PATTERNS)
            or cls._matches_any(stripped, cls.CORRECTION_PATTERNS)
        )

    @staticmethod
    def _starts_with_user_echo(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text or "").strip().lower()
        echo_starts = (
            "you said",
            "great! you said",
            "great, you said",
            "that's correct. you",
            "that is correct. you",
            "i see! so you're saying",
            "i see, so you're saying",
            "so you're saying",
            "your sentence is",
        )
        return any(normalized.startswith(prefix) for prefix in echo_starts)

    @staticmethod
    def _asks_user_to_drive(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text or "").strip().lower()
        patterns = (
            r"what would you like to ask",
            r"would you like to ask",
            r"do you want to ask",
            r"you can ask me",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def _repeats_recent_question(cls, response: str, recent_questions: List[str]) -> bool:
        response_questions = cls._extract_questions(response)
        if not response_questions or not recent_questions:
            return False
        normalized_recent = {cls._normalize_question(question) for question in recent_questions}
        recent_keywords = [cls._question_keywords(question) for question in recent_questions]
        for question in response_questions:
            normalized = cls._normalize_question(question)
            if normalized and normalized in normalized_recent:
                return True
            keywords = cls._question_keywords(question)
            if keywords and any(cls._questions_overlap(keywords, old_keywords) for old_keywords in recent_keywords):
                return True
        return False

    @staticmethod
    def _normalize_question(question: str) -> str:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (question or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _question_keywords(question: str) -> set:
        stop_words = {
            "a", "an", "the", "is", "are", "am", "do", "does", "did", "to", "from", "of", "for",
            "you", "your", "me", "my", "i", "it", "this", "that", "what", "where", "when", "how",
            "would", "could", "can", "please", "about", "tell", "like",
        }
        words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", (question or "").lower())
        return {word for word in words if word not in stop_words and len(word) > 1}

    @staticmethod
    def _questions_overlap(new_keywords: set, old_keywords: set) -> bool:
        if not new_keywords or not old_keywords:
            return False
        overlap = new_keywords & old_keywords
        smaller = min(len(new_keywords), len(old_keywords))
        return len(overlap) >= 3 or (smaller <= 3 and len(overlap) >= 2)

    @classmethod
    def _extract_questions(cls, text: str) -> List[str]:
        questions: List[str] = []
        for match in re.finditer(r"([^?？。.!]*[?？])", text or ""):
            question = cls._compact(match.group(1), 140)
            if question:
                questions.append(question)
        return questions

    @staticmethod
    def _extract_target_from_prompt(prompt_or_text: str) -> str:
        match = re.search(r"^TARGET:\s*(.+)$", prompt_or_text or "", flags=re.MULTILINE)
        if not match:
            return ""
        target = match.group(1).strip()
        if target == "(no English sentence found)":
            return ""
        return target

    @classmethod
    def _extract_explanation_target(cls, text: str) -> str:
        raw = re.sub(r"\s+", " ", text or "").strip()
        if not raw:
            return ""

        spelled = re.search(r"\b([a-zA-Z](?:\s+[a-zA-Z]){2,})\b", raw)
        if spelled:
            candidate = spelled.group(1).replace(" ", "")
            if len(candidate) >= 2:
                return candidate.lower()

        lowered = raw.lower()
        patterns = (
            r"meaning of ([a-zA-Z][a-zA-Z '\-]{0,40})",
            r"([a-zA-Z][a-zA-Z '\-]{0,40}) meaning",
            r"what'?s (?:the )?(?:word|world)?\s*([a-zA-Z][a-zA-Z '\-]{1,40})",
            r"i don'?t know (?:the )?meaning of ([a-zA-Z][a-zA-Z '\-]{0,40})",
        )
        stop_words = {"the", "word", "world", "english", "meaning", "of"}
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if not match:
                continue
            candidate = match.group(1).strip(" ?.,")
            words = [word for word in re.findall(r"[a-zA-Z]+", candidate) if word.lower() not in stop_words]
            if words:
                return " ".join(words[-3:]).lower()
        return ""

    @staticmethod
    def _fallback_word_explanation(target: str) -> str:
        normalized = re.sub(r"\s+", " ", target or "").strip().lower()
        if normalized == "coach":
            return (
                "coach 的意思通常是“教练”或“辅导老师”。比如 English coach 就是“英语教练”，"
                "指帮助你练习英语、纠正表达、带你对话的人。例句：I have an English coach."
            )
        return (
            f"“{target}” 是你想问的词或短语。它的具体意思要看上下文；"
            f"如果你把完整句子发给我，我可以帮你解释它在那句话里的准确含义。"
        )

    @staticmethod
    def _mostly_chinese(text: str) -> bool:
        letters = re.findall(r"[A-Za-z]", text or "")
        cjk = re.findall(r"[\u4e00-\u9fff]", text or "")
        return len(cjk) >= max(4, len(letters))

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

    @staticmethod
    def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            normalized = re.sub(r"\s+", " ", item or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(normalized)
        return result

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
