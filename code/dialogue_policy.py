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
from difflib import SequenceMatcher
from typing import Iterable, List, Mapping


@dataclass
class TeachingSessionState:
    """Compact state extracted from chat history."""

    scenario_setup: str = ""
    practice_language: str = ""
    last_user_english: str = ""
    last_assistant_text: str = ""
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

    GREETING_PATTERNS = (
        r"^\s*(\u4f60\u597d|\u55e8|\u54c8\u55bd|\u60a8\u597d)+[!\uff01\u3002.\s]*$",
        r"^\s*(ä½ å¥½|å—¨|å“ˆå–½|æ‚¨å¥½)[!ï¼ã€‚.\s]*$",
        r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)[!.\s]*$",
    )

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

    CORRECTION_SETUP_PATTERNS = (
        r"\bplease correct me if\b",
        r"\bcorrect me if\b",
        r"\bplease correct me when\b",
        r"\bcorrect me when\b",
        r"\bplease correct my mistakes\b",
        r"\bcorrect my mistakes\b",
        r"\bif there (are|is) any mistakes?\b",
        r"\bif i make (a )?mistakes?\b",
        r"\bi will .*please correct me\b",
        r"\bi'?ll .*please correct me\b",
        r"\bi am going to .*please correct me\b",
        r"\bi will introduce .*correct me\b",
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
        r"(let'?s|lets|i want to|i would like to|i'?d like to).*\bpractice\b",
        r"\bpractice\b.*\b(english|speaking|conversation|travel|business|daily)\b",
        r"\btravel english\b",
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
        r"\bdon[\u2019'?]t correct\b",
        r"\bdo not correct\b",
        r"\bdon'?t correct me\b",
        r"\bdon[\u2019'?]t correct me\b",
        r"\bdo not correct me\b",
    )

    CLOSING_PATTERNS = (
        r"\bthat'?s all\b",
        r"\bthat is all\b",
        r"\bnothing else\b",
        r"\bno more\b",
        r"\bno need\b",
        r"\ball done\b",
        r"^\s*(done|finished)\s*[.!?]*\s*$",
        r"\bi'?m done\b",
        r"\bi am done\b",
        r"\bi'?m finished\b",
        r"\bi am finished\b",
        r"\bwe'?re done\b",
        r"\bwe are done\b",
        r"\balready finished\b",
        r"\bwe already finished\b",
        r"\bi think we already finished\b",
        r"\bfinished ordering\b",
        r"\balready finished ordering\b",
        r"\bthat should be all\b",
        r"\bthat'?s enough\b",
        r"\bthat is enough\b",
        r"\bi think that'?s enough\b",
        r"\bi'?m done for now\b",
        r"\bi want to stop\b",
        r"\bstop here\b",
        r"\bstop here please\b",
        r"\bcan we finish here\b",
        r"\bwe can end\b",
        r"\bend this practice\b",
        r"\bend the role[- ]?play\b",
        r"\bfinish this conversation\b",
        r"\bi need to go\b",
        r"\bno more questions\b",
        r"\bno need to ask more\b",
        r"\bdon'?t want to continue\b",
        r"\bi don'?t want to continue\b",
        r"\bthank you[, ]+bye\b",
        r"\bthanks[, ]+see you next time\b",
        r"\bsee you next time\b",
        r"\u7ed3\u675f\u4e86",
        r"\u7ed3\u675f\u5427",
        r"\u4e0d\u7528\u4e86",
        r"\u4e0d\u7528\u7ee7\u7eed",
        r"\u4e0d\u7528\u518d\u7ee7\u7eed",
        r"\u4e0d\u7528\u7ee7\u7eed\u95ee",
        r"\u4e0d\u60f3\u7ee7\u7eed",
        r"\u6ca1\u6709\u4e86",
        r"\u5c31\u8fd9\u4e9b",
        r"\u5230\u8fd9\u91cc",
        r"\u5230\u8fd9\u513f",
        r"\u5c31\u5230\u8fd9",
        r"\u5df2\u7ecf\u5b8c\u6210",
        r"\u5df2\u7ecf\u7ed3\u675f",
        r"\u4eca\u5929\u7684\u7ec3\u4e60\u7ed3\u675f",
        r"\u4eca\u5929.*\u7ec3\u5230",
        r"\u4eca\u5929.*\u4e0d\u804a",
        r"\u4eca\u5929.*\u591f\u4e86",
        r"\u7ec3\u5f97.*\u5dee\u4e0d\u591a",
        r"\u5148\u4e0d\u7ec3",
        r"\u5148\u4e0d\u804a",
        r"\u5148\u5230\u8fd9",
        r"\u5148\u53bb\u5fd9",
        r"\u5148\u4e0b\u4e86",
        r"\u6211\u60f3\u7ed3\u675f",
        r"\u53ef\u4ee5\u7ed3\u675f",
        r"\u53ef\u4ee5\u4e86",
        r"\u8fd9\u8f6e\u7ed3\u675f",
        r"\u8fd9\u4e2a\u7ec3\u4e60\u5148\u7ed3\u675f",
        r"\u8fd9\u6bb5\u5bf9\u8bdd\u7ed3\u675f",
        r"\u660e\u5929\u518d\u7ee7\u7eed",
        r"\u62dc\u62dc",
    )

    CLOSING_CONTINUATION_PATTERNS = (
        r"\bwould you like\b",
        r"\bdo you want\b",
        r"\bwant to\b",
        r"\blet'?s move on\b",
        r"\bmove on\b",
        r"\btry (this|another|something)\b",
        r"\bpractice (another|something|more)\b",
        r"\bcontinue\b",
        r"\banother (sentence|topic|scene|time)\b",
        r"\bnext (topic|question|step|scene)\b",
        r"\bmore help\b",
        r"\bagain later\b",
        r"\bwhat would you say\b",
        r"\bwhat would you like\b",
        r"\bhow would you like\b",
        r"\bproceed\b",
        r"\bdestination\b",
        r"\bcheck in again\b",
        r"\bgreat job\b",
        r"\bnice work\b",
        r"\bnice effort\b",
        r"\bperfect\b",
        r"\bwell done\b",
        r"\byou did well\b",
        r"\byou'?re ready\b",
        r"\bwe'?ve covered\b",
        r"\ball (the )?(key points|basics)\b",
        r"\bwe finished\b",
        r"\bwe'?re done here\b",
        r"\bmeal is confirmed\b",
        r"\bfinished talking about\b",
        r"\byou ordered\b",
        r"\bis complete\b",
        r"\bhere is complete\b",
        r"\bairport check-in complete\b",
        r"\byour meal will be ready\b",
        r"\u518d\u7ec3",
        r"\u6362\u4e2a",
        r"\u4e0b\u4e00\u4e2a",
        r"\u8981\u4e0d\u8981",
        r"\u60f3\u4e0d\u60f3",
        r"\u8981\u7ee7\u7eed",
    )

    TRANSLATION_REQUEST_PATTERNS = (
        "\u7ffb\u8bd1",
        "\u8bd1\u4e00\u4e0b",
        "\u8bd1\u6210",
        r"\btranslate\b",
    )

    @classmethod
    def build_chat_prompt(
        cls,
        history: Iterable[Mapping[str, str]] | None,
        user_text: str,
        max_history_messages: int | None = None,
    ) -> str:
        normalized_user_text = (user_text or "").strip()
        raw_history = list(history or [])
        prompt_history = cls._without_duplicate_latest_user(
            cls._normalized_history(raw_history),
            normalized_user_text,
        )
        max_messages = max_history_messages or cls.MAX_DEFAULT_HISTORY_MESSAGES
        prompt_history = prompt_history[-max(2, max_messages) :]

        state = cls.extract_state(raw_history or prompt_history, normalized_user_text)
        focus = cls.classify_user_intent(normalized_user_text, prompt_history)
        return cls._build_unified_prompt(state, prompt_history, normalized_user_text, focus)

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
        if cls._is_closing_request(text):
            return "closing"
        if cls._is_explicit_translation_request_v2(text):
            return "translation"
        if cls._is_explicit_correction_request_v2(text):
            return "correction"
        return "general"

    @classmethod
    def generation_config(cls, user_text: str, history: Iterable[Mapping[str, str]] | None = None) -> dict:
        mode = cls.classify_user_intent(user_text, history or [])
        if mode == "translation":
            return {"max_new_tokens": 180, "temperature": 0.25, "top_p": 0.80, "repetition_penalty": 1.08}
        if mode == "correction":
            return {"max_new_tokens": 220, "temperature": 0.30, "top_p": 0.82, "repetition_penalty": 1.08}
        if mode == "closing":
            return {"max_new_tokens": 120, "temperature": 0.25, "top_p": 0.80, "repetition_penalty": 1.10}
        return {"max_new_tokens": 220, "temperature": 0.45, "top_p": 0.90, "repetition_penalty": 1.08}

    @classmethod
    def build_context_card(
        cls,
        history: Iterable[Mapping[str, str]] | None,
        user_text: str,
        mode: str,
    ) -> str:
        state = cls.extract_state(cls._normalized_history(history or []), user_text)
        return cls._simplified_context_card(state, mode)

    @classmethod
    def _build_unified_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
        focus: str,
    ) -> str:
        translation_target = cls._infer_explicit_translation_target(history, user_text)
        correction_target = state.last_user_english
        strategy_hints = cls._strategy_hints(
            state,
            history,
            user_text,
            focus,
            translation_target,
        )

        system_prompt = (
            "You are a patient English speaking coach for Chinese learners.\n"
            "Use the conversation naturally instead of following rigid mode templates.\n"
            "Understand the user's real intention from the latest turn and recent context.\n"
            f"Current focus: {focus}\n"
            f"{cls._simplified_context_card(state, focus)}\n"
            f"{strategy_hints}"
            "Core rules:\n"
            "- Preserve the ongoing scene or topic when it still matters, but do not force role-play if the user changes task.\n"
            "- If the user asks to correct a sentence, give a natural correction. A rigid output format is not required.\n"
            "- If the user asks to translate 'this sentence', 'that sentence', or 'what you just said', translate the referenced recent text instead of translating the command itself.\n"
            "- If the user asks for explanation, explain briefly and clearly.\n"
            "- If the user asks to switch language, follow that request naturally.\n"
            "- Do not mechanically repeat your previous assistant reply unless the user explicitly asks you to repeat it.\n"
            "- Ask at most one useful next question when a question is actually needed. If no question is needed, do not force one.\n"
            "- Use plain text only: no Markdown, no headings, no bullet lists.\n"
            "- Output only the assistant reply.\n"
        )
        if translation_target:
            system_prompt += (
                "Translation requirement: translate the full target sentence faithfully and completely.\n"
                "Do not answer the topic. Do not paraphrase. Do not add or remove meaning.\n"
                "If the target is English, translate it into natural Chinese. "
                "If the target is Chinese, translate it into natural English.\n"
                f"Referenced translation target: {translation_target}\n"
            )
        if focus == "correction" and correction_target:
            system_prompt += f"Correction target: {correction_target}\n"
        if focus == "closing":
            system_prompt += (
                "Closing requirement: the user is ending or completing the current practice/task.\n"
                "Reply with one short natural closing sentence.\n"
                "Briefly acknowledge completion and stop naturally.\n"
                "Match the user's language when it is obvious.\n"
                "Do not ask any question. Do not start a new scenario. Do not invite a new topic.\n"
                "Do not suggest continuing, moving on, trying another sentence, or practicing something else.\n"
                "Avoid adding unconfirmed scene details or unnecessary praise.\n"
            )

        parts: List[str] = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for message in history:
            role = message.get("role", "user")
            content = message.get("content", "").strip()
            if content:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @classmethod
    def _simplified_context_card(cls, state: TeachingSessionState, focus: str) -> str:
        lines = ["Conversation context:"]
        lines.append(f"- focus: {focus}")
        if state.scenario_setup:
            lines.append(f"- scenario_setup: {state.scenario_setup}")
        if state.practice_language:
            lines.append(f"- preferred_language: {state.practice_language}")
        if state.last_user_english:
            lines.append(f"- last_user_english: {state.last_user_english}")
        if state.last_assistant_text:
            lines.append(f"- last_assistant_text: {state.last_assistant_text}")
        if state.recent_ai_questions:
            lines.append("- recent_ai_questions:")
            for question in state.recent_ai_questions[-3:]:
                lines.append(f"  * {question}")
        if state.user_facts:
            lines.append("- user_facts:")
            for fact in state.user_facts[-4:]:
                lines.append(f"  * {fact}")
        return "\n".join(lines)

    @classmethod
    def _strategy_hints(
        cls,
        state: TeachingSessionState,
        history: Iterable[Mapping[str, str]],
        user_text: str,
        focus: str,
        translation_target: str = "",
    ) -> str:
        hints: List[str] = []

        if state.has_roleplay:
            hints.append(
                "Scene continuity: treat scenario_setup, user_facts, last_assistant_text, "
                "and recent_ai_questions as the current state. Do not restart completed steps, "
                "do not invent a completed order/check-in, and do not ask the same topic again."
            )

        if cls._is_short_contextual_user_turn(user_text, history):
            hints.append(
                "ASR/context recovery: the latest user text is likely a short spoken answer "
                "to the previous assistant question. Infer it from last_assistant_text, "
                "do not explain the words themselves, and continue the same task with one "
                "focused next step or one focused clarification."
            )

        if state.correction_mode == "gentle" and focus == "general":
            hints.append(
                "Correction preference: if the latest English sentence has a clear mistake, "
                "briefly give a more natural version before continuing. If it is acceptable, "
                "continue normally without forcing a correction."
            )

        if cls._is_explanation_like_request_v2(user_text):
            hints.append(
                "Explanation: answer the meaning or usage question directly. Prefer Chinese "
                "when the user asks in Chinese or mixes Chinese. Give one short example when useful. "
                "Do not force a new practice question after the explanation."
            )

        if translation_target:
            hints.append(
                "Translation: output the translation or meaning of the referenced target only. "
                "Do not answer the target topic and do not add a follow-up practice question."
            )

        if state.teacher_should_ask:
            hints.append(
                "Teacher-led practice: lead with one specific next question tied to the current "
                "topic or scene, not an open 'what do you want to do next' question."
            )

        if not hints:
            return ""
        return "Strategy hints:\n" + "".join(f"- {hint}\n" for hint in hints)

    @classmethod
    def _is_explicit_translation_request(cls, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        return (
            "翻译" in (text or "")
            or "译一下" in (text or "")
            or bool(re.search(r"\btranslate\b", lowered))
            or bool(re.search(r"\btranslation\b", lowered))
        )

    @classmethod
    def _is_explicit_correction_request(cls, text: str) -> bool:
        raw = (text or "").strip()
        lowered = raw.lower()
        if not raw:
            return False
        if any(
            phrase in raw
            for phrase in (
                "语法错误",
                "有语法错误吗",
                "纠正",
                "改一下",
                "更自然",
                "帮我改",
                "帮我纠错",
            )
        ):
            return True
        return bool(
            re.search(
                r"\b(correct|fix|grammar|grammatical|natural|better version|better way|is this wrong)\b",
                lowered,
            )
        )

    @classmethod
    def _infer_explicit_translation_target(cls, history: Iterable[Mapping[str, str]], user_text: str) -> str:
        if not cls._is_explicit_translation_request_v2(user_text):
            return ""
        direct_target = cls._extract_direct_translation_text(user_text)
        if direct_target:
            return cls._compact(direct_target, 500)

        for message in reversed(cls._normalized_history(history or [])):
            if message.get("role") == "assistant":
                return cls._compact(message.get("content", "").strip(), 500)
        return ""

    @classmethod
    def _is_explicit_translation_request_v2(cls, text: str) -> bool:
        raw = text or ""
        lowered = raw.strip().lower()
        if not lowered:
            return False
        return (
            "\u7ffb\u8bd1" in raw
            or "\u8bd1\u4e00\u4e0b" in raw
            or "\u7528\u4e2d\u6587\u7ffb\u8bd1" in raw
            or "\u7528\u82f1\u6587\u7ffb\u8bd1" in raw
            or "\u82f1\u6587\u600e\u4e48\u8bf4" in raw
            or "\u7528\u82f1\u6587\u600e\u4e48\u8bf4" in raw
            or "\u4e2d\u6587\u600e\u4e48\u8bf4" in raw
            or "\u7528\u4e2d\u6587\u600e\u4e48\u8bf4" in raw
            or "\u82f1\u6587\u662f\u4ec0\u4e48" in raw
            or "\u4e2d\u6587\u662f\u4ec0\u4e48" in raw
            or bool(re.search(r"(\u8fd9\u53e5\u8bdd|\u8fd9\u53e5|\u521a\u624d\u90a3\u53e5|\u521a\u521a\u90a3\u53e5).*(\u4ec0\u4e48\u610f\u601d|\u610f\u601d)", raw))
            or bool(re.search(r"\btranslate\b", lowered))
            or bool(re.search(r"\btranslation\b", lowered))
            or bool(re.search(r"\bhow (do|should|can|to) (i )?say\b", lowered))
            or bool(re.search(r"\bwhat (does|is) .{1,80} (mean )?in chinese\b", lowered))
            or bool(re.search(r"\bwhat (does|is) .{1,80} (mean )?in english\b", lowered))
        )

    @classmethod
    def _is_explicit_correction_request_v2(cls, text: str) -> bool:
        raw = (text or "").strip()
        lowered = raw.lower()
        if not raw:
            return False
        if any(
            phrase in raw
            for phrase in (
                "\u8bed\u6cd5\u9519\u8bef",
                "\u6709\u8bed\u6cd5\u9519\u8bef\u5417",
                "\u7ea0\u6b63",
                "\u6539\u4e00\u4e0b",
                "\u66f4\u81ea\u7136",
                "\u5e2e\u6211\u6539",
                "\u5e2e\u6211\u7ea0\u9519",
            )
        ):
            return True
        return bool(
            re.search(
                r"\b(correct|fix|grammar|grammatical|natural|better version|better way|is this wrong)\b",
                lowered,
            )
        )

    @classmethod
    def extract_state(
        cls,
        history: Iterable[Mapping[str, str]],
        latest_user_text: str = "",
    ) -> TeachingSessionState:
        state = TeachingSessionState()
        for message in history or []:
            if str(message.get("role", "")).strip().lower() == "system":
                cls._apply_system_context_to_state(state, str(message.get("content", "")))

        normalized = cls._normalized_history(history)

        for message in normalized:
            if message.get("role") == "assistant":
                assistant_text = message.get("content", "").strip()
                if assistant_text:
                    state.last_assistant_text = cls._compact(assistant_text, 500)
                state.recent_ai_questions.extend(cls._extract_questions(assistant_text))
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

    @classmethod
    def _apply_system_context_to_state(cls, state: TeachingSessionState, content: str) -> None:
        text = content or ""
        scene_match = re.search(r"(?im)^\s*-\s*scene\s*:\s*(.+)$", text)
        if scene_match and not state.scenario_setup:
            state.scenario_setup = cls._compact(f"scene: {scene_match.group(1).strip()}", 220)

        language_match = re.search(r"(?im)^\s*-\s*practice_language\s*:\s*(.+)$", text)
        if language_match and not state.practice_language:
            state.practice_language = cls._compact(language_match.group(1).strip(), 40)

        if re.search(r"\bcorrect\b|\bcorrection\b", text, flags=re.IGNORECASE) and not state.correction_mode:
            state.correction_mode = "gentle"

    @staticmethod
    def extract_latest_user_text(prompt_or_text: str) -> str:
        text = prompt_or_text or ""
        matches = re.findall(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return text.strip()

    @staticmethod
    def extract_system_prompt(prompt_or_text: str) -> str:
        text = prompt_or_text or ""
        match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

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
        if cls.classify_user_intent(latest_user_text, history) == "closing":
            return cls._direct_closing_response(latest_user_text, history)

        # Keep normal dialogue generation model-driven. This hook is only kept
        # for narrow high-confidence responses, not broad language choice.
        return ""

    @classmethod
    def finalize_response(cls, prompt_or_text: str, response_text: str) -> str:
        """Light cleanup after model generation without replacing model behavior."""
        response = cls._strip_special_tokens(response_text)
        history = cls.extract_prompt_history(prompt_or_text)
        latest_user_text = cls.extract_latest_user_text(prompt_or_text)
        mode = cls.classify_user_intent(latest_user_text, history)

        if mode == "translation":
            target = cls._infer_explicit_translation_target(history, latest_user_text)
            response = cls._strip_translation_preface(response, target)
            response = cls._trim_unwanted_followup(response)
            return response.strip()

        if mode == "correction":
            return response.strip()

        if cls._is_explanation_like_request_v2(latest_user_text):
            response = cls._trim_unwanted_followup(response)
        return response.strip()

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

        if mode == "translation":
            if cls._looks_like_translation_command_echo(response):
                return True
            translation_target = cls._infer_explicit_translation_target(history, latest_user_text)
            if translation_target and cls._translation_has_wrong_target_language(translation_target, response):
                return True
            if cls._has_unwanted_followup(response):
                return True
            return False

        if mode == "closing":
            if cls._has_question_mark(response) or cls._extract_questions(response):
                return True
            if cls._matches_any(response, cls.CLOSING_CONTINUATION_PATTERNS):
                return True
            return False

        if state.last_assistant_text and not cls._is_repeat_request(latest_user_text) and cls._repeats_recent_statement(
            response,
            [state.last_assistant_text],
        ):
            return True
        if state.recent_ai_questions and cls._repeats_recent_question(response, state.recent_ai_questions):
            return True
        if cls._misreads_short_answer_as_explanation(latest_user_text, response, state):
            return True
        if cls._looks_like_context_break(latest_user_text, response, state):
            return True
        if cls._is_explanation_like_request_v2(latest_user_text) and cls._has_unwanted_followup(response):
            return True
        if mode == "correction" and not cls._looks_like_relaxed_correction_response(
            response,
            state.last_user_english,
        ):
            return True
        if not state.repeat_user_answer and cls._starts_with_user_echo(response):
            return True
        return False

    @classmethod
    def fallback_response(cls, prompt_or_text: str) -> str:
        history = cls.extract_prompt_history(prompt_or_text)
        latest_user_text = cls.extract_latest_user_text(prompt_or_text)
        mode = cls.classify_user_intent(latest_user_text, history)
        state = cls.extract_state(history, latest_user_text)

        if mode == "translation":
            target = cls._infer_explicit_translation_target(history, latest_user_text)
            if target:
                clean_target = cls._strip_special_tokens(target)
                return f"我找到要翻译的内容了，但这次模型没有稳定生成译文：{clean_target}"
            return "请把你想翻译的句子单独发给我。"

        if mode == "correction":
            target = cls._strip_special_tokens(state.last_user_english)
            if target:
                return f"More natural: {target}"
            return "Please send the sentence you want me to correct."

        if mode == "closing":
            if re.search(r"[\u4e00-\u9fff]", latest_user_text or ""):
                return "\u597d\u7684\uff0c\u4eca\u5929\u5148\u5230\u8fd9\u91cc\u3002"
            return "Got it. We can stop here for now."

        if state.last_assistant_text and not cls._is_repeat_request(latest_user_text):
            return "Let me answer that more directly."
        return "Tell me what you'd like to practice next."

        if mode == "translation":
            target = cls._infer_translation_target(history, latest_user_text)
            if target:
                return (
                    "\u6211\u6ca1\u80fd\u7a33\u5b9a\u5b8c\u6210\u8fd9\u6b21\u7ffb\u8bd1\u3002"
                    "\u8bf7\u628a\u8981\u7ffb\u8bd1\u7684\u90a3\u53e5\u8bdd\u5355\u72ec\u53d1\u7ed9\u6211\uff0c"
                    "\u6211\u4f1a\u76f4\u63a5\u7ed9\u51fa\u8bd1\u6587\u3002"
                )
            return "\u8bf7\u628a\u4f60\u60f3\u7ffb\u8bd1\u7684\u53e5\u5b50\u53d1\u7ed9\u6211\uff0c\u6216\u8005\u8bf4\u201c\u7ffb\u8bd1\u4e0a\u4e00\u53e5\u201d\u3002"

        if mode == "closing":
            return "You're right. We can stop here. Nice job finishing this practice."

        if mode == "correction_setup":
            return "Sure. Start your introduction, and I'll correct any mistakes as you go."

        if mode == "greeting":
            return "你好！很高兴见到你。今天想练习英语，还是先随便聊聊？"

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
        mode = DialoguePolicy.classify_user_intent(latest_user_text, history)

        if mode == "closing":
            instruction = (
                "The previous draft was rejected because the user is closing or completing the current task. "
                "Reply with one short natural closing sentence. "
                "Match the user's language when it is obvious. "
                "Do not ask any question. Do not start a new scenario. Do not invite a new topic. "
                "Do not suggest continuing, moving on, trying another sentence, or practicing something else. "
                "Use plain text only and output only the final assistant reply."
            )
            return f"{prompt}\n<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        if mode == "translation":
            target = DialoguePolicy._infer_explicit_translation_target(history, latest_user_text)
            instruction = (
                "The previous draft mishandled a translation or meaning request. "
                "Translate the referenced target only, not the user's command. "
                "If the target is English, answer in natural Chinese. "
                "If the target is Chinese, answer in natural English. "
                "Do not add a follow-up question, teaching prompt, or role-play continuation. "
                f"Referenced target: {target or '(infer from recent assistant text)'} "
                "Use plain text only and output only the final assistant reply."
            )
            return f"{prompt}\n<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        if mode == "correction":
            state = DialoguePolicy.extract_state(history, latest_user_text)
            instruction = (
                "The previous draft mishandled a correction request. "
                "Give a natural corrected English version for the target sentence, with at most one brief reason if useful. "
                "Do not ask the user to resend the sentence and do not continue role-play. "
                f"Target sentence: {state.last_user_english or latest_user_text} "
                "Use plain text only and output only the final assistant reply."
            )
            return f"{prompt}\n<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        if DialoguePolicy._is_explanation_like_request_v2(latest_user_text):
            instruction = (
                "The previous draft mishandled an explanation request. "
                "Explain the word, phrase, or usage directly and briefly. "
                "Prefer Chinese if the user asked in Chinese or mixed Chinese. "
                "Include one short example when useful. "
                "Do not add a new practice question. "
                "Use plain text only and output only the final assistant reply."
            )
            return f"{prompt}\n<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

        generic_instruction = (
            "The previous draft was rejected because it was empty, repeated the previous assistant reply, "
            "echoed a translation command, or otherwise failed to respond directly. "
            "Now answer the latest user message naturally using the recent context. "
            "Do not repeat your previous assistant reply unless the user explicitly asked you to repeat it. "
            "If the latest user message is a short answer, infer it from the previous assistant question and continue the same scene. "
            "If the user asks to translate a referenced sentence, translate the referenced recent text instead of the command itself. "
            "If the user asks for correction, provide a natural correction without using a rigid template. "
            "Use plain text only and output only the final assistant reply."
        )
        return f"{prompt}\n<|im_start|>system\n{generic_instruction}<|im_end|>\n<|im_start|>assistant\n"

        if mode == "translation":
            instruction = (
                "The previous draft handled a translation request incorrectly. "
                "Use the recent chat history to resolve references such as 'this sentence', 'the previous sentence', "
                "or 'what you just said'. Translate the referenced assistant text, not the user's command. "
                "If the referenced text is English, translate it into natural Chinese. "
                "If the referenced text is Chinese, translate it into natural English. "
                "Output only the translation, with no labels."
            )
        elif mode == "closing":
            instruction = (
                "The previous draft was rejected. The user is closing or completing the current task. "
                "Do not ask a new question. Do not start a new scenario. "
                "Briefly confirm completion and close the practice naturally. "
                "Use plain text only. Output only the final assistant reply."
            )
        elif mode == "correction":
            instruction = (
                "The previous draft was rejected. The user wants a correction, not a new scenario. "
                "Provide a natural correction reply for the target English sentence. "
                "Include a corrected English version and, if helpful, a brief Chinese explanation. "
                "Do not require a rigid format. Do not ask the user to resend the sentence. "
                "Do not continue role-play and do not translate unless explicitly asked. "
                "Output only the final assistant reply."
            )
        elif mode == "language_switch":
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
        elif mode == "explanation":
            instruction = (
                "The previous draft was rejected. Answer the user's English-usage question in Chinese only. "
                "Do not reveal or quote private context. Do not output labels such as State, mode, roleplay, setup, "
                "practice_language, or user_facts. Output only the final assistant reply."
            )
        else:
            state = DialoguePolicy.extract_state(history, latest_user_text)
            recent_questions = "; ".join(state.recent_ai_questions[-4:]) or "(none)"
            instruction = (
                "The previous draft was rejected, usually because it repeated an earlier assistant question, "
                "leaked instructions, used the wrong language, or ignored the latest user message. "
                f"Latest user message: {latest_user_text}\n"
                f"Recent assistant questions that must not be repeated: {recent_questions}\n"
                "Now write a fresh reply that responds directly to the latest user message and advances the same scenario. "
                "Never restart the scenario. Never repeat your previous assistant reply or previous question. "
                "Ask at most one new useful question. Use plain text only: no Markdown, no bold markers, no bullet lists, no headings. "
                "Output only the final assistant reply."
            )
        return f"{prompt}\n<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

    @classmethod
    def normalize_history(cls, history: Iterable[Mapping[str, str]]) -> List[dict]:
        return cls._normalized_history(history)

    @classmethod
    def _build_translation_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
    ) -> str:
        target = cls._infer_translation_target(history, user_text)
        target_line = target or "(no referenced sentence found)"
        direction = cls._translation_direction(target)
        system_prompt = (
            "You are a precise translator for an English learning chat.\n"
            f"{cls._state_card(state, 'translation')}\n"
            "Task: translate the referenced sentence only.\n"
            "The user's latest message may be a command such as 'translate this sentence'; do not translate the command itself.\n"
            "Do not continue the previous role-play or practice flow.\n"
            "Do not ask a new practice question.\n"
            "Output only the translation, with no labels, no explanation, no Markdown, and no quotation marks unless they are needed in the translation.\n"
            f"Translation direction: {direction}\n"
            f"TARGET: {target_line}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        parts: List[str] = [f"<|im_start|>system\n{system_prompt}"]
        return "\n".join(parts)

    @classmethod
    def _build_greeting_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
    ) -> str:
        system_prompt = (
            "You are a warm English speaking coach for Chinese learners.\n"
            f"{cls._state_card(state, 'greeting')}\n"
            "Task: respond to a simple greeting naturally.\n"
            "Do not turn the greeting into a quiz about how to greet someone.\n"
            "Briefly greet the user back, then offer to practice English or chat.\n"
            "If the user greeted in Chinese, replying in Chinese is allowed.\n"
            "Use plain text only: no Markdown, no bullet lists, no headings.\n"
            "Ask at most one light next question.\n"
            "Output only the assistant reply.\n"
        )
        parts: List[str] = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
        for message in history[-4:]:
            role = message.get("role", "user")
            content = message.get("content", "").strip()
            if content:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append(f"<|im_start|>user\n{user_text}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    @classmethod
    def _build_closing_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
    ) -> str:
        system_prompt = (
            "You are an English speaking coach for Chinese learners.\n"
            f"{cls._state_card(state, 'closing')}\n"
            "Task: the user is closing or completing the current practice/task.\n"
            "Confirm completion naturally and stop the current flow.\n"
            "Do not ask a new question unless the user explicitly asks to continue.\n"
            "Do not start another scenario or introduce a new topic.\n"
            "Do not repeat the user's answer or the previous assistant reply.\n"
            "If role-play is active, close that scenario politely in character if appropriate.\n"
            "Use plain text only: no Markdown, no bold markers, no bullet lists, no headings.\n"
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
    def _build_correction_setup_prompt(
        cls,
        state: TeachingSessionState,
        history: List[dict],
        user_text: str,
    ) -> str:
        system_prompt = (
            "You are an English speaking coach for Chinese learners.\n"
            f"{cls._state_card(state, 'correction_setup')}\n"
            "Task: the user is setting a practice rule, not asking you to correct this sentence now.\n"
            "Briefly confirm that you will correct mistakes during the upcoming practice.\n"
            "Then invite the user to start their introduction or next spoken answer.\n"
            "Do not rewrite the user's setup sentence as a correction.\n"
            "Do not output correction labels such as '更自然的说法' or '说明' unless the user gives a sentence to correct.\n"
            "Use plain text only: no Markdown, no bold markers, no bullet lists, no headings.\n"
            "Ask at most one short next question.\n"
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
    def _build_correction_prompt(cls, state: TeachingSessionState, user_text: str) -> str:
        target = state.last_user_english
        target_line = target or "(no English sentence found)"
        return (
            "<|im_start|>system\n"
            "You are an English grammar correction tutor for Chinese learners.\n"
            "Task: correct the target English sentence only.\n"
            "Use plain text only: no Markdown, no bold markers, no bullet lists, no headings.\n"
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
    def _build_correction_prompt_relaxed(cls, state: TeachingSessionState, user_text: str) -> str:
        target = state.last_user_english
        target_line = target or "(no English sentence found)"
        return (
            "<|im_start|>system\n"
            "You are an English grammar correction tutor for Chinese learners.\n"
            "Task: correct the target English sentence only.\n"
            "Use plain text only: no Markdown, no bold markers, no bullet lists, no headings.\n"
            "Prefer a natural correction reply instead of a rigid template.\n"
            "A good reply should include a corrected English version and, if helpful, a brief Chinese explanation.\n"
            "You may use formats like 'More natural: ...' or '更自然可以说：...' but exact labels are not required.\n"
            "Do not continue role-play. Do not ask the user to resend the sentence if TARGET is provided.\n"
            "Do not translate unless the user explicitly asked for translation.\n"
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
            "Use plain text only: no Markdown, no bold markers like **word**, no bullet lists, no headings.\n"
            "If you compare words, write short natural sentences instead of a formatted list.\n"
            "If the user asks the meaning of a word or phrase, explain it directly; do not ask them to provide the phrase again.\n"
            f"{cls._translation_context_rule(history, user_text)}"
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
            f"{cls._translation_context_rule(history, user_text)}"
            "Task: handle a language-switch request, not a new scenario.\n"
            f"{language_rule}\n"
            f"{style_rule}\n"
            "Keep the previous scenario and user facts if role-play is active.\n"
            "Do not repeat the user's previous answer unless you are correcting it.\n"
            "Do not ask a question that was already asked recently.\n"
            "Use plain text only: no Markdown, no bold markers, no bullet lists, no headings.\n"
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
            f"{cls._translation_context_rule(history, user_text)}"
            "Rules:\n"
            "- If role-play is active, play your assigned role in that scenario.\n"
            "- Reply in English when the practice language is English.\n"
            "- Reply in Chinese when the practice language is Chinese.\n"
            "- Preserve user-provided facts and corrections.\n"
            "- Treat user speech as ASR text: infer obvious speech-recognition mistakes from context before responding.\n"
            "- Do not mechanically repeat or paraphrase the user's answer. Use brief acknowledgement only.\n"
            "- Never repeat your previous assistant reply. If the user repeats or slightly changes the same sentence, respond to it directly instead of restarting the scenario.\n"
            "- If the user asked not to repeat their answers, never start with 'You said', 'Great, you said', or a full restatement.\n"
            "- Do not ask the same question or same topic that appears in recent_ai_questions.\n"
            "- Do not ask a question every turn. Ask only when the current task genuinely needs one.\n"
            "- If the user says the task is finished, complete, or that's all, close the scenario instead of starting a new topic.\n"
            "- When the user gives a final answer such as 'No, that's all', confirm completion and stop.\n"
            "- If the user says you are the teacher, you should ask the next practice question instead of asking the user to ask you.\n"
            "- If the user says a topic is already discussed, move to a new angle or follow-up topic.\n"
            "- If the user asks to translate 'this sentence', 'that sentence', or 'what you just said', translate the referenced assistant text from the recent context; do not translate the user's command itself.\n"
            "- Use plain text only: no Markdown, no bold markers like **word**, no bullet lists, no headings.\n"
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
        if state.last_assistant_text:
            lines.append(f"- last_assistant_text: {state.last_assistant_text}")
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
            if role == "system":
                continue
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
    def _is_translation_request(
        cls,
        text: str,
        history: Iterable[Mapping[str, str]] | None = None,
    ) -> bool:
        if cls._matches_any(text or "", cls.TRANSLATION_REQUEST_PATTERNS):
            return True
        return cls._is_translation_followup_request(text, history)

    @classmethod
    def _is_translation_followup_request(
        cls,
        text: str,
        history: Iterable[Mapping[str, str]] | None = None,
    ) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        has_sentence_reference = bool(re.search(r"(\u8fd9\u53e5\u8bdd|\u8fd9\u53e5|\u8fd9\u4e2a\u53e5\u5b50)", raw))
        has_latin = bool(re.search(r"[A-Za-z]{2,}", raw))
        if has_sentence_reference and has_latin:
            return True

        normalized_history = cls._normalized_history(history or [])
        recent_user_messages = [
            message.get("content", "")
            for message in normalized_history[-4:]
            if message.get("role") == "user"
        ]
        recent_translation_request = any(
            cls._matches_any(message, cls.TRANSLATION_REQUEST_PATTERNS)
            for message in recent_user_messages
        )
        return recent_translation_request and has_latin and has_sentence_reference

    @classmethod
    def _translation_context_rule(cls, history: List[dict], user_text: str) -> str:
        if not cls._is_translation_request(user_text, history):
            return ""
        target = cls._infer_translation_target(history, user_text)
        lines = [
            "Translation reference rule:\n",
            "- The latest user message is a translation request, not the text to translate.\n",
            "- Resolve references like 'this sentence', 'the previous sentence', and 'what you just said' from recent assistant messages.\n",
            "- Translate the referenced text only. Do not translate the user's command itself.\n",
            "- If the referenced text is English, translate it into natural Chinese. If it is Chinese, translate it into natural English.\n",
        ]
        if target:
            lines.append(f"- Likely referenced text: {target}\n")
        else:
            lines.append("- If no referenced text is available, ask the user to provide the sentence.\n")
        return "".join(lines)

    @classmethod
    def _infer_translation_target(cls, history: Iterable[Mapping[str, str]], user_text: str) -> str:
        if not cls._is_translation_request(user_text, history):
            return ""
        direct_target = cls._extract_direct_translation_text(user_text)
        if direct_target:
            return cls._compact(direct_target, 500)

        last_assistant = ""
        for message in reversed(cls._normalized_history(history or [])):
            if message.get("role") == "assistant":
                last_assistant = message.get("content", "").strip()
                break
        if not last_assistant:
            return ""

        selected_sentence = cls._select_referenced_sentence(last_assistant, user_text)
        return cls._compact(selected_sentence or last_assistant, 500)

    @staticmethod
    def _extract_direct_translation_text(user_text: str) -> str:
        raw = re.sub(r"\s+", " ", user_text or "").strip()
        if not raw:
            return ""

        quoted = re.findall(r'["“‘](.+?)["”’]', raw)
        if quoted:
            return quoted[-1].strip()
        single_quoted = re.findall(r"'([^']{1,120})'", raw)
        if single_quoted:
            return single_quoted[-1].strip()

        match = re.search(
            "(?:\u7ffb\u8bd1|\\btranslate\\b)(?:\u4e00\u4e0b|\u6210\u4e2d\u6587|\u6210\u82f1\u6587| into chinese| into english)?\\s*[:：]\\s*(.+)$",
            raw,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        suffix_patterns = (
            r"(.+?)\s*(?:\u7528\u4e2d\u6587\u7ffb\u8bd1\u8fd9\u53e5\u8bdd|\u7528\u4e2d\u6587\u7ffb\u8bd1\u8fd9\u53e5|\u7ffb\u8bd1\u8fd9\u53e5\u8bdd|\u7ffb\u8bd1\u8fd9\u53e5|\u8bd1\u4e00\u4e0b\u8fd9\u53e5\u8bdd)$",
            r"(.+?)\s*(?:translate this sentence|translate this|translate it into chinese|translate it into english)$",
        )
        for pattern in suffix_patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" ：:，,。.\"'")
                if candidate:
                    return candidate
        if re.search(r"(\u8fd9\u53e5\u8bdd|\u8fd9\u53e5|\u8fd9\u4e2a\u53e5\u5b50)", raw) and re.search(r"[A-Za-z]{2,}", raw):
            cleaned = re.sub(
                r"(\u7528\u4e2d\u6587\u7ffb\u8bd1|\u7528\u82f1\u6587\u7ffb\u8bd1|\u7ffb\u8bd1|\u8bd1\u4e00\u4e0b|\u8fd9\u53e5\u8bdd|\u8fd9\u53e5|\u8fd9\u4e2a\u53e5\u5b50)",
                "",
                raw,
            ).strip(" ：:，,。.")
            if re.search(r"[A-Za-z]{2,}", cleaned):
                return cleaned
        return ""

    @staticmethod
    def _translation_direction(target: str) -> str:
        target_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", target or ""))
        target_has_latin = bool(re.search(r"[A-Za-z]{2,}", target or ""))
        if target_has_latin and not target_has_cjk:
            return "English to Chinese"
        if target_has_cjk and not target_has_latin:
            return "Chinese to English"
        return "infer from TARGET language"

    @classmethod
    def _select_referenced_sentence(cls, assistant_text: str, user_text: str) -> str:
        sentences = cls._split_reference_sentences(assistant_text)
        if len(sentences) <= 1:
            return assistant_text.strip()

        keywords = cls._translation_reference_keywords(user_text)
        for sentence in sentences:
            lowered_sentence = sentence.lower()
            if any(keyword in lowered_sentence for keyword in keywords):
                return sentence.strip()
        return ""

    @staticmethod
    def _split_reference_sentences(text: str) -> List[str]:
        parts = re.findall(r"[^.!?\u3002\uff01\uff1f]+[.!?\u3002\uff01\uff1f]?", text or "")
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _translation_reference_keywords(user_text: str) -> List[str]:
        lowered = (user_text or "").lower()
        keywords = re.findall(r"[a-zA-Z]{3,}", lowered)
        alias_map = {
            "\u4f26\u6566": "london",
            "\u5df4\u9ece": "paris",
            "\u6cd5\u56fd": "france",
            "\u6b27\u6d32": "europe",
            "\u5c3c\u7f57\u6cb3": "nile",
            "\u6cb3": "river",
            "\u673a\u573a": "airport",
        }
        for zh_word, en_word in alias_map.items():
            if zh_word in user_text:
                keywords.append(en_word)
        return list(dict.fromkeys(keywords))

    @staticmethod
    def _looks_like_translation_command_echo(response: str) -> bool:
        normalized = re.sub(r"\s+", " ", response or "").strip().lower()
        return (
            "translate this sentence" in normalized
            or "this sentence translates to" in normalized
            or "\u7ffb\u8bd1\u8fd9\u53e5\u8bdd" in (response or "")
            or "\u7ffb\u8bd1\u8fd9\u4e2a\u53e5\u5b50" in (response or "")
        )

    @staticmethod
    def _translation_has_wrong_target_language(target: str, response: str) -> bool:
        target_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", target or ""))
        target_has_latin = bool(re.search(r"[A-Za-z]{2,}", target or ""))
        response_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", response or ""))
        response_has_latin = bool(re.search(r"[A-Za-z]{2,}", response or ""))
        if target_has_latin and not target_has_cjk:
            return not response_has_cjk
        if target_has_cjk and not target_has_latin:
            return not response_has_latin
        return False

    @classmethod
    def _is_meta_instruction(cls, text: str) -> bool:
        stripped = (text or "").strip()
        lowered = stripped.lower()
        return (
            cls._matches_any(stripped, cls.SCENE_SETUP_PATTERNS)
            or cls._is_language_switch_to_english(stripped)
            or cls._is_language_switch_to_chinese(stripped)
            or cls._is_practice_setup_request(stripped)
            or cls._matches_any(stripped, cls.NO_REPEAT_PATTERNS)
            or cls._matches_any(stripped, cls.TEACHER_ASK_PATTERNS)
            or cls._matches_any(stripped, cls.STOP_CORRECTION_PATTERNS)
            or cls._is_closing_request(stripped)
            or cls._is_correction_setup_request(stripped)
            or cls._is_explicit_translation_request_v2(stripped)
            or cls._matches_any(stripped, cls.TRANSLATION_REQUEST_PATTERNS)
            or lowered in {"in english", "speak english", "in chinese", "speak chinese"}
        )

    @classmethod
    def _is_correction_setup_request(cls, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        if cls._matches_any(stripped, cls.CORRECTION_SETUP_PATTERNS):
            return True
        lowered = stripped.lower()
        return (
            "correct me" in lowered
            and re.search(r"\b(if|when|while|during|as i|after i)\b", lowered)
            and not re.search(r"\b(this sentence|this phrase|the sentence|the phrase)\b", lowered)
        )

    @classmethod
    def _is_practice_setup_request(cls, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if cls._matches_any(stripped, cls.SCENE_SETUP_PATTERNS):
            return True
        return (
            re.search(r"\b(let'?s|lets|want to|would like to|like to)\b.*\bpractice\b", lowered)
            or re.search(r"\bpractice\b.*\b(english|speaking|conversation|travel|business|daily)\b", lowered)
            or "you ask me" in lowered
        )

    @classmethod
    def _is_closing_request(cls, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        return cls._matches_any(stripped, cls.CLOSING_PATTERNS)

    @classmethod
    def _is_short_contextual_user_turn(
        cls,
        text: str,
        history: Iterable[Mapping[str, str]] | None = None,
    ) -> bool:
        stripped = (text or "").strip()
        if not stripped or cls._has_question_mark(stripped):
            return False
        if (
            cls._is_closing_request(stripped)
            or cls._is_explicit_translation_request_v2(stripped)
            or cls._is_explicit_correction_request_v2(stripped)
            or cls._is_explanation_like_request_v2(stripped)
            or cls._is_meta_instruction(stripped)
        ):
            return False
        normalized_history = cls._normalized_history(history or [])
        has_recent_assistant = any(message.get("role") == "assistant" for message in normalized_history[-4:])
        if not has_recent_assistant:
            return False
        words = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{1,4}", stripped)
        return len(words) <= 7 or len(stripped) <= 36

    @classmethod
    def _is_explanation_like_request_v2(cls, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False
        lowered = stripped.lower()
        if cls._is_explicit_translation_request_v2(stripped) or cls._is_explicit_correction_request_v2(stripped):
            return False
        if cls._matches_any(stripped, cls.EXPLANATION_PATTERNS):
            return True
        if re.search(r"\bwhat (does|do|is|are) .{1,80}\b(mean|means)\b", lowered):
            return True
        if re.search(r"\bhow to use\b", lowered):
            return True
        if re.search(r"\b(is|are) .{1,60}\b(same|different)\b", lowered):
            return True
        if re.search(r"\b(what about|and|not)\s+['\"“‘][^'\"”’]{1,60}['\"”’]\??$", stripped, flags=re.IGNORECASE):
            return True
        if re.search(r"^[A-Za-z][A-Za-z '\-]{1,35}\?$", stripped):
            return True
        return False

    @staticmethod
    def _strip_special_tokens(text: str) -> str:
        cleaned = re.sub(r"<\|[^>]+?\|>", "", text or "")
        cleaned = cleaned.replace("**", "")
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _strip_translation_preface(cls, response: str, target: str = "") -> str:
        raw = (response or "").strip()
        if not raw:
            return ""

        target_has_cjk = bool(re.search(r"[\u4e00-\u9fff]", target or ""))
        quoted = re.findall(r'["“](.+?)["”]', raw)
        if target_has_cjk and quoted:
            english_quotes = [item.strip() for item in quoted if re.search(r"[A-Za-z]{2,}", item)]
            if english_quotes:
                return english_quotes[0]

        raw = re.sub(r"^(sure|of course|okay|ok)[,，!！\s\-—:：]+", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"^(you can say|i can say|say)\s*[:：\-—]?\s*", "", raw, flags=re.IGNORECASE).strip()

        if target_has_cjk:
            match = re.match(r"^(it'?s|it is)\s+called\s+(?:a|an|the)?\s*([^。.!?？]+)[。.!?？]?$", raw, flags=re.IGNORECASE)
            if match:
                return match.group(2).strip(" \"'“”")
        return raw

    @classmethod
    def _trim_unwanted_followup(cls, response: str) -> str:
        raw = (response or "").strip()
        if not raw:
            return ""
        followup_start = (
            r"would you like",
            r"do you want",
            r"can you tell me",
            r"try saying",
            r"let'?s try",
            r"now try",
            r"repeat after me",
            r"say it",
        )
        english_pattern = r"([。.!?？][\"”']?\s+)(?:" + "|".join(followup_start) + r")\b.*$"
        raw = re.sub(english_pattern, r"\1", raw, flags=re.IGNORECASE).strip()
        chinese_pattern = r"([。.!?？]\s*)(需要我|要不要|你要不要|你想不想|请你试着|跟我说).*$"
        raw = re.sub(chinese_pattern, r"\1", raw).strip()
        return raw

    @classmethod
    def _has_unwanted_followup(cls, response: str) -> bool:
        return cls._trim_unwanted_followup(response or "") != (response or "").strip()

    @classmethod
    def _misreads_short_answer_as_explanation(
        cls,
        latest_user_text: str,
        response: str,
        state: TeachingSessionState,
    ) -> bool:
        if not state.last_assistant_text:
            return False
        if not cls._is_short_contextual_user_turn(latest_user_text, [{"role": "assistant", "content": state.last_assistant_text}]):
            return False
        normalized = re.sub(r"\s+", " ", response or "").strip().lower()
        bad_patterns = (
            r"\bwant me to explain\b",
            r"\bwould you like me to explain\b",
            r"\bwhat does .{1,50} mean\b",
            r"\bwhat does that word mean\b",
            r"\bin your own words\b",
        )
        return any(re.search(pattern, normalized) for pattern in bad_patterns)

    @classmethod
    def _looks_like_context_break(
        cls,
        latest_user_text: str,
        response: str,
        state: TeachingSessionState,
    ) -> bool:
        if not state.last_assistant_text and not state.has_roleplay:
            return False
        if not cls._is_short_contextual_user_turn(latest_user_text, [{"role": "assistant", "content": state.last_assistant_text}]):
            return False
        normalized = re.sub(r"\s+", " ", response or "").strip().lower()
        context_break_patterns = (
            r"\bwhat would you like to do next\b",
            r"\bwhat else would you like help with\b",
            r"\bwhat would you like to talk about\b",
            r"\btalk about something else\b",
            r"\bwould you like help with something else\b",
            r"\bwhat'?s next after that\b",
        )
        return any(re.search(pattern, normalized) for pattern in context_break_patterns)

    @staticmethod
    def _has_question_mark(text: str) -> bool:
        return bool(re.search(r"[?\uff1f]", text or ""))

    @classmethod
    def _direct_closing_response(cls, user_text: str, history: Iterable[Mapping[str, str]] | None = None) -> str:
        """Return a concise, natural closing without asking the model to improvise."""
        text = (user_text or "").strip()
        lowered = text.lower()
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))

        if has_chinese:
            if "\u660e\u5929" in text:
                return "\u597d\u7684\uff0c\u660e\u5929\u6211\u4eec\u518d\u7ee7\u7eed\u3002"
            if "\u62dc\u62dc" in text or "\u518d\u89c1" in text:
                return "\u597d\u7684\uff0c\u62dc\u62dc\uff0c\u4e0b\u6b21\u89c1\u3002"
            if "\u8c22\u8c22" in text:
                if "\u4e0d\u7528\u7ee7\u7eed\u95ee" in text or "\u4e0d\u7528\u518d\u95ee" in text:
                    return "\u4e0d\u5ba2\u6c14\uff0c\u4e0d\u7ee7\u7eed\u95ee\u4e86\u3002"
                return "\u4e0d\u5ba2\u6c14\uff0c\u4eca\u5929\u5148\u5230\u8fd9\u91cc\u3002"
            if "\u5fd9" in text:
                return "\u597d\u7684\uff0c\u5148\u53bb\u5fd9\u5427\u3002"
            if "\u7d2f" in text:
                return "\u597d\u7684\uff0c\u5148\u4f11\u606f\u4e00\u4e0b\u3002"
            if "\u4e0d\u804a" in text:
                return "\u597d\u7684\uff0c\u4eca\u5929\u5148\u4e0d\u804a\u4e86\u3002"
            if "\u4e0d\u7ec3" in text:
                return "\u597d\u7684\uff0c\u6211\u4eec\u5148\u505c\u5728\u8fd9\u91cc\u3002"
            if "\u7ec3\u5f97" in text and "\u5dee\u4e0d\u591a" in text:
                return "\u597d\u7684\uff0c\u4eca\u5929\u7ec3\u5f97\u5dee\u4e0d\u591a\u4e86\u3002"
            if "\u8fd9\u6bb5\u5bf9\u8bdd" in text:
                return "\u597d\u7684\uff0c\u8fd9\u6bb5\u5bf9\u8bdd\u7ed3\u675f\u3002"
            if "\u8fd9\u4e2a\u7ec3\u4e60" in text:
                return "\u597d\u7684\uff0c\u8fd9\u4e2a\u7ec3\u4e60\u5148\u7ed3\u675f\u3002"
            if "\u8fd9\u8f6e" in text:
                return "\u597d\u7684\uff0c\u8fd9\u8f6e\u7ec3\u4e60\u7ed3\u675f\u3002"
            if "\u4eca\u5929\u7684\u7ec3\u4e60" in text:
                return "\u597d\u7684\uff0c\u4eca\u5929\u7684\u7ec3\u4e60\u7ed3\u675f\u3002"
            if "\u8fd9\u6b21\u7ec3\u4e60" in text:
                return "\u6ca1\u95ee\u9898\uff0c\u8fd9\u6b21\u7ec3\u4e60\u5230\u8fd9\u91cc\u7ed3\u675f\u3002"
            if "\u53ef\u4ee5\u7ed3\u675f" in text:
                return "\u53ef\u4ee5\uff0c\u4eca\u5929\u7684\u7ec3\u4e60\u5230\u8fd9\u91cc\u7ed3\u675f\u3002"
            return "\u597d\u7684\uff0c\u4eca\u5929\u5148\u5230\u8fd9\u91cc\u3002"

        if "thank you, bye" in lowered or lowered.strip() in {"bye", "goodbye"}:
            return "You're welcome. Bye."
        if "see you next time" in lowered:
            return "You're welcome. See you next time."
        if "no more questions" in lowered:
            return "Of course. No more questions."
        if "i need to go" in lowered:
            return "Okay. See you later."
        if "tired" in lowered:
            return "Sure. Rest a bit."
        if "can we finish here" in lowered:
            return "Yes. We can finish here."
        if "end the role-play" in lowered:
            return "Sure. The role-play is finished."
        if "end this practice" in lowered:
            return "Okay. We can end this practice."
        if "stop here" in lowered or "let's stop here" in lowered:
            return "Sure. We'll stop here."
        if "that's all for today" in lowered:
            return "Okay. That's all for today."
        if "that's enough" in lowered:
            return "Okay. That's enough for now."
        if "i'm finished" in lowered:
            return "Sure. We are finished for today."
        if "i'm done" in lowered:
            return "No problem. We can stop here."
        return cls._scene_aware_english_closing(history or [])

    @classmethod
    def _scene_aware_english_closing(cls, history: Iterable[Mapping[str, str]]) -> str:
        text = " ".join((message.get("content", "") or "") for message in cls._normalized_history(history)).lower()
        if re.search(r"\b(airport|check[- ]?in|passport|flight|boarding)\b", text):
            return "Thank you. Have a safe trip!"
        if re.search(r"\b(restaurant|order|ordering|waiter|food|meal|beef noodles|cola)\b", text):
            return "Thank you. Have a great day!"
        if re.search(r"\b(interview|job|excel)\b", text):
            return "Thank you for practicing. Have a great day!"
        if re.search(r"\b(campus|library|class)\b", text):
            return "Okay, see you later!"
        return "Thanks for the conversation. Have a great day!"

    @classmethod
    def _is_greeting(cls, text: str) -> bool:
        return cls._matches_any(text or "", cls.GREETING_PATTERNS)

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
    def _looks_like_relaxed_correction_response(cls, text: str, target_text: str = "") -> bool:
        raw = (text or "").strip()
        if not raw:
            return False

        normalized = re.sub(r"\s+", " ", raw).strip().lower()
        if any(
            phrase in normalized
            for phrase in (
                "please send it again",
                "please send it once more",
                "please send the sentence",
                "please provide the sentence",
            )
        ):
            return False
        if any(
            phrase in raw
            for phrase in (
                "请再发一次",
                "请把你想纠正",
                "请把句子发给我",
            )
        ):
            return False

        has_english = bool(re.search(r"[A-Za-z]{2,}", raw))
        if not has_english:
            return False

        if "更自然" in raw or "语法" in raw or "说明" in raw:
            return True
        if any(
            cue in normalized
            for cue in (
                "more natural",
                "better version",
                "better way",
                "you can say",
                "a natural way",
                "correct version",
                "i'd say",
                "try saying",
            )
        ):
            return True

        target_norm = cls._normalize_statement(target_text)
        response_norm = cls._normalize_statement(raw)
        if target_norm and response_norm:
            if response_norm == target_norm:
                return False
            similarity = SequenceMatcher(None, response_norm, target_norm).ratio()
            if similarity >= 0.95:
                return False
            return True

        return True

    @classmethod
    def _fallback_roleplay_response(cls, state: TeachingSessionState, latest_user_text: str) -> str:
        text = (latest_user_text or "").strip()
        lowered = text.lower()
        scenario_context = " ".join([state.scenario_setup, *state.user_facts, text]).lower()

        if cls._matches_any(text, cls.NO_REPEAT_PATTERNS):
            return "好的，我不再重复你的回答。Let's continue: What date would you like to fly?"

        if cls._matches_any(text, cls.TEACHER_ASK_PATTERNS):
            return "Okay, I'll ask the questions. What date would you like to travel?"

        if "pay" in lowered or "ticket" in lowered:
            return "Of course. You can pay by card or cash. Which payment method would you prefer?"

        if re.search(r"\bshop\b|\bbuy\b|\bstore\b|\bitem\b|\bblack\b|\bcolor\b|\bcolour\b|\bsize\b|\bcaller\b", scenario_context):
            if re.search(r"\bblack\b|\bcolor\b|\bcolour\b|\bcaller\b", lowered):
                return "Yes, we have it in black. What size would you like?"
            if re.search(r"\bhow much\b|\bprice\b|\bcost\b", lowered):
                return "It costs 25 dollars. Would you like to try it on?"
            return "Sure. What color or size would you like?"

        if re.search(r"\bbook\b|\bflight\b|\bticket\b|\bplane\b|\bairport\b", lowered):
            if re.search(r"\bfrom\b.+\bto\b", lowered):
                return "Good. What date would you like to travel?"
            return "Sure. Where would you like to fly from and to?"

        if re.search(r"\bwaiter\b|\brestaurant\b|\border\b|\bfood\b|\bmenu\b|\bmeal\b|\bchicken\b|\brice\b", scenario_context):
            return "Sure. Would you like anything to drink with your meal?"

        if re.search(r"\bcold\b|\bdrink\b|\bwater\b|\bcoffee\b|\btea\b", lowered):
            return "Sure. Would you like a cold drink or something warm?"

        if re.search(r"\bweather\b|\brain\b|\braining\b|\braining\b|\bsunny\b|\bcold\b|\bhot\b", lowered):
            return "Let's continue with the weather. What did you do when it started raining?"

        if state.teacher_should_ask:
            return "Okay, I'll lead the practice. Can you answer this in English: what is your next request?"

        if re.search(r"[A-Za-z]{2,}", text):
            return "Good. Can you add one more detail in English?"
        return "Okay, let's continue. Please answer the next step in English."

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
            or cls._is_closing_request(stripped)
            or cls._is_language_switch_to_chinese(stripped)
            or cls._is_translation_request(stripped)
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

    @staticmethod
    def _is_repeat_request(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text or "").strip().lower()
        patterns = (
            r"\brepeat\b",
            r"\bsay (it|that) again\b",
            r"\bone more time\b",
            r"\bagain please\b",
            r"\u518d\u8bf4\u4e00\u904d",
            r"\u91cd\u590d\u4e00\u4e0b",
            r"\u518d\u6765\u4e00\u904d",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def _repeats_recent_statement(cls, response: str, recent_texts: List[str]) -> bool:
        response_norm = cls._normalize_statement(response)
        if not response_norm:
            return False
        for recent_text in recent_texts:
            recent_norm = cls._normalize_statement(recent_text)
            if not recent_norm:
                continue
            if response_norm == recent_norm:
                return True
            similarity = SequenceMatcher(None, response_norm, recent_norm).ratio()
            if similarity >= 0.92:
                return True
        return False

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
    def _normalize_statement(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (text or "").lower())
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
