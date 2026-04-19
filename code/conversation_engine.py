#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation engine for Qwen-based dialogue generation.
Supports loading a local base model and optionally attaching a LoRA adapter.
"""

import os
import re
import threading
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import torch

from config import DEFAULT_QWEN_MODEL, LEARNING_KEYWORDS, GREETING_KEYWORDS
from dialogue_policy import DialoguePolicy


def _env_lora_enabled():
    raw_variant = (
        os.environ.get("QWEN_MODEL_VARIANT")
        or os.environ.get("QWEN_DEFAULT_MODEL_VARIANT")
        or os.environ.get("BACKEND_MODEL_VARIANT")
        or ""
    )
    return str(raw_variant).strip().lower() in {"lora", "finetuned", "fine_tuned", "ft", "sft"}


class ConversationEngine:
    """Handles model loading and response generation."""

    def __init__(self):
        self.qwen_tokenizer = None
        self.qwen_model = None
        self.model_name = DEFAULT_QWEN_MODEL
        self.local_model_path = None
        self.lora_adapter_path = None

    def init_model(self, model_name=None, local_model_path=None, lora_adapter_path=None):
        """Initialize the base model and optionally load a LoRA adapter."""
        if model_name:
            self.model_name = model_name
        if local_model_path:
            self.local_model_path = local_model_path
        if (
            self.local_model_path
            and not os.path.exists(self.local_model_path)
            and os.environ.get("QWEN_BASE_MODEL_PATH")
            and os.path.exists(os.environ.get("QWEN_BASE_MODEL_PATH"))
        ):
            self.local_model_path = os.environ.get("QWEN_BASE_MODEL_PATH")
        if lora_adapter_path:
            self.lora_adapter_path = lora_adapter_path
        elif _env_lora_enabled() and os.environ.get("QWEN_LORA_ADAPTER_PATH"):
            self.lora_adapter_path = os.environ.get("QWEN_LORA_ADAPTER_PATH")
        else:
            self.lora_adapter_path = None

        print("Starting Qwen model initialization...")
        if self.lora_adapter_path:
            print(f"LoRA adapter requested: {self.lora_adapter_path}")

        if self.local_model_path and os.path.exists(self.local_model_path):
            print(f"Using local base model path: {self.local_model_path}")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Running on device: {device}")

                if torch.cuda.is_available():
                    print(f"CUDA version: {torch.version.cuda}")
                    print(f"GPU: {torch.cuda.get_device_name(0)}")

                # LoRA adapters are weight deltas. Keep the tokenizer tied to the
                # base model to avoid adapter-side tokenizer_config incompatibilities.
                tokenizer_path = self.local_model_path
                print(f"Using tokenizer path: {tokenizer_path}")

                self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=True,
                    local_files_only=True
                )

                if torch.cuda.is_available():
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        self.local_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=True
                    )

                    try:
                        self.qwen_model = self.qwen_model.to_bettertransformer()
                        print("BetterTransformer optimization enabled")
                    except Exception as exc:
                        print(f"BetterTransformer optimization skipped: {exc}")
                else:
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        self.local_model_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    print("Base model loaded on CPU")

                if self.lora_adapter_path:
                    if os.path.exists(self.lora_adapter_path):
                        from peft import PeftModel

                        print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
                        self.qwen_model = PeftModel.from_pretrained(
                            self.qwen_model,
                            self.lora_adapter_path,
                            local_files_only=True
                        )
                        print("LoRA adapter loaded successfully")
                    else:
                        print(f"LoRA adapter path not found: {self.lora_adapter_path}")
                        return False

                self.qwen_model.eval()
                return True
            except Exception as exc:
                print(f"Local model loading failed: {exc}")
                traceback.print_exc()
                if self.lora_adapter_path:
                    print("LoRA model initialization failed; not falling back to remote base model.")
                    return False
                print("Falling back to remote model...")

        try:
            print("Loading remote model from Hugging Face...")
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if torch.cuda.is_available():
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.qwen_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
            self.qwen_model.eval()
            print("Remote Qwen model loaded successfully")
            return True
        except Exception as exc:
            print(f"Remote model loading failed: {exc}")
            print("Conversation model initialization failed")
            return False

    def generate_response(self, text_input, max_length=2048, use_context=True, allow_long_response=False, medium_response=False):
        """Generate a conversational response."""
        print("Generating response...")

        if self.qwen_model is None or self.qwen_tokenizer is None:
            return "Hello! How can I help you?"

        try:
            if use_context:
                prompt = DialoguePolicy.build_chat_prompt([], text_input)
            else:
                prompt = DialoguePolicy.ensure_system_prompt(text_input)
            prompt = self._disable_qwen_thinking_if_supported(prompt)

            direct_response = DialoguePolicy.direct_response(prompt)
            if direct_response:
                print(f"Generated response: {direct_response}")
                return direct_response

            latest_user_text = DialoguePolicy.extract_latest_user_text(prompt)
            prompt_history = DialoguePolicy.extract_prompt_history(prompt)
            generation_config = DialoguePolicy.generation_config(latest_user_text, prompt_history)
            max_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", generation_config["max_new_tokens"]))
            temperature = float(os.environ.get("QWEN_TEMPERATURE", generation_config["temperature"]))
            top_p = float(os.environ.get("QWEN_TOP_P", generation_config["top_p"]))
            repetition_penalty = float(
                os.environ.get("QWEN_REPETITION_PENALTY", generation_config["repetition_penalty"])
            )

            response = self._generate_text_from_prompt(
                prompt,
                max_length=max_length,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            response = self._repair_response_with_policy(
                prompt,
                response,
                max_length=max_length,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if not response or len(response) < 3:
                return "Hello! How can I help you?"

            print(f"Generated response: {response}")
            return response

        except Exception as exc:
            print(f"Response generation failed: {exc}")
            return "Hello! How can I help you?"

    def _generate_text_from_prompt(
        self,
        prompt,
        max_length,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
    ):
        """Run one model generation pass and return cleaned assistant text."""
        prompt = self._disable_qwen_thinking_if_supported(prompt)
        inputs = self.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
                eos_token_id=self.qwen_tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )

        response = self.qwen_tokenizer.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        return self._clean_generated_response(response)

    def _repair_response_with_policy(
        self,
        prompt,
        response,
        max_length,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
    ):
        """Retry policy-rejected responses before using deterministic fallback."""
        if os.environ.get("DIALOGUE_POLICY_REPAIR_ENABLED", "1").lower() not in {"1", "true", "yes", "on"}:
            return response

        max_retries = self._policy_retry_count()
        for retry_index in range(1, max_retries + 1):
            if not DialoguePolicy.response_needs_retry(prompt, response):
                return response

            print(
                "Generated response rejected by dialogue policy, "
                f"retry {retry_index}/{max_retries}..."
            )
            print(f"Rejected draft: {response[:240]}")
            retry_prompt = DialoguePolicy.build_retry_prompt(prompt, response)
            latest_user_text = DialoguePolicy.extract_latest_user_text(prompt)
            prompt_history = DialoguePolicy.extract_prompt_history(prompt)
            retry_mode = DialoguePolicy.classify_user_intent(latest_user_text, prompt_history)
            retry_temperature = max(temperature, 0.45) if retry_mode == "roleplay" else min(temperature, 0.25)
            retry_top_p = max(top_p, 0.85) if retry_mode == "roleplay" else min(top_p, 0.8)
            response = self._generate_text_from_prompt(
                retry_prompt,
                max_length=max_length,
                max_tokens=max_tokens,
                temperature=retry_temperature,
                top_p=retry_top_p,
                repetition_penalty=max(repetition_penalty, 1.12),
            )

        if DialoguePolicy.response_needs_retry(prompt, response):
            print(
                "Generated response rejected after "
                f"{max_retries} retries, using dialogue fallback."
            )
            return DialoguePolicy.fallback_response(prompt)
        return response

    @staticmethod
    def _policy_retry_count():
        """Default to at least three repair attempts before fallback."""
        try:
            configured = int(os.environ.get("DIALOGUE_POLICY_MAX_RETRIES", "3"))
        except ValueError:
            configured = 3
        return max(3, configured)

    @staticmethod
    def _clean_generated_response(response):
        from text_processor import TextProcessor

        cleaned = TextProcessor.clean_response(response)
        cleaned = re.sub(r"^(Human|Assistant|User|System)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(Human|Assistant|User|System)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def build_prompt(self, text_input, use_context=True):
        """Build a chat prompt that matches the model's current formatting."""
        if use_context:
            return DialoguePolicy.build_chat_prompt([], text_input)

        if not text_input.startswith("<|im_start|>"):
            return DialoguePolicy.ensure_system_prompt(text_input)
        return DialoguePolicy.ensure_system_prompt(text_input)

    def generate_response_stream(
        self,
        text_input,
        max_length=2048,
        max_new_tokens=120,
        use_context=True,
        temperature=0.7,
        top_p=0.9,
    ):
        """Yield incremental response text chunks for realtime clients."""
        if self.qwen_model is None or self.qwen_tokenizer is None:
            yield "Hello! How can I help you?"
            return

        prompt = self.build_prompt(text_input, use_context=use_context)
        prompt = self._disable_qwen_thinking_if_supported(prompt)
        inputs = self.qwen_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        streamer = TextIteratorStreamer(
            self.qwen_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=float(os.environ.get("QWEN_STREAM_TIMEOUT", "30")),
        )

        latest_user_text = DialoguePolicy.extract_latest_user_text(prompt)
        prompt_history = DialoguePolicy.extract_prompt_history(prompt)
        policy_config = DialoguePolicy.generation_config(latest_user_text, prompt_history)
        effective_max_new_tokens = int(
            os.environ.get("QWEN_STREAM_MAX_NEW_TOKENS", max(max_new_tokens, policy_config["max_new_tokens"]))
        )
        effective_temperature = float(os.environ.get("QWEN_STREAM_TEMPERATURE", policy_config["temperature"]))
        effective_top_p = float(os.environ.get("QWEN_STREAM_TOP_P", policy_config["top_p"]))
        effective_repetition_penalty = float(
            os.environ.get("QWEN_STREAM_REPETITION_PENALTY", policy_config["repetition_penalty"])
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=effective_max_new_tokens,
            do_sample=True,
            temperature=effective_temperature,
            top_p=effective_top_p,
            pad_token_id=self.qwen_tokenizer.eos_token_id,
            eos_token_id=self.qwen_tokenizer.eos_token_id,
            repetition_penalty=effective_repetition_penalty,
        )

        def generation_worker():
            try:
                self.qwen_model.generate(**generation_kwargs)
            except Exception as exc:
                print(f"Streaming response generation failed: {exc}")

        worker = threading.Thread(target=generation_worker, daemon=True)
        worker.start()

        try:
            for piece in streamer:
                from text_processor import TextProcessor
                cleaned_piece = TextProcessor.remove_thinking(piece.replace("<|im_end|>", ""))
                if cleaned_piece:
                    yield cleaned_piece
        except Exception as exc:
            print(f"Streaming response iterator stopped: {exc}")

    def _disable_qwen_thinking_if_supported(self, prompt: str) -> str:
        """Use Qwen3 chat template switch when the installed tokenizer supports it."""
        if os.environ.get("QWEN_ENABLE_THINKING", "0").lower() in {"1", "true", "yes", "on"}:
            return prompt
        if not prompt or not prompt.startswith("<|im_start|>"):
            return prompt
        if not hasattr(self.qwen_tokenizer, "apply_chat_template"):
            return prompt

        try:
            messages = self._parse_chat_prompt(prompt)
            if not messages:
                return prompt
            return self.qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return prompt
        except Exception as exc:
            print(f"Qwen thinking disable skipped: {exc}")
            return prompt

    @staticmethod
    def _parse_chat_prompt(prompt: str) -> list:
        matches = re.findall(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", prompt, flags=re.DOTALL)
        messages = [
            {"role": role, "content": content.strip()}
            for role, content in matches
            if content.strip()
        ]
        if prompt.rstrip().endswith("<|im_start|>assistant"):
            return messages
        if messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
            messages = messages[:-1]
        return messages
