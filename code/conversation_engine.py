#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation engine for Qwen-based dialogue generation.
Supports loading a local base model and optionally attaching a LoRA adapter.
"""

import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

from config import DEFAULT_QWEN_MODEL, LEARNING_KEYWORDS, GREETING_KEYWORDS


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
        elif not self.lora_adapter_path:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env_adapter_path = os.environ.get("QWEN_LORA_ADAPTER_PATH")
            default_adapter_path = os.path.join(project_root, "distill", "output", "lora_sft_v7")
            candidate_adapter_path = env_adapter_path or default_adapter_path
            if os.path.exists(candidate_adapter_path):
                self.lora_adapter_path = candidate_adapter_path

        print("Starting Qwen model initialization...")

        if self.local_model_path and os.path.exists(self.local_model_path):
            print(f"Using local base model path: {self.local_model_path}")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Running on device: {device}")

                if torch.cuda.is_available():
                    print(f"CUDA version: {torch.version.cuda}")
                    print(f"GPU: {torch.cuda.get_device_name(0)}")

                tokenizer_path = self.local_model_path
                if (
                    self.lora_adapter_path
                    and os.path.exists(self.lora_adapter_path)
                    and os.path.exists(os.path.join(self.lora_adapter_path, "tokenizer_config.json"))
                ):
                    tokenizer_path = self.lora_adapter_path

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
                        print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
                        self.qwen_model = PeftModel.from_pretrained(
                            self.qwen_model,
                            self.lora_adapter_path,
                            local_files_only=True
                        )
                        print("LoRA adapter loaded successfully")
                    else:
                        print(f"LoRA adapter path not found: {self.lora_adapter_path}")

                self.qwen_model.eval()
                return True
            except Exception as exc:
                print(f"Local model loading failed: {exc}")
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

    def generate_response(self, text_input, max_length=512, use_context=True, allow_long_response=False, medium_response=False):
        """Generate a conversational response."""
        print("Generating response...")

        if self.qwen_model is None or self.qwen_tokenizer is None:
            return "Hello! How can I help you?"

        try:
            system_msg = """You are a friendly conversational partner helping someone practice English.
Rules:
1. Give detailed, natural responses (20-40 words)
2. Be conversational, warm and encouraging
3. After answering, guide the conversation forward by:
   - Asking relevant follow-up questions
   - Showing interest in what they said
   - Naturally extending the topic
4. Use complete sentences and natural English expressions
5. NEVER simulate dialogues like "Human: ... Assistant: ..."
6. Help maintain an engaging, flowing conversation
7. Adapt to any topic naturally - daily life, travel, business, casual chat, etc.
8. Respond as a real person would in that situation

Example:
User: "I'm planning a trip next month"
You: "That sounds exciting! Where are you planning to go? Have you been there before, or will this be your first time visiting?\""""

            if use_context:
                prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{text_input}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if not text_input.startswith("<|im_start|>"):
                    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n{text_input}"
                else:
                    if "<|im_start|>system" not in text_input:
                        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n{text_input}"
                    else:
                        prompt = text_input

            inputs = self.qwen_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            max_tokens = 120

            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.qwen_tokenizer.eos_token_id,
                    eos_token_id=self.qwen_tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )

            response = self.qwen_tokenizer.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            from text_processor import TextProcessor
            response = TextProcessor.clean_response(response)

            response = re.sub(r"^(Human|Assistant|User|System)\s*:\s*", "", response, flags=re.IGNORECASE)
            response = re.sub(r"\b(Human|Assistant|User|System)\s*:\s*", "", response, flags=re.IGNORECASE)

            if not response or len(response) < 3:
                return "Hello! How can I help you?"

            print(f"Generated response: {response}")
            return response

        except Exception as exc:
            print(f"Response generation failed: {exc}")
            return "Hello! How can I help you?"
