# deepseek_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from deep_translator import GoogleTranslator
import langdetect
from config import get_config


class DeepSeekAssistant:
    def __init__(self, mode="balanced"):
        self.model = None
        self.tokenizer = None
        self.context_window = []
        self.config = get_config(mode)()
        self.mode = mode
        self.load_model()

    def load_model(self):
        print(f"กำลังโหลดโมเดลในโหมด {self.mode}...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME, use_fast=True, legacy=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=self.config.DEVICE_MAP,
            max_memory=self.config.MAX_MEMORY,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        print(f"โหลดโมเดลเสร็จแล้ว! พร้อมใช้งานในโหมด {self.mode}")

    def _build_prompt(self, message, history):
        """Convert chat history to prompt format"""
        context = ""
        if self.context_window:
            formatted_history = []
            for msg in history:
                if msg["role"] == "user":
                    formatted_history.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    formatted_history.append(f"Assistant: {msg['content']}")
            context = "\nPrevious context:\n" + "\n".join(formatted_history[-6:])

        return f"{self.config.SYSTEM_PROMPT}\n{context}\nUser: {message}\nAssistant: "

    def generate_response(self, message, history):
        try:
            print(f"Original message: {message}")
            chinese_message = self.translate_text(message, "zh-CN")
            print(f"Chinese translation: {chinese_message}")

            prompt = self._build_prompt(chinese_message, history)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.MAX_INPUT_LENGTH,
                padding=True,
            ).to("cuda:0")

            generation_config = {
                "max_new_tokens": self.config.MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "top_k": self.config.TOP_K,
                "num_beams": self.config.NUM_BEAMS,
                "pad_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": self.config.REPETITION_PENALTY,
                "no_repeat_ngram_size": self.config.NO_REPEAT_NGRAM_SIZE,
            }

            if hasattr(self.config, "LENGTH_PENALTY") and self.config.NUM_BEAMS > 1:
                generation_config["length_penalty"] = self.config.LENGTH_PENALTY

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()
            print(f"AI response: {response}")

            detected_lang = self.detect_language(response)
            if detected_lang == "zh-CN":
                response = self.translate_text(response, "th")
                print(f"Translated response: {response}")

            if len(self.context_window) >= self.config.MAX_CONTEXT_LENGTH:
                self.context_window.pop(0)
            self.context_window.append({"role": "user", "content": message})
            self.context_window.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            return f"ขออภัย เกิดข้อผิดพลาด: {str(e)}"

    def detect_language(self, text):
        try:
            return langdetect.detect(text)
        except:
            return "en"

    def translate_text(self, text, target_language):
        try:
            if not text.strip():
                return text
            translator = GoogleTranslator(source="auto", target=target_language)
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

    def clear_context(self):
        self.context_window.clear()

    def change_mode(self, new_mode):
        if new_mode != self.mode:
            self.mode = new_mode
            self.config = get_config(new_mode)()
            print(f"เปลี่ยนโหมดเป็น {new_mode} สำเร็จ")
