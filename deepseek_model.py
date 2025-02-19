import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from deep_translator import GoogleTranslator # type: ignore
import langdetect # type: ignore
import time


class DeepSeekAssistant:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.context_window = []
        self.load_model()

    def load_model(self):
        print("กำลังโหลดโมเดล...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base", use_fast=True, legacy=False
        )

        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.layers": "cuda:0",
            "model.norm": "cuda:0",
            "lm_head": "cuda:0",
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory={"cuda:0": "3GB"},
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        print("โหลดโมเดลเสร็จแล้ว!")

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

    def _build_prompt(self, message, history):
        system_prompt = """You are an intelligent AI assistant. Please provide detailed and accurate responses.
        If the question is about:
        - Programming: Include code examples and explanations
        - Math: Show step-by-step solutions
        - General Knowledge: Provide comprehensive explanations with examples
        - Technical Topics: Break down complex concepts
        """

        context = ""
        if self.context_window:
            context = "\nPrevious context:\n" + "\n".join(
                [f"Q: {q}\nA: {a}" for q, a in self.context_window[-3:]]
            )

        return f"{system_prompt}\n{context}\nQuestion: {message}\nAnswer: "

    def generate_response(self, message, history):
        try:
            print(f"Original message: {message}")

            # 1. แปลเป็นภาษาจีน
            chinese_message = self.translate_text(message, "zh-CN")
            print(f"Chinese translation: {chinese_message}")

            # 2. สร้าง prompt และรับคำตอบจาก AI
            prompt = self._build_prompt(chinese_message, history)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to("cuda:0")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.92,
                    top_k=40,
                    num_beams=2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Answer:")[-1].strip()
            print(f"AI response: {response}")

            # 3. ตรวจสอบภาษาและแปลถ้าจำเป็น
            detected_lang = self.detect_language(response)
            if detected_lang not in ["th", "en"]:
                response = self.translate_text(response, "th")
                print(f"Translated response: {response}")

            if len(self.context_window) >= 5:
                self.context_window.pop(0)
            self.context_window.append((message, response))

            return response

        except Exception as e:
            return f"ขออภัย เกิดข้อผิดพลาด: {str(e)}"

    def clear_context(self):
        self.context_window.clear()
