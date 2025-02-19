# deepseek_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from deep_translator import GoogleTranslator
import langdetect
from config import get_config


class DeepSeekAssistant:
    def __init__(self, mode="balanced"):
        """
        Initialize DeepSeek Assistant with specified mode

        Args:
            mode (str): Operation mode ("fast", "balanced", or "smart")
        """
        self.model = None
        self.tokenizer = None
        self.context_window = []
        self.config = get_config(
            mode
        )()  # Get and instantiate config for specified mode
        self.mode = mode
        self.load_model()

    def load_model(self):
        """Load the model with appropriate configuration"""
        print(f"กำลังโหลดโมเดลในโหมด {self.mode}...")

        # Quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME, use_fast=True, legacy=False
        )

        # Load model
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

    def detect_language(self, text):
        """
        Detect the language of input text

        Args:
            text (str): Text to detect language

        Returns:
            str: Detected language code
        """
        try:
            return langdetect.detect(text)
        except:
            return "en"

    def translate_text(self, text, target_language):
        """
        Translate text to target language

        Args:
            text (str): Text to translate
            target_language (str): Target language code

        Returns:
            str: Translated text
        """
        try:
            if not text.strip():
                return text
            translator = GoogleTranslator(source="auto", target=target_language)
            return translator.translate(text)
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

    def _build_prompt(self, message, history):
        """
        Build prompt with system prompt and context

        Args:
            message (str): User message
            history (list): Conversation history

        Returns:
            str: Complete prompt
        """
        context = ""
        if self.context_window:
            context = "\nPrevious context:\n" + "\n".join(
                [f"Q: {q}\nA: {a}" for q, a in self.context_window[-3:]]
            )
        return f"{self.config.SYSTEM_PROMPT}\n{context}\nQuestion: {message}\nAnswer: "

    def generate_response(self, message, history):
        """
        Generate response for user message

        Args:
            message (str): User message
            history (list): Conversation history

        Returns:
            str: Generated response
        """
        try:
            print(f"Original message: {message}")
            chinese_message = self.translate_text(message, "zh-CN")
            print(f"Chinese translation: {chinese_message}")

            # Build and tokenize prompt
            prompt = self._build_prompt(chinese_message, history)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.MAX_INPUT_LENGTH,
                padding=True,
            ).to("cuda:0")

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                    top_k=self.config.TOP_K,
                    num_beams=self.config.NUM_BEAMS,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=self.config.REPETITION_PENALTY,
                    length_penalty=self.config.LENGTH_PENALTY,
                    no_repeat_ngram_size=self.config.NO_REPEAT_NGRAM_SIZE,
                )

            # Process response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Answer:")[-1].strip()
            print(f"AI response: {response}")

            # Translate if needed
            detected_lang = self.detect_language(response)
            if detected_lang not in ["th", "en"]:
                response = self.translate_text(response, "th")
                print(f"Translated response: {response}")

            # Update context window
            if len(self.context_window) >= self.config.MAX_CONTEXT_LENGTH:
                self.context_window.pop(0)
            self.context_window.append((message, response))

            return response

        except Exception as e:
            return f"ขออภัย เกิดข้อผิดพลาด: {str(e)}"

    def clear_context(self):
        """Clear conversation context"""
        self.context_window.clear()

    def change_mode(self, new_mode):
        """
        Change operation mode

        Args:
            new_mode (str): New mode to switch to ("fast", "balanced", or "smart")
        """
        if new_mode != self.mode:
            self.mode = new_mode
            self.config = get_config(new_mode)()
            print(f"เปลี่ยนโหมดเป็น {new_mode} สำเร็จ")

