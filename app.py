import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import time


class DeepSeekChat:
    def __init__(self):
        self.model = None
        self.tokenizer = None
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

    def generate_response(self, message, history):
        prompt = f"You are a helpful AI assistant. Please respond in Thai.\nQuestion: {message}\nAnswer in Thai: "

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256, padding=True
        ).to("cuda:0")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                length_penalty=1.0,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Answer in Thai:")[-1].strip()
        time.sleep(0.5)
        return response


# สร้างอินสแตนซ์ของ DeepSeek
assistant = DeepSeekChat()


def user_message(message, history):
    if message.strip() == "":
        return "", history
    history = history + [[message, None]]
    return "", history


def bot_message(history):
    if len(history) == 0:
        return history
    if history[-1][1] is not None:
        return history

    message = history[-1][0]
    response = assistant.generate_response(message, history)
    history[-1][1] = response
    return history


css = """
.container {max-width: 800px; margin: auto; padding: 20px;}
.chat-message {padding: 15px; border-radius: 10px; margin: 5px;}
.user-message {background-color: #e3f2fd; text-align: right;}
.bot-message {background-color: #f5f5f5;}
.header {text-align: center; padding: 20px; background: #2196F3; color: white; border-radius: 10px;}
"""

with gr.Blocks(css=css) as interface:
    gr.Markdown(
        """
        <div class="header">
            <h1>🤖 DeepSeek Chat Assistant</h1>
            <p>AI ผู้ช่วยอัจฉริยะ พร้อมตอบทุกคำถามของคุณ</p>
        </div>
        """
    )

    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        layout="bubble",
        bubble_full_width=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="พิมพ์ข้อความของคุณที่นี่...",
            container=False,
            scale=8,
        )
        submit = gr.Button("ส่ง", variant="primary", scale=1, min_width=100)

    with gr.Row():
        clear = gr.Button("ล้างการสนทนา", variant="secondary")
        retry = gr.Button("ลองใหม่", variant="secondary")

    gr.Examples(
        examples=[
            "ช่วยอธิบายเรื่อง Artificial Intelligence",
            "เขียนโค้ด Python สำหรับการคำนวณเลขคณิต",
            "แนะนำวิธีการเรียนรู้การเขียนโปรแกรม",
        ],
        inputs=msg,
        label="ตัวอย่างคำถาม",
    )

    submit.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_message, chatbot, chatbot
    )

    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_message, chatbot, chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)
    retry.click(bot_message, chatbot, chatbot)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=3000, share=True, debug=True)
