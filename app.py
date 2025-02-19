# app.py
import gradio as gr
from deepseek_model import DeepSeekAssistant


class DeepSeekUI:
    def __init__(self):
        self.assistant = DeepSeekAssistant()
        self.css = """
        .container {
            max-width: 850px;
            margin: auto;
            padding: 20px;
        }
        .chat-message {
            padding: 15px;
            border-radius: 15px;
            margin: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
            margin-left: 20%;
        }
        .bot-message {
            background-color: #f5f5f5;
            margin-right: 20%;
        }
        .header {
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .mode-info {
            margin-top: 10px;
            padding: 10px;
            border-radius: 8px;
            background-color: rgba(255,255,255,0.1);
        }
        .examples {
            background-color: #fff;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border: 1px solid #e0e0e0;
        }
        .mode-description {
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
            background-color: #f8f9fa;
            border-left: 4px solid #2196F3;
        }
        """

    def user_message(self, message, history):
        if message.strip() == "":
            return "", history
        history = history + [[message, None]]
        return "", history

    def bot_message(self, history):
        if len(history) == 0 or history[-1][1] is not None:
            return history
        message = history[-1][0]
        response = self.assistant.generate_response(message, history)
        history[-1][1] = response
        return history

    def clear_chat(self):
        self.assistant.clear_context()
        return None

    def change_mode(self, mode, history):
        """
        Change AI mode and update interface
        """
        self.assistant.change_mode(mode)
        return history

    def get_mode_description(self, mode):
        """
        Get description for selected mode
        """
        descriptions = {
            "fast": """🚀 โหมดเร็ว (Fast Mode)
            • ตอบสนองรวดเร็ว ตอบสั้นกระชับ
            • เหมาะสำหรับคำถามทั่วไป การสนทนาเบื้องต้น
            • ใช้ทรัพยากรน้อย""",
            "balanced": """⚖️ โหมดสมดุล (Balanced Mode)
            • สมดุลระหว่างความเร็วและความละเอียด
            • เหมาะสำหรับงานทั่วไป การเขียนโค้ด การอธิบายแนวคิด
            • ให้คำตอบที่ครบถ้วนในเวลาที่เหมาะสม""",
            "smart": """🧠 โหมดฉลาด (Smart Mode)
            • วิเคราะห์ละเอียด ตอบครบถ้วนที่สุด
            • เหมาะสำหรับงานที่ต้องการความแม่นยำสูง
            • ให้คำอธิบายเชิงลึก พร้อมตัวอย่างและแนวทางปฏิบัติ""",
        }
        return descriptions.get(mode, "")

    def create_interface(self):
        with gr.Blocks(css=self.css, theme=gr.themes.Soft()) as interface:
            # Header
            gr.Markdown(
                """
                <div class="header">
                    <h1>🤖 DeepSeek AI Assistant</h1>
                    <p>ผู้ช่วย AI อัจฉริยะ พร้อมตอบทุกคำถามของคุณ</p>
                </div>
                """
            )

            # Mode selector and description
            with gr.Row():
                with gr.Column(scale=1):
                    mode_selector = gr.Radio(
                        choices=["fast", "balanced", "smart"],
                        value="balanced",
                        label="เลือกโหมดการทำงาน",
                        interactive=True,
                    )
                with gr.Column(scale=3):
                    mode_info = gr.Markdown(
                        value=self.get_mode_description("balanced"),
                        elem_classes=["mode-description"],
                    )

            # Chat interface
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=600,
                        show_label=False,
                        layout="bubble",
                        bubble_full_width=False,
                        container=True,
                    )

            # Input area
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="พิมพ์ข้อความของคุณที่นี่...",
                    container=False,
                    scale=8,
                )
                submit = gr.Button("ส่ง", variant="primary", scale=1)

            # Control buttons
            with gr.Row():
                clear = gr.Button("ล้างการสนทนา", variant="secondary")
                retry = gr.Button("ลองใหม่", variant="secondary")

            # Example questions
            gr.Examples(
                examples=[
                    "ช่วยเขียนโค้ด Python สำหรับการจัดการไฟล์ Excel",
                    "อธิบายหลักการทำงานของ Machine Learning แบบเข้าใจง่าย",
                    "วิธีการออกแบบ Database ที่ดีควรทำอย่างไร",
                    "อธิบายวิธีการทำ Data Visualization ด้วย Python",
                ],
                inputs=msg,
                label="ตัวอย่างคำถาม",
            )

            # Event handlers
            mode_selector.change(
                fn=lambda x: self.get_mode_description(x),
                inputs=[mode_selector],
                outputs=[mode_info],
            ).then(
                fn=self.change_mode, inputs=[mode_selector, chatbot], outputs=[chatbot]
            )

            submit.click(
                self.user_message, [msg, chatbot], [msg, chatbot], queue=False
            ).then(self.bot_message, chatbot, chatbot)

            msg.submit(
                self.user_message, [msg, chatbot], [msg, chatbot], queue=False
            ).then(self.bot_message, chatbot, chatbot)

            clear.click(self.clear_chat, None, chatbot, queue=False)
            retry.click(self.bot_message, chatbot, chatbot)

        return interface


if __name__ == "__main__":
    ui = DeepSeekUI()
    interface = ui.create_interface()
    interface.launch(server_name="127.0.0.1", server_port=3000, share=False)
