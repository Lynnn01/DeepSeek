# app.py
import gradio as gr
from deepseek_model import DeepSeekAssistant


class DeepSeekUI:
    def __init__(self):
        self.assistant = DeepSeekAssistant()
        self.current_mode = "balanced"
        self.css = """
        .container { max-width: 850px; margin: auto; padding: 20px; }
        .chat-message { padding: 15px; border-radius: 15px; margin: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .user-message { background-color: #e3f2fd; text-align: right; margin-left: 20%; }
        .bot-message { background-color: #f5f5f5; margin-right: 20%; }
        .header { text-align: center; padding: 25px; background: linear-gradient(135deg, #2196F3, #1976D2);
                 color: white; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .mode-info { margin-top: 10px; padding: 10px; border-radius: 8px; background-color: rgba(255,255,255,0.1); }
        .examples { background-color: #fff; padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid #e0e0e0; }
        .mode-description { margin: 10px 0; padding: 10px; border-radius: 8px; background-color: #f8f9fa; 
                          border-left: 4px solid #2196F3; }
        .mode-status { text-align: center; padding: 8px; margin: 10px 0; border-radius: 4px;
                      background-color: #e8f5e9; color: #2e7d32; font-weight: 500; }
        """

    def user_message(self, message, history):
        """Format and add user message to chat"""
        if message.strip() == "":
            return "", history

        history.append({"role": "user", "content": message})
        return "", history

    def bot_message(self, history):
        """Generate and format bot response"""
        if not history or history[-1]["role"] != "user":
            return history

        message = history[-1]["content"]
        response = self.assistant.generate_response(message, history)

        history.append({"role": "assistant", "content": response})
        return history

    def clear_chat(self):
        """Clear chat history"""
        self.assistant.clear_context()
        return []

    def update_mode_and_ui(self, new_mode, chatbot):
        """Update mode and UI elements"""
        try:
            if new_mode != self.current_mode:
                self.assistant.change_mode(new_mode)
                self.current_mode = new_mode

            mode_names = {"fast": "เร็ว", "balanced": "สมดุล", "smart": "ฉลาด"}
            status = f"✓ เปลี่ยนเป็นโหมด{mode_names.get(new_mode, new_mode)}แล้ว"

            return (
                self.get_mode_description(new_mode),
                status,
                chatbot if chatbot else [],
            )
        except Exception as e:
            return (
                self.get_mode_description(self.current_mode),
                f"❌ เกิดข้อผิดพลาด: {str(e)}",
                chatbot if chatbot else [],
            )

    def get_mode_description(self, mode):
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
            gr.Markdown(
                """
                <div class="header">
                    <h1>🤖 DeepSeek AI Assistant</h1>
                    <p>ผู้ช่วย AI อัจฉริยะ พร้อมตอบทุกคำถามของคุณ</p>
                </div>
                """
            )

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

            status_msg = gr.Markdown(
                value="✓ เริ่มต้นในโหมดสมดุล", elem_classes=["mode-status"]
            )

            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=600,
                        show_label=False,
                        layout="bubble",
                        bubble_full_width=False,
                        container=True,
                        type="messages",
                        value=[],
                    )

            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="พิมพ์ข้อความของคุณที่นี่...",
                    container=False,
                    scale=8,
                )
                submit = gr.Button("ส่ง", variant="primary", scale=1)

            with gr.Row():
                clear = gr.Button("ล้างการสนทนา", variant="secondary")
                retry = gr.Button("ลองใหม่", variant="secondary")

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
                fn=self.update_mode_and_ui,
                inputs=[mode_selector, chatbot],
                outputs=[mode_info, status_msg, chatbot],
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
