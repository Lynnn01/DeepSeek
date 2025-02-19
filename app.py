import gradio as gr # type: ignore
import time
from deepseek_model import DeepSeekAssistant

# สร้างอินสแตนซ์ของ AI
assistant = DeepSeekAssistant()


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


def clear_chat():
    assistant.clear_context()
    return None


css = """
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
.examples {
    background-color: #fff;
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    border: 1px solid #e0e0e0;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as interface:
    gr.Markdown(
        """
        <div class="header">
            <h1>🤖 DeepSeek AI Assistant</h1>
            <p>ผู้ช่วย AI อัจฉริยะ พร้อมตอบทุกคำถามของคุณ</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=600,
                show_label=False,
                layout="bubble",
                bubble_full_width=False,
                container=True,
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
    submit.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_message, chatbot, chatbot
    )

    msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot_message, chatbot, chatbot
    )

    clear.click(clear_chat, None, chatbot, queue=False)
    retry.click(bot_message, chatbot, chatbot)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=3000, share=False)
