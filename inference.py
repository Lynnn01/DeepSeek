import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse


def initialize_model():
    print("กำลังโหลดโมเดล...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            use_fast=True,
            legacy=False,  # ใช้ tokenizer รุ่นใหม่
        )

        device_map = {
            "model.embed_tokens": "cuda:0",
            "model.layers": "cuda:0",
            "model.norm": "cuda:0",
            "lm_head": "cuda:0",
        }

        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-base",
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=device_map,
            max_memory={"cuda:0": "3GB"},
            low_cpu_mem_usage=True,
        )

        if model is None:
            raise ValueError("โมเดลโหลดไม่สำเร็จ")

        model.eval()
        return model, tokenizer

    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
        raise


def generate_response(model, tokenizer, user_input):
    # ปรับ prompt ให้ชัดเจนขึ้น
    prompt = f"You are a helpful AI assistant. Please respond in Thai.\nQuestion: {user_input}\nAnswer in Thai: "

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=256, padding=True
    ).to("cuda:0")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # เพิ่มความยาวคำตอบ
            do_sample=True,  # เปิดใช้ sampling
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # ป้องกันการพูดซ้ำ
            length_penalty=1.0,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # ตัดส่วน prompt ออก
    response = response.split("Answer in Thai:")[-1].strip()
    return response


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        model, tokenizer = initialize_model()
        print("\nพร้อมใช้งานแล้ว! พิมพ์ 'exit' เพื่อออก")

        while True:
            try:
                user_input = input("\nคุณ: ")
                if user_input.lower() == "exit":
                    break

                response = generate_response(model, tokenizer, user_input)
                print("\nDeepSeek:", response)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"\nเกิดข้อผิดพลาดระหว่างการทำงาน: {str(e)}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {str(e)}")


if __name__ == "__main__":
    main()
