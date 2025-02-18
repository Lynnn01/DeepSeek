import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def initialize_model(model_path):
    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, model_max_length=2048
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder="offload",
    )

    model.eval()
    return model, tokenizer


def generate_response(
    model, tokenizer, user_input, past_key_values=None, max_new_tokens=128
):
    inputs = tokenizer(
        user_input, return_tensors="pt", padding=True, truncation=True
    ).to(model.device)

    with torch.no_grad(), torch.amp.autocast("cuda"):  # แก้ไขการใช้ autocast
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # ใช้แค่ max_new_tokens
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,
            use_cache=True,
            past_key_values=past_key_values
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, outputs.past_key_values


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model, tokenizer = initialize_model(args.model_path)

    print("\nModel loaded successfully! You can start chatting (type 'exit' to quit)")

    past_key_values = None

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break

            response, past_key_values = generate_response(
                model,
                tokenizer,
                user_input,
                past_key_values,
                max_new_tokens=args.max_new_tokens,
            )

            print("\nDeepSeek:", response)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("\nMemory error occurred. Clearing cache...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                past_key_values = None
                continue
            else:
                raise e


if __name__ == "__main__":
    main()
