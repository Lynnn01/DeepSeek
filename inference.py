import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto"
    )

    print("\nModel loaded successfully! You can start chatting (type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nDeepSeek:", response)


if __name__ == "__main__":
    main()
