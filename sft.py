import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME =  "Qwen/Qwen3.5-4B"
DATASET_NAME = "Chuanlight/Qwen3.5-4B_GAOKAO_Essay"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs_qwen35_4b_lora")
MAX_SEQ_LEN = 8192


def to_text(example, tokenizer):
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each sample must contain a non-empty 'messages' list.")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(
        lambda x: to_text(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(test_size=0.1, seed=3407)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LEN,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            warmup_steps=70,
            max_steps=700,
            logging_steps=10,
            eval_steps=100,
            save_steps=100,
            bf16=True,
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/adapter")


if __name__ == "__main__":
    main()
