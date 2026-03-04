import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import (
    AutoTokenizer,
    MambaForCausalLM,
    MambaConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    AutoModelForCausalLM
)
import argparse
from datasets import load_dataset
import os

MODEL_NAME = "state-spaces/mamba-130m-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def reinit_mamba(config: MambaConfig, num_layers: int) -> MambaForCausalLM:
    """Reinitializes Mamba weights"""

    # Substitute number of layers in config
    config.num_hidden_layers = num_layers
    config.n_layer = num_layers

    # Return the updated model
    model = AutoModelForCausalLM.from_config(config)
    return model

# Define the callback for measure qualitative progress on epoch end
class GenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt, num_samples = 10, max_new_tokens = 50):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()
        input_ids = self.tokenizer(self.prompt, return_tensors="pt")["input_ids"].to(DEVICE)

        print(f"\n--- Epoch {int(state.epoch)}: Sample Generations... ---")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_k=50,
                num_return_sequences=self.num_samples
            )
        
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, text in enumerate(decoded, 1):
            print(f"\n [Sample {i}]\n{text}\n")

        model.train()

def main():

    # Parsing arguments for training
    p = argparse.ArgumentParser()

    p.add_argument("--lr", type=float, required=True, help="Learning Rate required for model's training.")
    p.add_argument("--bs", type=int, required=True, help="Batch size required for training.")
    p.add_argument("--dataset", type=str, required=True, help="Dataset used for Mamba training")
    p.add_argument("--output_dir", type=str, required=True, help="Where to store training results.")
    p.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    p.add_argument("--logging_dir", type=str, required=True, help="Where to store log files.")
    p.add_argument("--layers", type=int, required=True, help="Number of Mamba Layers.")

    args = p.parse_args()

    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    # Initializes tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # Start with Mamba-130M from scratch
    config = MambaConfig.from_pretrained(MODEL_NAME)
    model = reinit_mamba(config, args.layers)

    # Take dataset (train subsetted for memory saving)
    train_dataset = load_dataset(args.dataset, split="train[:5%]")
    test_dataset = load_dataset(args.dataset, split="validation")

    # Tokenize the datasets
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding=True, return_tensors="pt")
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=test_dataset.column_names)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Define data collator for batches
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        logging_steps=10,
        logging_dir=args.logging_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Define the callback
    prompt = "Once upon a time"
    generation_callback = GenerationCallback(tokenizer, prompt=prompt)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[generation_callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()

