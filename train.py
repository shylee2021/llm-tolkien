import time
from pathlib import Path

import torch
import vessl
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from llm.dataset import prepare_dataset
from llm.util import prepare_model

LORA_CONFIG = {
    "task_type": "CAUSAL_LM",
    "r": 16,  # attention heads
    "lora_alpha": 32,  # alpha scaling
    "lora_dropout": 0.05,
    "bias": "none",
}

TRAINER_CONFIG = {
    "per_device_train_batch_size": 1,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "logging_steps": 10,
    "output_dir": "/output",
    "eval_interval": 50,
}


def main():
    model_name = "bigscience/bloom-3b"

    n_epochs = TRAINER_CONFIG["num_train_epochs"]
    batch_size = TRAINER_CONFIG["per_device_train_batch_size"]
    learning_rate = TRAINER_CONFIG["learning_rate"]
    eval_interval = TRAINER_CONFIG["eval_interval"]
    log_interval = TRAINER_CONFIG["logging_steps"]
    output_dir = Path(TRAINER_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TODO: Fix train/test split
    dataset_dict = prepare_dataset(
        model_name=model_name,
        dataset_path=Path("./extracted_text.jsonl"),
        min_length=100,
        context_length=2048,
        test_size=0.1,
        shuffle=True,
    )
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    # scaler for half precision operation
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    model = prepare_model(model)
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAINER_CONFIG["weight_decay"],
    )

    num_training_steps = n_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.train()

    step = 0
    best_val_loss = 1e9
    for _ in range(n_epochs):
        for batch in train_loader:
            start = time.monotonic()

            with torch.autocast("cuda"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                scaler.scale(loss).backward()

            scaler.step(optimizer)
            lr_scheduler.step()
            scaler.update()
            optimizer.zero_grad()

            end = time.monotonic()

            if step % log_interval == 0:
                elapsed_time = end - start
                print(
                    f"step {step}, loss: {loss:.4f}, time: {elapsed_time * 1000:.2f}ms"
                )

            if step % eval_interval == 0:
                model.eval()

                val_loss = 0
                for batch in eval_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast("cuda"):
                        with torch.no_grad():
                            outputs = model(**batch)

                    val_loss += outputs.loss

                val_loss /= len(eval_loader)
                print(
                    f"step {step}: train loss {loss.item():.4f}, val loss {val_loss:.4f}"
                )
                vessl.log(
                    step=step,
                    payload={
                        "train_loss": loss,
                        "val_loss": val_loss,
                        "lr": learning_rate,
                    },
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if step > 0:
                        checkpoint = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step,
                            "best_val_loss": best_val_loss,
                        }
                        print(f"saving checkpoint to {output_dir}")
                        torch.save(checkpoint, output_dir.joinpath("ckpt.pt"))

                model.train()

            step += 1


if __name__ == "__main__":
    main()
