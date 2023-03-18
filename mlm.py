import tensorflow as tf
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    create_optimizer,
)


class MLM:
    def __init__(self, dataset_name, model_checkpoint, output_model_name):
        self.dataset_name = dataset_name
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.output_model_name = output_model_name

    def tokenize_function(self, examples):
        result = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    def group_texts(self, examples):
        chunk_size = 128
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    def prepare_tf_dataset(self, model, lm_datasets, data_collator):
        tf_train_dataset = model.prepare_tf_dataset(
            lm_datasets["train"], collate_fn=data_collator, shuffle=True, batch_size=32,
        )

        tf_eval_dataset = model.prepare_tf_dataset(
            lm_datasets["valid"], collate_fn=data_collator, shuffle=False, batch_size=32,
        )

        return tf_train_dataset, tf_eval_dataset

    def compile_tf_model(self, model, steps):
        optimizer, schedule = create_optimizer(
            init_lr=2e-5, num_warmup_steps=1_000, num_train_steps=steps, weight_decay_rate=0.01,
        )
        model.compile(optimizer=optimizer)

        # Train in mixed-precision float16
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    def train_and_push_to_hub(self):
        dataset = load_dataset(self.dataset_name)
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True, remove_columns=["text", "task1", "id"])

        lm_datasets = tokenized_datasets.map(self.group_texts, batched=True)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)

        tf_train_dataset, tf_eval_dataset = self.prepare_tf_dataset(model, lm_datasets, data_collator)

        self.compile_tf_model(model, steps=len(tf_train_dataset))

        model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=3)

        model.push_to_hub(self.output_model_name)
