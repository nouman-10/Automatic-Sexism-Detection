from datasets import load_dataset, DatasetDict
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import classification_report

import helper_functions as f


class FineTune:
    def __init__(self, dataset_name, output_model_name, model_checkpoint, num_labels, epochs, tokenizer_checkpoint=None) -> None:
        self.dataset_name = dataset_name
        self.model_checkpoint = model_checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint if tokenizer_checkpoint else model_checkpoint
        self.num_labels = num_labels
        self.output_model_name = output_model_name

        self.tokenizer = None
        self.model = None
        self.data_collator = None

        self.learning_rate = 2e-5
        self.batch_size = 16
        self.epochs = epochs
        self.training_args = None

        self.task = "task1" if num_labels == 2 else "task2"
        if self.task == 'task1':
          self.mapping = {
              'sexist': 1,
              'non-sexist': 0
          }
        else:
          self.mapping = {
              "non-sexist": 0,
              "objectification": 1,
              "misogyny-non-sexual-violence": 2,
              "ideological-inequality": 3,
              "stereotyping-dominance": 4,
              "sexual-violence": 5
          }
        
    def load_dataset(self) -> DatasetDict:
        """Load dataset from huggingface hub

        Returns:
            DatasetDict: dataset dict containing different splits
        """
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def load_tokenizer(self) -> None:
        """Load tokenizer from huggingface hub

        Returns:
            AutoTokenizer: tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)

    def load_model(self) -> None:
        """Load model from huggingface hub

        Returns:
            AutoModelForSequenceClassification: model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=self.num_labels, from_tf=True)

    def load_data_collator(self) -> None:
        """Load data collator"""
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def preprocess_function(self, examples: DatasetDict) -> DatasetDict:
        """Preprocess function for the dataset
        
        Args:
            examples (DatasetDict): dict containing the dataset

        Returns:
            DatasetDict: preprocessed dataset
        """
        examples["text"] = [remove_hashtags_links_mentions(text) for text in examples["text"]]
        examples["label"] = [self.mapping[label] for label in examples[self.task]]
        return self.tokenizer(examples["text"], truncation=True)
    
    def define_training_args(self) -> None:
        """Define training arguments"""
        self.training_args = TrainingArguments(
            output_dir=self.output_model_name,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            push_to_hub=True,
            metric_for_best_model="f1",
        )

    def define_trainer(self, dataset: DatasetDict) -> None:
        """Define trainer

        Args:
            dataset (DatasetDict): dataset dict containing different splits
        """
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics
        )

    def evaluate(self, test_dataset) -> None:
        """Evaluate model

        Args:
            test_dataset (DatasetDict): test dataset
        """
        trained_model = pipeline("text-classification", model=self.output_model_name, tokenizer=self.tokenizer_checkpoint, truncation=True)
        texts = [f.remove_hashtags_links_mentions(text) for text in test_dataset["text"]]
        labels = [label for label in test_dataset["task1"]]

        preds = trained_model(texts)
        final_preds = ['non-sexist' if p['label'] == "LABEL_0" else "sexist" for p in preds]
        
        print(classification_report(labels, final_preds))

        labels = [label for label in test_dataset["task2"]]

        reverse_mapping = {value: key for key, value in self.mapping.items()}
        final_preds = [reverse_mapping[int(p['label'].split('_')[1])] for p in preds]

        print(classification_report(labels, final_preds))



    def fine_tune(self) -> None:
        """Fine tune model"""
        dataset = self.load_dataset()

        self.load_tokenizer()
        self.load_model()
        self.load_data_collator()

        columns_to_remove = ["text", "task1", "task2"] if self.dataset_name == "nouman-10/exist-en" else ["text", "task1"]

        dataset = dataset.map(
            self.preprocess_function, batched=True, remove_columns=columns_to_remove
        )

        self.define_training_args()
        self.define_trainer(dataset)

        self.trainer.train()
        self.trainer.push_to_hub()
