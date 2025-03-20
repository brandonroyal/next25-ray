import ray
from ray import train, data
from ray.train import ScalingConfig, CheckpointConfig, RunConfig
from ray.train.huggingface import TransformersTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset

# Initialize Ray
ray.init()

# Define your dataset (replace with your actual data loading logic)
def load_text_to_sql_dataset(file_path):
    """Loads a text-to-SQL dataset from a file (e.g., JSONL).

    Args:
        file_path: Path to the dataset file.

    Returns:
        A Hugging Face Dataset object.
    """
    import json

    data_list = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    def preprocess_function(examples):
        inputs = [f"translate question to sql: {question} <table> {table}" for question, table in zip(examples["question"], examples["table"])]
        targets = [sql for sql in examples["sql"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)
        return model_inputs

    dataset = Dataset.from_list(data_list)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset

# Configuration
model_name = "google/gemma-3b-it"  # or "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token #Gemma needs this.

dataset_path = "your_text_to_sql_dataset.jsonl"  # Replace with your dataset path
tokenized_dataset = load_text_to_sql_dataset(dataset_path)

# Split dataset into train and validation
train_dataset = tokenized_dataset.train_test_split(test_size=0.1)["train"]
eval_dataset = tokenized_dataset.train_test_split(test_size=0.1)["test"]

# Convert Hugging Face Datasets to Ray Datasets
ray_train_dataset = data.from_huggingface(train_dataset)
ray_eval_dataset = data.from_huggingface(eval_dataset)

# Define the training function
def train_function(train_dataset, eval_dataset, **config):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer)) #important for gemma

    training_args = TrainingArguments(
        output_dir="./gemma_finetuned",
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.get("learning_rate", 2e-5),
        fp16=True, #use fp16
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4), #adjust according to your gpu memory.
        gradient_checkpointing=True, #saves memory
        push_to_hub=False, #set to True if you wish to push to hub.
        dataloader_num_workers = 4, #adjust based on your cpu cores.
        remove_unused_columns=False, #important to avoid errors
        label_names=["labels"], #important to avoid errors.
        report_to="none" # important for ray integration.
    )

    trainer = train.huggingface.transformers.TransformersTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    return trainer.train()

# Configure Ray Train
scaling_config = ScalingConfig(
    num_workers=1,  # Adjust based on your resources
    use_gpu=True,  # Enable GPU usage
    resources_per_worker={"GPU": 1}, #adjust based on the gpus you have.
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="eval_loss",
        checkpoint_score_order="min",
    )
)

trainer = TransformersTrainer(
    train_func=train_function,
    train_dataset=ray_train_dataset,
    eval_dataset=ray_eval_dataset,
    scaling_config=scaling_config,
    run_config=run_config,
    tokenizer=tokenizer,
    datasets={"train": ray_train_dataset, "evaluation": ray_eval_dataset},
    trainer_init_per_worker=False,
    datasets_iter_config={},
    trainer_init_config={},
    trainer_kwargs={},
    datasets_to_split=None,
    datasets_split_config=None,
    datasets_to_ray_datasets=None,
    datasets_ray_datasets_config=None,
    param_dict={"epochs": 1, "batch_size": 1, "learning_rate": 2e-5, "gradient_accumulation_steps":4} #adjust hyper parameters here.
)

# Run the training
result = trainer.fit()

print(f"Training finished. Checkpoints saved at: {result.checkpoint.path}")

# Example Inference after training.
trained_model = AutoModelForCausalLM.from_pretrained(result.checkpoint.path)

def generate_sql(question, table):
    prompt = f"translate question to sql: {question} <table> {table}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = trained_model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = "How many employees are in the sales department?"
table = "Employees (employee_id, employee_name, department_id), Departments (department_id, department_name)"

sql_query = generate_sql(question, table)
print(f"Generated SQL: {sql_query}")

ray.shutdown()