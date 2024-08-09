import logging
import torch
torch.cuda.empty_cache()
import pandas as pd
from unsloth import FastLanguageModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os
import requests
from transformers import AutoTokenizer

# Disable SSL warnings (use with caution)
os.environ["CURL_CA_BUNDLE"] = ""
requests.packages.urllib3.disable_warnings()

# Initialize logging
logging.basicConfig(filename='gemma_model.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load and prepare the diabetes dataset
data = pd.read_csv('Diabetes Dataset_Training Part.csv')
X = data.drop(['Outcome'], axis=1)
y = data['Outcome'].astype(str)  # Convert labels to strings for compatibility

logging.info('Data loaded successfully.')
logging.info(f'X shape: {X.shape}')
logging.info(f'y shape: {y.shape}')

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
logging.info('Data split into training and test sets.')
logging.info(f'X_train shape: {X_train.shape}')
logging.info(f'X_test shape: {X_test.shape}')
logging.info(f'y_train shape: {y_train.shape}')
logging.info(f'y_test shape: {y_test.shape}')

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logging.info('Data standardized using StandardScaler.')

# Convert to Dataset format
train_dataset = Dataset.from_pandas(pd.DataFrame(X_train_scaled, columns=X.columns))
test_dataset = Dataset.from_pandas(pd.DataFrame(X_test_scaled, columns=X.columns))
train_dataset = train_dataset.add_column("labels", y_train.tolist())
test_dataset = test_dataset.add_column("labels", y_test.tolist())

# Add a "text" column to the datasets for compatibility
train_dataset = train_dataset.add_column("text", [" ".join(map(str, row)) for row in X_train_scaled])
test_dataset = test_dataset.add_column("text", [" ".join(map(str, row)) for row in X_test_scaled])

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Prepare model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/gemma-2-2b",
        padding_side='left',  # This ensures proper padding for generation
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-2-2b",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        #use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Skipping for_inference() since it causes the model to be None
except Exception as e:
    logging.error(f'Error loading model: {e}')
    raise

# Fine-tune the model
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=10,
    max_steps=200,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    max_seq_length=1024,
    dataset_num_proc=2,
    packing=False,
    dataset_text_field="text",
    args=training_args,
)

trainer_stats = trainer.train()
logging.info('Model fine-tuning completed.')

# Make predictions
if model is not None:
    def predict(inputs, chunk_size=10):
        predictions = []  # Ensure this is correctly indented inside the function
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            tokens = tokenizer(chunk, return_tensors="pt", padding="longest", truncation=True).to("cuda")
        
        # Get the model's outputs
            outputs = model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        
        # If the output is logits or another dict, extract sequences (for example)
            if isinstance(outputs, dict):
                outputs = outputs['logits']  # or 'sequences' based on your model's output
        
        # Ensure outputs is a tensor of token IDs
            token_ids = torch.argmax(outputs, dim=-1)  # Example for classification
        
        # Decode the token IDs
            predictions.extend(tokenizer.batch_decode(token_ids, skip_special_tokens=True))
        
            torch.cuda.empty_cache()  # Clear cache to free up memory
        return predictions


    # Prepare input data for prediction
    input_data = [" ".join(map(str, X_test_scaled[i])) for i in range(X_test_scaled.shape[0])]

    # Perform prediction
    predictions = predict(input_data)
    y_pred = [str(1) if "1" in pred else str(0) for pred in predictions]  # Ensure that y_pred contains '0' and '1' as strings
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='1')  # Ensure pos_label is a string
    recall = recall_score(y_test, y_pred, pos_label='1')  # Ensure pos_label is a string
    f1 = f1_score(y_test, y_pred, pos_label='1')  # Ensure pos_label is a string

    logging.info(f'Accuracy Score: {accuracy * 100:.2f}%')
    logging.info(f'Precision Score: {precision * 100:.2f}%')
    logging.info(f'Recall Score: {recall * 100:.2f}%')
    logging.info(f'F1 Score: {f1 * 100:.2f}%')

    # Print the evaluation metrics
    print(f'Accuracy Score: {accuracy * 100:.2f}%')
    print(f'Precision Score: {precision * 100:.2f}%')
    print(f'Recall Score: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')

    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
else:
    logging.error('Model is None, cannot perform predictions.')
    print('Model is None, cannot perform predictions.')


