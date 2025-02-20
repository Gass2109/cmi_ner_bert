import yaml
import json
import os
from transformers import AutoTokenizer, BertForTokenClassification
import torch


# Load configuration settings from the config.yaml file
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Initialize the model and tokenizer based on config and set device
def initialize_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = BertForTokenClassification.from_pretrained(config["model_name"])
    
    # Get the device from config and move the model to the right device
    device = config.get("device", "cpu")
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    model.to(device)
    return tokenizer, model, device


# Tokenize the input text and get predictions
def process_text(input_text, tokenizer, model, device):
    inputs = tokenizer(input_text, add_special_tokens=True, return_tensors="pt")
    
    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_token_class_ids = logits.argmax(-1)
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    
    return tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), predicted_tokens_classes


# Reconstruct entity words from token fragments
def extract_entities(tokens, predicted_tokens_classes):
    entities = {}
    current_entity = ""
    current_label = ""

    for token, label in zip(tokens, predicted_tokens_classes):
        if token.startswith("##"):
            current_entity += token[2:]  # Merge subword token
        else:
            if current_entity and current_label:
                entities[current_entity] = current_label  # Save previous entity
            
            if label != "O":  # Start a new entity
                current_entity = token
                current_label = label
            else:
                current_entity = ""
                current_label = ""

    # Add last entity if exists
    if current_entity and current_label:
        entities[current_entity] = current_label

    return entities


# Read the input text from a file
def read_input_text(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Save the extracted entities to a file in the output folder
def save_entities(entities, output_folder, input_filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Dynamically name the output file as 'parsed_{input_filename}.json'
    output_filename = f"parsed_{os.path.splitext(input_filename)[0]}.json"
    output_file_path = os.path.join(output_folder, output_filename)
    
    with open(output_file_path, "w") as f:
        json.dump(entities, f, indent=4)
    print(f"Entities saved to {output_file_path}")


# Process the input folder, validate files, and extract entities
def process_input_folder(config):
    input_folder = config["input_folder"]
    
    # Get all .txt files from the input folder
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    if not input_files:
        print("No text files found in the input folder.")
        return
    
    tokenizer, model, device = initialize_model(config)
    
    for input_file in input_files:
        input_file_path = os.path.join(input_folder, input_file)
        
        # Read the text file
        input_text = read_input_text(input_file_path)
        
        # Process the text and get token labels
        tokens, predicted_tokens_classes = process_text(input_text, tokenizer, model, device)
        
        # Extract named entities
        extracted_entities = extract_entities(tokens, predicted_tokens_classes)
        
        # Print the extracted entities
        print(f"Extracted Entities from {input_file}:", extracted_entities)
        
        # Save the extracted entities to the output folder
        save_entities(extracted_entities, config["output_folder"], input_file)


# Main function to load config and process the input folder
def main(config_path="config.yaml"):
    config = load_config(config_path)
    
    # Process the input folder and extract entities from all text files
    process_input_folder(config)


if __name__ == "__main__":
    main()
