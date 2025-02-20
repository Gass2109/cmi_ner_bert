# 🏦 Financial NER with BERT 🧠💰

This project extracts named entities from text files using a pre-trained Named Entity Recognition (NER) model from Hugging Face. The extracted entities are saved as JSON files in the output folder. The aim of this code is to give an overview of how to download and run a general purpose NER model to extract named entities. 



## Project Overview

This project uses the ```dbmdz/bert-large-cased-finetuned-conll03-english``` model from Hugging Face to perform Named Entity Recognition (NER) on ```.txt``` files stored in the ```input/``` folder. The extracted entities are saved as JSON files in the ```output/``` folder. 
```dbmdz/bert-large-cased-finetuned-conll03-english``` is designed for named-entity recognition (NER), capable of finding person, organization, and other entities in the text.
This NER model is used for illustration purposes only. 



## Installation

### Clone the Repository
```bash
git clone https://github.com/cmi_ner_bert.git
cd cmi_ner_bert
```

### Install Dependencies (python==3.10)
```bash
pip install -r requirements.txt
```

## Configuration

This project uses a config.yaml file to manage settings. Update config.yaml as needed:
```bash
model_name: "dbmdz/bert-large-cased-finetuned-conll03-english" #The pre-trained model for NER.
input_folder: "./input" #Folder where input .txt files are stored.
output_folder: "./output" #Folder where parsed JSON files are saved.
device: "cuda"  # Set to "cpu" for CPU, "cuda" for GPU (if available)
```

## Usage

### Prepare Input Files

Place all ```.txt``` files you want to process in the input/ folder.

### Run the Script
```bash
python extract_entities.py
```

### Check the Output

Extracted entities will be saved as JSON files in the ```output/``` folder, named as:
```
parsed_{input_filename}.json
```

For example, if ```input/sample1.txt``` is processed, the output will be:

```
output/parsed_sample1.json
```

## Project Structure
```bash
project_folder/
├── config.yaml           # Configuration file
├── requirements.txt      # Dependencies list
├── input/                # Folder for input text files
│   ├── sample1.txt
│   ├── sample2.txt
├── output/               # Folder for output JSON files
│   ├── parsed_sample1.json
│   ├── parsed_sample2.json
├── extract_entities.py   # Main script to process files
└── README.md             # Documentation
```

## Example Output

If the input file ```sample1.txt``` contains:
```txt
11:49:05 I'll revert regarding BANK ABC to try to do another 200 mio at 2Y
FR001400QV82	AVMAFC FLOAT	06/30/28
offer 2Y EVG estr+45bps
estr average Estr average / Quarterly interest payment
```


The extracted JSON file (parsed_sample1.json) will look like this:
```JSON
{
    "BANK": "I-ORG",
    "ABC": "I-ORG"
}
```
