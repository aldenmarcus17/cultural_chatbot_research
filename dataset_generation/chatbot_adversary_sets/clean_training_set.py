"""Dataset Cleaner"""
import json

with open('synthetic_training_data.json', 'r') as file:
    dataset = json.load(file)

for entry in dataset:
    if '\n' in entry['target_output']:
        entry['target_output'] = entry['target_output'].split('\n')[0]

with open('cleaned_synthetic_dataset.json', 'w') as file:
    json.dump(dataset, file, indent=2)
