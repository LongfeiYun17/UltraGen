import json
import re
import random
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file")
parser.add_argument("--output_train_path", type=str, required=True, help="Path to save the train JSON file")
parser.add_argument("--output_valid_path", type=str, required=True, help="Path to save the valid JSON file")
args = parser.parse_args()

# Predefined attribute type parsing rules
PATTERN_RULES = [
    # Keyword frequency
    {
        "pattern": r"The word '(.+?)' should appear (\d+) times? in your response.",
        "type": "keyword",
        "instruction": "keywords frequency",
        "fields": {"token": 1, "freq": 2}
    },
    # Include keyword
    {
        "pattern": r"Include the keyword '(.+?)' in your response.",
        "type": "keyword",
        "instruction": "include keywords",
        "fields": {"token": 1}
    },
    # Number of paragraphs
    {
        "pattern": r"The text should contain (\d+) paragraphs?",
        "type": "length",
        "instruction": "number of paragraphs",
        "fields": {"value": 1}
    },
    # Word count range
    {
        "pattern": r"The text should contain between (\d+) and (\d+) words.",
        "type": "length",
        "instruction": "number of words",
        "fields": {"min": 1, "max": 2}
    },
    # Sentence count range
    {
        "pattern": r"The text should contain between (\d+) and (\d+) sentences.",
        "type": "length",
        "instruction": "number of sentences",
        "fields": {"min": 1, "max": 2}
    },
    # All uppercase
    {
        "pattern": r"Your response should be in all uppercase.",
        "type": "case",
        "instruction": "all uppercase"
    },
    # All lowercase
    {
        "pattern": r"Your response should be in all lowercase.",
        "type": "case",
        "instruction": "all lowercase"
    },
    # Start with
    {
        "pattern": r"Your response should start with the word '(.+?)'.",
        "type": "start with",
        "instruction": "start with",
        "fields": {"token": 1}
    }
]

def parse_attribute(description):
    """Parse attribute description into structured data"""
    for rule in PATTERN_RULES:
        match = re.fullmatch(rule["pattern"], description, re.IGNORECASE)
        if match:
            parsed = {
                "type": rule["type"],
                "instruction": rule["instruction"],
                "description": description
            }
            return parsed
    # Unmatched as soft attribute
    return {
        "type": "soft",
        "description": description
    }

def resolve_conflicts(attributes):
    """Resolve attribute conflicts"""
    conflict_rules = {
        "keywords frequency": lambda a, b: b,
        "number of paragraphs": lambda a, b: b,  # Keep the last one
        "number of words": lambda a, b: b,
        "number of sentences": lambda a, b: b,
        "all uppercase": lambda a, b: a,  # Boolean type, effective if exists
        "all lowercase": lambda a, b: a,
        "start with": lambda a, b: b  # Keep the last specified start word
    }
    
    groups = defaultdict(list)
    # Group by conflict type
    for attr in attributes:
        if attr["type"] == "soft":
            continue
        key = (attr["type"], attr["instruction"], attr.get("token"))
        groups[key].append(attr)
    
    # Apply conflict resolution rules
    resolved = []
    for key, groups_attrs in groups.items():
        if len(groups_attrs) == 1:
            resolved.append(groups_attrs[0])
            continue
        
        # Get resolver function
        resolver = conflict_rules.get(key[1], lambda a, b: b)
        current = groups_attrs[0]
        for attr in groups_attrs[1:]:
            current = resolver(current, attr)
        resolved.append(current)
    
    return resolved + [attr for attr in attributes if attr["type"] == "soft"]

def process_attribute_set(attr_set):
    """Process a single attribute set"""
    # Step 1: Remove duplicates
    unique_attrs = list(set(attr_set))
    
    # Step 2: Parse and resolve conflicts
    parsed_attrs = [parse_attribute(desc) for desc in unique_attrs]
    hard_attrs = [a for a in parsed_attrs if a["type"] != "soft"]
    soft_attrs = [a["description"] for a in parsed_attrs if a["type"] == "soft"]
    
    # Step 3: Structured output
    structured_hard = []
    for attr in hard_attrs:
        structured = {
            "type": attr["type"],
            "instruction": attr["instruction"],
            "description": attr["description"]
        }
        structured_hard.append(structured)
    
    return {
        "soft_attributes": soft_attrs,
        "hard_attributes": structured_hard
    }

# Main processing flow
with open(args.input_path) as f:
    input_data = json.load(f)

processed_data = [process_attribute_set(attr_set) for attr_set in input_data]

# If there's multiple "start with" type, only keep the first one
for i, data in enumerate(processed_data):
    start_with_attrs = [attr for attr in data["hard_attributes"] if attr["type"] == "start with"]
    case_attrs = [attr for attr in data["hard_attributes"] if attr["type"] == "case"]
    other_attrs = [attr for attr in data["hard_attributes"] if attr["type"] != "start with" and attr["type"] != "case"]
    if len(start_with_attrs) > 1:
        data["hard_attributes"] = [start_with_attrs[0]] + other_attrs
    if len(case_attrs) >= 1:
        data["hard_attributes"] = [case_attrs[0]] + other_attrs

# Save results
# Split into train and valid sets (80% train, 20% valid)
random.shuffle(processed_data)
train_size = int(0.8 * len(processed_data))
train_data = processed_data[:train_size]
valid_data = processed_data[train_size:]

# Save train set
with open(args.output_train_path, "w") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

# Save valid set  
with open(args.output_valid_path, "w") as f:
    json.dump(valid_data, f, indent=2, ensure_ascii=False)