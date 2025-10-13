import json
import os
import numpy as np


# Base dir is the folder containing test.py
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "fitness_books.json")


print("Loading from:", file_path)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

"""
if isinstance(data, list):
    print(f"✅ The file contains {len(data)} elements.")
elif isinstance(data, dict):
    print(f"✅ The file contains {len(data.keys())} keys.")

data_numbers = np.zeros((len(data),1536))

for i in range(len(data)):
    if i==0:
        print(data[0])
"""
m = 0
text = data[0]['text']
for i in range(len(text)):
    if text[i] == " ": m+=1

print(m)


