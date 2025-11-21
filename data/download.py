from datasets import load_dataset

# Load the English training split
dataset = load_dataset("PleIAs/SYNTH", split="train")
print(dataset)
print(dataset[0])  # first example
