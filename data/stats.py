from collections import Counter
import numpy as np

lengths = []
for ex in ds:  # your HF dataset
    lengths.append(len(ex["input_ids"]))

lengths = np.array(lengths)

print("Total samples:", len(lengths))
print("Min / median / 95th / 99th:", lengths.min(),
      np.median(lengths),
      np.percentile(lengths, 95),
      np.percentile(lengths, 99))
