import numpy as np

data = np.load("data/processed/wsi/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E_patches.npz")

print(data.files)

for key in data.files:
    print(key, data[key].shape)