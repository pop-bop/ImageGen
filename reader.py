import pandas as pd
import numpy as np
import torch


df = pd.read_csv('mnist_train.csv')

raw_data = df.iloc[:, 1:].to_numpy()
data = torch.from_numpy(raw_data / 255.0).to(torch.float32)

pre_labels = df.iloc[:, 0].to_numpy()

labels = torch.from_numpy(np.eye(10)[pre_labels]).to(torch.float32)

if __name__ == "__main__":
    print("labels shape:", labels.shape)

    print("first few one-hot labeels:\n", labels[:5])

    print("\nData tensor Sshape:", data.shape)
    print("data tensor:\n", data)



