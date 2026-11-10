import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from torch.utils.data import TensorDataset, DataLoader

from reader import data, labels

def print_sq(value: int):

    value = max(0, min(255, int(value)))
    color_code = f"\033[48;2;{value};{value};{value}m"
    reset_code = "\033[0m"

    square_width = 2
    square_height = 1

    for _ in range(square_height):
        sys.stdout.write(f"{color_code}{' ' * square_width}{reset_code}")

    sys.stdout.flush()


device = "cuda" if torch.cuda.is_available() else "cpu"

x_train = labels
y_train = data

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

def noise_fn(i,a,b):
    return abs(a*i + b)

class PredictorHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PredictorHead, self).__init__()
        self.Head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.Head(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.head = PredictorHead(794, 794, 784).to(device)

    def forward(self, prompt):

        prompt = prompt.to(device)
        img = torch.rand(prompt.size(0),784).to(device)

        for i in range(40):

            model_inp = torch.cat((img, prompt), dim=1).to(device)
            output = self.head(model_inp).to(device)
            img = img - output

        return img

def train():
    epochs = 10

    loss_fn = nn.MSELoss().to(device)

    diffuser = Model().to(device)
    optimizer = optim.AdamW(params = diffuser.parameters(), lr=0.001)


    for epoch in range(epochs):
        diffuser.train()

        for batch_x, batch_y in loader:

            output = diffuser(batch_x.to(device))
            loss = loss_fn(output, batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss)

    torch.save(diffuser.state_dict(), "model.pth")

def inference():
    diffuser = Model().to(device)
    weights = torch.load("model.pth")
    diffuser.load_state_dict(weights)
    diffuser.eval()

    inpt = int(input("Enter prompt: "))

    inputt = torch.zeros(10, dtype=torch.float32)
    inputt[min(max(inpt, 0), 9)] = 1.0
    inputt = inputt.unsqueeze(0).to(device)

    with torch.no_grad():
        output = diffuser(inputt)

        flat_values = (output * 255).round().int().flatten().tolist()
        print(flat_values)

        for i, val in enumerate(flat_values):
            print_sq(val)
            if (i + 1) % 28 == 0:
                print("\n", end="")

if __name__ == "__main__":
    file_path = Path("model.pth")
    if file_path.is_file():
        inference()
    else:
        train()