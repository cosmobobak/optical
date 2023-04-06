import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ttt_state import TTT
from tqdm import tqdm

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

all_states = TTT.get_every_state()
print(f"Number of states: {len(all_states)}")

inputs = []
outputs = []

for state in tqdm(all_states, desc="Vectorising/Computing best move"):
    if state.is_terminal():
        continue
    inputs.append(state.vectorise()) # (2, 3, 3)
    bm = state.best_move()
    # bm is an index, needs to be one-hot
    outputs.append(np.zeros(9))
    outputs[-1][bm] = 1

print(f"Number of non-terminal states: {len(inputs)}")

inputs = np.array(inputs)
outputs = np.array(outputs)
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = torch.tensor(outputs, dtype=torch.float32)

# shuffle & split
indices = np.arange(len(inputs))
np.random.shuffle(indices)
inputs = inputs[indices]
outputs = outputs[indices]

print(inputs.shape)
# => [4520, 2, 3, 3]
print(outputs.shape)
# => [4520, 9]

train_inputs = inputs[:4000]
train_outputs = outputs[:4000]
test_inputs = inputs[4000:]
test_outputs = outputs[4000:]

# move to GPU
train_inputs = train_inputs.to(device)
train_outputs = train_outputs.to(device)
test_inputs = test_inputs.to(device)
test_outputs = test_outputs.to(device)

class TTTModel(nn.Module):
    def __init__(self):
        super(TTTModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2 * 3 * 3, 9)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 2 * 3 * 3)
        x = torch.softmax(self.fc1(x), dim=1)
        return x

model = TTTModel().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

EPOCHS = 100000
for epoch in range(EPOCHS):
    # Forward pass
    predictions = model(train_inputs)
    loss = criterion(predictions, train_outputs)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the progress
    if (epoch+1) % 10 == 0:
        t_loss = criterion(model(test_inputs), test_outputs)
        print('Epoch [{}/{}], Loss: {:.4f} (train), {:.4f} (test), Accuracy: {:.4f}'.format(
            epoch+1,
            EPOCHS,
            loss.item(),
            t_loss.item(),
            (predictions.argmax(dim=1) == train_outputs.argmax(dim=1)).float().mean().item()
        ))

        if t_loss.item() < 0.001:
            print("Converged")
            break

# Test the model on the test set
with torch.no_grad():
    predictions = model(test_inputs)
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == test_outputs).float().mean()
    print('Accuracy: ', accuracy.item())

    print('Predicted: ', predicted)
    print('Outputs: ', test_outputs)

# save the model
torch.save(model.state_dict(), 'ttt_model.pth')