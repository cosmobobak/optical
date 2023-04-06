import torch
import torch.nn as nn
import torch.optim as optim

# Define the inputs and outputs
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
outputs = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the model
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = XORModel()

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the model
for epoch in range(10000):
    # Forward pass
    predictions = model(inputs)
    loss = criterion(predictions, outputs)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the progress
    if (epoch+1) % 1000 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 10000, loss.item()))

    if loss.item() < 0.01:
        print("Converged")
        break

# Test the model
with torch.no_grad():
    predictions = model(inputs)
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == outputs).float().mean()
    print('Accuracy: ', accuracy.item())

    print('Predicted: ', predicted)
    print('Outputs: ', outputs)