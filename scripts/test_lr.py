import torch
import torch.nn as nn
import torch.optim as optim

# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Dummy dataset
data = torch.randn(100, 10)
targets = torch.randn(100, 1)

# DataLoader
dataset = torch.utils.data.TensorDataset(data, targets)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Model, loss, optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Number of total iterations
num_epochs = 2
num_iterations = num_epochs * len(dataloader)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

# Training loop
iteration = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    for i, (inputs, labels) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Step scheduler
        scheduler.step()

        # Print learning rate
        lr = optimizer.param_groups[0]['lr']
        print(f"Iteration {iteration + 1}, Learning Rate: {lr:.6f}, Loss: {loss.item():.6f}")

        iteration += 1
