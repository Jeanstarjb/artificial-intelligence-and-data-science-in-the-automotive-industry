import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network for predictive maintenance in the automotive industry
class PredictiveMaintenanceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictiveMaintenanceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Generate dummy data for training and testing
def generate_dummy_data(num_samples, input_size):
    X = np.random.rand(num_samples, input_size)
    y = (np.sum(X, axis=1) > input_size / 2).astype(np.float32)  # Binary classification
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

if __name__ == '__main__':
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_samples = 1000
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 10

    # Generate dummy data
    X, y = generate_dummy_data(num_samples, input_size)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = PredictiveMaintenanceModel(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Test the model on new dummy data
    test_X, test_y = generate_dummy_data(100, input_size)
    with torch.no_grad():
        test_outputs = model(test_X)
        predictions = (test_outputs > 0.5).float()
        accuracy = (predictions.eq(test_y).sum().item()) / test_y.size(0)
        print(f'Test Accuracy: {accuracy:.4f}')