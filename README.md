# Deep-Learning-Gravity
5 dimensional Space
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assume input_data is a 2D tensor (100 samples, 5 features)
input_data = torch.randn(100, 5)
target_data = torch.randn(100, 1)

# Normalize input_data
scaler = StandardScaler()
input_data = torch.tensor(scaler.fit_transform(input_data), dtype=torch.float32)

# Split the data into training and validation sets
input_train, input_val, target_train, target_val = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

# Define the General Relativity-inspired neural network
class GeneralRelativityModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GeneralRelativityModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 5
hidden_size = 10
output_size = 1

gravity_model = GeneralRelativityModel(input_size, hidden_size, output_size).to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(gravity_model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
input_train = input_train.to(device)
target_train = target_train.to(device)
input_val = input_val.to(device)
target_val = target_val.to(device)

for epoch in range(num_epochs):
    # Forward pass
    outputs = gravity_model(input_train)
    loss = criterion(outputs, target_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation loss
    val_outputs = gravity_model(input_val)
    val_loss = criterion(val_outputs, target_val)

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Evaluate the model on test data if available
# test_outputs = gravity_model(test_input)
# test_loss = criterion(test_outputs, test_target)

