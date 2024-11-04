import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from offshore_wind_nj.data_processing import flatten_data, scaler_flattened_data, scale_flat_data
from offshore_wind_nj.data_loader import all_arrays
from offshore_wind_nj.config import MODELS_DIR, FIGURES_DIR
import matplotlib.pyplot as plt
import numpy as np
import os 

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)  # Encoder to compress input
        self.decoder = nn.Linear(hidden_dim, input_dim)  # Decoder to reconstruct input

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))  # Activation function
        decoded = self.decoder(encoded)          # Reconstruction
        return decoded
    
def train_model(model, train_tensor, val_tensor, optimizer, criterion, num_epochs=600):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        optimizer.zero_grad()
        
        output = model(train_tensor)
        train_loss = criterion(output, train_tensor)
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_output = model(val_tensor)
            val_loss = criterion(val_output, val_tensor)
            val_losses.append(val_loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    return train_losses, val_losses 

def save_model_and_plot(model, train_losses, val_losses, model_path, plot_path):
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training and validation losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Over Epochs")
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")
    plt.close()



if __name__=='__main__':
    # Data processing
    flattened_data_list = flatten_data(all_arrays, mask=True)
    scaler = scaler_flattened_data(flattened_data_list)
    scaled_data_list = scale_flat_data(flattened_data_list, scaler)[0]

    '''
    need to edit scale_flattened_data to scaler_flattened_data -> It will return the scaler, then it needs to be applied to each array
    '''
    scaled_data_np = np.vstack([file for file in scaled_data_list if not np.isnan(file).any()])
    
    model = Autoencoder(input_dim=5, hidden_dim=4)  # Adjust input_dim based on the number of features
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(scaled_data_np, test_size=0.2, random_state=42)  # 80% train, 20% validation

    # Convert the split data into PyTorch tensors
    train_tensor = torch.FloatTensor(train_data)
    val_tensor = torch.FloatTensor(val_data)

    # Check if GPU is available and move data and model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_tensor = train_tensor.to(device)
    val_tensor = val_tensor.to(device)

        # Train the model
    train_losses, val_losses = train_model(model, train_tensor, val_tensor, optimizer, criterion, num_epochs=600)

    # File paths for saving
    model_path = os.path.join(MODELS_DIR, "autoencoder_model.pt")
    plot_path = os.path.join(FIGURES_DIR, "training_plot.png")

    # Save model and training plot
    save_model_and_plot(model, train_losses, val_losses, model_path, plot_path)
    