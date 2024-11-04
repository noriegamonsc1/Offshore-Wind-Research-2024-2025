import torch
import torch.nn as nn

class AdaptiveAutoencoder(nn.Module):
    def __init__(self):
        super(AdaptiveAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x (H/2) x (W/2)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x (H/4) x (W/4)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x (H/8) x (W/8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x (H/16) x (W/16)
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x (H/8) x (W/8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x (H/4) x (W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Output: 32 x (H/2) x (W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 5, kernel_size=4, stride=2, padding=1)    # Output: 5 x (H) x (W)
        )
        
    def forward(self, x):
        # Get input dimensions
        batch_size, channels, height, width = x.size()
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Resize the output to match the input dimensions
        decoded = nn.functional.interpolate(decoded, size=(height, width), mode='bilinear', align_corners=False)
        
        return decoded