import torch
import torch.nn as nn
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 5 convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the flattened size for FC layers
        # For MNIST (28x28):
        # - After conv1 + pool: (28 / 2) = 14
        # - After conv2 + pool: (14 / 2) = 7
        # - After conv3 + pool: (7 / 2) = 3.5 -> floor to 3
        # - After conv4 + pool: (3 / 2) = 1 (rounded down)
        self.fc1 = nn.Linear(512 * 1 * 1, 512)  # Adjusted flattened size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Activation and Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        #x = self.pool(x)
        # Flatten the output of the last conv layer to feed into the FC layer
        x = x.view(x.size(0), -1)  # Flatten the feature map
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


import torch
import torch.nn as nn
from collections import OrderedDict

def count_conv_flops(layer, input_shape):
    Cin = layer.in_channels
    Cout = layer.out_channels
    Kh, Kw = layer.kernel_size
    Hout, Wout = input_shape[2], input_shape[3]
    flops = 2 * Cin * Cout * Kh * Kw * Hout * Wout
    return flops

def count_fc_flops(layer):
    return 2 * layer.in_features * layer.out_features

def compute_flops_per_layer(model, input_shape=(1, 1, 28, 28)):
    model.eval()
    x = torch.randn(input_shape)
    layer_flops = OrderedDict()

    def make_hook(name, layer_type):
        def hook(layer, inp, out):
            if layer_type == 'conv':
                flops = count_conv_flops(layer, out.shape)
            elif layer_type == 'fc':
                flops = count_fc_flops(layer)
            else:
                flops = 0
            layer_flops[name] = {
                'forward': flops,
                'backward': 2 * flops,
                'total': 3 * flops
            }
        return hook

    # Register hooks
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(make_hook(name, 'conv')))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(make_hook(name, 'fc')))

    # Trigger forward pass
    with torch.no_grad():
        model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Print results
    print(f"{'Layer':<20} {'FWD (MFLOPs)':>15} {'BWD (MFLOPs)':>15} {'Total (MFLOPs)':>18}")
    print("=" * 70)
    total_fwd, total_bwd, total_all = 0, 0, 0
    for name, stats in layer_flops.items():
        fwd = stats['forward'] / 1e6
        bwd = stats['backward'] / 1e6
        total = stats['total'] / 1e6
        total_fwd += fwd
        total_bwd += bwd
        total_all += total
        print(f"{name:<20} {fwd:>15.2f} {bwd:>15.2f} {total:>18.2f}")
    print("=" * 70)
    print(f"{'Total':<20} {total_fwd:>15.2f} {total_bwd:>15.2f} {total_all:>18.2f}")

# Example usage
model = SimpleCNN()
compute_flops_per_layer(model)



