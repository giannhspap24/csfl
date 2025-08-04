import torch
import torch.nn as nn
from collections import OrderedDict

class OurVGG11(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(OurVGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def count_conv_flops(layer, input_shape):
    Cin = layer.in_channels
    Cout = layer.out_channels
    Kh, Kw = layer.kernel_size
    Hout, Wout = input_shape[2], input_shape[3]
    return 2 * Cin * Cout * Kh * Kw * Hout * Wout

def count_fc_flops(layer):
    return 2 * layer.in_features * layer.out_features

def compute_flops_per_layer(model, input_shape=(1, 3, 32, 32)):
    model.eval()
    x = torch.randn(input_shape)
    layer_flops = OrderedDict()

    def make_hook(name, layer_type):
        def hook(layer, inp, out):
            if layer_type == 'conv':
                flops = count_conv_flops(layer, out.shape)
                param_count = layer.weight.numel()
                if layer.bias is not None:
                    param_count += layer.bias.numel()
            elif layer_type == 'fc':
                flops = count_fc_flops(layer)
                param_count = layer.weight.numel()
                if layer.bias is not None:
                    param_count += layer.bias.numel()
            else:
                flops = 0
                param_count = 0

            param_mem_bytes = param_count * 4  # float32
            activation_elements = out.numel()
            activation_mem_bytes = activation_elements * 4  # float32

            layer_flops[name] = {
                'forward': flops,
                'backward': 2 * flops,
                'total': 3 * flops,
                'param_MB': param_mem_bytes / 1e6,
                'activation_MB': activation_mem_bytes / 1e6
            }
        return hook

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(make_hook(name, 'conv')))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(make_hook(name, 'fc')))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    print(f"{'Layer':<30} {'FWD (MFLOPs)':>15} {'BWD (MFLOPs)':>15} {'Total (MFLOPs)':>18} {'Param Mem (MB)':>17} {'Act Mem (MB)':>17}")
    print("=" * 130)
    total_fwd, total_bwd, total_all, total_mem, total_act = 0, 0, 0, 0, 0
    for name, stats in layer_flops.items():
        fwd = stats['forward'] / 1e6
        bwd = stats['backward'] / 1e6
        total = stats['total'] / 1e6
        mem = stats['param_MB']
        act = stats['activation_MB']
        total_fwd += fwd
        total_bwd += bwd
        total_all += total
        total_mem += mem
        total_act += act
        print(f"{name:<30} {fwd:>15.2f} {bwd:>15.2f} {total:>18.2f} {mem:>17.2f} {act:>17.2f}")
    print("=" * 130)
    print(f"{'Total':<30} {total_fwd:>15.2f} {total_bwd:>15.2f} {total_all:>18.2f} {total_mem:>17.2f} {total_act:>17.2f}")

# Run the analysis
model = OurVGG11()
compute_flops_per_layer(model)

