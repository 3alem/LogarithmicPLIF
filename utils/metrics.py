import torch
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional
from spikingjelly.activation_based.encoding import PoissonEncoder  # Import PoissonEncoder

def train_model(net, train_loader, optimizer, device, T=20, encoder=None):
    net.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    for frames, labels in train_loader:
        optimizer.zero_grad()
        frames, labels = frames.to(device), labels.to(device)
        
        # Apply Poisson encoding if an encoder is provided
        if encoder is not None:
            frames = encoder(frames)  # Convert frames to spike trains
        
        label_onehot = F.one_hot(labels, 10).float()
        
        with amp.autocast():
            out = sum(net(frames) for _ in range(T)) / T
            loss = F.mse_loss(out, label_onehot)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.numel()
        total_correct += (out.argmax(1) == labels).sum().item()
        total_samples += labels.numel()
        functional.reset_net(net)
    
    return total_loss / total_samples, total_correct / total_samples


def evaluate_model(net, test_loader, device, T=20, encoder=None):
    net.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            
            # Apply Poisson encoding if an encoder is provided
            if encoder is not None:
                frames = encoder(frames)  # Convert frames to spike trains
            
            label_onehot = F.one_hot(labels, 10).float()
            out = sum(net(frames) for _ in range(T)) / T
            loss = F.mse_loss(out, label_onehot)
            
            total_loss += loss.item() * labels.numel()
            total_correct += (out.argmax(1) == labels).sum().item()
            total_samples += labels.numel()
            functional.reset_net(net)
    
    return total_loss / total_samples, total_correct / total_samples
