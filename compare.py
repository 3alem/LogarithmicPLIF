import argparse, os , time
import pandas as pd
import torch
from models import lif, plif, imp_plif
from utils.dataloader import get_dataloaders
from utils.metrics import train_model, evaluate_model
from torch.utils.tensorboard import SummaryWriter
from spikingjelly.activation_based.encoding import PoissonEncoder  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lif', 'plif', 'implif'], required=True)
    args = parser.parse_args()
    
    log_name = 'runs/' + args.model 
    #writer = SummaryWriter( log_name )
    csv_name = args.model + '_results.csv'

    # Shared configuration
    config = {
        'epochs': 50,  # Number of epochs to train
        'batch_size': 128,
        'T': 20,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'data_dir': '/home/ahmed/datasets/NMNIST',
        'num_workers': 2
    }

    # Load data
    train_loader, test_loader = get_dataloaders(
        config['data_dir'], config['batch_size'], config['num_workers'], config['T']
    )

    # Initialize the Poisson Encoder
    poisson_encoder = PoissonEncoder()

    # Model selection
    if args.model == 'lif':
        net = lif.SNN_LIF().to(config['device'])
    elif args.model == 'plif':
        net = plif.SNN_Parametric().to(config['device'])
    elif args.model == 'implif':
        net = imp_plif.SNN_ImprovedParametric().to(config['device'])

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9,  nesterov=True)

    # Training and evaluation loop
    for epoch in range(config['epochs']):
        
        # Train for one epoch
        start_time = time.time()
        train_loss, train_acc = train_model(net, train_loader, optimizer, config['device'], config['T'], poisson_encoder)
        epoch_time = time.time() - start_time
        
        # Evaluate on the test set
        val_loss, val_acc = evaluate_model(net, test_loader, config['device'], config['T'], poisson_encoder)
        
        """
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Acc/train_epoch', train_acc, epoch)        
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)        

        writer.add_scalar('Time/train_epoch', epoch_time, epoch)
        """
        
        # Log results
        print(f"epoch: {epoch}, TR_Loss: {train_loss:.4f}, TR_Acc: {train_acc:.4f}, Val_Loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f}, Time: {epoch_time:.4f}")

        
        # Save results to CSV
        results = pd.DataFrame({
            'epoch': [epoch +1],
            'train_loss': [train_loss],
            'train_acc': [train_acc],
            'val_loss': [val_loss],
            'val_acc': [val_acc],
            'epoch_time': [epoch_time]
        })
        results.to_csv( csv_name, mode='a', index=False, header=not os.path.exists( csv_name) )

    #writer.close()

if __name__ == '__main__':
    main()
