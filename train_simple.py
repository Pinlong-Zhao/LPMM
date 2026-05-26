"""
Simplified training script for quick testing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.encoder import RelevantEncoder, IrrelevantEncoder
from models.decoder import Decoder
from models.generator import SyntheticGenerator
from models.classifier import AuxiliaryClassifier
from modules.mdc import MultiDescriptorConsensus
from modules.information_purification import InformationPurification


def main():
    # Simple configuration
    config = {
        'dataset': 'cifar10',
        'num_classes': 10,
        'image_size': 32,
        'ipc': 10,
        'latent_dim_R': 128,
        'latent_dim_I': 64,
        'alpha': 0.5,
        'beta': 0.3,
        'batch_size': 128,
        'lr': 0.001,
        'epochs': 100
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    
    # Initialize models
    input_dim = 32 * 32 * 3
    encoder_R = RelevantEncoder(input_dim, config['latent_dim_R']).to(device)
    encoder_I = IrrelevantEncoder(input_dim, config['latent_dim_I']).to(device)
    decoder = Decoder(config['latent_dim_R'], config['latent_dim_I'], input_dim).to(device)
    generator = SyntheticGenerator(config['latent_dim_R'], 32, config['num_classes'], config['ipc']).to(device)
    classifier_R = AuxiliaryClassifier(config['latent_dim_R'], config['num_classes']).to(device)
    classifier_I = AuxiliaryClassifier(config['latent_dim_I'], config['num_classes']).to(device)
    
    # Initialize modules
    mdc = MultiDescriptorConsensus(
        descriptors=['training_loss', 'uncertainty'],
        weight_method='uniform'
    )
    
    ip = InformationPurification(
        config['latent_dim_R'],
        config['latent_dim_I'],
        config['alpha'],
        config['beta']
    )
    
    # Optimizers
    optimizer = optim.Adam(
        list(encoder_R.parameters()) + 
        list(encoder_I.parameters()) + 
        list(decoder.parameters()) + 
        list(generator.parameters()) + 
        list(classifier_R.parameters()) + 
        list(classifier_I.parameters()),
        lr=config['lr']
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(config['epochs']):
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            images_flat = images.view(images.size(0), -1)
            
            # Forward pass
            z_R_mu, z_R_logvar = encoder_R(images_flat)
            z_I_mu, z_I_logvar = encoder_I(images_flat)
            
            # Reparameterization
            z_R = z_R_mu + torch.exp(0.5 * z_R_logvar) * torch.randn_like(z_R_mu)
            z_I = z_I_mu + torch.exp(0.5 * z_I_logvar) * torch.randn_like(z_I_mu)
            
            # Decode
            recon_images = decoder(z_R, z_I)
            
            # Generate
            synthetic_images = generator(z_R, labels)
            
            # Classify
            pred_R = classifier_R(z_R)
            pred_I = classifier_I(z_I)
            
            # Compute losses
            loss_ip = ip.compute_loss(
                images_flat, recon_images, z_R, z_I,
                synthetic_images, labels, pred_R, pred_I, generator
            )
            
            loss = loss_ip['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_R.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(encoder_I.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epochs']}] Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'encoder_R': encoder_R.state_dict(),
                'encoder_I': encoder_I.state_dict(),
                'decoder': decoder.state_dict(),
                'generator': generator.state_dict(),
            }, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()
