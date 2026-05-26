"""
Visualization utilities for distilled datasets.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def visualize_synthetic_images(images, labels, num_classes, save_path=None):
    """
    Visualize synthetic images organized by class.
    
    Args:
        images: Tensor of images (N, C, H, W)
        labels: Tensor of labels (N,)
        num_classes: Number of classes
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(num_classes, 10, figsize=(20, 2*num_classes))
    
    for class_idx in range(num_classes):
        class_images = images[labels == class_idx][:10]
        
        for img_idx in range(min(10, len(class_images))):
            ax = axes[class_idx, img_idx] if num_classes > 1 else axes[img_idx]
            
            # Denormalize and convert to numpy
            img = class_images[img_idx].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            ax.imshow(img)
            ax.axis('off')
            
            if img_idx == 0:
                ax.set_ylabel(f'Class {class_idx}', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(losses, save_path=None):
    """
    Plot training loss curves.
    
    Args:
        losses: Dictionary of loss lists
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(losses['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # MDC loss
    axes[0, 1].plot(losses['mdc'])
    axes[0, 1].set_title('Multi-Descriptor Consensus Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Reconstruction loss
    axes[1, 0].plot(losses['rec'])
    axes[1, 0].set_title('Reconstruction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Disentanglement loss
    axes[1, 1].plot(losses['dis'])
    axes[1, 1].set_title('Disentanglement Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_latent_space(z_R, z_I, labels, save_path=None):
    """
    Visualize latent space using t-SNE.
    
    Args:
        z_R: Relevant latent codes
        z_I: Irrelevant latent codes
        labels: Ground truth labels
        save_path: Path to save figure (optional)
    """
    from sklearn.manifold import TSNE
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_R_2d = tsne.fit_transform(z_R.cpu().numpy())
    z_I_2d = tsne.fit_transform(z_I.cpu().numpy())
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Z_R
    scatter1 = axes[0].scatter(z_R_2d[:, 0], z_R_2d[:, 1], 
                               c=labels.cpu().numpy(), cmap='tab10', alpha=0.6)
    axes[0].set_title('Relevant Latent Space (Z_R)')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot Z_I
    scatter2 = axes[1].scatter(z_I_2d[:, 0], z_I_2d[:, 1], 
                               c=labels.cpu().numpy(), cmap='tab10', alpha=0.6)
    axes[1].set_title('Irrelevant Latent Space (Z_I)')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def compare_real_vs_synthetic(real_images, synthetic_images, num_samples=10, save_path=None):
    """
    Compare real and synthetic images side by side.
    
    Args:
        real_images: Real images tensor
        synthetic_images: Synthetic images tensor
        num_samples: Number of samples to display
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    
    for i in range(num_samples):
        # Real image
        real_img = real_images[i].cpu().numpy().transpose(1, 2, 0)
        real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min())
        axes[0, i].imshow(real_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Real', fontsize=12)
        
        # Synthetic image
        syn_img = synthetic_images[i].cpu().numpy().transpose(1, 2, 0)
        syn_img = (syn_img - syn_img.min()) / (syn_img.max() - syn_img.min())
        axes[1, i].imshow(syn_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Synthetic', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
