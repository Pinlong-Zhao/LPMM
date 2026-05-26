"""
Information Purification (IP) module.
Implements the disentangled information bottleneck objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InformationPurification(nn.Module):
    """
    Information Purification module based on Information Bottleneck principle.
    Disentangles relevant (Z_R) and irrelevant (Z_I) latent codes.
    """
    
    def __init__(self, latent_dim_R, latent_dim_I, alpha=0.5, beta=0.3):
        """
        Args:
            latent_dim_R: Dimension of relevant latent code
            latent_dim_I: Dimension of irrelevant latent code
            alpha: Weight for semantic disentanglement loss
            beta: Weight for purified synthesis loss
        """
        super(InformationPurification, self).__init__()
        
        self.latent_dim_R = latent_dim_R
        self.latent_dim_I = latent_dim_I
        self.alpha = alpha
        self.beta = beta
        
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
    def compute_reconstruction_loss(self, images_flat, recon_images):
        """
        Compute reconstruction loss J_rec.
        Measures how well the decoder reconstructs original images from Z_R and Z_I.
        
        Args:
            images_flat: Original flattened images
            recon_images: Reconstructed images from decoder
            
        Returns:
            Reconstruction loss
        """
        return self.reconstruction_loss(recon_images, images_flat)
    
    def compute_disentanglement_loss(self, z_R, z_I, labels, pred_R, pred_I):
        """
        Compute semantic disentanglement loss J_dis.
        Maximizes I(Y; Z_R) and minimizes I(Y; Z_I).
        
        Args:
            z_R: Relevant latent code
            z_I: Irrelevant latent code
            labels: Ground truth labels
            pred_R: Predictions from Z_R
            pred_I: Predictions from Z_I
            
        Returns:
            Disentanglement loss
        """
        # Maximize I(Y; Z_R) by minimizing cross-entropy
        loss_R = self.classification_loss(pred_R, labels)
        
        # Minimize I(Y; Z_I) by maximizing cross-entropy (unlearning)
        # We want the classifier to be uncertain about labels given Z_I
        loss_I = -self.classification_loss(pred_I, labels)
        
        return loss_I - loss_R
    
    def compute_synthesis_loss(self, synthetic_images, z_R, z_I, generator):
        """
        Compute purified synthesis loss J_syn.
        Maximizes I(X_tilde; Z_R) and minimizes I(X_tilde; Z_I).
        
        Args:
            synthetic_images: Generated synthetic images
            z_R: Relevant latent code
            z_I: Irrelevant latent code
            generator: Generator network
            
        Returns:
            Synthesis loss
        """
        # Flatten synthetic images
        syn_flat = synthetic_images.view(synthetic_images.size(0), -1)
        
        # Try to reconstruct synthetic images from Z_R (should succeed)
        # This is approximated by checking if generator uses Z_R effectively
        recon_from_R = generator.decoder(z_R, torch.zeros_like(z_I))
        loss_R = self.reconstruction_loss(recon_from_R, syn_flat)
        
        # Try to reconstruct synthetic images from Z_I (should fail)
        recon_from_I = generator.decoder(torch.zeros_like(z_R), z_I)
        loss_I = self.reconstruction_loss(recon_from_I, syn_flat)
        
        # We want low loss_R (high dependence on Z_R) and high loss_I (low dependence on Z_I)
        return loss_I - loss_R
    
    def compute_loss(self, images_flat, recon_images, z_R, z_I, 
                    synthetic_images, labels, pred_R, pred_I, generator):
        """
        Compute total information purification loss.
        
        Args:
            images_flat: Original flattened images
            recon_images: Reconstructed images
            z_R: Relevant latent code
            z_I: Irrelevant latent code
            synthetic_images: Generated synthetic images
            labels: Ground truth labels
            pred_R: Predictions from Z_R
            pred_I: Predictions from Z_I
            generator: Generator network
            
        Returns:
            Dictionary containing all loss components
        """
        # Component losses
        loss_rec = self.compute_reconstruction_loss(images_flat, recon_images)
        loss_dis = self.compute_disentanglement_loss(z_R, z_I, labels, pred_R, pred_I)
        loss_syn = self.compute_synthesis_loss(synthetic_images, z_R, z_I, generator)
        
        # Total loss
        total_loss = loss_rec + self.alpha * loss_dis + self.beta * loss_syn
        
        return {
            'total': total_loss,
            'rec': loss_rec,
            'dis': loss_dis,
            'syn': loss_syn
        }
    
    def compute_mutual_information(self, z, labels, num_classes):
        """
        Estimate mutual information I(Z; Y) using variational bound.
        
        Args:
            z: Latent code
            labels: Ground truth labels
            num_classes: Number of classes
            
        Returns:
            Estimated mutual information
        """
        # Compute class-conditional distributions
        mi = 0.0
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                z_c = z[mask]
                # Estimate entropy using Gaussian assumption
                cov = torch.cov(z_c.T)
                sign, logdet = torch.slogdet(cov + 1e-6 * torch.eye(z.size(1), device=z.device))
                entropy_c = 0.5 * logdet
                mi += entropy_c * (mask.sum().float() / labels.size(0))
        
        # Marginal entropy
        cov_marginal = torch.cov(z.T)
        sign, logdet_marginal = torch.slogdet(cov_marginal + 1e-6 * torch.eye(z.size(1), device=z.device))
        entropy_marginal = 0.5 * logdet_marginal
        
        return entropy_marginal - mi
