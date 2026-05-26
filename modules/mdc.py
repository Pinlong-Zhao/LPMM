"""
Multi-Descriptor Consensus (MDC) module.
Implements the consensus matching objective with entropy-based weighting.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy

from .descriptors import get_descriptor


class MultiDescriptorConsensus(nn.Module):
    """
    Multi-Descriptor Consensus module.
    Combines multiple statistical descriptors with adaptive weighting.
    """
    
    def __init__(self, descriptors, weight_method='ewm'):
        """
        Args:
            descriptors: List of descriptor names to use
            weight_method: Weighting method ('ewm', 'uniform', 'learned')
        """
        super(MultiDescriptorConsensus, self).__init__()
        
        self.descriptor_names = descriptors
        self.descriptors = [get_descriptor(name) for name in descriptors]
        self.weight_method = weight_method
        self.num_descriptors = len(descriptors)
        
        # Initialize weights
        if weight_method == 'learned':
            self.weights = nn.Parameter(torch.ones(self.num_descriptors))
        else:
            self.register_buffer('weights', torch.ones(self.num_descriptors))
        
    def compute_ewm_weights(self, descriptor_values):
        """
        Compute Entropy Weight Method (EWM) weights.
        
        Args:
            descriptor_values: List of descriptor value tensors
            
        Returns:
            Normalized weights (num_descriptors,)
        """
        weights = []
        
        for values in descriptor_values:
            # Normalize values to [0, 1]
            values_np = values.cpu().numpy()
            values_norm = (values_np - values_np.min()) / (values_np.max() - values_np.min() + 1e-10)
            
            # Compute probability distribution
            hist, _ = np.histogram(values_norm, bins=10, range=(0, 1))
            prob = hist / (hist.sum() + 1e-10)
            
            # Compute entropy
            ent = entropy(prob + 1e-10)
            
            # Weight is proportional to entropy (higher entropy = more informative)
            weights.append(ent)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-10)
        
        return torch.tensor(weights, dtype=torch.float32, device=descriptor_values[0].device)
    
    def compute_descriptor_distance(self, desc_real, desc_synthetic):
        """
        Compute distance between real and synthetic descriptor distributions.
        
        Args:
            desc_real: Descriptor values for real data
            desc_synthetic: Descriptor values for synthetic data
            
        Returns:
            Distance value (scalar)
        """
        # Compute mean and variance
        mean_real = desc_real.mean()
        mean_syn = desc_synthetic.mean()
        var_real = desc_real.var()
        var_syn = desc_synthetic.var()
        
        # L2 distance between statistics
        mean_dist = (mean_real - mean_syn) ** 2
        var_dist = (var_real - var_syn) ** 2
        
        return mean_dist + 0.1 * var_dist
    
    def compute_loss(self, synthetic_images, real_images, labels, model):
        """
        Compute multi-descriptor consensus loss.
        
        Args:
            synthetic_images: Generated synthetic images
            real_images: Real training images
            labels: Ground truth labels
            model: Classification model for computing descriptors
            
        Returns:
            Total consensus loss
        """
        descriptor_values_real = []
        descriptor_values_syn = []
        
        # Compute all descriptors
        for descriptor in self.descriptors:
            desc_real = descriptor.compute(real_images, labels, model)
            desc_syn = descriptor.compute(synthetic_images, labels, model)
            
            descriptor_values_real.append(desc_real)
            descriptor_values_syn.append(desc_syn)
        
        # Update weights if using EWM
        if self.weight_method == 'ewm':
            self.weights = self.compute_ewm_weights(descriptor_values_real)
        elif self.weight_method == 'uniform':
            self.weights = torch.ones(self.num_descriptors, device=real_images.device) / self.num_descriptors
        
        # Compute weighted loss
        total_loss = 0.0
        for i, (desc_real, desc_syn) in enumerate(zip(descriptor_values_real, descriptor_values_syn)):
            dist = self.compute_descriptor_distance(desc_real, desc_syn)
            total_loss += self.weights[i] * dist
        
        return total_loss
    
    def get_weights(self):
        """Get current descriptor weights."""
        if self.weight_method == 'learned':
            return torch.softmax(self.weights, dim=0)
        return self.weights
