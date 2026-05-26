"""
Test script to verify installation and basic functionality.
"""

import torch
import sys

def test_imports():
    """Test if all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.encoder import RelevantEncoder, IrrelevantEncoder
        from models.decoder import Decoder
        from models.generator import SyntheticGenerator
        from models.classifier import AuxiliaryClassifier
        from modules.mdc import MultiDescriptorConsensus
        from modules.information_purification import InformationPurification
        from modules.descriptors import get_descriptor
        from utils.data_loader import get_dataset
        from utils.metrics import evaluate_distilled_dataset
        print("✓ All imports successful!")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_models():
    """Test if models can be instantiated."""
    print("\nTesting model instantiation...")
    
    try:
        from models.encoder import RelevantEncoder, IrrelevantEncoder
        from models.decoder import Decoder
        from models.generator import SyntheticGenerator
        from models.classifier import AuxiliaryClassifier
        
        # Test encoder
        encoder_R = RelevantEncoder(input_dim=3072, latent_dim=128)
        encoder_I = IrrelevantEncoder(input_dim=3072, latent_dim=64)
        print("✓ Encoders instantiated")
        
        # Test decoder
        decoder = Decoder(latent_dim_R=128, latent_dim_I=64, output_dim=3072)
        print("✓ Decoder instantiated")
        
        # Test generator
        generator = SyntheticGenerator(latent_dim=128, output_size=32, num_classes=10, ipc=10)
        print("✓ Generator instantiated")
        
        # Test classifier
        classifier = AuxiliaryClassifier(latent_dim=128, num_classes=10)
        print("✓ Classifier instantiated")
        
        return True
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through models."""
    print("Testing forward pass...")
    
    try:
        from models.encoder import RelevantEncoder, IrrelevantEncoder
        from models.decoder import Decoder
        from models.generator import SyntheticGenerator
        
        # Create dummy data
        batch_size = 4
        input_dim = 3072
        x = torch.randn(batch_size, input_dim)
        labels = torch.randint(0, 10, (batch_size,))
        
        # Encoder forward
        encoder_R = RelevantEncoder(input_dim=input_dim, latent_dim=128)
        z_R_mu, z_R_logvar = encoder_R(x)
        print(f"✓ Encoder output shape: {z_R_mu.shape}")
        
        encoder_I = IrrelevantEncoder(input_dim=input_dim, latent_dim=64)
        z_I_mu, z_I_logvar = encoder_I(x)
        print(f"✓ Encoder I output shape: {z_I_mu.shape}")
        
        # Decoder forward
        decoder = Decoder(latent_dim_R=128, latent_dim_I=64, output_dim=input_dim)
        recon = decoder(z_R_mu, z_I_mu)
        print(f"✓ Decoder output shape: {recon.shape}")
        
        # Generator forward
        generator = SyntheticGenerator(latent_dim=128, output_size=32, num_classes=10, ipc=10)
        synthetic = generator(z_R_mu, labels)
        print(f"✓ Generator output shape: {synthetic.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("Testing CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("✗ CUDA is not available (CPU only)")
        return False


def main():
    """Run all tests."""
    print("="*50)
    print("LPMM Framework Installation Test")
    print("="*50)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Model Instantiation", test_models()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("CUDA", test_cuda()))
    
    # Summary
    print("" + "="*50)
    print("Test Summary")
    print("="*50)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(result[1] for result in results[:-1])  # Exclude CUDA test
    
    if all_passed:
        print("✓ All critical tests passed! Installation is successful.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
