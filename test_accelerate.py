#!/usr/bin/env python
"""
Test script for AccelerateTrainer integration.
This script tests basic functionality without requiring full dataset setup.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import tempfile

def test_accelerate_integration():
    """Test basic AccelerateTrainer functionality."""
    
    print("Testing AccelerateTrainer integration...")
    
    # Import detectron2 components
    from detectron2.engine.train_loop import AccelerateTrainer
    from detectron2.config import get_cfg
    from detectron2.config.defaults import _C
    
    print("✓ Successfully imported AccelerateTrainer")
    
    # Test configuration
    cfg = get_cfg()
    
    # Check if ACCELERATE config exists
    assert hasattr(cfg.SOLVER, 'ACCELERATE'), "ACCELERATE config not found in SOLVER"
    assert hasattr(cfg.SOLVER.ACCELERATE, 'ENABLED'), "ACCELERATE.ENABLED not found"
    assert hasattr(cfg.SOLVER.ACCELERATE, 'MIXED_PRECISION'), "ACCELERATE.MIXED_PRECISION not found"
    
    print("✓ ACCELERATE configuration is properly set up")
    
    # Create dummy model, data, optimizer for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, data):
            # Simulate detectron2 model interface
            x = data['input']
            loss = self.linear(x).mean()
            return {"loss": loss}
    
    class DummyDataset(Dataset):
        def __init__(self, size=100):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {"input": torch.randn(10)}
    
    # Create model, data, optimizer
    model = DummyModel()
    dataset = DummyDataset(20)
    dataloader = DataLoader(dataset, batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    
    print("✓ Created dummy model, dataset, and optimizer")
    
    # Test AccelerateTrainer creation
    try:
        accelerate_cfg = {
            "mixed_precision": "no",  # Use "no" for CPU testing
            "cpu": True,  # Force CPU for testing
            "gradient_accumulation_steps": 1,
        }
        
        trainer = AccelerateTrainer(
            model=model,
            data_loader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerate_cfg=accelerate_cfg,
        )
        
        print("✓ Successfully created AccelerateTrainer")
        
        # Test a single training step with EventStorage context
        from detectron2.utils.events import EventStorage
        
        with EventStorage():
            trainer.iter = 0
            trainer.run_step()
        
        print("✓ Successfully ran one training step")
        
        # Test state dict
        state_dict = trainer.state_dict()
        assert 'optimizer' in state_dict
        print("✓ Successfully created state dict")
        
    except Exception as e:
        print(f"✗ Error creating or running AccelerateTrainer: {e}")
        raise
    
    print("\n🎉 All tests passed! AccelerateTrainer integration is working correctly.")

if __name__ == "__main__":
    # Set environment to avoid GPU if not available
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    test_accelerate_integration()
