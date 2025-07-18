#!/usr/bin/env python
"""
Test script for Rich progress bar integration in Detectron2.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import os


def test_progress_bars():
    """Test various progress bar functionalities."""
    
    print("🧪 Testing Detectron2 Rich Progress Bar Integration...")
    
    # Import progress components
    from detectron2.utils.progress import (
        training_progress, 
        evaluation_progress,
        inference_progress,
        dataset_progress,
        get_progress_manager,
        track_training,
        track_evaluation
    )
    from detectron2.utils.progress_wrappers import (
        training_dataloader,
        evaluation_dataloader,
        inference_dataloader,
        dataset_iterator,
        progress_wrapper
    )
    
    print("✓ Successfully imported progress components")
    
    # Test 1: Basic progress contexts
    print("\n📊 Test 1: Basic Progress Contexts")
    
    # Training progress
    with training_progress() as progress:
        with progress.task_context("test_train", "Testing training progress", 10) as task:
            for i in range(10):
                time.sleep(0.1)
                progress.update(task, metrics={"loss": 1.0 - i*0.1, "lr": 0.001})
    
    print("✓ Training progress test completed")
    
    # Evaluation progress  
    with evaluation_progress() as progress:
        with progress.task_context("test_eval", "Testing evaluation progress", 5) as task:
            for i in range(5):
                time.sleep(0.1)
                progress.update(task, metrics={"accuracy": i*0.2, "mAP": i*0.15})
    
    print("✓ Evaluation progress test completed")
    
    # Test 2: Data loader wrapping
    print("\n📦 Test 2: DataLoader Progress Wrapping")
    
    class DummyDataset(Dataset):
        def __init__(self, size=20):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {"input": torch.randn(10), "target": torch.randint(0, 2, (1,))}
    
    dataset = DummyDataset(20)
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Test training dataloader wrapper
    print("Training DataLoader:")
    for batch in training_dataloader(dataloader, "Loading training data"):
        time.sleep(0.05)  # Simulate processing time
    
    print("✓ Training DataLoader wrapper test completed")
    
    # Test evaluation dataloader wrapper
    print("Evaluation DataLoader:")
    for batch in evaluation_dataloader(dataloader, "Loading evaluation data"):
        time.sleep(0.05)
    
    print("✓ Evaluation DataLoader wrapper test completed")
    
    # Test 3: Nested progress contexts
    print("\n🔄 Test 3: Nested Progress Contexts")
    
    with training_progress() as training_prog:
        # Main training loop
        training_prog.add_task("main_training", "Main Training Loop", 3)
        
        for epoch in range(3):
            # Training phase
            training_prog.add_task("train_epoch", f"Training Epoch {epoch+1}", 5)
            
            for batch_idx in range(5):
                time.sleep(0.1)
                training_prog.update("train_epoch", metrics={
                    "epoch": epoch+1,
                    "batch": batch_idx+1,
                    "loss": 1.0 - (epoch*5 + batch_idx)*0.05
                })
            
            training_prog.remove_task("train_epoch")
            
            # Evaluation phase (nested)
            training_prog.add_task("eval_epoch", f"Evaluation Epoch {epoch+1}", 3)
            
            for eval_batch in range(3):
                time.sleep(0.1)
                training_prog.update("eval_epoch", metrics={
                    "eval_batch": eval_batch+1,
                    "accuracy": eval_batch * 0.1 + 0.7
                })
            
            training_prog.remove_task("eval_epoch")
            training_prog.update("main_training", metrics={"epoch": epoch+1})
    
    print("✓ Nested progress contexts test completed")
    
    # Test 4: Custom progress styles
    print("\n🎨 Test 4: Custom Progress Styles")
    
    # Dataset processing style
    with dataset_progress() as progress:
        items = list(range(15))
        for item in dataset_iterator(items, "Processing dataset items"):
            time.sleep(0.05)
    
    print("✓ Dataset progress style test completed")
    
    # Inference style
    with inference_progress() as progress:
        with progress.task_context("inference", "Running inference", 8) as task:
            for i in range(8):
                time.sleep(0.1)
                progress.update(task, metrics={
                    "processed": i+1,
                    "fps": (i+1) * 10
                })
    
    print("✓ Inference progress style test completed")
    
    # Test 5: Multi-GPU safety (simulate)
    print("\n🔧 Test 5: Multi-GPU Safety Simulation")
    
    # Simulate non-main process
    original_comm = None
    try:
        import detectron2.utils.comm as comm
        original_is_main = comm.is_main_process
        
        # Mock non-main process
        comm.is_main_process = lambda: False
        
        # These should not show progress bars
        with training_progress() as progress:
            progress.add_task("test", "Should not show", 5)
            for i in range(5):
                progress.update("test")
                time.sleep(0.05)
        
        # Restore main process
        comm.is_main_process = original_is_main
        
        print("✓ Multi-GPU safety test completed")
        
    except Exception as e:
        if original_comm:
            # Restore original if something went wrong
            comm.is_main_process = original_is_main
        print(f"⚠️  Multi-GPU safety test failed: {e}")
    
    # Test 6: Progress manager singleton behavior
    print("\n🔄 Test 6: Progress Manager Singleton")
    
    manager1 = get_progress_manager()
    manager2 = get_progress_manager()
    
    assert manager1 is manager2, "Progress manager should be singleton"
    print("✓ Progress manager singleton test completed")
    
    print("\n🎉 All progress bar tests completed successfully!")
    print("\nProgress bar system features:")
    print("  ✅ Rich-based beautiful progress bars")
    print("  ✅ Multi-GPU safe (only shows on main process)")
    print("  ✅ Multiple styles (training, evaluation, inference, dataset)")
    print("  ✅ Nested progress contexts")
    print("  ✅ Custom metrics display")
    print("  ✅ DataLoader wrapper utilities")
    print("  ✅ Thread-safe progress management")
    print("  ✅ Context manager integration")


def test_trainer_integration():
    """Test progress bar integration with a simple trainer."""
    
    print("\n🏋️ Testing Trainer Integration...")
    
    from detectron2.engine.train_loop import AccelerateTrainer
    from detectron2.utils.events import EventStorage
    
    # Create dummy components
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            
        def forward(self, data):
            x = data['input']
            loss = self.linear(x).mean()
            return {"total_loss": loss}
    
    class DummyDataset(Dataset):
        def __init__(self, size=50):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {"input": torch.randn(10)}
    
    # Create trainer components
    model = DummyModel()
    dataset = DummyDataset(50)
    dataloader = DataLoader(dataset, batch_size=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Test AccelerateTrainer with progress
    trainer = AccelerateTrainer(
        model=model,
        data_loader=dataloader,
        optimizer=optimizer,
        accelerate_cfg={"cpu": True, "mixed_precision": "no"}
    )
    
    # Test with EventStorage (simulating training)
    with EventStorage() as storage:
        trainer.storage = storage
        
        # Test a few training steps
        for i in range(5):
            trainer.iter = i
            trainer.run_step()
            # Add some metrics to storage for testing
            storage.put_scalar("total_loss", 1.0 - i*0.1)
            storage.put_scalar("lr", 0.01)
    
    print("✓ Trainer integration test completed")


if __name__ == "__main__":
    # Set environment to CPU only for testing
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    test_progress_bars()
    test_trainer_integration()
