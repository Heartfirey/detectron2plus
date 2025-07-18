# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Progress bar decorators and utilities for data processing in Detectron2.
"""

from typing import Iterator, Any, Optional, Iterable
from functools import wraps

from detectron2.utils.progress import (
    get_progress_manager,
    track_training,
    track_evaluation,
    dataset_progress,
    inference_progress,
)

__all__ = [
    "progress_wrapper",
    "training_dataloader", 
    "evaluation_dataloader",
    "inference_dataloader",
    "dataset_iterator",
]


def progress_wrapper(
    dataloader, 
    description: str = "Processing", 
    style: str = "training",
    task_name: Optional[str] = None
):
    """
    Wrap a dataloader with a progress bar.
    
    Args:
        dataloader: The dataloader to wrap
        description: Description for the progress bar
        style: Style of progress bar ("training", "evaluation", "inference", "dataset")
        task_name: Optional custom task name
    
    Returns:
        An iterator that yields items from dataloader with progress tracking
    """
    if task_name is None:
        task_name = f"{style}_dataloader"
    
    total = None
    try:
        total = len(dataloader)
    except:
        pass  # Some dataloaders don't support len()
    
    manager = get_progress_manager()
    
    if manager.is_active():
        # Use existing progress context
        with manager.task_context(task_name, description, total) as name:
            for item in dataloader:
                yield item
                manager.update(name)
    else:
        # Create appropriate progress context
        if style == "training":
            for item in track_training(dataloader, description, total, task_name):
                yield item
        elif style == "evaluation":
            for item in track_evaluation(dataloader, description, total, task_name):
                yield item
        elif style == "inference":
            with inference_progress():
                with manager.task_context(task_name, description, total) as name:
                    for item in dataloader:
                        yield item
                        manager.update(name)
        elif style == "dataset":
            with dataset_progress():
                with manager.task_context(task_name, description, total) as name:
                    for item in dataloader:
                        yield item
                        manager.update(name)
        else:
            # Default to training style
            for item in track_training(dataloader, description, total, task_name):
                yield item


def training_dataloader(dataloader, description: str = "Training Data"):
    """Wrap dataloader with training-style progress bar."""
    return progress_wrapper(dataloader, description, "training", "training_data")


def evaluation_dataloader(dataloader, description: str = "Evaluation Data"):
    """Wrap dataloader with evaluation-style progress bar.""" 
    return progress_wrapper(dataloader, description, "evaluation", "eval_data")


def inference_dataloader(dataloader, description: str = "Inference Data"):
    """Wrap dataloader with inference-style progress bar."""
    return progress_wrapper(dataloader, description, "inference", "inference_data")


def dataset_iterator(
    iterable, 
    description: str = "Processing Dataset",
    total: Optional[int] = None
):
    """
    Wrap any iterable with dataset processing progress bar.
    
    Args:
        iterable: The iterable to wrap
        description: Description for the progress bar
        total: Total number of items (if known)
    
    Returns:
        An iterator with progress tracking
    """
    if total is None:
        try:
            total = len(iterable)
        except:
            pass
    
    manager = get_progress_manager()
    
    if manager.is_active():
        with manager.task_context("dataset_iter", description, total) as name:
            for item in iterable:
                yield item
                manager.update(name)
    else:
        with dataset_progress():
            with manager.task_context("dataset_iter", description, total) as name:
                for item in iterable:
                    yield item
                    manager.update(name)


def progress_decorator(style: str = "training", description: str = "Processing"):
    """
    Decorator to automatically add progress bars to functions that return iterables.
    
    Args:
        style: Progress bar style
        description: Progress bar description
    
    Example:
        @progress_decorator("evaluation", "Running inference")
        def inference_on_dataset(model, data_loader):
            for batch in data_loader:
                yield model(batch)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # If result is an iterator/generator, wrap it with progress
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
                return progress_wrapper(result, description, style)
            else:
                return result
        
        return wrapper
    return decorator
