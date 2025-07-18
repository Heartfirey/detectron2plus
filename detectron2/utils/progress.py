# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Rich-based progress bar system for Detectron2.
Provides beautiful, informative progress bars for training, evaluation, and data processing.
"""

import os
import time
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Any, Union
from collections import defaultdict

from rich.progress import (
    Progress,
    TaskID,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    ProgressColumn,
)
from rich.text import Text
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

import detectron2.utils.comm as comm

__all__ = [
    "ProgressManager",
    "get_progress_manager", 
    "training_progress",
    "evaluation_progress",
    "dataset_progress",
    "inference_progress",
]


class MetricsColumn(ProgressColumn):
    """Custom column to display training metrics like loss, accuracy etc."""
    
    def __init__(self, metrics_key: str = "metrics"):
        self.metrics_key = metrics_key
        super().__init__()
    
    def render(self, task) -> Text:
        """Render the metrics text."""
        metrics = task.fields.get(self.metrics_key, {})
        if not metrics:
            return Text("", style="dim")
        
        # Format metrics nicely
        metric_strs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if key.lower() in ['loss', 'lr', 'learning_rate']:
                    metric_strs.append(f"{key}: {value:.6f}")
                else:
                    metric_strs.append(f"{key}: {value:.4f}")
            else:
                metric_strs.append(f"{key}: {value}")
        
        return Text(" | ".join(metric_strs), style="cyan")


class SpeedColumn(ProgressColumn):
    """Custom column to display processing speed (e.g., iterations/sec)."""
    
    def render(self, task) -> Text:
        """Render the speed text."""
        speed = task.speed
        if speed is None:
            return Text("", style="dim")
        
        if speed > 1:
            return Text(f"{speed:.1f} it/s", style="green")
        else:
            return Text(f"{1/speed:.1f} s/it", style="yellow")


class ProgressManager:
    """
    Centralized progress bar manager for Detectron2.
    
    Features:
    - Multi-GPU safe (only shows progress on main process)
    - Multiple progress styles for different tasks
    - Automatic cleanup and context management
    - Rich integration with custom columns
    """
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._progress: Optional[Progress] = None
        self._live: Optional[Live] = None
        self._tasks: Dict[str, TaskID] = {}
        self._lock = threading.Lock()
        self._is_main_process = comm.is_main_process()
        self._active = False
        
    def _get_training_columns(self):
        """Get columns for training progress bar."""
        return [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            SpeedColumn(),
            TextColumn("•"),
            MetricsColumn(),
        ]
    
    def _get_evaluation_columns(self):
        """Get columns for evaluation progress bar."""
        return [
            SpinnerColumn("dots"),
            TextColumn("[bold green]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            SpeedColumn(),
            TextColumn("•"),
            MetricsColumn(),
        ]
    
    def _get_dataset_columns(self):
        """Get columns for dataset processing progress bar."""
        return [
            SpinnerColumn("line"),
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            SpeedColumn(),
        ]
    
    def _get_inference_columns(self):
        """Get columns for inference progress bar."""
        return [
            SpinnerColumn("arrow3"),
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            SpeedColumn(),
        ]
    
    def start(self, style: str = "training"):
        """Start the progress manager with specified style."""
        if not self._is_main_process:
            return
        
        with self._lock:
            if self._active:
                return
                
            # Choose columns based on style
            if style == "training":
                columns = self._get_training_columns()
            elif style == "evaluation":
                columns = self._get_evaluation_columns()
            elif style == "dataset":
                columns = self._get_dataset_columns()
            elif style == "inference":
                columns = self._get_inference_columns()
            else:
                columns = self._get_training_columns()  # Default
            
            self._progress = Progress(*columns, console=self.console)
            self._live = Live(self._progress, console=self.console, refresh_per_second=10)
            self._live.start()
            self._active = True
    
    def stop(self):
        """Stop the progress manager and cleanup."""
        if not self._is_main_process:
            return
            
        with self._lock:
            if not self._active:
                return
                
            if self._live:
                self._live.stop()
                self._live = None
            self._progress = None
            self._tasks.clear()
            self._active = False
    
    def add_task(
        self, 
        name: str, 
        description: str, 
        total: Optional[int] = None,
        **kwargs
    ) -> str:
        """Add a new progress task."""
        if not self._is_main_process or not self._active:
            return name
            
        with self._lock:
            if self._progress is None:
                return name
                
            task_id = self._progress.add_task(
                description=description,
                total=total,
                **kwargs
            )
            self._tasks[name] = task_id
            return name
    
    def update(
        self, 
        name: str, 
        advance: int = 1, 
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Update a progress task."""
        if not self._is_main_process or not self._active:
            return
            
        with self._lock:
            if self._progress is None or name not in self._tasks:
                return
                
            update_kwargs = kwargs.copy()
            if advance:
                update_kwargs['advance'] = advance
            if metrics:
                update_kwargs['metrics'] = metrics
                
            self._progress.update(self._tasks[name], **update_kwargs)
    
    def set_description(self, name: str, description: str):
        """Update task description."""
        if not self._is_main_process or not self._active:
            return
            
        with self._lock:
            if self._progress is None or name not in self._tasks:
                return
                
            self._progress.update(self._tasks[name], description=description)
    
    def remove_task(self, name: str):
        """Remove a progress task."""
        if not self._is_main_process or not self._active:
            return
            
        with self._lock:
            if self._progress is None or name not in self._tasks:
                return
                
            self._progress.remove_task(self._tasks[name])
            del self._tasks[name]
    
    def is_active(self) -> bool:
        """Check if progress manager is active."""
        return self._active and self._is_main_process
    
    @contextmanager
    def task_context(
        self, 
        name: str, 
        description: str, 
        total: Optional[int] = None,
        **kwargs
    ):
        """Context manager for automatic task management."""
        self.add_task(name, description, total, **kwargs)
        try:
            yield name
        finally:
            self.remove_task(name)


# Global progress manager instance
_global_progress_manager: Optional[ProgressManager] = None
_manager_lock = threading.Lock()


def get_progress_manager() -> ProgressManager:
    """Get or create the global progress manager instance."""
    global _global_progress_manager
    
    with _manager_lock:
        if _global_progress_manager is None:
            _global_progress_manager = ProgressManager()
        return _global_progress_manager


@contextmanager
def training_progress():
    """Context manager for training progress bars."""
    manager = get_progress_manager()
    manager.start("training")
    try:
        yield manager
    finally:
        manager.stop()


@contextmanager
def evaluation_progress():
    """Context manager for evaluation progress bars."""
    manager = get_progress_manager()
    manager.start("evaluation")
    try:
        yield manager
    finally:
        manager.stop()


@contextmanager
def dataset_progress():
    """Context manager for dataset processing progress bars."""
    manager = get_progress_manager()
    manager.start("dataset")
    try:
        yield manager
    finally:
        manager.stop()


@contextmanager
def inference_progress():
    """Context manager for inference progress bars."""
    manager = get_progress_manager()
    manager.start("inference")
    try:
        yield manager
    finally:
        manager.stop()


# Convenience functions for quick progress bars
def track_training(
    iterable, 
    description: str = "Training", 
    total: Optional[int] = None,
    task_name: str = "main_training"
):
    """Track an iterable with training-style progress bar."""
    manager = get_progress_manager()
    
    if not manager.is_active():
        # If no progress context is active, create a temporary one
        with training_progress():
            with manager.task_context(task_name, description, total or len(iterable)) as name:
                for item in iterable:
                    yield item
                    manager.update(name)
    else:
        # Use existing progress context
        with manager.task_context(task_name, description, total or len(iterable)) as name:
            for item in iterable:
                yield item
                manager.update(name)


def track_evaluation(
    iterable, 
    description: str = "Evaluating", 
    total: Optional[int] = None,
    task_name: str = "main_evaluation"
):
    """Track an iterable with evaluation-style progress bar."""
    manager = get_progress_manager()
    
    if not manager.is_active():
        with evaluation_progress():
            with manager.task_context(task_name, description, total or len(iterable)) as name:
                for item in iterable:
                    yield item
                    manager.update(name)
    else:
        with manager.task_context(task_name, description, total or len(iterable)) as name:
            for item in iterable:
                yield item
                manager.update(name)
