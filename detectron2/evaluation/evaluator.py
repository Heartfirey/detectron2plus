# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    callbacks=None,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    from detectron2.utils.progress import get_progress_manager
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    # Setup progress tracking
    progress_manager = get_progress_manager()
    use_existing_progress = progress_manager.is_active()
    
    if not use_existing_progress:
        # Create evaluation progress context if none exists
        from detectron2.utils.progress import evaluation_progress
        progress_context = evaluation_progress()
        progress_context.__enter__()
    else:
        progress_context = None

    # Add inference task
    progress_manager.add_task(
        "inference",
        f"Running inference on {total} batches",
        total=total
    )

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    
    try:
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            dict.get(callbacks or {}, "on_start", lambda: None)()
            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                dict.get(callbacks or {}, "before_inference", lambda: None)()
                outputs = model(inputs)
                dict.get(callbacks or {}, "after_inference", lambda: None)()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                # Update progress with metrics
                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                if iters_after_start > 0:
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    eval_seconds_per_iter = total_eval_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                    
                    progress_metrics = {
                        "data_time": f"{data_seconds_per_iter:.4f}s",
                        "compute_time": f"{compute_seconds_per_iter:.4f}s", 
                        "eval_time": f"{eval_seconds_per_iter:.4f}s"
                    }
                    
                    progress_manager.update(
                        "inference",
                        advance=1,
                        metrics=progress_metrics
                    )
                    progress_manager.set_description(
                        "inference",
                        f"Inference [{idx+1}/{total}]"
                    )
                    
                    if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            n=5,
                        )
                else:
                    # Warmup phase
                    progress_manager.update("inference", advance=1)
                    progress_manager.set_description(
                        "inference", 
                        f"Warmup [{idx+1}/{total}]"
                    )
                start_data_time = time.perf_counter()
            dict.get(callbacks or {}, "on_end", lambda: None)()
    
    finally:
        # Clean up progress tracking
        progress_manager.remove_task("inference")
        
        if progress_context is not None:
            progress_context.__exit__(None, None, None)

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str,
            total_compute_time / (total - num_warmup),
            num_devices,
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
