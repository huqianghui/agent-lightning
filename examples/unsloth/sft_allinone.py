# Copyright (c) Microsoft. All rights reserved.

"""This sample provides an all-in-one script for SFT algorithm.

It's equivalent to running the following commands in parallel:

```bash
agl store
python sft_rollout_runners.py
python sft_algorithm.py
```
"""

from typing import Optional

from math_agent import GsmProblem, load_math_dataset, math_agent
from rich.console import Console
from sft_algorithm import sft_one_iter

from agentlightning import Trainer, setup_logging
from agentlightning.adapter import TraceToTripletBase
from agentlightning.algorithm import Algorithm
from agentlightning.llm_proxy import LLMProxy
from agentlightning.types import Dataset

console = Console()


class UnslothSupervisedFinetuning(Algorithm):
    """Supervised Fine-Tuning (SFT) algorithm implementation using Unsloth.

    This class implements a complete SFT training loop that:
    1. Runs rollouts with the current model
    2. Collects and filters training data by reward
    3. Fine-tunes the model on rewarded examples
    4. Iterates for multiple rounds of improvement

    Args:
        max_iterations: Optional safety limit for SFT iterations. If None, run until SFT data stops changing.
        vllm_port: The port to use for the vLLM inference server.
        reward_threshold: Only triplets with rewards greater than this threshold are used for training.
        initial_model_path: The path to the initial model to start training from.
    """

    def __init__(
        self,
        *,
        max_iterations: Optional[int],
        vllm_port: int,
        reward_threshold: float,
        initial_model_path: str,
    ):
        # LLM proxy and data adapter are created by the trainer and we can directly use them
        self.max_iterations = max_iterations
        self.vllm_port = vllm_port
        self.reward_threshold = reward_threshold
        self.initial_model_path = initial_model_path

    async def run(
        self, train_dataset: Optional[Dataset[GsmProblem]] = None, val_dataset: Optional[Dataset[GsmProblem]] = None
    ):
        """Execute the SFT training loop. Managed by trainer.

        Args:
            train_dataset: The training dataset of GSM problems to use for rollouts.
            val_dataset: Optional validation dataset (not currently used in SFT).

        Raises:
            ValueError: If train_dataset is None, or required components are missing.
        """
        store = self.get_store()
        llm_proxy = self.get_llm_proxy()
        data_adapter = self.get_adapter()

        # SFT trainer relies on the adapter to convert the trace data to triplets
        if not isinstance(data_adapter, TraceToTripletBase):
            raise ValueError("Data adapter must be a TracerTraceToTriplet.")
        if train_dataset is None:
            raise ValueError("Train dataset must be provided.")
        if val_dataset is not None:
            console.print("[bold red][Algo][/bold red] Validation dataset is not supported in SFT.")
        if llm_proxy is None:
            raise ValueError("LLM proxy must be provided.")

        if self.max_iterations is None:
            console.print("[bold red][Algo][/bold red] Starting SFT until SFT data stops changing.")
        else:
            console.print(f"[bold red][Algo][/bold red] Starting SFT with up to {self.max_iterations} iterations.")
        console.print(f"[bold red][Algo][/bold red] Initial model path: {self.initial_model_path}")
        model_path = self.initial_model_path
        iteration = 0
        previous_data_fingerprint: Optional[str] = None
        while self.max_iterations is None or iteration < self.max_iterations:
            result = await sft_one_iter(
                iteration=iteration,
                store=store,
                model_path=model_path,
                train_dataset=train_dataset,
                llm_proxy=llm_proxy,
                data_adapter=data_adapter,
                reward_threshold=self.reward_threshold,
                vllm_port=self.vllm_port,
                previous_data_fingerprint=previous_data_fingerprint,
            )
            model_path = result.model_path
            if not result.data_changed:
                break
            previous_data_fingerprint = result.data_fingerprint
            iteration += 1

        console.print(f"[bold red][Algo][/bold red] Final model path: {model_path}")


if __name__ == "__main__":
    setup_logging()

    algo = UnslothSupervisedFinetuning(
        max_iterations=None,
        vllm_port=12316,
        reward_threshold=0.0,
        initial_model_path="models/version_0",
    )
    trainer = Trainer(
        n_runners=4,
        algorithm=algo,
        llm_proxy=LLMProxy(port=12358),
        # Uncomment the following two lines if you want to rely on proxy-side trace data collection
        # Otherwise, the rollout runner will have an agentops tracer to collect the trace data,
        # and the adapter will be a TracerTraceToTriplet that parses the trace data generated by this tracer
        # adapter=LlmProxyTraceToTriplet(),
        # tracer=OtelTracer(),
    )
    trainer.fit(math_agent, load_math_dataset())
