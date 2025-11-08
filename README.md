# LLMSFTComBenchmarking
**Benchmarking Parallelism Strategies for LLM Fine-Tuning on Jean Zay Supercomputer.**

A comprehensive benchmarking study conducted on the Jean Zay supercomputer to evaluate and compare the most efficient parallelism strategies for Instruct Fine-Tuning of large language models (LLMs) of various scales (14B, 32B, and 72B parameters).

The experiments focus on how communication bandwidth impacts performance across different distributed training techniques (Data, Tensor, Pipeline, and Sequence parallelism).
The goal is to identify optimal configurations under varying interconnect bandwidth conditions to maximize throughput, scalability, and training stability.
