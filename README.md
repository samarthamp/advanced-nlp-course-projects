# Advanced NLP Systems Implementation

This repository contains the implementation of advanced Natural Language Processing architectures and optimization techniques, built from scratch. The projects focus on understanding the internal mechanics of Transformers, LLM efficiency through quantization, and scaling via Mixture-of-Experts (MoE).

## Assignment 1: Transformer from Scratch (Machine Translation)
**Goal:** Build a complete Encoder-Decoder Transformer for Finnish-to-English translation without using pre-built PyTorch Transformer modules.

* **Core Architecture:** Implemented the full Transformer architecture (Encoder, Decoder, Multi-Head Attention) entirely from scratch.
* **Positional Encodings:** Implemented and compared two strategies: Rotary Positional Embeddings (RoPE) and Relative Position Bias.
* **Decoding Algorithms:** Implemented custom inference strategies including Greedy Decoding, Beam Search, and Top-k Sampling to analyze translation quality.
* **Training Loop:** Developed a custom training loop with teacher forcing to train the model on parallel corpora.

## Assignment 2: Large Model Fine-tuning & Quantization
**Goal:** Optimize Large Language Models (LLMs) for deployment by analyzing trade-offs between precision, memory footprint, and inference latency.

* **Full Fine-Tuning:** Fine-tuned a GPT-2 Large model on the AG News dataset to establish a high-precision performance baseline.
* **Quantization from Scratch:** Implemented linear Post-Training Quantization (PTQ) algorithms (FP32 to INT8) manually to understand the mapping of floating-point values to integer ranges.
* **Library-Based Optimization:** Leveraged `bitsandbytes` to implement 8-bit and 4-bit (NF4) quantization, comparing the results against the scratch implementation.
* **Performance Analysis:** Conducted a rigorous analysis of memory usage (MB/GB), inference latency (ms), and accuracy metrics across different precision levels.

## Assignment 3: Mixture-of-Experts (MoE) & Scaling
**Goal:** Implement a sparse MoE layer to replace dense Feed-Forward Networks, enabling scalable modeling with conditional computation.

* **Sparse MoE Layer:** Built a Mixture-of-Experts layer from scratch, where different inputs are routed to specific "expert" networks.
* **Routing Algorithms:** Implemented Hash Routing and Token-Choice Top-k Routing to manage expert selection dynamically.
* **Load Balancing:** Designed a custom load balancer to prevent expert collapse and ensure efficient distribution of tokens during training.
* **Summarization Task:** Trained the MoE model on the XSum dataset and compared performance against dense baselines (BART, T5, Llama-3) using ROUGE and BERTScore.

---