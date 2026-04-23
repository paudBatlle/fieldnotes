Projects (proposed by Claude)
0. Full ML pipeline


1. Quantization from first principles — on MLX
Compression · Scholar-leaning
Don't use a library that does it for you. Implement INT8 quantization by hand — compute scale factors, zero points, simulate quantized inference in PyTorch or MLX, then measure the accuracy/latency tradeoff on a small transformer. Then compare your implementation against Apple's MLX built-in quantization on the same model.
The non-obvious part: implement both post-training quantization and a minimal quantization-aware training loop and compare them empirically. Most tutorials do one or the other. Doing both and explaining why QAT recovers accuracy that PTQ loses is the actual insight.
Deliverable: a notebook with your implementation, benchmark plots, and a written explanation of where quantization error actually comes from mathematically.

2. Build a inference server that respects latency SLOs
Inference serving · Engineer-leaning
Take a small LLM (something like SmolLM or Qwen 0.5B that runs on M4) and build a minimal inference server from scratch — not FastAPI slapped on top of a model. Implement continuous batching manually, track time-to-first-token and throughput separately, and add a simple SLO enforcement mechanism that rejects or queues requests when latency targets are at risk.
The non-obvious part: most people optimize for throughput. Optimizing for latency percentiles under load (p95, p99) is a different problem entirely and much closer to what production actually cares about.
Deliverable: a working server with a load testing script and a dashboard showing latency distributions under increasing load.

3. Mixed precision training on Metal
Training infra · Engineer + Scholar
Apple's MLX supports float16 and bfloat16 on the M4 unified memory architecture, but the memory model is fundamentally different from CUDA — no separate GPU VRAM, which changes everything about how you think about memory pressure. Train a small model (a few layers, toy dataset) in full precision, then implement a manual mixed precision training loop — forward in fp16, loss scaling, backward in fp32 — and measure the memory and speed tradeoffs.
The non-obvious part: implement loss scaling from scratch rather than using GradScaler. Understanding why loss scaling exists (underflow in fp16 gradients) and building the dynamic scaling logic yourself is the whole lesson.
Deliverable: a clean training loop implementation with ablations showing what breaks without loss scaling, and memory/speed benchmarks comparing precision modes.

4. A distributed RL training system in miniature
RL infra · Engineer-leaning
Implement a small Ape-X style distributed RL system using Python multiprocessing — separate actor processes that collect experience, a shared prioritized replay buffer, and a learner process that pulls batches and pushes updated weights back. Use a simple environment (CartPole or LunarLander) so the RL itself isn't the hard part. The hard part is the distributed coordination.
The non-obvious part: implement the prioritized experience replay buffer as a sum-tree from scratch. It's a data structure problem disguised as an RL problem, and understanding why a naive replay buffer is a bottleneck at scale is the real insight.
Deliverable: a working multi-process training system with a real-time plot of actor throughput, buffer size, and learner updates — showing the system actually scaling with more actors.

5. Sim-to-real pipeline without real hardware
Robotics infra · All three personalities
This is the most ambitious one. Use MuJoCo or PyBullet to train a policy (PPO or SAC) on a simple locomotion or manipulation task, then implement the domain randomization pipeline that makes sim-to-real transfer possible: randomize physics parameters (friction, mass, damping) during training, and measure how the policy degrades as you increase the sim-to-real gap systematically.
The non-obvious part: implement Automatic Domain Randomization (ADR) — the idea from OpenAI's Dexterous Hand work where the randomization range itself is adapted based on policy performance. You don't need the full system, just the core feedback loop. It's a meta-learning idea hiding inside an infra problem.
Deliverable: a training pipeline with domain randomization, an evaluation framework that measures robustness to parameter shifts, and a visualization showing the policy's performance envelope across the randomization space.

6. CoreML model compilation pipeline
Compression + Serving · Engineer-leaning
Build a pipeline that takes a PyTorch model, converts it to CoreML, applies CoreML's built-in quantization and palettization, and benchmarks it against the original on M4 using the Neural Engine. The twist: instrument the pipeline to catch the silent failures — ops that fall back to CPU, layers that lose too much accuracy, numerical mismatches between PyTorch and CoreML outputs.
The non-obvious part: CoreML conversion fails silently more often than it fails loudly. Building a validation layer that automatically checks numerical equivalence and flags suspicious layers teaches you more about model portability than any tutorial.
Deliverable: a reusable conversion + validation pipeline that you could actually use on future projects, with a report on what breaks and why for a few different model architectures.

The natural order if you want to build progressively: start with 1 (quantization from scratch) to understand the theory, then 3 (mixed precision training) to see it in a training context, then 2 (inference server) to see it in a serving context, then 6 (CoreML pipeline) to tie it all together on Apple Silicon specifically. The RL infra (4) and robotics (5) are more self-contained and can slot in anywhere.
The M4 constraint actually makes projects 2, 3, and 6 more interesting than their CUDA equivalents — fewer people have done them, and the unified memory architecture raises genuinely different questions.