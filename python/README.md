# Bitty (Python)

The Python side of this repository is **Bitty**, an autonomous software-development agent scaffold. It orchestrates a team of LLM-backed agents through a fixed development cycle and persists knowledge between cycles.

This is a scaffold: agents are thin wrappers, and the actual inference is delegated to an external LLM (see [Local AI support](#local-ai-support)).

## Core Components

| Agent | Responsibility |
|---|---|
| `PlannerAgent` | Decomposes a goal into prioritized milestones |
| `ArchitectAgent` | Systems design with tradeoff analysis and technical-debt management |
| `DeveloperAgent` | Code generation, refactoring automation |
| `TesterAgent` | Test-case generation, bug detection, regression prevention |
| `OptimizerAgent` | Performance profiling and memory optimization |
| `MemorySystem` | Pickle-backed knowledge storage shared across cycles |

## Communication Flow

`Bitty.run_development_cycle(goal)` runs the pipeline sequentially:

```
Planner ──> tasks ──> Architect ──> design ──> Developer ──> implementation
                                                              │
    Optimizer <── optimized <── test_results <── Tester <─────┘
        │
        └──> MemorySystem.store_cycle(...)
```

`run_continuous(goals)` loops this cycle over a list of goals, recording errors to memory.

## Framework

- **Python**: orchestration (asyncio, one `Bitty` orchestrator)
- **Rust**: the BitLLM inference engine (see the workspace `crates/`)
- Agents are pure Python; no REST/JWT bridge between the two is implemented yet.

## Local AI Support

- Local LLMs via Ollama (e.g. `llama3.1`, `qwen2.5`)
- Quantized models via llama.cpp variants
- Runtime model switching

## Hardware Awareness

- GPU support: CUDA / ROCm
- Quantization: 1-bit ternary weights via the BitLLM engine
- Memory management: arena allocation (planned)

## Modular Design

Vertical separation of concerns with horizontal integration points.

## Future-Proofing

- Model-agnostic API layer
- Distributed training support
- Hardware abstraction layer
