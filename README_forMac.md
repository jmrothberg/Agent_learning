# Running and Testing Multi-Model Configurations on Apple Silicon (Mac Studio)

This guide walks you through configuring, running, and testing the multi-agent coding engine on high-end macOS hardware like a **512 GB Mac Studio**.

Apple Silicon uses a **Unified Memory Architecture** where all system RAM is dynamically shared as VRAM. Because of this, loading multiple large models simultaneously is incredibly fast and cheap—but since all tasks share a single unified GPU queue, we want to maximize efficiency by preventing double-loading identical models.

---

## 1. Staging Roles with Single-Model Inheritance (VRAM Efficiency)

Since you are running on a single physical SOC (one GPU queue), you can get the full multi-agent experience (Coder + Visual PlayTester + Architect) using a **single large model** (like Qwen 3.6 27B) and letting the agent loop handle different roles out-of-band.

To prevent having to load multiple copies or specify model names repeatedly, you can stage secondary and tertiary roles **using Model 1 inheritance**:

```bash
# 1. Stage your main model on Model 1 (this is the baseline Coder)
/model qwen3.6:27b

# 2. Stage Model 2 as the Visual Critic by inheriting Model 1 directly (just pass the role!)
/model2 --role critic

# 3. Stage Model 3 as the Architect by inheriting Model 1 directly
/model3 --role architect
```

### Why this is incredibly optimized:
- **Zero Redundant Loading:** The codebase will detect that the model names and backends match. It will **reuse the exact same Backend memory instance** across all three slots, skipping any secondary loading pauses or cold starts.
- **Auto-Balanced Roles:** If you configure Model 2 as `critic`, the system automatically defaults Model 3 to `architect` (and vice-versa) if you omit the role flags!

---

## 2. Viewing the Active Multi-Model Stack

To verify your configuration is staged and running correctly:

### A. The `/list` Command
Run `/list` inside the TUI to see where your staged and active roles live:
```
O * qwen3.6:27b-q8_0  [VLM]  ← active (coder) ← staged (coder)  ← staged (model2: critic)  ← staged (model3: architect)
```

### B. The `/status` Command
Run `/status` to print the full, structured multi-model hierarchy:
```
── status ──
  backend (active):     OLLAMA
  model (active):       qwen3.6:27b-q8_0
  model2 (active):      qwen3.6:27b (critic)
  model3 (active):      qwen3.6:27b (architect)
```

### C. Live Status Panel
The right-hand status panel scroll-pane in `chat.py` will render distinct, parallel cards showing the real-time activity of your Coder, Architect, and Critic concurrently.

---

## 3. How to Test & Verify

1. Launch the TUI on your Mac:
   ```bash
   .venv/bin/python chat.py
   ```
2. Run `/list` to ensure your models are registered.
3. Type the inheritance commands:
   ```bash
   /model <number_for_qwen_27b_vlm>
   /model2 --role critic
   /model3 --role architect
   ```
4. Verify the staged roles are correct via `/status`.
5. Run `/new build a platformer` to start the game development loop.
6. Verify in the logs that:
   - **Model 2 (Architect)** coordinates high-level planning.
   - **Model 1 (Coder)** writes raw code.
   - **Model 3 (Critic)** playtests screenshots out-of-band and feeds back visual bugs.
