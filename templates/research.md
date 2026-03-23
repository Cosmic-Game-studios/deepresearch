# research.md — DeepResearch Session Instructions
# This is YOUR file. Edit it to steer the research direction.
# The agent reads this at session start and follows it autonomously.

## Goal
Minimize val_bpb (validation bits per byte) for a 5-minute training run.
# ↑ Replace with your actual goal

## Target
- `train.py` — the single file the agent modifies
# ↑ Replace with your actual target file(s)

## Constraints
- TIME_BUDGET = 300 seconds (fixed, non-negotiable)
- Peak VRAM must stay under 80GB
- Do NOT modify prepare.py or this file
- Do NOT change the evaluation metric or harness
# ↑ Add your domain-specific constraints

## Strategy Guidance
# Tell the agent what to prioritize. Examples:

### High Priority (try first)
- Architecture changes (depth, width, attention patterns)
- Optimizer tuning (learning rate, schedule shape)

### Medium Priority
- Regularization experiments
- Data processing changes

### Low Priority (try later if stuck)
- Radical architecture changes (non-transformer)
- Completely different optimizer families

### Do NOT Try
- Anything that changes the tokenizer
- Multi-GPU approaches (single GPU only)
# ↑ This section prevents the agent from wasting time on dead ends

## Session Config
- **Population size:** 3 branches
- **Temperature:** moderate (start 0.5, decay 0.95)
- **Max experiments:** unlimited (run overnight)
- **Report frequency:** every 25 experiments

## Notes
# Free-form notes for the agent. Context it should know.
# Example: "Last session found that depth=10 is optimal. Focus on width now."
# Example: "The dataset has high entropy. Consider narrower vocab."
