# DeepResearch 🔬

**An evolution of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — smarter, persistent, parallel.**

> Same core loop (modify → execute → evaluate → keep/revert), but with a brain.

## What's Different from Autoresearch?

| Feature | Autoresearch | DeepResearch |
|---|---|---|
| Search strategy | Greedy hill-climbing | Bayesian bandit + simulated annealing |
| Branches | Single best | Population of K competing branches |
| Memory | None (per-session only) | Persistent knowledge base across sessions |
| Agents | Single sequential | Multi-agent parallel (when supported) |
| Reports | Manual analysis | Auto-generated research reports |
| Domain | ML training only | Universal (ML, code, prompts, games, docs) |
| Plateau handling | Manual intervention | Auto-reheat + guided random restart |
| Quality control | None | Periodic ablation analysis |

## Quick Start

```bash
# 1. Initialize for your domain
bash init.sh --domain ml     # or: code, prompt, game, doc, custom

# 2. Edit your config
vim .deepresearch/config.json   # Set target_files, metric, etc.

# 3. Create your research goals
cp templates/research.md .      # Edit with your priorities

# 4. Point your AI agent here
# In Claude Code:
#   "Read SKILL.md and start deepresearch"
```

## Files

```
SKILL.md           — The full skill specification (agent reads this)
strategy.py        — Strategy engine (bandit, annealing, population mgmt)
init.sh            — Project initializer
templates/
  research.md      — Template for human-written research goals
```

## How It Works

1. **You** write `research.md` — what to optimize, what constraints apply
2. **The agent** reads `SKILL.md` and runs autonomously
3. **Strategy Engine** picks what to try next (Thompson sampling over mutation categories)
4. **Simulated Annealing** lets the agent accept worse results early to escape local optima
5. **Population Search** maintains multiple competing approaches
6. **Knowledge Base** remembers what works across sessions — no repeat failures
7. **Auto Reports** generate research papers from experiment logs

## Architecture

```
Human Layer          research.md (goals + constraints)
      ↓
Strategy Engine      Bandit selector → Temperature scheduler → Population manager
      ↓
Experiment Loop      Mutate → Execute → Score → Keep/Revert → Log
      ↓
Memory + Reports     .deepresearch/ knowledge base → Auto-generated reports
      ↑______________|  (feedback loop)
```

## License

MIT
