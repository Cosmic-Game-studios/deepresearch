#!/bin/bash
# DeepResearch — Universal Project Initializer
#
# Usage:
#   bash init.sh                              # Interactive
#   bash init.sh --level 1 --domain ml        # Level 1, ML training
#   bash init.sh --level 3 --domain web_api --spec "Build a REST API for tasks"
#
# Levels:
#   1   — Parameter tuning (classic autoresearch)
#   2   — Generative mutations (add/remove/replace code)
#   3   — Autonomous engineer (spec → research → build → optimize)
#
# Domains: ml, code, prompt, game, web_api, library, optimization, custom

set -e

# Parse args
LEVEL=1
DOMAIN="custom"
SPEC=""
TARGET=""
METRIC=""
BUDGET=300

while [[ $# -gt 0 ]]; do
  case $1 in
    --level) LEVEL="$2"; shift 2;;
    --domain) DOMAIN="$2"; shift 2;;
    --spec) SPEC="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    --metric) METRIC="$2"; shift 2;;
    --budget) BUDGET="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "╔══════════════════════════════════════════════╗"
echo "║  DeepResearch — Level $LEVEL Initialization          ║"
echo "║  Domain: $DOMAIN                                 "
echo "╚══════════════════════════════════════════════╝"

# Create directory structure
mkdir -p .deepresearch/{memos,reports,research,backups,populations}

# Mutation levels based on Level
if [ "$LEVEL" -eq 1 ]; then
  MUT_LEVELS="[1]"
  MUT_CATS='["parametric"]'
elif [ "$LEVEL" -eq 2 ]; then
  MUT_LEVELS="[1, 2]"
  MUT_CATS='["parametric", "structural_addition", "structural_removal", "structural_replacement", "integration"]'
else
  MUT_LEVELS="[1, 2, 3]"
  MUT_CATS='["parametric", "structural_addition", "structural_removal", "structural_replacement", "integration", "architectural"]'
fi

# Domain-specific defaults
case $DOMAIN in
  ml)
    TARGET="${TARGET:-train.py}"
    METRIC="${METRIC:-val_loss}"
    DIRECTION="lower"
    TEST_CMD=""
    CURRICULUM_DOMAIN="ml_training"
    ;;
  web_api)
    TARGET="${TARGET:-src/}"
    METRIC="${METRIC:-p99_latency_ms}"
    DIRECTION="lower"
    TEST_CMD="pytest tests/ -q"
    CURRICULUM_DOMAIN="web_api"
    ;;
  game)
    TARGET="${TARGET:-src/}"
    METRIC="${METRIC:-ai_vs_random_winrate}"
    DIRECTION="higher"
    TEST_CMD="pytest tests/ -q"
    CURRICULUM_DOMAIN="game"
    ;;
  library)
    TARGET="${TARGET:-src/}"
    METRIC="${METRIC:-benchmark_ops_sec}"
    DIRECTION="higher"
    TEST_CMD="pytest tests/ -q"
    CURRICULUM_DOMAIN="library"
    ;;
  code|optimization)
    TARGET="${TARGET:-target.py}"
    METRIC="${METRIC:-primary_metric}"
    DIRECTION="lower"
    TEST_CMD=""
    CURRICULUM_DOMAIN="optimization"
    ;;
  prompt)
    TARGET="${TARGET:-prompt.txt}"
    METRIC="${METRIC:-score}"
    DIRECTION="higher"
    TEST_CMD=""
    CURRICULUM_DOMAIN="optimization"
    ;;
  *)
    TARGET="${TARGET:-src/}"
    METRIC="${METRIC:-primary_metric}"
    DIRECTION="higher"
    TEST_CMD=""
    CURRICULUM_DOMAIN="custom"
    ;;
esac

# Write config.json
cat > .deepresearch/config.json << CONFEOF
{
  "level": $LEVEL,
  "domain": "$DOMAIN",
  "target_files": ["$TARGET"],
  "read_only_files": ["tests/", "eval.sh"],
  "metric": "$METRIC",
  "metric_direction": "$DIRECTION",
  "budget_seconds": $BUDGET,
  "experiment_budget": 200,
  "mutation_levels": $MUT_LEVELS,
  "mutation_categories": $MUT_CATS,
  "test_command": "$TEST_CMD",
  "hard_constraints": [],
  "spec": "$SPEC"
}
CONFEOF

# Initialize strategy state
cat > .deepresearch/strategy-state.json << STATEEOF
{
  "temperature": 1.0,
  "total_experiments": 0,
  "bandit_arms": {},
  "population": [],
  "no_improvement_streak": 0,
  "best_metric": null,
  "baseline_metric": null
}
STATEEOF

# Initialize knowledge base
cat > .deepresearch/knowledge.json << KNOWEOF
{
  "patterns": [],
  "anti_patterns": [],
  "domain_insights": [],
  "cross_domain": []
}
KNOWEOF

# Initialize dependencies tracker
echo '{"dependencies":[]}' > .deepresearch/dependencies.json

# Initialize empty experiment log
touch .deepresearch/experiments.jsonl

echo ""
echo "✓ Config:     .deepresearch/config.json"
echo "✓ Strategy:   .deepresearch/strategy-state.json"
echo "✓ Knowledge:  .deepresearch/knowledge.json"
echo "✓ Log:        .deepresearch/experiments.jsonl"

# Level 2-3: Create curriculum
if [ "$LEVEL" -ge 2 ]; then
  echo ""
  echo "Setting up Level $LEVEL features..."
  python3 -c "
import sys; sys.path.insert(0, '.')
from engine.curriculum import CurriculumRunner
runner = CurriculumRunner.create_from_template('$CURRICULUM_DOMAIN')
print(f'✓ Curriculum: .deepresearch/curriculum.json ({len(runner.stages)} stages)')
for i, s in enumerate(runner.stages, 1):
    op = '≥' if s.get('direction','higher')=='higher' else '≤'
    print(f'    Stage {i}: {s[\"name\"]} — {s[\"metric\"]} {op} {s[\"target\"]}')
" 2>/dev/null || echo "  (curriculum requires engine/ modules — run from project root)"

  # Initialize knowledge acquisition with domain-specific search queries
  python3 -c "
import sys; sys.path.insert(0, '.')
from engine.knowledge import KnowledgeAcquisition
ka = KnowledgeAcquisition(domain='$DOMAIN', spec='$SPEC', language='python')
queries = ka.generate_searches()
print(f'✓ Knowledge:  Ready for domain research')
print(f'  Top search queries:')
for q in queries[:3]:
    print(f'    [{q[\"priority\"]:.2f}] {q[\"query\"]}')
print(f'  Run: python -m engine.level3 knowledge --domain $DOMAIN')
" 2>/dev/null || echo "  (knowledge requires engine/ modules)"
fi

# Level 3: Initialize orchestrator
if [ "$LEVEL" -ge 3 ]; then
  python3 -c "
import sys; sys.path.insert(0, '.')
from engine.autonomous import Orchestrator
orch = Orchestrator(spec='$SPEC')
orch.save_state()
print('✓ Orchestrator: .deepresearch/orchestrator_state.json')
print(f'  Spec: ${SPEC:-"(not set — pass --spec)"}'[:60])
print('  Phase: research → architect → bootstrap → build → test → optimize → report')
" 2>/dev/null || echo "  (orchestrator requires engine/ modules)"
fi

echo ""
echo "═══════════════════════════════════════════"
if [ "$LEVEL" -eq 1 ]; then
  echo "  Ready! Tell your agent:"
  echo "  'Read SKILL.md and start deepresearch'"
elif [ "$LEVEL" -eq 2 ]; then
  echo "  Ready! Tell your agent:"
  echo "  'Read SKILL.md, run: python -m engine.level3 next'"
else
  echo "  Ready! Tell your agent:"
  echo "  'Read SKILL.md, run: python -m engine.level3 next'"
  echo "  The orchestrator will guide it through all 7 phases."
fi
echo "═══════════════════════════════════════════"
