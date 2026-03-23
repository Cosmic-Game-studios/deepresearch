#!/usr/bin/env bash
# DeepResearch — Project Initializer
# Run this once in your project root to set up the .deepresearch directory.
# Usage: bash init.sh [--domain ml|code|prompt|game|doc]

set -e

DOMAIN="${2:-custom}"
DR_DIR=".deepresearch"

echo "🔬 Initializing DeepResearch..."

# Create directory structure
mkdir -p "$DR_DIR/populations/branch-0" "$DR_DIR/reports"

# Domain-specific defaults
case "$DOMAIN" in
  ml)
    METRIC="val_bpb"
    DIRECTION="lower"
    BUDGET=300
    POP_SIZE=1
    TEMP="moderate"
    CATEGORIES='["architecture","hyperparameters","optimizer","regularization","scheduling","efficiency"]'
    ;;
  code)
    METRIC="benchmark_ms"
    DIRECTION="lower"
    BUDGET=60
    POP_SIZE=3
    TEMP="moderate"
    CATEGORIES='["algorithm","memory","parallelism","io","language_features","architecture"]'
    ;;
  prompt)
    METRIC="judge_score"
    DIRECTION="higher"
    BUDGET=30
    POP_SIZE=3
    TEMP="aggressive"
    CATEGORIES='["structure","specificity","tone","examples","guardrails","persona"]'
    ;;
  game)
    METRIC="balance_score"
    DIRECTION="higher"
    BUDGET=120
    POP_SIZE=3
    TEMP="moderate"
    CATEGORIES='["economy","combat","progression","map_balance","ai_behavior"]'
    ;;
  doc)
    METRIC="quality_score"
    DIRECTION="higher"
    BUDGET=15
    POP_SIZE=3
    TEMP="aggressive"
    CATEGORIES='["structure","clarity","completeness","conciseness","formatting"]'
    ;;
  *)
    METRIC="score"
    DIRECTION="higher"
    BUDGET=60
    POP_SIZE=3
    TEMP="moderate"
    CATEGORIES='["category_a","category_b","category_c"]'
    echo "⚠  Custom domain — edit .deepresearch/config.json with your specifics."
    ;;
esac

# Write config
cat > "$DR_DIR/config.json" << EOF
{
  "target_files": [],
  "metric": "$METRIC",
  "metric_direction": "$DIRECTION",
  "budget_seconds": $BUDGET,
  "population_size": $POP_SIZE,
  "temperature_schedule": "$TEMP",
  "mutation_categories": $CATEGORIES,
  "domain": "$DOMAIN",
  "created": "$(date -Iseconds)",
  "session_count": 0
}
EOF

# Initialize strategy state
cat > "$DR_DIR/strategy-state.json" << EOF
{
  "temperature": 0.5,
  "total_experiments": 0,
  "bandit_arms": {},
  "population": [],
  "best_metric": null,
  "baseline_metric": null,
  "metric_direction": "$DIRECTION"
}
EOF

# Initialize knowledge base
if [ ! -f "$DR_DIR/knowledge.json" ]; then
  echo '{"patterns":[],"anti_patterns":[],"domain_insights":[],"cross_domain":[]}' > "$DR_DIR/knowledge.json"
  echo "   Created fresh knowledge base"
else
  echo "   Existing knowledge base preserved ✓"
fi

# Initialize experiment log
touch "$DR_DIR/experiments.jsonl"

# Add to .gitignore if not already there
if [ -f .gitignore ]; then
  if ! grep -q ".deepresearch/" .gitignore 2>/dev/null; then
    echo ".deepresearch/" >> .gitignore
    echo "   Added .deepresearch/ to .gitignore"
  fi
else
  echo ".deepresearch/" > .gitignore
  echo "   Created .gitignore with .deepresearch/"
fi

echo ""
echo "✅ DeepResearch initialized!"
echo "   Directory: $DR_DIR/"
echo "   Domain:    $DOMAIN"
echo "   Metric:    $METRIC ($DIRECTION is better)"
echo "   Budget:    ${BUDGET}s per experiment"
echo "   Population: $POP_SIZE branches"
echo "   Temperature: $TEMP"
echo ""
echo "Next steps:"
echo "  1. Edit $DR_DIR/config.json — set your target_files"
echo "  2. Create research.md — your research goals (see templates/)"
echo "  3. Create an eval harness ($DR_DIR/eval.sh or eval.py)"
echo "  4. Tell your agent: 'Read SKILL.md and start deepresearch'"
