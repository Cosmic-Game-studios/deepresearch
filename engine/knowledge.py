"""
DeepResearch Knowledge Acquisition — External Domain Knowledge

The missing bridge between "the agent knows what's in the code" and
"the agent knows what's possible in this domain."

Before experimenting, a good researcher reads:
  - Documentation of libraries they're using
  - Papers/articles about the problem domain
  - Source code of existing solutions
  - Community discussions about common pitfalls

This module structures HOW the agent acquires external knowledge
and integrates it into the Reasoning Layer (R1 Deep Read, R2 Hypothesis).

Components:
  1. SearchStrategy    — Generates targeted search queries per domain
  2. SourceManager     — Tracks what's been read, what's pending, priorities
  3. KnowledgeExtractor — Structures raw reading into actionable insights
  4. TechniqueLibrary  — Builds a domain-specific technique list FROM research
  5. KnowledgeIntegration — Connects knowledge to experiment hypotheses

Usage:
    from engine.knowledge import KnowledgeAcquisition

    ka = KnowledgeAcquisition(domain="web_api", spec="Optimize REST API")
    
    # Phase 1: Generate search queries
    queries = ka.generate_searches()
    # → ["REST API performance optimization techniques",
    #    "python web framework benchmarks 2026", ...]
    
    # Phase 2: Agent reads sources, registers findings
    ka.register_source(url="https://...", title="FastAPI Performance Guide",
                       source_type="documentation")
    ka.extract_technique(source_url="https://...",
                        name="Connection pooling",
                        description="Reuse DB connections across requests",
                        expected_impact="30-50% latency reduction",
                        complexity="moderate",
                        evidence="Benchmarks show 3x throughput improvement")
    
    # Phase 3: Get prioritized technique list for experiments
    techniques = ka.prioritized_techniques()
    # → [{"name": "Connection pooling", "priority": 0.9, "reason": "..."}, ...]
    
    # Phase 4: Generate hypothesis from knowledge
    hypothesis = ka.generate_hypothesis_context(technique="connection_pooling",
                                                  current_bottleneck="DB latency")
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

DR_DIR = Path(".deepresearch")


# ════════════════════════════════════════════════════════════
# 1. SEARCH STRATEGY — What to search for
# ════════════════════════════════════════════════════════════

class SearchStrategy:
    """
    Generates targeted search queries based on domain and current bottleneck.
    
    The agent uses these queries with web_search to find relevant sources.
    Queries are ordered by priority: most likely to help first.
    """

    # Universal search patterns (domain-agnostic)
    UNIVERSAL_PATTERNS = [
        "{domain} best practices 2026",
        "{domain} performance optimization techniques",
        "{domain} common mistakes pitfalls",
        "{domain} architecture patterns",
        "{spec_keywords} tutorial guide",
        "{spec_keywords} benchmark comparison",
        "awesome {domain} github list",
    ]

    # Domain-specific search patterns
    DOMAIN_PATTERNS = {
        "web_api": [
            "{language} web framework performance comparison",
            "REST API latency optimization",
            "database connection pooling {language}",
            "API caching strategies",
            "load testing {language} web application",
            "async vs sync {language} web performance",
            "{language} profiling web requests",
        ],
        "ml_training": [
            "training speed optimization {framework}",
            "learning rate schedule comparison",
            "mixed precision training guide",
            "data loading bottleneck {framework}",
            "model architecture efficiency {task}",
            "gradient accumulation vs large batch",
            "training loss plateau solutions",
        ],
        "cli_tool": [
            "{language} CLI framework comparison",
            "command line argument parsing best practices",
            "streaming file processing {language}",
            "{language} memory efficient file reading",
            "CLI error handling UX patterns",
            "unix philosophy CLI design",
        ],
        "game": [
            "game balance algorithm design",
            "game AI decision making",
            "procedural generation techniques",
            "game economy modeling",
            "player engagement metrics game design",
            "ELO rating system implementation",
        ],
        "library": [
            "{language} library API design best practices",
            "{language} package performance optimization",
            "backward compatibility API versioning",
            "property-based testing {language}",
            "{language} documentation generation",
        ],
        "data_pipeline": [
            "ETL pipeline optimization",
            "streaming data processing patterns",
            "data validation frameworks {language}",
            "batch vs stream processing tradeoffs",
            "idempotent data pipeline design",
        ],
        "optimization": [
            "profiling {language} application bottleneck",
            "algorithmic complexity optimization",
            "memory optimization {language}",
            "CPU cache friendly programming",
            "zero-copy techniques {language}",
        ],
    }

    # Bottleneck-specific search patterns
    BOTTLENECK_PATTERNS = {
        "latency": [
            "reducing {component} latency",
            "{component} response time optimization",
            "p99 latency reduction techniques",
        ],
        "throughput": [
            "increasing {component} throughput",
            "concurrent request handling {language}",
            "batch processing optimization",
        ],
        "memory": [
            "{language} memory profiling",
            "reducing memory usage {component}",
            "memory leak detection {language}",
            "streaming processing vs loading into memory",
        ],
        "accuracy": [
            "improving {component} accuracy",
            "{domain} evaluation metrics",
            "error analysis techniques",
        ],
        "reliability": [
            "fault tolerance patterns",
            "retry strategy design",
            "circuit breaker implementation {language}",
            "graceful degradation patterns",
        ],
        "scalability": [
            "horizontal scaling {component}",
            "distributed system design patterns",
            "sharding strategies",
            "load balancing algorithms",
        ],
    }

    @classmethod
    def generate(cls, domain: str, spec: str = "",
                 language: str = "python", bottleneck: str = None,
                 component: str = None, framework: str = None,
                 task: str = None) -> list:
        """
        Generate prioritized search queries.
        
        Returns list of {"query": str, "priority": float, "category": str}
        """
        queries = []
        spec_keywords = " ".join(spec.split()[:5]) if spec else domain

        # Fill template variables
        vars = {
            "domain": domain, "spec_keywords": spec_keywords,
            "language": language, "component": component or domain,
            "framework": framework or language, "task": task or domain,
        }

        def fill(template):
            result = template
            for k, v in vars.items():
                result = result.replace("{" + k + "}", v or "")
            return result.strip()

        # Universal (always include)
        for i, pattern in enumerate(cls.UNIVERSAL_PATTERNS):
            queries.append({
                "query": fill(pattern),
                "priority": 0.8 - i * 0.05,
                "category": "universal",
            })

        # Domain-specific
        domain_patterns = cls.DOMAIN_PATTERNS.get(domain, [])
        for i, pattern in enumerate(domain_patterns):
            queries.append({
                "query": fill(pattern),
                "priority": 0.9 - i * 0.05,
                "category": "domain",
            })

        # Bottleneck-specific (highest priority if known)
        if bottleneck:
            bn_patterns = cls.BOTTLENECK_PATTERNS.get(bottleneck, [])
            for i, pattern in enumerate(bn_patterns):
                queries.append({
                    "query": fill(pattern),
                    "priority": 0.95 - i * 0.03,
                    "category": "bottleneck",
                })

        # Sort by priority, deduplicate
        seen = set()
        unique = []
        for q in sorted(queries, key=lambda x: -x["priority"]):
            if q["query"] not in seen:
                seen.add(q["query"])
                unique.append(q)

        return unique


# ════════════════════════════════════════════════════════════
# 2. SOURCE MANAGER — Track what's been read
# ════════════════════════════════════════════════════════════

@dataclass
class Source:
    """An external source of domain knowledge."""
    url: str
    title: str
    source_type: str  # documentation, paper, article, code, discussion, tutorial
    domain_relevance: float = 0.5  # 0-1, how relevant to our problem
    read: bool = False
    summary: str = ""
    key_insights: list = field(default_factory=list)
    techniques_found: list = field(default_factory=list)  # names of extracted techniques
    added_at: str = field(default_factory=lambda: datetime.now().isoformat())
    read_at: str = ""


class SourceManager:
    """Manages the reading list and tracks what's been read."""

    def __init__(self):
        self.path = DR_DIR / "research" / "sources.json"
        self.sources = []
        self.load()

    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.sources = [Source(**s) for s in data.get("sources", [])]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"sources": [asdict(s) for s in self.sources],
                "total": len(self.sources),
                "read": sum(1 for s in self.sources if s.read),
                "last_updated": datetime.now().isoformat()}
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, url: str, title: str, source_type: str,
            domain_relevance: float = 0.5) -> Source:
        """Add a source to the reading list."""
        # Deduplicate by URL
        for s in self.sources:
            if s.url == url:
                return s
        source = Source(url=url, title=title, source_type=source_type,
                       domain_relevance=domain_relevance)
        self.sources.append(source)
        self.save()
        return source

    def mark_read(self, url: str, summary: str, key_insights: list = None):
        """Mark a source as read with findings."""
        for s in self.sources:
            if s.url == url:
                s.read = True
                s.summary = summary
                s.key_insights = key_insights or []
                s.read_at = datetime.now().isoformat()
                break
        self.save()

    def unread(self) -> list:
        """Get prioritized list of unread sources."""
        return sorted([s for s in self.sources if not s.read],
                      key=lambda s: -s.domain_relevance)

    def all_insights(self) -> list:
        """Get all insights from all read sources."""
        insights = []
        for s in self.sources:
            if s.read:
                for insight in s.key_insights:
                    insights.append({"insight": insight, "source": s.title,
                                     "url": s.url, "type": s.source_type})
        return insights

    def reading_progress(self) -> str:
        total = len(self.sources)
        read = sum(1 for s in self.sources if s.read)
        insights = sum(len(s.key_insights) for s in self.sources if s.read)
        techniques = sum(len(s.techniques_found) for s in self.sources if s.read)
        return (f"Sources: {read}/{total} read | "
                f"Insights: {insights} | Techniques: {techniques}")


# ════════════════════════════════════════════════════════════
# 3. KNOWLEDGE EXTRACTOR — Structure raw reading into insights
# ════════════════════════════════════════════════════════════

@dataclass
class Technique:
    """A specific technique extracted from external knowledge."""
    name: str
    description: str
    source_url: str
    source_title: str = ""
    expected_impact: str = ""  # "30-50% latency reduction"
    complexity: str = "moderate"  # trivial, simple, moderate, complex, major
    prerequisites: list = field(default_factory=list)  # other techniques needed first
    evidence: str = ""  # what supports this (benchmarks, case studies, theory)
    applicable_when: str = ""  # "when DB calls are the bottleneck"
    not_applicable_when: str = ""  # "when CPU is the bottleneck"
    mutation_type: str = "structural_addition"  # what kind of mutation this requires
    estimated_experiments: int = 3  # how many experiments to implement + tune
    tried: bool = False
    result: str = ""  # "worked: +40% improvement" or "failed: incompatible with async"
    priority: float = 0.5  # computed from evidence + expected impact + complexity


class TechniqueLibrary:
    """
    Builds a domain-specific technique library FROM research.
    
    Unlike a pre-built feature library (which is domain-specific and static),
    this library is built dynamically by the agent as it reads external sources.
    The agent discovers what techniques exist, evaluates their relevance,
    and adds them to the library for future experiments.
    """

    def __init__(self):
        self.path = DR_DIR / "research" / "techniques.json"
        self.techniques = []
        self.load()

    def load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.techniques = [Technique(**t) for t in data.get("techniques", [])]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "techniques": [asdict(t) for t in self.techniques],
            "total": len(self.techniques),
            "tried": sum(1 for t in self.techniques if t.tried),
            "last_updated": datetime.now().isoformat(),
        }
        self.path.write_text(json.dumps(data, indent=2))

    def add(self, name: str, description: str, source_url: str, **kwargs) -> Technique:
        """Add a technique discovered from reading."""
        # Deduplicate by name
        for t in self.techniques:
            if t.name.lower() == name.lower():
                # Update with new info if provided
                for k, v in kwargs.items():
                    if v and hasattr(t, k):
                        setattr(t, k, v)
                self.save()
                return t
        tech = Technique(name=name, description=description,
                        source_url=source_url, **kwargs)
        # Auto-compute priority
        tech.priority = self._compute_priority(tech)
        self.techniques.append(tech)
        self.save()
        return tech

    def _compute_priority(self, t: Technique) -> float:
        """Compute priority from multiple factors."""
        score = 0.5

        # Evidence strength
        if t.evidence:
            if "benchmark" in t.evidence.lower() or "measured" in t.evidence.lower():
                score += 0.2  # empirical evidence
            elif "paper" in t.evidence.lower() or "study" in t.evidence.lower():
                score += 0.15  # academic evidence
            else:
                score += 0.05  # anecdotal

        # Expected impact
        impact_keywords = {"2x": 0.15, "3x": 0.2, "50%": 0.15, "30%": 0.1,
                          "10x": 0.25, "significant": 0.1, "major": 0.12,
                          "minor": 0.02, "small": 0.02}
        for kw, bonus in impact_keywords.items():
            if kw in t.expected_impact.lower():
                score += bonus
                break

        # Complexity penalty (simpler = try first)
        complexity_penalty = {"trivial": 0, "simple": 0.02, "moderate": 0.05,
                             "complex": 0.1, "major": 0.2}
        score -= complexity_penalty.get(t.complexity, 0.05)

        # Prerequisites penalty
        score -= len(t.prerequisites) * 0.05

        return min(1.0, max(0.0, score))

    def mark_tried(self, name: str, result: str):
        """Record the result of trying a technique."""
        for t in self.techniques:
            if t.name.lower() == name.lower():
                t.tried = True
                t.result = result
                break
        self.save()

    def untried(self) -> list:
        """Get prioritized list of untried techniques."""
        return sorted([t for t in self.techniques if not t.tried],
                      key=lambda t: -t.priority)

    def successful(self) -> list:
        """Get techniques that worked."""
        return [t for t in self.techniques if t.tried and "worked" in t.result.lower()]

    def failed(self) -> list:
        """Get techniques that didn't work."""
        return [t for t in self.techniques if t.tried and "failed" in t.result.lower()]


# ════════════════════════════════════════════════════════════
# 4. KNOWLEDGE INTEGRATION — Connect knowledge to experiments
# ════════════════════════════════════════════════════════════

class KnowledgeIntegration:
    """
    Connects acquired knowledge to the Reasoning Layer.
    
    This is the bridge between "I read about connection pooling" and
    "My hypothesis for experiment #12 is to add connection pooling because
    the domain research showed it reduces latency by 30-50% when DB calls
    are the bottleneck, and my Deep Read (R1) confirms DB calls are our
    bottleneck (142ms of 200ms total request time)."
    """

    def __init__(self, techniques: TechniqueLibrary, sources: SourceManager):
        self.techniques = techniques
        self.sources = sources

    def suggest_next_technique(self, current_bottleneck: str = "",
                                failed_approaches: list = None) -> Optional[Technique]:
        """
        Suggest the best technique to try next, given:
        - The current bottleneck (from R1 Deep Read)
        - What we've already tried and failed
        """
        failed = set(f.lower() for f in (failed_approaches or []))
        candidates = []

        for t in self.techniques.untried():
            # Skip if we already failed with this
            if t.name.lower() in failed:
                continue
            # Boost if applicable to current bottleneck
            relevance = t.priority
            if current_bottleneck and t.applicable_when:
                if current_bottleneck.lower() in t.applicable_when.lower():
                    relevance += 0.2
            if current_bottleneck and t.not_applicable_when:
                if current_bottleneck.lower() in t.not_applicable_when.lower():
                    relevance -= 0.3
            candidates.append((t, relevance))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def generate_hypothesis_context(self, technique_name: str,
                                     current_bottleneck: str = "") -> str:
        """
        Generate the knowledge-backed context for a hypothesis.
        
        This feeds into R2 (Causal Hypothesis). Instead of:
          "I'll try adding caching"
        The agent writes:
          "Based on the FastAPI Performance Guide (source #3), connection pooling
           reduces p99 latency by 30-50% when DB connections are the bottleneck.
           My Deep Read confirms DB connections consume 142ms of our 200ms p99.
           I predict this technique will bring p99 below 100ms."
        """
        tech = None
        for t in self.techniques.techniques:
            if t.name.lower() == technique_name.lower():
                tech = t
                break

        if not tech:
            return f"No knowledge found for technique '{technique_name}'."

        lines = [f"## Knowledge-backed hypothesis context for: {tech.name}",
                 f"",
                 f"**Source:** {tech.source_title} ({tech.source_url})",
                 f"**Technique:** {tech.description}",
                 f"**Expected impact:** {tech.expected_impact}",
                 f"**Evidence:** {tech.evidence}",
                 f"**Complexity:** {tech.complexity}",
                 f"**Prerequisites:** {', '.join(tech.prerequisites) if tech.prerequisites else 'none'}",
                 f"**Applicable when:** {tech.applicable_when}",
                 f"**NOT applicable when:** {tech.not_applicable_when}"]

        if current_bottleneck:
            lines.append(f"")
            lines.append(f"**Current bottleneck (from R1 Deep Read):** {current_bottleneck}")
            if tech.applicable_when and current_bottleneck.lower() in tech.applicable_when.lower():
                lines.append(f"→ MATCH: Current bottleneck matches the technique's use case.")
            elif tech.not_applicable_when and current_bottleneck.lower() in tech.not_applicable_when.lower():
                lines.append(f"→ WARNING: Current bottleneck matches the technique's NOT-applicable case.")

        # Check what similar techniques worked/failed
        similar_results = []
        for t in self.techniques.techniques:
            if t.tried and t.name != tech.name:
                if any(p in tech.prerequisites for p in [t.name]):
                    similar_results.append(f"  Prerequisite '{t.name}': {t.result}")
                if t.mutation_type == tech.mutation_type:
                    similar_results.append(f"  Similar mutation '{t.name}': {t.result}")

        if similar_results:
            lines.append(f"")
            lines.append(f"**Related experiment results:**")
            lines.extend(similar_results)

        return "\n".join(lines)

    def research_summary(self) -> str:
        """Generate a complete research summary for the agent."""
        lines = [
            f"{'═'*60}",
            f"  Domain Knowledge Summary",
            f"{'═'*60}",
            f"",
            f"  {self.sources.reading_progress()}",
            f"",
        ]

        # Key insights
        insights = self.sources.all_insights()
        if insights:
            lines.append(f"  Key insights ({len(insights)} total):")
            for ins in insights[:10]:  # top 10
                lines.append(f"    • {ins['insight']} (from: {ins['source']})")
            lines.append("")

        # Techniques
        untried = self.techniques.untried()
        successful = self.techniques.successful()
        failed = self.techniques.failed()

        if untried:
            lines.append(f"  Untried techniques (by priority):")
            for t in untried[:5]:
                lines.append(f"    {t.priority:.2f} | {t.name}: {t.expected_impact} [{t.complexity}]")
            lines.append("")

        if successful:
            lines.append(f"  Successful techniques ({len(successful)}):")
            for t in successful:
                lines.append(f"    ✅ {t.name}: {t.result}")
            lines.append("")

        if failed:
            lines.append(f"  Failed techniques ({len(failed)}):")
            for t in failed:
                lines.append(f"    ❌ {t.name}: {t.result}")

        lines.append(f"{'═'*60}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 5. UNIFIED INTERFACE
# ════════════════════════════════════════════════════════════

class KnowledgeAcquisition:
    """
    Unified interface for the complete knowledge acquisition pipeline.
    
    Usage:
        ka = KnowledgeAcquisition(domain="web_api", spec="Optimize REST API")
        
        # Step 1: Get search queries
        queries = ka.generate_searches()
        
        # Step 2: Agent searches, reads, and registers findings
        ka.register_source(url, title, source_type)
        ka.mark_source_read(url, summary, key_insights)
        ka.extract_technique(source_url, name, description, ...)
        
        # Step 3: Get experiment guidance
        next_tech = ka.suggest_next(current_bottleneck="DB latency")
        context = ka.hypothesis_context("connection_pooling", "DB latency")
        
        # Step 4: After experiment, record result
        ka.record_result("connection_pooling", "worked: p99 from 142ms to 85ms")
    """

    def __init__(self, domain: str = "", spec: str = "",
                 language: str = "python", **kwargs):
        self.domain = domain
        self.spec = spec
        self.language = language
        self.extra = kwargs
        self.sources = SourceManager()
        self.techniques = TechniqueLibrary()
        self.integration = KnowledgeIntegration(self.techniques, self.sources)

    def generate_searches(self, bottleneck: str = None) -> list:
        """Generate prioritized search queries."""
        return SearchStrategy.generate(
            domain=self.domain, spec=self.spec,
            language=self.language, bottleneck=bottleneck,
            **{k: v for k, v in self.extra.items()
               if k in ("component", "framework", "task")})

    def register_source(self, url: str, title: str, source_type: str,
                       relevance: float = 0.5) -> Source:
        """Register an external source."""
        return self.sources.add(url, title, source_type, relevance)

    def mark_source_read(self, url: str, summary: str, key_insights: list = None):
        """Mark a source as read with findings."""
        self.sources.mark_read(url, summary, key_insights)

    def extract_technique(self, source_url: str, name: str,
                         description: str, **kwargs) -> Technique:
        """Extract a technique from a source."""
        source_title = ""
        for s in self.sources.sources:
            if s.url == source_url:
                source_title = s.title
                if name not in s.techniques_found:
                    s.techniques_found.append(name)
                    self.sources.save()
                break
        return self.techniques.add(name=name, description=description,
                                   source_url=source_url,
                                   source_title=source_title, **kwargs)

    def suggest_next(self, current_bottleneck: str = "",
                     failed: list = None) -> Optional[Technique]:
        """Suggest the best technique to try next."""
        return self.integration.suggest_next_technique(current_bottleneck, failed)

    def hypothesis_context(self, technique_name: str,
                          current_bottleneck: str = "") -> str:
        """Get knowledge-backed hypothesis context."""
        return self.integration.generate_hypothesis_context(
            technique_name, current_bottleneck)

    def record_result(self, technique_name: str, result: str):
        """Record the result of trying a technique."""
        self.techniques.mark_tried(technique_name, result)

    def summary(self) -> str:
        """Full research summary."""
        return self.integration.research_summary()

    def reading_protocol(self) -> str:
        """
        Generate the reading protocol for the agent.
        
        This tells the agent HOW to read external sources effectively —
        not just "read this URL" but "read this URL and extract:
        techniques, pitfalls, architecture patterns, and benchmarks."
        """
        return f"""
## Reading Protocol — How to Extract Knowledge from External Sources

When reading a source (documentation, paper, article, code), extract:

### 1. Techniques (the most valuable)
For each technique mentioned:
- **Name:** A short identifier (e.g., "connection_pooling")
- **Description:** One sentence of what it does
- **Expected impact:** Quantified if possible ("30-50% latency reduction")
- **Complexity:** trivial / simple / moderate / complex / major
- **Prerequisites:** What must exist first
- **Evidence:** Benchmarks, case studies, or theoretical reasoning
- **When applicable:** Under what conditions this helps
- **When NOT applicable:** Under what conditions this hurts

Call: ka.extract_technique(source_url, name, description, ...)

### 2. Architecture patterns
- What is the standard way to structure this kind of system?
- What are the main components and how do they connect?
- What design decisions are most impactful?

### 3. Common pitfalls
- What do beginners always get wrong?
- What looks right but performs poorly at scale?
- What are the subtle bugs that only appear under load?

### 4. Benchmarks and comparisons
- How do different approaches compare quantitatively?
- What is the baseline performance to expect?
- What is the state-of-the-art performance?

### After reading each source:
Call: ka.mark_source_read(url, summary="2-3 sentences", 
                          key_insights=["insight 1", "insight 2"])

### Time budget: 
- Read at most 5-7 sources before starting experiments
- Spend at most 2-3 minutes per source
- Prioritize: documentation > tutorials > articles > papers > discussions
- Stop reading when you have 3+ actionable techniques to try

### Domain: {self.domain}
### Spec: {self.spec}
### Current search queries (by priority):
"""  + "\n".join(f"  {i+1}. [{q['priority']:.2f}] {q['query']}"
                for i, q in enumerate(self.generate_searches()[:10]))
