# Architectural Diagrams

This directory contains progressive architectural diagrams for the **Distributive Justice Experiment Framework** (Master Thesis of Lucas Mueller), designed to guide understanding from high-level concepts to detailed implementation.

## Quick Start

**New to the project?** Start here:
1. [01_experiment_overview.md](01_experiment_overview.md) - Quick overview
2. [02_system_context.md](02_system_context.md) - External dependencies
3. [04_phase2_services.md](04_phase2_services.md) - Most important for developers

**Working on a specific feature?** Jump to:
- **Discussion logic**: [06_discussion_sequence.md](06_discussion_sequence.md)
- **Voting system**: [07_voting_sequence.md](07_voting_sequence.md)
- **Memory management**: [08_memory_flow.md](08_memory_flow.md)
- **Payoff calculation**: [09_payoff_calculation.md](09_payoff_calculation.md)

---

## Diagram Hierarchy

### Level 1: Executive Overview
Quick orientation for researchers, managers, and first-time readers.

| Diagram | Purpose | Complexity |
|---------|---------|------------|
| [01_experiment_overview.md](01_experiment_overview.md) | Two-phase experiment flow | 3-5 boxes |
| [02_system_context.md](02_system_context.md) | System boundaries and external dependencies | 5-7 boxes |

**Read these first** if you're new to the project or giving a high-level presentation.

---

### Level 2: Conceptual Architecture
Understand the major components and their responsibilities.

| Diagram | Purpose | Complexity |
|---------|---------|------------|
| [03_phase1_architecture.md](03_phase1_architecture.md) | Phase 1 parallel execution model | 10-12 boxes |
| [04_phase2_services.md](04_phase2_services.md) | **Phase 2 services-first architecture** | 12-15 boxes |
| [05_data_model_core.md](05_data_model_core.md) | Core data types and relationships | 8-10 classes |

**Read these** if you're starting to work with the codebase or reviewing architecture decisions.

**Priority**: Start with **04_phase2_services.md** - it's the most important diagram for understanding how Phase 2 works.

---

### Level 3: Detailed Workflows
Deep dive into specific processes for implementation and debugging.

| Diagram | Purpose | Complexity |
|---------|---------|------------|
| [06_discussion_sequence.md](06_discussion_sequence.md) | One complete discussion round with memory updates | 15-20 interactions |
| [07_voting_sequence.md](07_voting_sequence.md) | Complete voting flow (initiation → ballot → consensus) | 20-25 interactions |
| [08_memory_flow.md](08_memory_flow.md) | Memory routing logic and truncation rules | 12-15 decision boxes |
| [09_payoff_calculation.md](09_payoff_calculation.md) | Payoff calculation and counterfactual analysis | 15-18 decision boxes |

**Read these** when implementing features, debugging issues, or understanding complex logic.

---

## Reading Paths

### Path 1: "I'm a researcher trying to understand the experiment"
1. [01_experiment_overview.md](01_experiment_overview.md) - What happens in an experiment?
2. [02_system_context.md](02_system_context.md) - How does it interact with LLMs?
3. [05_data_model_core.md](05_data_model_core.md) - What data is collected?
4. Done! (Optional: Read TECHNICAL_README.md Section 1-2 for philosophical background)

---

### Path 2: "I'm a developer starting to work on the codebase"
1. [01_experiment_overview.md](01_experiment_overview.md) - High-level flow
2. [04_phase2_services.md](04_phase2_services.md) - Services architecture (most important!)
3. [03_phase1_architecture.md](03_phase1_architecture.md) - Phase 1 structure
4. [05_data_model_core.md](05_data_model_core.md) - Core data types
5. Done! (Optional: Read specific workflow diagrams as needed)

---

### Path 3: "I'm implementing a feature in Phase 2"
1. [04_phase2_services.md](04_phase2_services.md) - Service boundaries and ownership
2. Choose your area:
   - **Discussion features** → [06_discussion_sequence.md](06_discussion_sequence.md)
   - **Voting features** → [07_voting_sequence.md](07_voting_sequence.md)
   - **Memory features** → [08_memory_flow.md](08_memory_flow.md)
   - **Results features** → [09_payoff_calculation.md](09_payoff_calculation.md)
3. Done! (Check CLAUDE.md "Service Ownership and Modification Guide")

---

### Path 4: "I'm debugging a specific issue"
Jump directly to the relevant workflow diagram:
- **Agent statements too short**: [06_discussion_sequence.md](06_discussion_sequence.md) § Validation
- **Voting not detecting consensus**: [07_voting_sequence.md](07_voting_sequence.md) § Consensus Detection
- **Memory overflow errors**: [08_memory_flow.md](08_memory_flow.md) § Memory Overflow Prevention
- **Unexpected payoffs**: [09_payoff_calculation.md](09_payoff_calculation.md) § Principle Application

---

## Diagram Conventions

### Visual Language

**Colors** (consistent across all diagrams):
- 🔵 **Blue** (#4A90E2): Framework core components
- 🟢 **Green** (#7ED321): Services and successful outcomes
- 🟠 **Orange** (#F5A623): Agents and supporting components
- 🟣 **Purple** (#BD10E0): Data models and state
- 🔷 **Teal** (#50E3C2): External systems
- 🔴 **Red** (#D0021B): Errors and slow operations

**Box Styles**:
- **Solid border**: Active component
- **Dashed border**: Optional component
- **Thick border**: Entry point or key component
- **Double border**: Final output

**Arrow Types**:
- **Solid arrow** (→): Control flow or data flow
- **Dotted arrow** (⇢): Read-only access
- **Dashed arrow** (⇀): Optional or conditional

### Complexity Ratings

- **Executive**: 3-7 boxes, quick to understand
- **Conceptual**: 8-15 boxes, moderate detail
- **Detailed**: 15-25 boxes, comprehensive detail

### Target Audiences

Each diagram specifies its target audience:
- **Researchers**: Non-technical stakeholders
- **Managers**: Technical oversight, architecture decisions
- **Developers**: Implementation and maintenance
- **System Architects**: Design decisions and trade-offs

---

## Technology

### Rendering

All diagrams use **Mermaid** syntax for maximum portability:
- ✅ Renders directly in GitHub/GitLab
- ✅ No special software needed
- ✅ Version control friendly (text-based)
- ✅ Easy to update and review

### Viewing Options

**Option 1: GitHub** (Recommended)
- Open any `.md` file in GitHub - diagrams render automatically
- Best for browsing and sharing

**Option 2: Pre-rendered PNG files** 🆕
- See [rendered/](rendered/) directory for 13 high-resolution PNG files
- Perfect for presentations, documentation, and offline viewing
- Total size: ~1.9 MB with transparent backgrounds

**Option 3: VS Code**
- Install "Markdown Preview Mermaid Support" extension
- Preview diagrams while editing

**Option 4: Local Rendering**
```bash
# Regenerate all diagrams to PNG
cd docs/diagrams/
python3 render_diagrams.py
```

---

## Maintenance

### When to Update Diagrams

Update diagrams when:
- [ ] New service added to Phase 2
- [ ] Major refactoring of core components
- [ ] New experiment phase introduced
- [ ] Data model changes (new types, fields)
- [ ] Voting or memory systems modified

### Update Process

1. **Identify affected diagrams**: Check cross-references in each file
2. **Update Mermaid source**: Edit the markdown file directly
3. **Verify rendering**: Preview in VS Code or push to GitHub
4. **Update cross-references**: Ensure "Next Steps" links are current
5. **Update this README**: If adding/removing diagrams

### Review Schedule

- **Monthly**: Verify Level 1-2 diagrams match current implementation
- **Quarterly**: Review Level 3 workflow diagrams
- **Per release**: Full diagram audit

---

## Contributing

### Adding New Diagrams

Follow the template structure:
```markdown
# Diagram X.Y: Title

**Purpose**: One-sentence description

**Target Audience**: Who should read this

**Complexity Level**: ⭐/⭐⭐/⭐⭐⭐

---

## [Main content with Mermaid diagram]

---

## [Supporting sections]

---

## Related Files

**Core Implementation**:
- `path/to/file.py` - Description

---

## Next Steps

- **For related topic**: See Diagram X.Z
- **For technical details**: See TECHNICAL_README.md Section N
```

### Diagram Checklist

Before committing a new diagram:
- [ ] Clear title and purpose statement
- [ ] Target audience specified
- [ ] Complexity level indicated
- [ ] Mermaid syntax renders correctly
- [ ] Colors follow style guide
- [ ] Related files section included
- [ ] Cross-references to other diagrams
- [ ] Added to this README

---

## Related Documentation

**Primary Documentation**:
- [TECHNICAL_README.md](../../TECHNICAL_README.md) - Comprehensive technical documentation
- [CLAUDE.md](../../CLAUDE.md) - Repository guidelines and patterns
- [DIAGRAM_PLAN.md](../../DIAGRAM_PLAN.md) - Diagram strategy and rationale

**Code Documentation**:
- `core/` - Core orchestration logic
- `core/services/` - Phase 2 service implementations
- `models/` - Data model definitions

**Testing Documentation**:
- `tests/unit/` - Unit tests (fast, deterministic)
- `tests/component/` - Component tests (real API calls)
- `tests/integration/` - Integration tests (end-to-end workflows)

---

## Quick Reference

### File Index

| # | File | Title | Level | Priority |
|---|------|-------|-------|----------|
| 1.1 | `01_experiment_overview.md` | Two-Phase Experiment Flow | Executive | Medium |
| 1.2 | `02_system_context.md` | High-Level System Context | Executive | Medium |
| 2.1 | `03_phase1_architecture.md` | Phase 1 Architecture | Conceptual | Medium |
| 2.2 | `04_phase2_services.md` | Phase 2 Services Architecture | Conceptual | **HIGHEST** |
| 2.3 | `05_data_model_core.md` | Data Model Overview | Conceptual | Medium |
| 3.1 | `06_discussion_sequence.md` | Phase 2 Discussion Round | Detailed | High |
| 3.2 | `07_voting_sequence.md` | Voting Process Flow | Detailed | High |
| 3.3 | `08_memory_flow.md` | Memory Management Flow | Detailed | High |
| 3.4 | `09_payoff_calculation.md` | Payoff Calculation Process | Detailed | Medium |

---

## Feedback

**Found an issue?**
- Diagram out of date → Create GitHub issue with "docs: diagram" label
- Unclear explanation → Suggest improvements in PR review
- Missing diagram → Propose addition in GitHub discussion

**Have suggestions?**
- Propose new diagrams in `DIAGRAM_PLAN.md`
- Suggest alternative visual styles
- Share alternative reading paths for specific use cases

---

## Success Metrics

These diagrams are successful if:
- ✅ New contributors understand architecture quickly
- ✅ "How does X work?" questions are answered by pointing to diagrams
- ✅ Code reviews reference diagrams for architectural decisions
- ✅ Onboarding time is reduced significantly
- ✅ Less confusion about service boundaries and responsibilities

---

**Last Updated**: 2025-01-14 (Initial diagram suite completed)

**Diagram Count**: 10 diagrams (Levels 1-3 complete)

**Future Additions** (see DIAGRAM_PLAN.md):
- Level 4: Implementation details (service dependencies, complete data model)
- Level 5: Dynamic behavior (execution timeline, state machines)
