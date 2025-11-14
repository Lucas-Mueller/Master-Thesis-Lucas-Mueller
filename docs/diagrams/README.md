# Visual Documentation Index

This directory contains comprehensive visual documentation for the Frohlich Experiment Framework, organized into **2 layers** following industry best practices (Apple/Meta style).

---

## Documentation Structure

### Layer 1: High-Level Process Flows
*What happens and when - for researchers, managers, and new developers*

**Focus**: Complete process understanding without implementation details

| # | Diagram | Purpose | Audience |
|---|---------|---------|----------|
| 01 | [Complete Experiment Overview](01_experiment_complete_overview.md) | Two-phase experiment structure | All audiences |
| 02 | [Phase 1 Process Flow](02_phase1_process_flow.md) | Individual deliberation (5 steps including final ranking) | Researchers, Developers |
| 03 | [Phase 2 Process Flow](03_phase2_process_flow.md) | Group discussion with post-round memory updates | Researchers, Developers |

---

### Layer 2: Detailed Service Interactions
*How it works at the service level - for developers implementing features and debugging*

**Focus**: Service boundaries, API calls, timing, and implementation specifics

| # | Diagram | Purpose | Audience |
|---|---------|---------|----------|
| 04 | [Phase 1 Service Sequence](04_phase1_service_sequence.md) | Service interactions during Phase 1 | Developers |
| 05 | [Discussion Round Detailed](05_discussion_round_detailed.md) | Complete discussion workflow with memory management | Developers, Debugging |
| 06 | [Voting Process Detailed](06_voting_detailed.md) | Four-phase voting with multilingual support | Developers, Debugging |
| 07 | [Data Models](07_data_models.md) | Core data structures and relationships | Developers |

---

## Reading Paths

### For Researchers / Non-Technical Users
**Goal**: Understand experimental process and outputs

1. [01: Complete Overview](01_experiment_complete_overview.md) - Start here
2. [02: Phase 1 Process](02_phase1_process_flow.md) - Individual agent behavior
3. [03: Phase 2 Process](03_phase2_process_flow.md) - Group discussion dynamics

**Time**: ~15-20 minutes

---

### For New Developers
**Goal**: Understand architecture and implementation

1. [01: Complete Overview](01_experiment_complete_overview.md) - Big picture
2. [02: Phase 1 Process](02_phase1_process_flow.md) - Phase 1 flow
3. [03: Phase 2 Process](03_phase2_process_flow.md) - Phase 2 flow
4. [04: Phase 1 Services](04_phase1_service_sequence.md) - Service interactions
5. [07: Data Models](07_data_models.md) - Data structures

**Time**: ~45-60 minutes

---

### For Debugging / Implementation
**Goal**: Understand specific feature implementation

**Phase 1 Issues**:
- Start: [02: Phase 1 Process](02_phase1_process_flow.md)
- Deep dive: [04: Phase 1 Services](04_phase1_service_sequence.md)

**Phase 2 Discussion Issues**:
- Start: [03: Phase 2 Process](03_phase2_process_flow.md)
- Deep dive: [05: Discussion Round Detailed](05_discussion_round_detailed.md)

**Phase 2 Voting Issues**:
- Start: [03: Phase 2 Process](03_phase2_process_flow.md)
- Deep dive: [06: Voting Process Detailed](06_voting_detailed.md)

**Data Structure Questions**:
- Go directly to: [07: Data Models](07_data_models.md)

---

## Diagram Conventions

### Visual Style
Following Apple/Meta developer documentation principles:
- **Minimal diagrams**: Focus on essential information
- **Progressive disclosure**: High-level first, drill-down links for details
- **Consistent formatting**: Clear hierarchies and annotations

### Mermaid Diagrams
All diagrams use **Mermaid syntax** for:
- Version control compatibility
- GitHub/GitLab native rendering
- Easy maintenance and updates

### Color Coding
Consistent across all diagrams:
- **Blue** `#4A90E2`: Core framework components
- **Green** `#7ED321`: Services and successful outcomes
- **Orange** `#F5A623`: Agents and supporting components
- **Purple** `#BD10E0`: Data models and state
- **Teal** `#50E3C2`: External systems
- **Yellow** `#FDD835`: Notes and important information
- **Pink** `#E91E63`: Decision points

---

## Key Features Documented

### ⭐ Critical Accuracy Updates

**Phase 1 Final Ranking** (Previously Undocumented)
- **Step 1.5**: Final ranking after application rounds
- **Location**: `phase1_manager.py:581-625`
- **Documented in**: [02: Phase 1 Process](02_phase1_process_flow.md), [04: Phase 1 Services](04_phase1_service_sequence.md)

**Phase 2 Post-Round Memory Updates** (Previously Unclear)
- **Timing**: After ALL speakers finish, before voting
- **Purpose**: Ensure symmetric information for fair voting
- **Location**: `phase2_manager.py:873-900`
- **Documented in**: [03: Phase 2 Process](03_phase2_process_flow.md), [05: Discussion Round Detailed](05_discussion_round_detailed.md)

---

## Rendering Diagrams

### View in GitHub
Diagrams render automatically when viewing `.md` files on GitHub.

### Generate PNG Images
```bash
# Install Mermaid CLI (if not already installed)
npm install -g @mermaid-js/mermaid-cli

# Navigate to diagrams directory
cd docs/diagrams

# Generate all PNG renders
mmdc -i 01_experiment_complete_overview.md -o rendered/01_experiment_complete_overview.png
mmdc -i 02_phase1_process_flow.md -o rendered/02_phase1_process_flow.png
mmdc -i 03_phase2_process_flow.md -o rendered/03_phase2_process_flow.png
mmdc -i 04_phase1_service_sequence.md -o rendered/04_phase1_service_sequence.png
mmdc -i 05_discussion_round_detailed.md -o rendered/05_discussion_round_detailed.png
mmdc -i 06_voting_detailed.md -o rendered/06_voting_detailed.png
mmdc -i 07_data_models.md -o rendered/07_data_models.png
```

---

## Archive

Old diagram versions and drafts are preserved in `archive/` for reference:
- Previous 3-layer organization (01-09 from old structure)
- Draft versions (10, 10_v2, 11, 11_v2)
- Proposed and comparison diagrams (PROPOSED_*, COMPARISON_*)

**Note**: Archived diagrams may not reflect current implementation. Use current diagrams (01-07) for accurate information.

---

## Maintenance

### Updating Diagrams
When modifying code:
1. **Check Layer 1**: Does the high-level process change?
2. **Check Layer 2**: Do service interactions change?
3. Update relevant diagram(s)
4. Regenerate PNG renders if needed
5. Update this README if new diagrams added

### Diagram Ownership
- **Layer 1**: Product/research team + senior developers
- **Layer 2**: Development team

### Update Checklist
Before committing diagram changes:
- [ ] Mermaid syntax renders correctly
- [ ] Colors follow style guide
- [ ] Cross-references to other diagrams updated
- [ ] Code file references accurate (with line numbers)
- [ ] "Next Steps" section current
- [ ] This README updated if needed

---

## Related Documentation

- **[TECHNICAL_README.md](../../TECHNICAL_README.md)**: Complete technical documentation with 2-layer structure
- **[README.md](../../README.md)**: Project overview and quick start
- **[CLAUDE.md](../../CLAUDE.md)**: Developer guide and conventions

---

## Viewing Options

**Option 1: GitHub** (Recommended)
- Open any `.md` file in GitHub - diagrams render automatically
- Best for browsing and sharing

**Option 2: Pre-rendered PNG files**
- See [rendered/](rendered/) directory for high-resolution PNG files
- Perfect for presentations and offline viewing

**Option 3: VS Code**
- Install "Markdown Preview Mermaid Support" extension
- Preview diagrams while editing

**Option 4: Local CLI Rendering**
- Use Mermaid CLI (mmdc) as shown in "Generate PNG Images" section above

---

## Quick Reference

### Complete File Index

| Layer | # | File | Title | Priority |
|-------|---|------|-------|----------|
| **1** | 01 | [01_experiment_complete_overview.md](01_experiment_complete_overview.md) | Complete Experiment Overview | High |
| **1** | 02 | [02_phase1_process_flow.md](02_phase1_process_flow.md) | Phase 1 Process Flow (5 steps) | High |
| **1** | 03 | [03_phase2_process_flow.md](03_phase2_process_flow.md) | Phase 2 Process Flow (with post-round memory) | High |
| **2** | 04 | [04_phase1_service_sequence.md](04_phase1_service_sequence.md) | Phase 1 Service Sequence | Medium |
| **2** | 05 | [05_discussion_round_detailed.md](05_discussion_round_detailed.md) | Discussion Round Detailed | High |
| **2** | 06 | [06_voting_detailed.md](06_voting_detailed.md) | Voting Process Detailed | High |
| **2** | 07 | [07_data_models.md](07_data_models.md) | Core Data Models | Medium |

---

## Success Metrics

These diagrams are successful if:
- ✅ New contributors understand architecture in <1 hour
- ✅ "How does X work?" questions answered by pointing to diagrams
- ✅ Code reviews reference diagrams for architectural decisions
- ✅ Onboarding time reduced significantly
- ✅ Clear separation between process (what) and implementation (how)

---

*Last Updated*: 2025-01-14
*Diagram Structure*: 2-Layer (01-07)
*Total Diagrams*: 7 (3 Layer 1 + 4 Layer 2)
