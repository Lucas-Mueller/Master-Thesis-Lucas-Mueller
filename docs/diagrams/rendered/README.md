# Rendered Diagrams (PNG Format)

This directory contains PNG renderings of all Mermaid diagrams for the **Distributive Justice Experiment Framework** (Master Thesis of Lucas Mueller).

## Overview

**Total Diagrams**: 13 PNG files
**Total Size**: ~1.9 MB
**Format**: PNG with transparent background
**Generated**: 2025-01-14 using Mermaid CLI (mmdc)

---

## Files

### Level 1: Executive Overview

| File | Size | Description |
|------|------|-------------|
| `01_experiment_overview.png` | 32 KB | Two-phase experiment flow |
| `02_system_context_1.png` | 90 KB | System context diagram |
| `02_system_context_2.png` | 52 KB | Data flow sequence |

**Use Case**: Presentations, executive summaries, quick reference

---

### Level 2: Conceptual Architecture

| File | Size | Description |
|------|------|-------------|
| `03_phase1_architecture_1.png` | 148 KB | Phase 1 component architecture |
| `03_phase1_architecture_2.png` | 89 KB | Phase 1 data flow sequence |
| `04_phase2_services_1.png` | 155 KB | Phase 2 services architecture (main) |
| `04_phase2_services_2.png` | 121 KB | Phase 2 typical discussion round flow |
| `05_data_model_core_1.png` | 215 KB | Core data model class diagram |
| `05_data_model_core_2.png` | 44 KB | Data flow through experiment |

**Use Case**: Architecture reviews, developer onboarding, documentation

---

### Level 3: Detailed Workflows

| File | Size | Description |
|------|------|-------------|
| `06_discussion_sequence.png` | 248 KB | Complete discussion round sequence |
| `07_voting_sequence.png` | 312 KB | Complete voting process flow |
| `08_memory_flow.png` | 196 KB | Memory management decision flow |
| `09_payoff_calculation.png` | 216 KB | Payoff calculation process |

**Use Case**: Implementation guidance, debugging, detailed analysis

---

## Usage

### In Presentations
- Use Level 1 diagrams for executive presentations
- Use Level 2 diagrams for technical architecture reviews
- Use Level 3 diagrams for deep-dive implementation sessions

### In Documentation
- Embed in Markdown files: `![Description](docs/diagrams/rendered/filename.png)`
- Include in PDFs, Word documents, or slides
- Print for reference materials

### In Code Reviews
- Reference specific diagrams when discussing architectural decisions
- Compare implementation against architectural diagrams
- Use as visual aids for explaining complex flows

---

## Regenerating Diagrams

If you update the source `.md` files in the parent directory, regenerate the diagrams:

```bash
# From the docs/diagrams/ directory
python3 render_diagrams.py
```

The script will:
- Extract all Mermaid code blocks from markdown files
- Render each to PNG with transparent background
- Overwrite existing files in this directory

**Requirements**:
- Mermaid CLI (`mmdc`) installed: `npm install -g @mermaid-js/mermaid-cli`
- Python 3.6+

---

## File Naming Convention

Files follow the pattern: `{number}_{diagram_name}_{optional_index}.png`

- **Number**: Diagram level and sequence (01-09)
- **Diagram Name**: Descriptive identifier
- **Optional Index**: For files with multiple diagrams (e.g., `_1`, `_2`)

**Examples**:
- `01_experiment_overview.png` - Single diagram
- `02_system_context_1.png` - First of multiple diagrams
- `04_phase2_services_2.png` - Second diagram in Phase 2 services

---

## Technical Details

**Rendering Settings**:
- Background: Transparent
- Max Width: 1920px
- Max Height: 1080px (auto-scales based on content)
- Format: PNG
- Tool: Mermaid CLI (mmdc)

**Color Palette**:
- Blue (#4A90E2): Framework core
- Green (#7ED321): Services and success
- Orange (#F5A623): Agents and supporting components
- Purple (#BD10E0): Data models and state
- Teal (#50E3C2): External systems
- Red (#D0021B): Errors and slow operations

---

## Source Files

All rendered diagrams are generated from source markdown files in the parent directory:

- [01_experiment_overview.md](../01_experiment_overview.md)
- [02_system_context.md](../02_system_context.md)
- [03_phase1_architecture.md](../03_phase1_architecture.md)
- [04_phase2_services.md](../04_phase2_services.md)
- [05_data_model_core.md](../05_data_model_core.md)
- [06_discussion_sequence.md](../06_discussion_sequence.md)
- [07_voting_sequence.md](../07_voting_sequence.md)
- [08_memory_flow.md](../08_memory_flow.md)
- [09_payoff_calculation.md](../09_payoff_calculation.md)

**Prefer viewing source files on GitHub** - Mermaid renders automatically with interactive zoom and pan.

---

## Quality Notes

### Resolution
- Diagrams are rendered at high resolution suitable for:
  - ✅ Screen presentations (1920x1080)
  - ✅ Printed documentation (300 DPI equivalent)
  - ✅ Large displays and projectors
  - ❌ May be too large for web embedding (consider resizing)

### File Sizes
- Simple diagrams: 30-50 KB
- Medium diagrams: 80-150 KB
- Complex diagrams: 150-320 KB

**Optimization**: If file sizes are a concern, resize PNGs to smaller dimensions or convert to WebP format.

---

## Version Control

These PNG files **are included in version control** for convenience:
- ✅ Easy access for presentations and documentation
- ✅ Historical record of diagram evolution
- ✅ No build step required for documentation consumers

However, they can be regenerated at any time from source markdown files using the rendering script.

---

## Troubleshooting

### Problem: "mmdc: command not found"
**Solution**: Install Mermaid CLI
```bash
npm install -g @mermaid-js/mermaid-cli
```

### Problem: "Parse error" during rendering
**Solution**: Check for special characters in node labels (pipes `|`, brackets, quotes). The rendering script will show which diagram failed.

### Problem: Diagrams look different from GitHub preview
**Solution**: This is expected - Mermaid CLI may use slightly different rendering than GitHub's live preview. Both are correct.

---

## Related Documentation

- **Main Diagram Index**: [../README.md](../README.md)
- **Diagram Plan**: [../../DIAGRAM_PLAN.md](../../DIAGRAM_PLAN.md)
- **Technical README**: [../../TECHNICAL_README.md](../../TECHNICAL_README.md)

---

**Last Generated**: 2025-01-14
**Generator**: `render_diagrams.py`
**Total Diagrams**: 13 (100% success rate)
