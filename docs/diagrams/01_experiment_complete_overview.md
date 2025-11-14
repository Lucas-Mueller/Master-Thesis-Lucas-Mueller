# Diagram 01: Complete Experiment Overview

**Purpose**: High-level view of the two-phase experiment structure
**Audience**: Researchers, managers, new developers
**Layer**: 1 - High-Level Process Flow

---

```mermaid
flowchart TD
    Start([Experiment Start]) --> Config[Load Configuration]
    Config --> Phase1[Phase 1: Individual Deliberation]

    Phase1 --> P1_1[1.1 Initial Ranking]
    P1_1 --> P1_2[1.2 Principle Explanation]
    P1_2 --> P1_3[1.3 Post-Explanation Ranking]
    P1_3 --> P1_4[1.4 Application Rounds x4]
    P1_4 --> P1_5[1.5 Final Ranking]

    P1_5 --> Phase2[Phase 2: Group Discussion]

    Phase2 --> P2_1[Discussion Rounds]
    P2_1 --> P2_2{Voting<br/>Initiated?}
    P2_2 -->|No| P2_1
    P2_2 -->|Yes| P2_3[Voting Process]
    P2_3 --> P2_4{Consensus<br/>Reached?}
    P2_4 -->|No| P2_1
    P2_4 -->|Yes| Results[Apply Principle & Calculate Payoffs]

    Results --> Rankings[Collect Final Rankings]
    Rankings --> End([Experiment Complete])

    %% Styling
    classDef phase1Style fill:#E8F4F8,stroke:#4A90E2,stroke-width:2px
    classDef phase2Style fill:#F0F9F4,stroke:#7ED321,stroke-width:2px
    classDef processStyle fill:#FFF4E6,stroke:#F5A623,stroke-width:2px
    classDef decisionStyle fill:#FFF0F5,stroke:#BD10E0,stroke-width:2px

    class Phase1,P1_1,P1_2,P1_3,P1_4,P1_5 phase1Style
    class Phase2,P2_1,P2_3 phase2Style
    class Results,Rankings processStyle
    class P2_2,P2_4 decisionStyle
```

---

## Phase 1: Individual Deliberation (Parallel)

Each agent independently:
1. **Initial Ranking**: Rank 4 justice principles without knowledge
2. **Principle Explanation**: Learn principles with payoff examples
3. **Post-Explanation Ranking**: Re-rank after learning
4. **Application Rounds** (x4): Select principle and distribution, receive payoffs
5. **Final Ranking**: Final preference after experiencing outcomes

**Duration**: ~5-10 API calls per agent
**Details**: See [02_phase1_process_flow.md](./02_phase1_process_flow.md)

---

## Phase 2: Group Discussion (Sequential)

Agents discuss and reach consensus:
- **Discussion Rounds**: Agents speak in rotating order
- **Voting**: End-of-round voting attempts (initiation → confirmation → ballot)
- **Consensus**: Unanimous agreement on principle and constraints required

**Duration**: 5-20 rounds typical, ~30-100 API calls total
**Details**: See [03_phase2_process_flow.md](./03_phase2_process_flow.md)

---

## Key Outputs

- **Phase1Results**: Rankings (initial, post-explanation, final), application choices, earnings
- **Phase2Results**: Discussion history, voting records, consensus outcome, final rankings
- **Payoffs**: Calculated counterfactuals and actual earnings under chosen principle

---

## Related Files

- `core/experiment_manager.py` - Orchestrates both phases
- `core/phase1_manager.py` - Phase 1 implementation
- `core/phase2_manager.py` - Phase 2 implementation

---

## Next Steps

- **Deep Dive Phase 1**: [02_phase1_process_flow.md](./02_phase1_process_flow.md)
- **Deep Dive Phase 2**: [03_phase2_process_flow.md](./03_phase2_process_flow.md)
- **Service Architecture**: [04_phase1_service_sequence.md](./04_phase1_service_sequence.md)
