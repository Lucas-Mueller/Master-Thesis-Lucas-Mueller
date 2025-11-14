# Diagram 2.1: Phase 1 Architecture

**Purpose**: Show how Phase 1 manages parallel agent familiarization with justice principles

**Target Audience**: Developers starting to work with the codebase

**Complexity Level**: Conceptual Conceptual (10-12 boxes)

---

## Phase 1 Component Architecture

```mermaid
graph TB
    Manager[Phase1Manager<br/>Orchestrator]

    subgraph "Parallel Execution (asyncio)"
        Agent1[ParticipantAgent 1<br/>Seed: 42]
        Agent2[ParticipantAgent 2<br/>Seed: 43]
        Agent3[ParticipantAgent 3<br/>Seed: 44]
        AgentN[... Agent 8<br/>Seed: 49]
    end

    subgraph "Supporting Components"
        DistGen[DistributionGenerator<br/>Income Distributions]
        Memory[SelectiveMemoryManager<br/>Context Truncation]
        Utility[UtilityAgent<br/>Parsing & Validation]
        SeedMgr[SeedManager<br/>Reproducibility]
    end

    subgraph "Phase 1 Tasks (Per Agent)"
        Task1[Task 1: Initial Ranking<br/>Rank 4 principles]
        Task2[Task 2: Principle Explanation<br/>Learn about principles]
        Task3[Task 3: Post-Explanation Ranking<br/>Re-rank after learning]
        Task4[Task 4: Application Rounds<br/>4 rounds of distribution selection]
    end

    Results[Phase1Results<br/>Rankings + Applications]

    Manager -->|Creates tasks| Agent1
    Manager -->|Creates tasks| Agent2
    Manager -->|Creates tasks| Agent3
    Manager -->|Creates tasks| AgentN

    Agent1 --> Task1
    Task1 --> Task2
    Task2 --> Task3
    Task3 --> Task4

    DistGen -->|Provides distributions| Task4
    Memory -->|Truncates context| Agent1
    Memory -->|Truncates context| Agent2
    SeedMgr -->|Deterministic seeding| Agent1
    SeedMgr -->|Deterministic seeding| Agent2
    Utility -->|Parses responses| Task1
    Utility -->|Parses responses| Task3

    Task4 --> Results

    style Manager fill:#4A90E2,stroke:#333,stroke-width:3px,color:#fff
    style Agent1 fill:#F5A623,stroke:#333,stroke-width:2px
    style Agent2 fill:#F5A623,stroke:#333,stroke-width:2px
    style Agent3 fill:#F5A623,stroke:#333,stroke-width:2px
    style AgentN fill:#F5A623,stroke:#333,stroke-width:2px
    style DistGen fill:#7ED321,stroke:#333
    style Memory fill:#7ED321,stroke:#333
    style Utility fill:#7ED321,stroke:#333
    style SeedMgr fill:#7ED321,stroke:#333
    style Results fill:#BD10E0,stroke:#333,stroke-width:2px
```

---

## Key Architectural Concepts

### 1. Parallel Execution Strategy
**Why Parallel?** Phase 1 has no inter-agent dependencies - each agent learns independently.

- **Implementation**: Uses Python `asyncio.gather()` to execute all 8 agents simultaneously
- **Performance**: Reduces Phase 1 time from ~8 minutes (sequential) to ~60 seconds (parallel)
- **Isolation**: Each agent receives its own seeded random number generator

### 2. Deterministic Seeding
**Why Seeded?** Ensures reproducible experiments for scientific rigor.

- **Global Seed**: Set in configuration (e.g., `seed: 42`)
- **Per-Agent Seed**: `global_seed + agent_index` (Agent 0 gets 42, Agent 1 gets 43, etc.)
- **What's Seeded**: Distribution generation, income class assignment
- **What's NOT Seeded**: LLM responses (inherently stochastic)

### 3. Memory Management
**Why Truncation?** Prevents context window overflow as experiments progress.

- **Statement Truncation**: Max 300 characters per statement
- **Reasoning Truncation**: Max 200 characters per reasoning block
- **Strategy**: Oldest content removed first (chronological FIFO)

### 4. Task Sequence (Per Agent)
Each agent executes 4 sequential tasks:

1. **Initial Ranking**: Rank 4 principles without any information (baseline preferences)
2. **Principle Explanation**: Receive detailed explanations with payoff examples
3. **Post-Explanation Ranking**: Re-rank after learning (measures preference stability)
4. **Application Rounds**: Apply principles to 4 different distributions (tests understanding)

---

## Data Flow

```mermaid
sequenceDiagram
    participant PM as Phase1Manager
    participant PA as ParticipantAgent
    participant DG as DistributionGenerator
    participant UA as UtilityAgent

    PM->>PA: 1. Task 1: Initial Ranking
    PA->>PA: 2. Generate ranking
    PA->>UA: 3. Parse ranking
    UA->>PM: 4. PrincipleRanking object

    PM->>PA: 5. Task 2: Explanations
    PA->>PA: 6. Process explanations

    PM->>PA: 7. Task 3: Post-Explanation Ranking
    PA->>UA: 8. Parse ranking
    UA->>PM: 9. Updated PrincipleRanking

    loop 4 Application Rounds
        PM->>DG: 10. Generate distributions
        DG->>PM: 11. DistributionSet (4 distributions)
        PM->>PA: 12. Apply principle to distributions
        PA->>PM: 13. ApplicationResult
    end

    PM->>PM: 14. Compile Phase1Results
```

---

## Component Responsibilities

| Component | Responsibility | Key Methods | File Location |
|-----------|----------------|-------------|---------------|
| **Phase1Manager** | Orchestrates Phase 1 execution | `run_phase1()`, `_execute_participant_tasks()` | `core/phase1_manager.py` |
| **ParticipantAgent** | Represents individual AI agent | `think()`, `rank_principles()` | `experiment_agents/participant_agent.py` |
| **DistributionGenerator** | Generates income distributions | `generate_distribution_for_round()` | `core/distribution_generator.py` |
| **SelectiveMemoryManager** | Manages agent memory | `update_memory_simple()`, `update_memory_complex()` | `utils/selective_memory_manager.py` |
| **UtilityAgent** | Parses agent responses | `parse_principle_ranking()`, `parse_principle_choice()` | `experiment_agents/utility_agent.py` |
| **SeedManager** | Manages reproducibility | `get_seed_for()`, `_build_participant_rngs()` | `utils/seed_manager.py` |

---

## Output: Phase1Results

Each agent produces a `Phase1Results` object containing:

```python
Phase1Results:
    participant_name: str                      # e.g., "Alice"
    initial_ranking: PrincipleRanking          # Before learning
    post_explanation_ranking: PrincipleRanking # After learning
    application_results: List[ApplicationResult] # 4 rounds of applications
    completion_status: str                     # "completed" or error
```

These results are:
1. **Passed to Phase 2** for contextual discussion prompts
2. **Saved in final experiment results** for analysis
3. **Used to measure learning** (compare initial vs post-explanation rankings)

---

## Performance Characteristics

**Timing** (8 agents, parallel execution):
- Initial Ranking: ~5-10 seconds
- Explanations: ~10-15 seconds
- Post-Explanation Ranking: ~5-10 seconds
- 4 Application Rounds: ~15-30 seconds
- **Total**: ~60 seconds (±20 seconds)

**Token Usage** (GPT-4o):
- Per agent: ~6,000 tokens
- Total (8 agents): ~50,000 tokens
- **Cost**: ~$1-2 at current pricing

---

## Related Files

**Core Implementation**:
- `core/phase1_manager.py` - Main orchestrator (220 lines)
- `core/distribution_generator.py` - Income distribution logic (180 lines)
- `experiment_agents/participant_agent.py` - Agent implementation (350 lines)

**Supporting Components**:
- `utils/selective_memory_manager.py` - Memory management (150 lines)
- `utils/seed_manager.py` - Seeding logic (90 lines)
- `experiment_agents/utility_agent.py` - Parsing utilities (200 lines)

**Data Models**:
- `models/principle_types.py` - JusticePrinciple, PrincipleRanking
- `models/phase1_results.py` - Phase1Results structure

---

## Next Steps

- **For Phase 2 architecture**: See Diagram 2.2 (Phase 2 Services Architecture)
- **For detailed task workflows**: See TECHNICAL_README.md Section 2.1
- **For understanding seeding**: See TECHNICAL_README.md Section 8
