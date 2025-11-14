# Diagram 04: Phase 1 Service Sequence (Detailed)

**Purpose**: Service-level interactions during Phase 1 execution
**Audience**: Developers implementing Phase 1 features, debugging
**Layer**: 2 - Detailed Service Interactions

---

```mermaid
sequenceDiagram
    participant PM as Phase1Manager
    participant PA as ParticipantAgent
    participant UA as UtilityAgent
    participant LM as LanguageManager
    participant MS as MemoryService
    participant DG as DistributionGenerator

    Note over PM: FOR EACH AGENT (Parallel)

    %% Step 1.1: Initial Ranking
    rect rgb(232, 244, 248)
        Note over PM,MS: Step 1.1: Initial Ranking
        PM->>LM: get_prompt("initial_ranking")
        LM-->>PM: Localized prompt
        PM->>PA: generate_ranking(prompt, context)
        PA-->>PM: Ranking text
        PM->>UA: parse_ranking(text)
        UA-->>PM: List[JusticePrinciple]
        PM->>MS: update_memory(agent, "ranking", data)
        MS-->>PM: Updated memory
    end

    %% Step 1.2: Principle Explanation
    rect rgb(240, 249, 244)
        Note over PM,MS: Step 1.2: Principle Explanation
        PM->>LM: get_explanations("detailed_principles")
        LM-->>PM: Localized explanations with examples
        PM->>PA: read_explanations(explanations)
        PA-->>PM: Acknowledgment
        PM->>MS: update_memory(agent, "explanation", content)
        MS-->>PM: Updated memory
    end

    %% Step 1.3: Post-Explanation Ranking
    rect rgb(232, 244, 248)
        Note over PM,MS: Step 1.3: Post-Explanation Ranking
        PM->>LM: get_prompt("post_explanation_ranking")
        LM-->>PM: Localized prompt
        PM->>PA: generate_ranking(prompt, context)
        PA-->>PM: Ranking text
        PM->>UA: parse_ranking(text)
        UA-->>PM: List[JusticePrinciple]
        PM->>MS: update_memory(agent, "informed_ranking", data)
        MS-->>PM: Updated memory
    end

    %% Step 1.4: Application Rounds (x4)
    rect rgb(255, 244, 230)
        Note over PM,DG: Step 1.4: Application Rounds (Loop 1-4)
        loop Round 1 to 4
            PM->>DG: generate_distributions(4)
            DG-->>PM: 4 income distributions
            PM->>LM: get_prompt("application_round", round_num)
            LM-->>PM: Localized prompt with distributions
            PM->>PA: select_principle_and_distribution(prompt)
            PA-->>PM: Selection text
            PM->>UA: parse_application_choice(text)
            UA-->>PM: {principle, distribution_index}
            PM->>PM: calculate_payoff(choice, agent_class)
            PM->>PM: update_earnings(payoff)
            PM->>MS: update_memory(agent, "application", result)
            MS-->>PM: Updated memory
        end
    end

    %% Step 1.5: Final Ranking ⭐
    rect rgb(255, 240, 245)
        Note over PM,MS: Step 1.5: Final Ranking ⭐
        PM->>LM: get_prompt("final_ranking")
        LM-->>PM: Localized prompt
        PM->>PA: generate_ranking(prompt, context)
        PA-->>PM: Ranking text
        PM->>UA: parse_ranking(text)
        UA-->>PM: List[JusticePrinciple]
        PM->>MS: update_memory(agent, "final_ranking", data)
        MS-->>PM: Updated memory
    end

    %% Collect Results
    rect rgb(224, 242, 241)
        Note over PM: Collect Phase1Results
        PM->>PM: assemble_results(all_data)
        Note over PM: Returns Phase1Results:<br/>• initial_ranking<br/>• post_explanation_ranking<br/>• application_results x4<br/>• final_ranking ⭐<br/>• total_earnings<br/>• final_memory_state
    end
```

---

## Service Responsibilities

### Phase1Manager
**Role**: Orchestrates complete Phase 1 flow for all agents

**Key Methods**:
- `run_phase1_for_agent()` - Main execution loop
- `_step_1_1_initial_ranking()` - Initial preference capture
- `_step_1_2_detailed_explanation()` - Principle education
- `_step_1_3_post_explanation_ranking()` - Informed preference
- `_step_1_4_application_rounds()` - Practical application (x4)
- `_step_1_5_final_ranking()` - Experience-based preference ⭐

**Code**: `core/phase1_manager.py`

---

### ParticipantAgent
**Role**: AI agent representing a participant

**Key Interactions**:
- Receives prompts in configured language
- Generates rankings (initial, post-explanation, final)
- Makes principle/distribution selections
- Maintains internal reasoning state

**Code**: `experiment_agents/participant_agent.py`

---

### UtilityAgent
**Role**: Parser and validator for agent responses

**Responsibilities**:
- Parse ranking text → structured `List[JusticePrinciple]`
- Parse application choices → `{principle, distribution_index}`
- Validate response format and content
- Handle multilingual responses

**Code**: `experiment_agents/utility_agent.py`

---

### LanguageManager
**Role**: Localization and translation management

**Provides**:
- Localized prompts from `translations/{language}/`
- Principle explanations with cultural context
- Consistent terminology across languages

**Code**: `utils/language_manager.py`

---

### MemoryService
**Role**: Agent memory state management

**Operations**:
- Update memory after each step
- Enforce character limits
- Apply guidance styles (narrative/structured)
- Maintain context continuity

**Code**: `core/services/memory_service.py` (conceptually; Phase 1 uses direct updates)

---

### DistributionGenerator
**Role**: Create income distribution scenarios

**Functionality**:
- Generate 4 distributions per application round
- Ensure diversity of scenarios
- Consistent calculation methods

**Code**: `utils/distribution_generator.py`

---

## Timing and Performance

### API Calls Per Agent
- **Initial Ranking**: 1 call (ParticipantAgent) + 1 call (UtilityAgent)
- **Explanation**: 1 call (ParticipantAgent)
- **Post-Explanation Ranking**: 1 + 1 calls
- **Application Rounds**: 4 × (1 + 1) calls = 8 calls
- **Final Ranking**: 1 + 1 calls ⭐

**Total**: ~13 API calls per agent (with retries may increase)

### Parallel Execution
- All agents run Phase 1 simultaneously
- No dependencies between agents
- Total wall-clock time ≈ single agent time

---

## Memory Update Pattern

After each step:
```python
memory_update = {
    "step": "step_name",
    "round": round_number,
    "stage": ExperimentStage,
    "content": step_specific_data,
    "timestamp": current_time
}

agent.memory.append(memory_update)
```

Memory grows incrementally, preserving full Phase 1 journey for Phase 2 initialization.

---

## Error Handling

### Retry Mechanisms
- **Ranking parsing failures**: Retry with clarified prompt
- **Application choice parsing**: Retry with format examples
- **Timeout**: Configurable timeout per step (default 60s)

### Fallbacks
- **Invalid ranking**: Request re-ranking with validation rules
- **Malformed selection**: Provide format template and retry
- **Persistent failures**: Log error, mark agent as failed

---

## Data Flow Summary

```
Configuration → Phase1Manager → (Parallel for each agent)
    ↓
1. Initial Ranking → UtilityAgent → Memory
2. Explanation → Direct to Memory
3. Post-Explanation Ranking → UtilityAgent → Memory
4. Application Rounds x4 → UtilityAgent → Payoff → Memory
5. Final Ranking → UtilityAgent → Memory ⭐
    ↓
Phase1Results (includes final_ranking) → Phase2Manager
```

---

## Critical Implementation Notes

### Final Ranking Step (Step 1.5) ⭐
**Previously Undocumented**

- **Location**: `phase1_manager.py:581-625`
- **Round Number**: 5
- **Stage**: `ExperimentStage.FINAL_RANKING`
- **Purpose**: Capture preference after experiencing all 4 application rounds
- **Output**: Included in `Phase1Results.final_ranking`

This step is **critical** for understanding preference evolution:
- Initial ranking: Pure veil of ignorance
- Post-explanation: After learning
- Final ranking: After concrete experience ⭐

---

## Related Files

- `core/phase1_manager.py` - Complete orchestration (lines 199-625)
- `experiment_agents/participant_agent.py` - Agent implementation
- `experiment_agents/utility_agent.py` - Response parsing
- `utils/language_manager.py` - Localization
- `utils/distribution_generator.py` - Distribution creation
- `models/phase1_models.py` - Data models

---

## Next Steps

- **Phase 2 Service Details**: [05_discussion_round_detailed.md](./05_discussion_round_detailed.md)
- **Data Models**: [07_data_models.md](./07_data_models.md)
- **High-Level Process**: [02_phase1_process_flow.md](./02_phase1_process_flow.md)
