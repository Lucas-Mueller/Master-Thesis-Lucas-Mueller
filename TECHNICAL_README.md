# Technical Documentation: Frohlich Experiment Framework

## Executive Summary

This framework implements a computational experiment system for studying distributive justice through AI agents. Based on Frohlich & Oppenheimer's empirical work (1992) and Rawlsian philosophy, it simulates agents making collective decisions about justice principles under a "veil of ignorance."

**Core Capabilities**:
- Two-phase experiment orchestration (individual → group)
- Services-based architecture for Phase 2
- Multilingual support (English, Spanish, Mandarin)
- Multi-provider LLM integration (OpenAI, Gemini, OpenRouter, Ollama)
- Formal voting system with consensus detection
- Sophisticated memory management and counterfactual analysis

---

## Documentation Structure

This technical documentation is organized into **2 layers**:

### Layer 1: High-Level Process Flows
*What happens and when*

- **[Overview](docs/diagrams/01_experiment_complete_overview.md)**: Complete two-phase experiment structure
- **[Phase 1 Process](docs/diagrams/02_phase1_process_flow.md)**: Individual deliberation (5 steps including final ranking)
- **[Phase 2 Process](docs/diagrams/03_phase2_process_flow.md)**: Group discussion with post-round memory updates

### Layer 2: Detailed Service Interactions
*How it works at the service level*

- **[Phase 1 Services](docs/diagrams/04_phase1_service_sequence.md)**: Service-level interactions during Phase 1
- **[Discussion Round Details](docs/diagrams/05_discussion_round_detailed.md)**: Complete discussion workflow with memory management
- **[Voting Process Details](docs/diagrams/06_voting_detailed.md)**: Four-phase voting with multilingual support
- **[Data Models](docs/diagrams/07_data_models.md)**: Core data structures and relationships

**Recommended Reading Paths**:
- **Researchers/Non-Technical**: Layer 1 diagrams (01, 02, 03)
- **New Developers**: 01 → 02 → 03 → 04
- **Debugging/Implementation**: Navigate directly to relevant Layer 2 diagram

---

# PART 1: Overview (High-Level Process)

*Audience: Researchers, managers, new developers*

---

## 1.1 Experiment Structure

The framework implements a two-phase experimental protocol:

### Phase 1: Individual Deliberation (Parallel)
**Duration**: ~5-10 API calls per agent
**Execution**: All agents run independently in parallel

Each agent:
1. **Initial Ranking**: Ranks 4 justice principles without knowledge
2. **Principle Explanation**: Learns principles with payoff examples
3. **Post-Explanation Ranking**: Re-ranks after understanding
4. **Application Rounds** (x4): Selects principle/distribution, receives payoffs
5. **Final Ranking** ⭐: Final preference after experiencing outcomes

**Key Feature**: Complete isolation—no inter-agent communication

**See**: [Phase 1 Process Flow](docs/diagrams/02_phase1_process_flow.md)

---

### Phase 2: Group Discussion (Sequential)
**Duration**: 5-20 rounds typical, ~30-100 API calls
**Execution**: Agents interact sequentially in discussion rounds

Group process:
1. **Discussion Rounds**: Agents speak in rotating order, statements added to public history
2. **Post-Round Memory Updates** ⭐: After all speakers finish, symmetric memory updates
3. **Voting Attempts**: End-of-round voting (initiation → confirmation → ballot)
4. **Consensus Detection**: Unanimous agreement required on principle + constraints
5. **Payoff Calculation**: Apply chosen principle, calculate counterfactuals
6. **Final Rankings**: Collect post-results preferences

**Key Feature**: Post-round memory updates ensure all agents have complete, consistent round information before voting

**See**: [Phase 2 Process Flow](docs/diagrams/03_phase2_process_flow.md)

---

## 1.2 Four Justice Principles

The framework operationalizes four distinct distributive justice approaches:

### Principle 1: Maximizing Floor (Rawlsian)
- **Strategy**: Maximize the lowest income in distribution
- **Philosophy**: Prioritize worst-off members (Rawls' difference principle)
- **Configuration**: No parameters required

### Principle 2: Maximizing Average (Utilitarian)
- **Strategy**: Maximize average income across all classes
- **Philosophy**: Total welfare maximization
- **Configuration**: No parameters required

### Principle 3: Floor Constraint (Hybrid)
- **Strategy**: Maximize average with minimum floor guarantee
- **Example**: "Maximize average but ensure minimum $15,000"
- **Configuration**: Requires floor amount (positive integer)

### Principle 4: Range Constraint (Hybrid)
- **Strategy**: Maximize average with limited inequality
- **Example**: "Maximize average but limit spread to $10,000"
- **Configuration**: Requires range limit (positive integer)

**Trade-offs**: Each principle produces different payoffs for different income classes, creating meaningful bargaining dynamics.

**Code**: `models/justice_principles.py`

---

## 1.3 Income Distribution System

### Five Income Classes
Based on Frohlich & Oppenheimer's original design:

- **Class 1** (High): $32,000 baseline
- **Class 2** (Medium-High): $27,000 baseline
- **Class 3** (Medium, modal): $24,000 baseline - *50% probability*
- **Class 4** (Medium-Low): $13,000 baseline
- **Class 5** (Low): $12,000 baseline

### Distribution Generation Modes

**Dynamic Mode**:
- Applies random scaling (0.8-1.2× multipliers) to baseline
- Seeded RNG ensures reproducibility

**Original Values Mode**:
- Uses historical distributions from Frohlich's empirical data
- Maps rounds to specific situations (A, B, C, D)

### Payoff Calculation
Uses **weighted expected value**: Sum of (income × probability) for each class.

**Code**: `utils/distribution_generator.py`

---

## 1.4 Key Results & Outputs

### Phase1Results
Complete record of individual agent experience:
- `initial_ranking`: Uninformed preferences
- `post_explanation_ranking`: After learning
- `application_results`: 4 rounds of choices and payoffs
- `final_ranking` ⭐: Experience-based preferences
- `total_earnings`: Cumulative application payoffs
- `final_memory_state`: Complete memory for Phase 2

### Phase2Results
Complete record of group process:
- `consensus_reached`: Boolean outcome
- `chosen_principle`: Selected principle (if consensus)
- `constraint_amount`: Floor/range value (if applicable)
- `discussion_history`: All statements with metadata
- `voting_records`: Complete voting attempts
- `payoffs`: Actual and counterfactual payoffs per agent
- `final_rankings`: Post-results preferences
- `rounds_completed`: Number of discussion rounds

**Code**: `models/phase1_models.py`, `models/phase2_models.py`

---

# PART 2: Implementation Details (Service-Level)

*Audience: Developers implementing features, debugging*

---

## 2.1 Services Architecture

Phase 2 uses a **services-first architecture** where specialized services handle specific responsibilities:

### Core Services

**SpeakingOrderService** (`core/services/speaking_order_service.py`)
- **Responsibility**: Turn allocation with finisher restrictions
- **Key Method**: `determine_speaking_order()` - randomizes with finisher limits

**DiscussionService** (`core/services/discussion_service.py`)
- **Responsibility**: Discussion prompts, statement validation, history management
- **Key Methods**:
  - `build_discussion_prompt()` - contextual prompts
  - `validate_statement()` - length/content validation
  - `manage_discussion_history_length()` - intelligent truncation
- **Configuration**: `Phase2Settings.public_history_max_length` (default: 100,000 chars)

**VotingService** (`core/services/voting_service.py`)
- **Responsibility**: Complete voting process coordination
- **Key Methods**:
  - `initiate_voting()` - end-of-round voting check
  - `coordinate_voting_confirmation()` - unanimous confirmation phase
  - `coordinate_secret_ballot()` - two-stage voting
  - `detect_consensus()` - unanimous agreement detection

**MemoryService** (`core/services/memory_service.py`)
- **Responsibility**: Unified memory management with guidance styles
- **Key Methods**:
  - `update_discussion_memory()` - post-round updates
  - `update_voting_memory()` - voting event updates
  - `update_results_memory()` - final results
- **Features**: Character limits, truncation, narrative/structured styles

**CounterfactualsService** (`core/services/counterfactuals_service.py`)
- **Responsibility**: Payoff calculations and results formatting
- **Key Methods**:
  - `calculate_payoffs()` - actual and counterfactual payoffs
  - `format_detailed_results()` - results presentation
  - `collect_final_rankings()` - post-results preferences

### Architecture Pattern

**Phase2Manager** acts as orchestrator:
```
Phase2Manager (Orchestrator)
    ↓
SpeakingOrderService → Discussion Phase
DiscussionService → Prompt Generation & Validation
MemoryService → Post-Round Memory Updates ⭐
VotingService → Voting Coordination
CounterfactualsService → Payoff Analysis
```

**Benefits**:
- Clean separation of concerns
- Protocol-based dependency injection
- Easy testing in isolation
- Configurable behavior via `Phase2Settings`

**See**: [Discussion Details](docs/diagrams/05_discussion_round_detailed.md), [Voting Details](docs/diagrams/06_voting_detailed.md)

---

## 2.2 Phase 1 Services

### Phase1Manager Orchestration

**File**: `core/phase1_manager.py`

**Responsibilities**:
- Execute all 5 Phase 1 steps for each agent
- Coordinate with ParticipantAgent and UtilityAgent
- Manage memory updates after each step
- Return complete Phase1Results

### Five-Step Process

**Step 1.1: Initial Ranking** (Lines 199-268)
- Prompt agent to rank 4 principles without explanation
- UtilityAgent parses ranking text → structured list
- Update memory with initial preferences

**Step 1.2: Principle Explanation** (Lines 271-329)
- Load localized explanations from `translations/{language}/`
- Agent reads detailed explanations with payoff examples
- Update memory with acquired knowledge

**Step 1.3: Post-Explanation Ranking** (Lines 332-401)
- Prompt agent to re-rank after understanding
- UtilityAgent parses ranking
- Update memory with informed preferences

**Step 1.4: Application Rounds** (Lines 404-578)
- For each of 4 rounds:
  1. Generate 4 distributions (DistributionGenerator)
  2. Agent selects principle and distribution
  3. UtilityAgent parses selection
  4. Calculate payoff for agent's income class
  5. Update running earnings total
  6. Update memory with outcome

**Step 1.5: Final Ranking** ⭐ (Lines 581-625)
- Prompt agent to rank after experiencing outcomes
- UtilityAgent parses ranking
- Update memory with experience-based preferences
- **Most informed ranking**: combines learning AND application experience

**Critical Note**: Step 1.5 was previously undocumented but exists in implementation.

### Supporting Services

**ParticipantAgent** (`experiment_agents/participant_agent.py`)
- Receives localized prompts
- Generates rankings and selections
- Maintains internal reasoning state

**UtilityAgent** (`experiment_agents/utility_agent.py`)
- Parses ranking text → `List[JusticePrinciple]`
- Parses selections → `{principle, distribution_index}`
- Handles multilingual responses

**LanguageManager** (`utils/language_manager.py`)
- Loads localized prompts from `translations/`
- Provides consistent terminology across languages

### API Call Budget
Per agent Phase 1 execution:
- Initial Ranking: 2 calls (agent + utility)
- Explanation: 1 call
- Post-Explanation Ranking: 2 calls
- Application Rounds: 8 calls (4 × 2)
- Final Ranking: 2 calls ⭐

**Total**: ~13 API calls per agent (may increase with retries)

**See**: [Phase 1 Service Sequence](docs/diagrams/04_phase1_service_sequence.md)

---

## 2.3 Phase 2 Services

### Discussion Round Flow

**File**: `core/phase2_manager.py` (Lines 677-900)

**Complete Round Sequence**:

1. **Speaking Order** (SpeakingOrderService)
   - Randomize speaking order for fairness
   - Apply finisher restrictions (prevent same agent always finishing)

2. **Discussion Phase** (For each speaker)
   - Build contextual prompt (DiscussionService)
     - Agent's Phase 1 results
     - Public discussion history
     - Group composition
     - Round number and guidelines
   - Agent generates statement
   - Validate statement (min length, format)
   - Append to public history
   - **No memory update yet** (deferred to post-round)

3. **Post-Round Memory Update Phase** ⭐ (Lines 873-900)
   - **Critical timing**: After ALL speakers finish, before voting
   - For each participant:
     - Assemble complete round context
     - Format with guidance style (narrative/structured)
     - Apply character limits and truncation
     - Update agent memory
   - **Why post-round?**: Ensures all agents have complete, symmetric information before voting decisions

4. **Countdown Message** (If 2 rounds remaining)
   - Insert "⏰ 2 rounds remaining" message
   - Encourages convergence

5. **Voting Attempt** (VotingService)
   - Covered in detail below

**Configuration**: `Phase2Settings` controls discussion behavior (history limits, validation rules, memory style)

**See**: [Discussion Round Detailed](docs/diagrams/05_discussion_round_detailed.md)

---

### Voting Process Flow

**File**: `core/services/voting_service.py`

**Four-Phase Voting**:

### Phase 1: Initiation (Sequential)
- Ask each agent: "Do you want to initiate voting? (1=Yes/0=No)"
- Extract numerical response (1 or 0)
- Early exit after first "Yes"
- Update state: `voting_in_progress = True`

**Why sequential?** Prevents groupthink, respects turn order

### Phase 2: Confirmation (Parallel)
- Ask all agents simultaneously: "Do you confirm participation? (1=Yes/0=No)"
- Unanimous requirement: ALL must say Yes (1)
- Any No (0) aborts voting → return to discussion

**Why unanimous?** Ethical requirement for voluntary participation

### Phase 3: Secret Ballot (Two-Stage)

**Stage 1: Principle Selection**
- Parallel ballot: "Which principle? (1=Floor, 2=Avg, 3=Avg+Floor, 4=Avg+Range)"
- Regex extraction: `\b([1-4])\b` pattern
- Fallback: Keyword matching in agent's language
- Examples:
  - "I vote for principle 1" → Extracted: 1
  - "My choice is 3" → Extracted: 3
  - "I prefer floor" → Fallback: keyword "floor" → 1

**Stage 2: Amount Specification** (Principles 3 & 4 only)
- Filter agents who voted 3 or 4
- Parallel ballot: "What constraint amount?"
- Regex extraction with cultural number format support
- Examples:
  - English: "15,000" → 15000
  - Spanish: "15.000" → 15000
  - Mandarin: "1万5千" → 15000

**Implementation**: `core/two_stage_voting_manager.py`, `utils/cultural_adaptation.py`

### Phase 4: Consensus Detection
- **Unanimous agreement required**: All votes identical
- Check: `all(vote == first_vote for vote in votes)`
- Consensus → Exit discussion loop
- No consensus → Continue to next round

**Consensus Examples**:
- ✅ All vote (1, None) → Consensus
- ✅ All vote (3, 15000) → Consensus
- ❌ Votes (1, None), (2, None), (1, None) → No consensus
- ❌ Votes (3, 15000), (3, 20000), (3, 15000) → No consensus

### Memory Updates (Post-Voting)

Three event types tracked:
1. **VOTING_INITIATED**: "Group considering voting"
2. **VOTING_CONFIRMED**: "All agents confirmed"
3. **BALLOT_COMPLETED**: "Secret ballot completed, consensus [reached/not reached]"

**Timing**: ~1.5-2.5 minutes per voting attempt (5 agents)

**See**: [Voting Process Detailed](docs/diagrams/06_voting_detailed.md)

---

### Memory Management

**File**: `core/services/memory_service.py`

**Architecture**:
- **MemoryService**: Public API (routing and coordination)
- **SelectiveMemoryManager**: Implementation (truncation and formatting)

**Key Features**:

**1. Event Routing**
- Discussion updates → Simple insertion
- Voting updates → Simple (initiation/confirmation) or complex (ballot with consensus)
- Results updates → Always complex (detailed analysis)

**2. Guidance Styles**
- **Narrative**: Natural language, story-like summaries
- **Structured**: Bullet points, clear sections

**3. Content Truncation**
- Statement limit: 300 characters
- Reasoning limit: 200 characters
- Public history: Configurable (default 100,000 chars)
- Strategy: Preserve recent content, summarize or remove older

**4. Consistency**
- Post-round updates ensure symmetric information
- All agents see complete round before voting
- Prevents information asymmetry during discussion

**Configuration**: `Phase2Settings.memory_guidance_style`, character limits

---

## 2.4 Data Models & Configuration

### Core Data Structures

**Phase1Results**
```python
{
    agent_id: str
    initial_ranking: List[JusticePrinciple]           # Step 1.1
    post_explanation_ranking: List[JusticePrinciple]  # Step 1.3
    application_results: List[ApplicationRoundResult] # Step 1.4 (x4)
    final_ranking: List[JusticePrinciple]             # Step 1.5 ⭐
    total_earnings: int
    final_memory_state: List[Dict]
}
```

**Phase2Results**
```python
{
    consensus_reached: bool
    chosen_principle: Optional[JusticePrinciple]
    constraint_amount: Optional[int]
    discussion_history: List[Dict]
    voting_records: List[VotingRecord]
    payoffs: Dict[str, PayoffResult]
    final_rankings: Dict[str, List[JusticePrinciple]]
    rounds_completed: int
}
```

**JusticePrinciple** (Enumeration)
- MAXIMIZING_FLOOR (1)
- MAXIMIZING_AVERAGE (2)
- FLOOR_CONSTRAINT (3)
- RANGE_CONSTRAINT (4)

**See**: [Data Models](docs/diagrams/07_data_models.md) for complete reference

---

### Configuration System

**YAML-Based Configuration** (`config/models.py`)

**Key Components**:

**ExperimentConfiguration**
- `experiment_id`: Unique identifier
- `num_agents`: Group size
- `agents`: List of AgentConfig (model, language, temperature, income class)
- `phase1_settings`: Phase 1 behavior
- `phase2_settings`: Phase 2 behavior (most important)
- `random_seed`: For reproducibility

**Phase2Settings** (`config/phase2_settings.py`)
```yaml
max_rounds: 10                        # Maximum discussion rounds
public_history_max_length: 100000     # Character limit
statement_min_length: 10              # Validation
statement_timeout: 60                 # Seconds
statement_retry_attempts: 3

memory_guidance_style: "narrative"    # or "structured"
memory_compression_threshold: 0.9

voting_initiation_timeout: 30         # Seconds
voting_confirmation_timeout: 30
voting_secret_ballot_timeout: 45
voting_retry_limit: 3
voting_retry_backoff_factor: 1.5
```

**AgentConfig**
- `model`: LLM model identifier
- `provider`: Auto-detected from model string
- `language`: en/es/zh
- `temperature`: 0.0-2.0
- `income_class`: 1-5
- `enable_internal_reasoning`: Boolean

---

### Multi-Provider Model Support

**Intelligent Provider Detection** (`experiment_agents/participant_agent.py`)

**Detection Rules**:
- `gpt-*`, `o1-*`, `o3-*` → OpenAI provider
- `gemini-*`, `gemma-*` → Google Gemini provider
- `provider/model` format (contains `/`) → OpenRouter provider
- `ollama/model` prefix → Ollama local provider

**Provider Configuration**:

**OpenAI**: Native Agents SDK
```python
client = Agents(api_key=os.getenv("OPENAI_API_KEY"))
```

**Gemini**: Native Google API
```python
client = anthropic.Anthropic(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/openai/"
)
```

**OpenRouter**: Universal proxy
```python
client = anthropic.Anthropic(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
```

**Ollama**: Local models
```python
client = anthropic.Anthropic(
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
)
```

**Usage**: Specify model in config YAML, provider auto-detected

---

## 2.5 Testing & Development

### Test Architecture

**Intelligent test acceleration system** with layered execution:

### Test Modes

**Ultra-Fast Mode** (`--mode=ultra_fast`)
- Unit tests only
- Duration: ~7 seconds
- API calls: 0
- Use: Daily development feedback

**Development Mode** (`--mode=dev`)
- Unit + component tests
- Duration: ~5 minutes
- API calls: Minimal
- Use: Pre-commit validation

**CI Mode** (`--mode=ci`)
- Comprehensive validation
- Duration: ~15 minutes
- API calls: Moderate
- Use: CI/CD pipelines

**Full Mode** (`--mode=full`)
- Complete validation
- Duration: ~30-45 minutes
- API calls: All
- Use: Release validation

### Test Layers

**Unit Tests** (`tests/unit/`)
- Component-level testing
- Protocol-based service testing
- No external dependencies
- Fast execution (~7 seconds)

**Fast Tests** (`tests/unit/test_fast_*`)
- Ultra-fast service boundary testing
- 43 tests in 0.04 seconds
- Multilingual response parsing with deterministic data
- Data flow validation with synthetic data

**Component Tests** (`tests/component/`)
- Mid-level integration
- Requires API key for LLM interactions
- Multilingual coverage (English, Spanish, Mandarin)

**Integration Tests** (`tests/integration/`)
- Cross-component testing
- End-to-end Phase 2 workflows
- Service interaction validation

**Snapshot Tests** (`tests/snapshots/`)
- Regression testing
- Golden snapshot comparisons

### Running Tests

```bash
# Ultra-fast development feedback
pytest --mode=ultra_fast

# Pre-commit validation
pytest --mode=dev

# CI/CD comprehensive testing
pytest --mode=ci

# Full release validation
pytest --mode=full

# Targeted execution
pytest tests/unit/test_fast_*                    # Fast tests
pytest tests/component -m "component and not live"  # Offline components
pytest tests/integration -m "live"               # Live integration

# With coverage
pytest --mode=ci --cov=. --cov-report=term-missing
```

### Environment Control

```bash
# Development mode (skips expensive tests)
DEVELOPMENT_MODE=1 pytest --mode=dev

# Force comprehensive testing
pytest --mode=ci --run-live

# Skip expensive tests even with API keys
pytest --mode=ci --skip-expensive

# Custom configuration
TEST_CONFIG_OVERRIDE=config/test_ultra_fast.yaml pytest
```

### Performance Improvements

- **Ultra-fast mode**: 99.3% improvement (7.6s vs 90-120 min)
- **Fast test suite**: 99.97% improvement (0.04s for 43 tests)
- **Development workflow**: 95% improvement (5min vs 90-120 min)
- **CI/CD pipeline**: 85% improvement (15min vs 90-120 min)

---

## Running Experiments

### Basic Usage

```bash
# Run with configuration file (required)
python main.py config/fast.yaml

# Specify output path
python main.py config/fast.yaml results/my_experiment.json
```

### Sample Configurations

**Quick Testing**: `config/fast.yaml`
- 5 rounds, minimal setup
- Recommended starting point

**Local Models**: `config/sample_ollama_gemma3.yaml`
- Ollama integration example

**Gemini Models**: `config/test_gemini.yaml`
- Google Gemini API example

**Mixed Providers**: `config/test_mixed_providers.yaml`
- Multiple providers in single experiment

### Batch Execution

**Hypothesis Testing Framework** (`hypothesis_testing/`)
- Organized experimental conditions
- Batch execution utilities (`utils_hypothesis_testing/runner.py`)
- Example conditions:
  - `hypothesis_1/`: 33 experimental conditions
  - `hypothesis_2/`: Cultural variations
  - `hypothesis_3/`: Income inequality variations

---

## Key Implementation Notes

### Critical Accuracy Points

**1. Phase 1 Final Ranking** ⭐
- **Location**: `phase1_manager.py:581-625`
- **Round Number**: 5
- **Stage**: `ExperimentStage.FINAL_RANKING`
- **Included in**: `Phase1Results.final_ranking`
- **Purpose**: Captures preference after experiencing all application outcomes
- **Previously**: Undocumented but implemented

**2. Phase 2 Post-Round Memory Updates** ⭐
- **Location**: `phase2_manager.py:873-900`
- **Timing**: After ALL speakers finish, before voting
- **Purpose**: Ensure symmetric information state for fair voting
- **Previously**: Not clearly documented

**3. Voting Consensus**
- **Requirement**: Unanimous agreement (100%)
- **Scope**: Both principle AND constraint amount must match
- **No partial consensus**: Any difference prevents consensus

### Reproducibility

**Seeding Strategy**:
- Experiment seed set in configuration
- Each agent receives: `seed = base_seed + agent_index`
- Ensures deterministic randomization across runs

**Configuration Files**:
- YAML-based, Pydantic-validated
- Version control for reproducibility
- Explicit parameter specification

---

## Related Documentation

### Code Files

**Core Orchestration**:
- `core/experiment_manager.py` - Complete experiment orchestration
- `core/phase1_manager.py` - Phase 1 implementation (Lines 199-625)
- `core/phase2_manager.py` - Phase 2 implementation (Lines 677-900)

**Phase 2 Services**:
- `core/services/speaking_order_service.py` - Turn management
- `core/services/discussion_service.py` - Discussion coordination
- `core/services/voting_service.py` - Voting orchestration
- `core/services/memory_service.py` - Memory management
- `core/services/counterfactuals_service.py` - Payoff calculations

**Agents**:
- `experiment_agents/participant_agent.py` - AI agents
- `experiment_agents/utility_agent.py` - Response parsing

**Models**:
- `models/phase1_models.py` - Phase 1 data structures
- `models/phase2_models.py` - Phase 2 data structures
- `models/justice_principles.py` - Justice principle enumeration
- `models/voting_types.py` - Voting data structures

**Configuration**:
- `config/models.py` - Experiment configuration
- `config/phase2_settings.py` - Phase 2 behavior settings

### Visual Documentation

**Layer 1: High-Level Process Flows**
- [01: Experiment Overview](docs/diagrams/01_experiment_complete_overview.md)
- [02: Phase 1 Process](docs/diagrams/02_phase1_process_flow.md)
- [03: Phase 2 Process](docs/diagrams/03_phase2_process_flow.md)

**Layer 2: Detailed Service Interactions**
- [04: Phase 1 Services](docs/diagrams/04_phase1_service_sequence.md)
- [05: Discussion Round](docs/diagrams/05_discussion_round_detailed.md)
- [06: Voting Process](docs/diagrams/06_voting_detailed.md)
- [07: Data Models](docs/diagrams/07_data_models.md)

---

## Acknowledgments

This framework implements computational experiments inspired by:
- **Frohlich, N., & Oppenheimer, J. A. (1992)**. *Choosing Justice: An Experimental Approach to Ethical Theory*. University of California Press.
- **Rawls, J. (1971)**. *A Theory of Justice*. Harvard University Press.

---

*Last Updated*: 2025-01-14
*Framework Version*: Master Thesis Lucas Mueller - Final Implementation
