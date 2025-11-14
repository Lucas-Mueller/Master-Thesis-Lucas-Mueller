# Diagram 1.2: High-Level System Context

**Purpose**: Show how the Distributive Justice Experiment Framework interacts with external systems

**Target Audience**: Researchers, system architects, first-time readers

**Complexity Level**: Executive (5-7 boxes)

---

## System Context

```mermaid
graph TB
    User([Researcher/User])

    Framework[Distributive Justice Framework<br/>Python Application]

    OpenAI[OpenAI API<br/>gpt-4o, o1-mini, etc.]
    Gemini[Google Gemini API<br/>gemini-2.0-flash, etc.]
    OpenRouter[OpenRouter<br/>Universal LLM Gateway]
    Ollama[Ollama Local<br/>Local Models]

    Config[YAML Configuration<br/>Experiment Settings]
    Results[JSON Results<br/>Experiment Outputs]
    Transcripts[Transcript Logs<br/>Optional Debugging]

    User -->|Configures| Config
    Config -->|Loaded by| Framework

    Framework -->|API Calls| OpenAI
    Framework -->|API Calls| Gemini
    Framework -->|API Calls| OpenRouter
    Framework -->|API Calls| Ollama

    Framework -->|Generates| Results
    Framework -->|Optionally Logs| Transcripts

    Results -->|Reviewed by| User
    Transcripts -->|Debugged by| User

    style User fill:#E8E8E8,stroke:#333,stroke-width:2px
    style Framework fill:#4A90E2,stroke:#333,stroke-width:4px,color:#fff
    style OpenAI fill:#50E3C2,stroke:#333,stroke-width:2px
    style Gemini fill:#50E3C2,stroke:#333,stroke-width:2px
    style OpenRouter fill:#50E3C2,stroke:#333,stroke-width:2px
    style Ollama fill:#50E3C2,stroke:#333,stroke-width:2px
    style Config fill:#F5A623,stroke:#333,stroke-width:2px
    style Results fill:#7ED321,stroke:#333,stroke-width:2px
    style Transcripts fill:#BD10E0,stroke:#333,stroke-width:2px
```

---

## System Components

### Core Application: Frohlich Experiment Framework
- **Technology**: Python 3.11+, OpenAI Agents SDK
- **Purpose**: Orchestrates multi-agent distributive justice experiments
- **Key Features**: Two-phase experiments, voting systems, multilingual support

### External LLM Providers (Choose One or Mix)
1. **OpenAI API**: Native support for GPT models (`gpt-4o`, `o1-mini`, etc.)
2. **Google Gemini**: Native support for Gemini models (`gemini-2.0-flash`, etc.)
3. **OpenRouter**: Universal gateway for any LLM provider (`anthropic/claude-3.5-sonnet`, etc.)
4. **Ollama**: Local model execution (`ollama/gemma3:1b`, etc.)

### Configuration Inputs
- **YAML Files**: Experiment configuration (agents, settings, principles)
- **Environment Variables**: API keys, feature flags
- **Seed Values**: Reproducibility settings

### Experiment Outputs
- **JSON Results**: Complete experiment data (rankings, votes, payoffs, transcripts)
- **Transcript Logs**: Optional detailed interaction logs for debugging
- **Statistical Reports**: R-based analysis outputs (separate pipeline)

---

## Data Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Framework
    participant LLM as LLM Provider
    participant R as Results Store

    U->>F: 1. Provide YAML config + seed
    U->>F: 2. Set API keys (env vars)
    U->>F: 3. Run experiment

    loop Phase 1 & Phase 2
        F->>LLM: 4. Send agent prompts
        LLM->>F: 5. Return agent responses
    end

    F->>R: 6. Save JSON results
    F->>R: 7. Optionally save transcripts

    R->>U: 8. Results available for analysis
```

---

## Integration Points

| Component | Direction | Data Format | Purpose |
|-----------|-----------|-------------|---------|
| YAML Config | Input | YAML | Experiment parameters, agent settings |
| API Keys | Input | Environment Variables | Authentication for LLM providers |
| LLM API Calls | Bidirectional | JSON (OpenAI format) | Agent reasoning and responses |
| Results | Output | JSON | Complete experiment data |
| Transcripts | Output | JSON | Debugging logs (optional) |

---

## Environment Requirements

**Required**:
- Python 3.11+
- At least one API key: `OPENAI_API_KEY` OR `GEMINI_API_KEY` OR `OPENROUTER_API_KEY`
- Network connection (except for Ollama local models)

**Optional**:
- `OLLAMA_BASE_URL` for local model deployment
- R programming language for statistical analysis
- Mermaid CLI for diagram rendering

---

## Related Files

- `main.py` - Entry point for running experiments
- `config/models.py` - Pydantic models for configuration validation
- `experiment_agents/participant_agent.py` - LLM provider detection and client creation

---

## Next Steps

- **For framework internals**: See Diagram 2.1 (Phase 1 Architecture) and Diagram 2.2 (Phase 2 Services)
- **For configuration details**: See TECHNICAL_README.md Section 5
- **For model provider setup**: See GEMINI.md and CLAUDE.md
