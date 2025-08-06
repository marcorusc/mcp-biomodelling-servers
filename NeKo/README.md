# NeKo MCP Server
## Model Context Protocol Integration for Biological Network Construction

This is a **Model Context Protocol (MCP) server** that exposes NeKo network construction capabilities to Large Language Models (LLMs). It enables LLMs to build biological networks from gene lists through natural language interactions and seamless tool chaining.

### What is an MCP Server?

Model Context Protocol (MCP) is a standardized way to connect LLMs with external tools and data sources. This server:

- **Exposes NeKo functionality** as callable tools for LLMs
- **Enables natural language network construction** from gene lists
- **Provides automated pathway database access** through simple prompts
- **Supports complex workflow chaining** with downstream analysis tools

### LLM Integration Patterns

#### 1. Direct Network Construction
LLMs can build networks through natural language requests:

**User Prompt**: *"Build a cancer network from TP53, MYC, and VEGFA using pathway databases"*

**LLM Tool Chain**:
```
1. create_network(['TP53', 'MYC', 'VEGFA'], database='omnipath', max_len=2)
2. network_dimension()  # Check connectivity
3. export_network('bnet')  # Prepare for Boolean modeling
```

#### 2. Database-Driven Exploration
**User Prompt**: *"I need a comprehensive hypoxia response network. What genes should I include?"*

**LLM Response Pattern**:
- Suggests relevant gene sets (HIF1A, VEGFA, EPAS1, etc.)
- Constructs network using pathway databases
- Analyzes connectivity and suggests improvements
- Exports in appropriate format for downstream analysis

#### 3. Cross-Server Workflow Initiation
**User Prompt**: *"Create a Boolean model of cell cycle regulation and simulate its dynamics"*

**Multi-Server Tool Chain**:
```
NeKo Server:    create_network(cell_cycle_genes) → export_network('bnet')
MaBoSS Server:  bnet_to_bnd_and_cfg() → run_simulation()
PhysiCell:      add_physiboss_model() → multiscale_simulation()
```

### Tool Categories Exposed

#### Network Construction
- `create_network()` - Build networks from gene lists using pathway databases
- `find_paths()` - Discover connections between specific genes
- Automatic database selection and parameter optimization

#### Network Analysis
- `network_dimension()` - Get topology statistics and connectivity metrics
- Built-in validation for disconnected components and optimization suggestions

#### Export Functions
- `export_network()` - Generate SIF files for analysis or BNET files for Boolean modeling
- Seamless integration with MaBoSS and PhysiCell workflows

### Prompt Engineering Patterns

#### Pattern 1: Gene-to-Network Automation
```
"Build a [pathway/disease] network from these genes: [gene_list]"
→ LLM automatically: selects_database → constructs_network → validates_connectivity → exports
```

#### Pattern 2: Exploratory Network Building
```
"What would a comprehensive apoptosis network look like?"
→ LLM suggests: gene_sets → constructs_multiple_variants → compares_connectivity → recommends_best
```

#### Pattern 3: Workflow Pipeline Initialization
```
"I want to model how hypoxia affects cell fate decisions"
→ LLM chains: NeKo_network_construction → MaBoSS_boolean_simulation → PhysiCell_multiscale_modeling
```

### Integration Benefits for LLMs

1. **Database Access**: LLMs gain access to comprehensive pathway databases (OmniPath, SIGNOR, Reactome)
2. **Automatic Parameter Tuning**: Smart suggestions for path length, database selection, and filtering
3. **Error Recovery**: Built-in guidance for gene symbol correction and connectivity optimization
4. **Workflow Initiation**: Starting point for complex Boolean and multiscale modeling pipelines
5. **Biological Validation**: Automatic detection of disconnected networks with improvement suggestions

### Example LLM Conversation Flow

**User**: *"I'm studying drug resistance in cancer. Help me build a relevant network."*

**LLM with MCP Access**:
1. **Suggests gene set**: "I'll focus on key resistance genes: TP53, MDR1, MYC, PIK3CA, EGFR"
2. **Calls**: `create_network(['TP53', 'MDR1', 'MYC', 'PIK3CA', 'EGFR'], database='omnipath')`
3. **Calls**: `network_dimension()` to check connectivity
4. **Analyzes results**: "The network has 23 nodes and 45 edges, showing strong connectivity"
5. **Suggests next steps**: "Would you like me to export this for Boolean simulation or extend it with additional resistance pathways?"

### Error Recovery and Optimization

#### Gene Symbol Correction
- Automatic detection of invalid gene symbols
- Suggestions for correct HGNC nomenclature
- Common alias resolution (e.g., p53 → TP53)

#### Connectivity Optimization
- Database selection recommendations for sparse networks
- Parameter tuning suggestions (max_len, only_signed)
- Alternative gene suggestions for disconnected components

#### Workflow Guidance
- Export format recommendations based on intended downstream analysis
- Integration tips for MaBoSS Boolean modeling
- PhysiCell multiscale modeling preparation

### Technical Architecture

- **Protocol**: Model Context Protocol (MCP) standard
- **Interface**: JSON-RPC tool calling
- **Database Integration**: Real-time access to pathway databases
- **Error Handling**: Structured responses with actionable recovery suggestions
- **Cross-Server Coordination**: Seamless handoff to MaBoSS and PhysiCell servers

### Getting Started

1. **Install MCP client** in your LLM environment
2. **Connect to NeKo MCP server** endpoint
3. **Use natural language** to request network construction
4. **Chain with other MCP servers** for complete modeling workflows

**Learn More About NeKo**: [NeKo Official Documentation](https://github.com/sysbio-curie/neko)

This MCP server transforms NeKo from a command-line network construction tool into an **LLM-accessible biological database interface**, enabling natural language-driven network biology and systems modeling workflows.
