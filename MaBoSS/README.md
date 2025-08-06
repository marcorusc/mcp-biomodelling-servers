# MaBoSS MCP Server
## Model Context Protocol Integration for Boolean Network Simulation

This is a 3. **Calls**: `run_simulation()`
4. **Calls**: `get_simulation_result()` to analyze dynamics
5. **Interprets results**: "The simulation shows bistable behavior with p53 activation leading to either DNA repair (oscillatory) or apoptosis (stable high state)"odel Context Protocol (MCP) server** that exposes MaBoSS Boolean network simulation capabilities to Large Language Models (LLMs). It enables LLMs to perform complex Boolean dynamics analysis through natural language interactions and tool chaining.

### What is an MCP Server?

Model Context Protocol (MCP) is a standardized way to connect LLMs with external tools and data sources. This server:

- **Exposes MaBoSS functionality** as callable tools for LLMs
- **Enables natural language interaction** with Boolean network simulations
- **Provides structured tool chaining** for complex workflows
- **Maintains session state** across multiple LLM interactions

### LLM Integration Patterns

#### 1. Direct Tool Invocation
LLMs can directly call MaBoSS functions through natural language:

**User Prompt**: *"Convert the BNET file to simulation format and run a Boolean dynamics analysis"*

**LLM Tool Chain**:
```
1. bnet_to_bnd_and_cfg("Network.bnet")
2. build_simulation("output.bnd", "output.cfg") 
3. run_simulation()
4. get_simulation_result()
```

#### 2. Exploratory Analysis Workflows
**User Prompt**: *"I have a cancer gene network. Help me understand the steady-state behavior and pathway dynamics."*

**LLM Response Pattern**:
- Identifies available BNET files
- Converts to simulation format
- Runs trajectory analysis
- Interprets biological significance of attractors
- Suggests parameter variations for exploration

#### 3. Cross-Server Coordination
**User Prompt**: *"Build a Boolean network from TP53, MYC, and BAX genes, then simulate the dynamics"*

**Multi-Server Tool Chain**:
```
NeKo Server:    create_network(['TP53', 'MYC', 'BAX']) → export_network('bnet')
MaBoSS Server:  bnet_to_bnd_and_cfg() → run_simulation() → analyze_results()
```

### Tool Categories Exposed

#### File Conversion Tools
- `bnet_to_bnd_and_cfg()` - Convert network definitions to simulation format
- Enable seamless integration with NeKo network construction

#### Simulation Management
- `build_simulation()` - Initialize Boolean dynamics engine
- `run_simulation()` - Execute stochastic simulations
- `update_maboss_parameters()` - Configure simulation parameters

#### Analysis Tools  
- `get_simulation_result()` - Extract probability trajectory results
- `simulate_mutation()` - Test gene knockout/overexpression effects
- `visualize_network_trajectories()` - Generate trajectory plots

### Prompt Engineering Patterns

#### Pattern 1: Workflow Automation
```
"Analyze the Boolean dynamics of [gene list] under [conditions]"
→ LLM chains: file_conversion → simulation → analysis → interpretation
```

#### Pattern 2: Parameter Exploration
```
"Test different time scales for this Boolean network"
→ LLM iterates: parameter_variation → simulation → comparison → optimization
```

#### Pattern 3: Biological Interpretation
```
"What do these Boolean attractors mean for cell fate decisions?"
→ LLM combines: simulation_results → biological_knowledge → pathway_analysis
```

### Integration Benefits for LLMs

1. **Computational Access**: LLMs gain ability to perform quantitative Boolean simulations
2. **Workflow Automation**: Complex multi-step analyses become single natural language requests
3. **Cross-Domain Integration**: Combine network construction (NeKo) with simulation (MaBoSS) and multicellular modeling (PhysiCell)
4. **State Persistence**: Maintain simulation context across conversation turns
5. **Error Recovery**: Built-in guidance for parameter adjustment and troubleshooting

### Example LLM Conversation Flow

**User**: *"I need to simulate the p53 pathway under DNA damage conditions"*

**LLM with MCP Access**:
1. **Identifies need for Boolean network simulation**
2. **Calls**: `bnet_to_bnd_and_cfg("p53_network.bnet")`
3. **Calls**: `build_simulation("output.bnd", "output.cfg")`
4. **Calls**: `run_simulation()`
5. **Calls**: `get_probtraj()` to analyze dynamics
6. **Interprets results**: "The simulation shows bistable behavior with p53 activation leading to either DNA repair (oscillatory) or apoptosis (stable high state)"

### Technical Architecture

- **Protocol**: Model Context Protocol (MCP) standard
- **Interface**: JSON-RPC tool calling
- **State Management**: Session-based simulation context
- **Error Handling**: Structured error responses with recovery suggestions
- **Integration**: Seamless connection with NeKo and PhysiCell MCP servers

### Getting Started

1. **Install MCP client** in your LLM environment
2. **Connect to MaBoSS MCP server** endpoint
3. **Use natural language** to request Boolean network analysis
4. **Chain tools** for complex workflows

**Learn More About MaBoSS**: [MaBoSS Official Documentation](https://maboss.curie.fr/)

This MCP server transforms MaBoSS from a command-line tool into an **LLM-accessible computational engine** for Boolean network analysis, enabling natural language-driven systems biology research.
