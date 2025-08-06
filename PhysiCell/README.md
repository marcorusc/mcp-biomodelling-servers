# PhysiCell MCP Server
## Model Context Protocol Integration for Multiscale Biological Simulation

This is a **Model Context Protocol (MCP) server** that exposes PhysiCell multicellular simulation capabilities to Large Language Models (LLMs). It enables LLMs to construct sophisticated physics-based tissue simulations with integrated Boolean gene networks through natural language interactions.

### What is an MCP Server?

Model Context Protocol (MCP) is a standardized way to connect LLMs with external tools and data sources. This server:

- **Exposes PhysiCell functionality** as callable tools for LLMs
- **Enables natural language simulation construction** from biological scenarios
- **Provides PhysiBoSS integration** for multiscale gene-to-tissue modeling
- **Supports complex workflow orchestration** across multiple biological scales

### LLM Integration Patterns

#### 1. Scenario-Driven Simulation Building
LLMs can construct complete simulations from biological descriptions:

**User Prompt**: *"Create a simulation of breast cancer cells in a hypoxic 3D environment with immune cell infiltration"*

**LLM Tool Chain**:
```
1. analyze_biological_scenario("Breast cancer in hypoxic 3D tissue with immune infiltration")
2. create_simulation_domain(3000, 3000, 500, max_time=7200)
3. add_single_substrate("oxygen", 100000, 0.01, 38)
4. add_single_cell_type("cancer_cell", "Ki67_basic")
5. add_single_cell_type("immune_cell", "live_cell")
6. add_single_cell_rule("cancer_cell", "oxygen", "decreases", "apoptosis rate")
7. export_xml_configuration("tumor_simulation.xml")
```

#### 2. Multiscale Integration Workflows
**User Prompt**: *"Integrate the p53 Boolean network into cancer cell behavior and simulate tumor growth under drug treatment"*

**Cross-Server Tool Chain**:
```
NeKo:       create_network(['TP53', 'MDM2', 'ATM']) → export_network('bnet')
MaBoSS:     bnet_to_bnd_and_cfg() → test Boolean dynamics
PhysiCell:  add_physiboss_model() → link genes to behaviors → simulate
```

#### 3. Interactive Simulation Design
**User Prompt**: *"What cell types and environmental factors should I include for studying drug resistance?"*

**LLM Response Pattern**:
- Analyzes biological scenario
- Suggests appropriate cell types and substrates
- Recommends signal-behavior rules
- Guides PhysiBoSS integration for gene-level control
- Provides complete simulation configuration

### Tool Categories Exposed

#### Simulation Framework
- `create_simulation_domain()` - Define 3D spatial and temporal boundaries
- `add_single_substrate()` - Add chemical environments (oxygen, drugs, nutrients)
- Session management with progress tracking across complex workflows

#### Cell Population Definition
- `add_single_cell_type()` - Define cancer, immune, stromal cell populations
- `configure_cell_parameters()` - Set size, motility, death rates
- `set_substrate_interaction()` - Define consumption and secretion

#### Behavioral Programming
- `add_single_cell_rule()` - Create environmental sensing and response
- `list_all_available_signals()` and `list_all_available_behaviors()` - Discovery tools
- Context-aware signal/behavior expansion based on simulation components

#### PhysiBoSS Multiscale Integration
- `add_physiboss_model()` - Integrate Boolean networks into cell behavior
- `add_physiboss_input_link()` - Connect environment to gene regulation
- `add_physiboss_output_link()` - Connect gene states to cell phenotypes
- `apply_physiboss_mutation()` - Simulate genetic perturbations

### Prompt Engineering Patterns

#### Pattern 1: Complete Simulation from Description
```
"Simulate [disease/scenario] with [environmental conditions] and [cell types]"
→ LLM automatically: scenario_analysis → domain_setup → cell_definition → rule_programming → export
```

#### Pattern 2: Multiscale Model Construction
```
"Connect this Boolean network to cell behavior in a tissue simulation"
→ LLM chains: physiboss_integration → input_output_linking → parameter_tuning → validation
```

#### Pattern 3: Iterative Simulation Refinement
```
"Add drug treatment effects to my existing simulation"
→ LLM extends: add_drug_substrate → modify_behavioral_rules → update_cell_interactions
```

### Advanced PhysiBoSS Integration

#### Multiscale Architecture through MCP
The server enables LLMs to seamlessly connect molecular and cellular scales:

```
Gene Regulation (Boolean) ↔ Cell Behavior (PhysiCell) ↔ Tissue Dynamics (3D Physics)
        ↓                         ↓                        ↓
   Input: Environment        Output: Phenotype      Emergent: Population
   (oxygen, drugs)          (death, proliferation)   (growth, invasion)
```

#### LLM-Orchestrated Multiscale Workflows
**User Prompt**: *"Model how TP53 mutations affect tumor response to chemotherapy"*

**LLM Multiscale Tool Chain**:
1. **Network Level**: Construct TP53 regulatory network (NeKo)
2. **Boolean Level**: Simulate pathway dynamics (MaBoSS)  
3. **Cellular Level**: Link TP53 states to apoptosis/survival (PhysiCell PhysiBoSS)
4. **Tissue Level**: Simulate drug treatment effects on tumor population
5. **Analysis**: Compare wild-type vs mutant tumor responses

### Session Management for Complex Workflows

#### Multi-Session Orchestration
- `create_session()` - Isolated simulation environments
- `switch_session()` - Compare different scenarios
- `get_workflow_status()` - Track progress across complex builds

#### Workflow State Tracking
LLMs can monitor and guide users through simulation construction:
- Domain setup → Substrates → Cell types → Rules → PhysiBoSS → Export
- Progress percentages and next-step recommendations
- Error recovery with specific correction guidance

### Integration Benefits for LLMs

1. **Biological Scenario Translation**: Convert narrative descriptions into quantitative simulations
2. **Multiscale Coordination**: Orchestrate gene→cell→tissue modeling workflows
3. **Parameter Discovery**: Access to extensive signal/behavior libraries with context awareness
4. **Error Prevention**: Built-in validation and progress tracking
5. **Cross-Server Integration**: Seamless coordination with NeKo and MaBoSS servers

### Example LLM Conversation Flow

**User**: *"I want to study how hypoxia drives cancer cell invasion with p53 mutations."*

**LLM with MCP Access**:
1. **Scenario Analysis**: "I'll create a hypoxic tumor simulation with p53-controlled invasion"
2. **Domain Setup**: Creates 3D environment with oxygen gradients
3. **Cell Definition**: Adds cancer cells with motility and invasion capabilities  
4. **PhysiBoSS Integration**: 
   - Links oxygen levels to HIF1A activation
   - Connects p53 mutations to survival/invasion decisions
   - Programs hypoxia-induced migration behaviors
5. **Simulation Export**: Generates complete XML configuration
6. **Next Steps**: "The simulation is ready. Would you like me to add immune cells or test different p53 mutation scenarios?"

### Technical Architecture

- **Protocol**: Model Context Protocol (MCP) standard
- **Interface**: JSON-RPC tool calling with complex object handling
- **State Management**: Session-based simulation building with persistence
- **PhysiBoSS Integration**: Direct Boolean network coupling to cellular physics
- **Cross-Server Coordination**: Seamless file handoff from NeKo/MaBoSS workflows

### Getting Started

1. **Install MCP client** in your LLM environment
2. **Connect to PhysiCell MCP server** endpoint  
3. **Use natural language** to describe biological scenarios
4. **Chain with NeKo/MaBoSS** for complete gene→tissue modeling

**Learn More About PhysiCell**: [PhysiCell Official Documentation](http://physicell.org/)  
**Learn More About PhysiBoSS**: [PhysiBoSS Publication](https://doi.org/10.1093/bioinformatics/btz279)

This MCP server transforms PhysiCell from a complex simulation framework into an **LLM-accessible multiscale modeling platform**, enabling natural language-driven construction of sophisticated gene-to-tissue simulations.
