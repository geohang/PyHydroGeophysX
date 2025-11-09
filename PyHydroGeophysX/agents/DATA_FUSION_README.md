# Multi-Method Data Fusion Agents

## Overview

The PyHydroGeophysX data fusion system provides intelligent coordination of multiple geophysical methods through natural language interfaces. This system automatically combines complementary geophysical datasets to improve subsurface characterization with quantified uncertainty.

## New Agents

### 1. DataFusionAgent

**Purpose**: Intelligent coordinator for multi-method geophysical workflows

**Capabilities**:
- Understands which geophysical methods work well together
- Automatically creates execution plans for complex workflows
- Recommends optimal fusion patterns based on available data
- Extensible architecture for future geophysical methods

**Supported Fusion Patterns**:

| Pattern | Methods | Description | Benefits |
|---------|---------|-------------|----------|
| `structure_constraint` | Seismic + ERT | Use velocity interfaces to constrain resistivity inversion | Sharper layer boundaries, reduced artifacts |
| `petrophysics_integration` | ERT + Petrophysics | Convert resistivity to water content | Direct hydrological interpretation |
| `full_integration` | Seismic + ERT + Petrophysics | Complete geological-to-hydrological workflow | Maximum constraint, quantified uncertainty |

**Example Usage**:

```python
from PyHydroGeophysX.agents import DataFusionAgent

# Initialize agent
fusion_agent = DataFusionAgent(api_key=api_key, model='gpt-4.1-nano')

# Get available patterns
patterns = fusion_agent.get_available_patterns()

# Create execution plan
fusion_input = {
    'fusion_pattern': 'full_integration',  # or 'auto' for LLM recommendation
    'methods': ['seismic', 'ert', 'petrophysics'],
    'workflow_config': {
        'velocity_threshold': 1200,  # m/s
        'seismic_params': {'lam': 50, 'zWeight': 0.2},
        'ert_params': {'lambda': 10, 'max_iterations': 20},
        'petrophysics_params': {'n_realizations': 100}
    },
    'output_dir': 'results/data_fusion'
}

plan = fusion_agent.execute(fusion_input)

# Plan contains step-by-step execution instructions
for step in plan['execution_plan']:
    print(f"{step['step']}: {step['description']}")
```

### 2. StructureConstraintAgent

**Purpose**: Apply seismic velocity interfaces as structural constraints to ERT inversion

**Capabilities**:
- Creates ERT meshes that honor geological boundaries from seismic data
- Runs structure-constrained resistivity inversion
- Preserves sharp layer contrasts while applying smoothing within layers
- Automatically adjusts regularization for constrained inversions

**Key Parameters**:
- `interface_coords`: (x, z) coordinates from seismic interface extraction
- `lambda`: Regularization (typically 5-20 for constrained, vs 50+ unconstrained)
- `mesh_quality`: Mesh refinement parameter (default: 31)

**Example Usage**:

```python
from PyHydroGeophysX.agents import StructureConstraintAgent

# Initialize agent
structure_agent = StructureConstraintAgent(api_key=api_key, model='gpt-4.1-nano')

# Prepare input
structure_input = {
    'ert_data': ertData,  # PyGIMLI ERT data object
    'interface_coords': (interface_x, interface_z),  # From seismic
    'inversion_params': {
        'lambda': 10.0,
        'max_iterations': 20,
        'limits': [1.0, 10000.0]  # Resistivity bounds (Ωm)
    },
    'output_dir': 'results/structure_constrained'
}

# Execute constrained inversion
results = structure_agent.execute(structure_input)

# Results include
resistivity_model = results['resistivity_model']
constrained_mesh = results['constrained_mesh']
coverage = results['coverage']
statistics = results['statistics']
```

**Advantages Over Unconstrained Inversion**:
- ✅ Sharp, geologically realistic layer boundaries
- ✅ Reduced regularization artifacts
- ✅ Better vertical resolution
- ✅ Incorporation of a priori structural information
- ✅ Lower regularization needed (10 vs 50+)

### 3. PetrophysicsAgent

**Purpose**: Convert resistivity to hydrological properties with uncertainty quantification

**Capabilities**:
- Converts resistivity to water content, saturation, and porosity
- Layer-specific petrophysical parameters (regolith vs bedrock)
- Monte Carlo uncertainty quantification
- Accounts for surface conductivity in clay-rich materials
- Statistical analysis and percentile calculations

**Petrophysical Models**:
- **Archie's Law**: For clean sands and fractured rock
- **Modified Archie's Law**: With surface conductivity for clay-rich materials
- **Layer-Specific Parameters**: Different m, n, porosity for each geological unit

**Example Usage**:

```python
from PyHydroGeophysX.agents import PetrophysicsAgent

# Initialize agent
petro_agent = PetrophysicsAgent(api_key=api_key, model='gpt-4.1-nano')

# Prepare input
petro_input = {
    'resistivity_model': resistivity_array,  # From ERT inversion
    'mesh': mesh,  # PyGIMLI mesh
    'cell_markers': cell_markers,  # Layer identifiers
    'n_realizations': 100,  # Monte Carlo samples
    'layer_params': {  # Optional - defaults provided
        3: {  # Layer marker 3 (regolith)
            'm': {'mean': 1.3, 'std': 0.1},
            'n': {'mean': 2.1, 'std': 0.1},
            'sigma_sur': {'mean': 0.005, 'std': 0.005},
            'porosity': {'mean': 0.42, 'std': 0.05},
            'rho_fluid': 20.0
        },
        2: {  # Layer marker 2 (bedrock)
            'm': {'mean': 1.9, 'std': 0.2},
            'n': {'mean': 1.7, 'std': 0.2},
            'sigma_sur': {'mean': 0.0, 'std': 0.0},
            'porosity': {'mean': 0.25, 'std': 0.15},
            'rho_fluid': 20.0
        }
    },
    'output_dir': 'results/petrophysics'
}

# Execute conversion
results = petro_agent.execute(petro_input)

# Results include
water_content_mean = results['water_content_mean']
water_content_std = results['water_content_std']
water_content_p10 = results['water_content_p10']  # 10th percentile
water_content_p90 = results['water_content_p90']  # 90th percentile
saturation_mean = results['saturation_mean']
```

**Output Statistics**:
- Mean water content and saturation
- Standard deviation (uncertainty)
- Percentiles (P10, P50, P90) for confidence bounds
- Layer-specific statistics
- Parameter distributions used in MC sampling

## Complete Workflow Example

See `examples/Ex_DataFusion_NaturalLanguage.ipynb` for a complete demonstration.

### Natural Language Input:

```python
user_request = """
I need to characterize subsurface water content using a multi-method approach:

1. First, use seismic refraction data to identify the boundary between regolith and bedrock.
   Use a velocity threshold of 1200 m/s to extract the interface.

2. Then, use this seismic structure to constrain ERT inversion.
   Apply moderate regularization (lambda=10) since we have structural constraints.

3. Finally, convert the resistivity model to water content.
   Use Monte Carlo uncertainty analysis with 100 realizations.
   Account for different petrophysical properties in regolith vs bedrock layers.
"""

workflow_config = context_agent.parse_request(user_request)
```

### Execution Flow:

```
Natural Language Request
         ↓
ContextInputAgent → Parse to structured config
         ↓
DataFusionAgent → Plan multi-method workflow
         ↓
    ┌────────────────────┐
    │ Seismic Inversion  │ → Velocity model
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Interface Extract  │ → (x, z) coordinates
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Constrained ERT    │ → Resistivity model
    │ StructureAgent     │    with sharp boundaries
    └────────────────────┘
         ↓
    ┌────────────────────┐
    │ Petrophysics       │ → Water content
    │ PetrophysicsAgent  │    with uncertainty
    └────────────────────┘
```

## Integration with Existing Agents

### SeismicAgent (Already Exists)

The existing `SeismicAgent` handles:
- Seismic travel time inversion
- Velocity model creation
- Interface extraction at specified thresholds
- LLM-powered interpretation

### ERTLoaderAgent (Already Exists)

Loads ERT data in various formats (E4D, BERT, etc.)

### Complete Agent Ecosystem:

```
ContextInputAgent ──────┐
                        ↓
                 DataFusionAgent (NEW - Coordinator)
                        ↓
         ┌──────────────┼──────────────┐
         ↓              ↓               ↓
    SeismicAgent    StructureAgent  PetrophysicsAgent
    (existing)      (NEW)           (NEW)
         ↓              ↓               ↓
    Velocity      Resistivity    Water Content
    Model         Model           + Uncertainty
```

## Extensibility for Future Methods

### Adding New Geophysical Methods:

The DataFusionAgent is designed to be easily extended:

1. **Create Specialized Agent**:
   ```python
   class GPRAgent(BaseAgent):
       """Agent for Ground Penetrating Radar"""
       def execute(self, input_data):
           # Process GPR data
           pass
   ```

2. **Define Fusion Pattern**:
   ```python
   FUSION_PATTERNS['gpr_ert'] = {
       'methods': ['gpr', 'ert'],
       'description': 'Combine EM and electrical methods',
       'workflow': ['gpr_processing', 'ert_inversion', 'joint_analysis'],
       'benefits': 'Complementary depth coverage and resolution'
   }
   ```

3. **Natural Language Automatically Adapts**:
   - No code changes needed for users
   - ContextInputAgent parses new method names
   - DataFusionAgent coordinates automatically

### Potential Future Combinations:

| Methods | Application | Benefits |
|---------|-------------|----------|
| GPR + ERT | Shallow subsurface | EM + electrical properties |
| Gravity + Seismic | Deep structure | Density constraints on velocity |
| Multi-temporal ERT | 4D monitoring | Temporal regularization |
| ERT + IP | Mineral exploration | Resistivity + chargeability |
| Seismic + GPR | Vadose zone | Complete velocity structure |

## Best Practices

### 1. Data Quality

- **Seismic**: Good spatial coverage, clear first arrivals
- **ERT**: Sufficient electrode spacing, appropriate array type
- **Both**: Overlapping survey areas for structural constraints

### 2. Parameter Selection

**Seismic Inversion**:
- `lam`: 20-100 (higher for noisy data)
- `zWeight`: 0.1-0.5 (balance horizontal/vertical smoothness)
- `vTop`: Expected shallow velocity (300-800 m/s)
- `vBottom`: Expected deep velocity (3000-6000 m/s)

**Structure-Constrained ERT**:
- `lambda`: 5-20 (lower than unconstrained due to structural info)
- Use higher `lambda` if structure constraints are uncertain

**Petrophysics**:
- `n_realizations`: 100+ for stable statistics
- Adjust parameter distributions based on local knowledge
- Regolith: Higher porosity (0.35-0.50), lower m (1.2-1.5)
- Bedrock: Lower porosity (0.15-0.30), higher m (1.8-2.2)

### 3. Uncertainty Interpretation

- **Low uncertainty**: Well-constrained parameters, good data coverage
- **High uncertainty**: Parameter variability, data gaps, structural complexity
- **Spatial patterns**: Higher uncertainty at boundaries and depth extremes

### 4. Validation

- Compare with direct measurements (cores, wells)
- Check physical plausibility of water content values
- Verify layer boundaries match geological expectations
- Assess coverage and resolution limits

## Troubleshooting

### Issue: Interface extraction fails

**Symptoms**: No interface points found, or scattered points

**Solutions**:
- Adjust `velocity_threshold` based on actual velocity model
- Increase `interval` parameter for smoother interface
- Check if velocity contrast is sufficient (>20%)

### Issue: High uncertainty in water content

**Symptoms**: Large standard deviations, wide percentile ranges

**Solutions**:
- Refine layer-specific parameter distributions
- Increase Monte Carlo realizations
- Check if cell markers correctly identify layers
- Verify resistivity model quality

### Issue: Unrealistic water content values

**Symptoms**: WC > porosity, or negative values

**Solutions**:
- Check resistivity model for extreme values
- Adjust petrophysical parameter distributions
- Verify `rho_fluid` is appropriate for site
- Check for saturation calculation issues

## Output Files

Each agent saves results automatically:

### StructureConstraintAgent:
```
results/structure_constrained/
  ├── resistivity_model.npy
  ├── coverage.npy
  ├── cell_markers.npy
  └── constrained_mesh.bms
```

### PetrophysicsAgent:
```
results/petrophysics/
  ├── water_content_mean.npy
  ├── water_content_std.npy
  ├── saturation_mean.npy
  └── saturation_std.npy
```

## References

### Structure-Constrained Inversion:
- Gallardo, L. A., & Meju, M. A. (2004). Joint two-dimensional DC resistivity and seismic travel time inversion with cross-gradients constraints. *JGR*, 109(B3).

### Petrophysical Models:
- Archie, G. E. (1942). The electrical resistivity log as an aid in determining some reservoir characteristics. *Petroleum Transactions of AIME*, 146, 54-62.
- Waxman, M. H., & Smits, L. J. M. (1968). Electrical conductivities in oil-bearing shaly sands. *SPE Journal*, 8(2), 107-122.

### Uncertainty Quantification:
- Hermans, T., et al. (2016). Uncertainty quantification of medium-term heat storage from short-term geophysical experiments using Bayesian Evidential Learning. *Water Resources Research*, 52(4), 2931-2948.

## Support

For questions or issues:
- See `examples/Ex_DataFusion_NaturalLanguage.ipynb` for complete working example
- Check individual agent docstrings for detailed parameter descriptions
- Review existing examples: `Ex_Structure_resinv.py`, `Ex_MC_Hydro.py`

## Future Development

Planned enhancements:
- Joint inversion frameworks (simultaneous optimization)
- Additional petrophysical models (CRIM, complex conductivity)
- Time-lapse multi-method workflows
- Integration with hydrological models
- Real-time monitoring dashboards
