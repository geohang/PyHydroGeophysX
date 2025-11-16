# PyHydroGeophysX Web Application Guide

## 🌍 Streamlit Web Interface for Geophysical Workflows

The PyHydroGeophysX web application provides an intuitive interface for running geophysical workflows using natural language.

## 🚀 Quick Start

### 1. Install Requirements

Make sure you have all dependencies installed:

```bash
pip install streamlit
pip install -r requirements.txt
```

### 2. Launch the Web App

From the `examples` directory, run:

```bash
streamlit run app_geophysics_workflow.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Configure API Key

In the sidebar:
1. Select your LLM provider (OpenAI, Gemini, or Claude)
2. Enter your API key
3. Click "🚀 Initialize System"

### 4. Run Your Workflow

1. **Describe your workflow** in the text area using natural language
2. **Upload data files** (optional) - or reference them in your description
3. Click **"🚀 Run Workflow"**
4. View results and download generated files

## 📚 Example Workflows

### Standard ERT Workflow

```
We have ERT data from DAS-1 instrument at 
examples/data/ERT/DAS/20171105_1418.Data 
with electrode file at examples/data/ERT/DAS/electrodes.dat 

Use petrophysical parameters: 
rho_sat=541, porosity=0.37, n=1.24

Apply lambda=15 for regularization
```

### Time-Lapse ERT Workflow

```
Run TIME-LAPSE ERT inversion on 4 E4D format files:
- File 1 (BASELINE): 2022-03-26_0030.ohm
- File 2: 2022-04-26_0030.ohm
- File 3: 2022-05-26_0030.ohm
- File 4: 2022-06-26_0030.ohm

Settings:
- Temporal regularization: 10
- Spatial regularization (lambda): 15

Fetch climate data for Mt. Snodgrass site:
- Coordinates: 38.92584°N, -106.97998°W
- Date range: March 2022 to June 2022
- Variables: precipitation, temperature, solar radiation
```

### Data Fusion Workflow

```
I need to characterize subsurface water content using multi-method data fusion:

1. Use field seismic refraction data at data/Seismic/srtfieldline2.dat
   - Velocity threshold: 1000 m/s to identify regolith/bedrock boundary

2. Use seismic structure to constrain ERT inversion
   - ERT data: data/ERT/Bert/fielddataline2.dat
   - Lambda: 20 (moderate regularization)

3. Convert to water content with layer-specific petrophysics:
   - Regolith: rho_sat (50-250 Ωm), n (1.3-2.2), porosity (0.25-0.5)
   - Fractured bedrock: rho_sat (165-350 Ωm), n (2.0-2.2), porosity (0.2-0.3)
   
4. Run Monte Carlo uncertainty analysis with 100 realizations
```

## 📁 File Upload Options

The app supports three types of file uploads:

### 1. ERT Data Files
- Formats: `.ohm`, `.dat`, `.Data`
- Single file → Standard ERT workflow
- Multiple files → Time-lapse workflow

### 2. Seismic Data Files
- Formats: `.dat`, `.txt`
- For data fusion workflows

### 3. Electrode Files
- Formats: `.dat`, `.txt`
- Optional: provides topography support

**Note:** Uploaded files override file paths specified in your natural language description.

## 📊 Results Display

After workflow completion, you'll see:

1. **Interpretation**: AI-generated explanation of the workflow
2. **Execution Plan**: Steps executed and agents used
3. **Results Summary**: Key metrics (resistivity, water content, etc.)
4. **Generated Files**: Download buttons for reports and visualizations
5. **Configuration**: Full workflow configuration (expandable)

## 🎯 Workflow Types (Auto-Detected)

The system automatically detects your workflow type:

| Workflow Type | Detected When | Key Features |
|--------------|---------------|--------------|
| **Standard ERT** | Single ERT file mentioned | ERT inversion + petrophysics |
| **Time-Lapse ERT** | Multiple ERT files or "time-lapse" keyword | Temporal inversion + climate integration |
| **Data Fusion** | Seismic + ERT mentioned | Structure-constrained inversion |

## 💡 Tips for Best Results

### 1. Be Specific
```
❌ "Run ERT on my data"
✅ "Run ERT inversion on data/ERT/field1.dat with lambda=20"
```

### 2. Include File Paths
```
❌ "Use seismic data"
✅ "Use seismic data at data/Seismic/srtfieldline2.dat"
```

### 3. Specify Parameters
```
❌ "Convert to water content"
✅ "Convert to water content with rho_sat=541, porosity=0.37, n=1.24"
```

### 4. For Time-Lapse: List Files Clearly
```
✅ Time-lapse files:
   - Baseline: 2022-03-26.ohm
   - Time 2: 2022-04-26.ohm
   - Time 3: 2022-05-26.ohm
```

### 5. For Data Fusion: Mention Both Methods
```
✅ "Use seismic at 1000 m/s threshold to constrain ERT inversion"
```

## 🔧 Troubleshooting

### App Won't Start
- Check Streamlit is installed: `pip install streamlit`
- Try: `python -m streamlit run app_geophysics_workflow.py`

### "API Key Not Found"
- Enter your API key in the sidebar
- Or set environment variable: `export OPENAI_API_KEY=your-key`

### "File Not Found"
- Use relative paths from the `examples` directory
- Or upload files using the file upload buttons

### Workflow Fails
- Check the error message in the app
- Verify file paths are correct
- Ensure parameters are valid (e.g., lambda > 0)
- Check your API key has sufficient credits

## 📖 Related Resources

- **Unified Workflow Notebooks**: See `Ex_Unified_Workflow*.ipynb` for examples
- **API Documentation**: Check `docs/` for detailed API reference
- **Example Data**: Located in `examples/data/`

## 🆘 Need Help?

1. Check the example workflows in the app
2. Review the unified workflow notebooks
3. Open an issue on GitHub
4. Consult the full documentation

---

**Happy analyzing! 🎉**

