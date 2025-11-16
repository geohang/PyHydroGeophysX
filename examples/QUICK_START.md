# PyHydroGeophysX - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Launch the Web App
```bash
cd examples
streamlit run app_geophysics_workflow.py
```
Or use the launcher scripts:
- **Windows**: `start_webapp.bat`
- **Linux/Mac**: `./start_webapp.sh`

### Step 2: Configure API Key
In the sidebar:
1. Select LLM provider (OpenAI recommended)
2. Enter your API key
3. Click "🚀 Initialize System"

### Step 3: Describe Your Workflow
Type your request in natural language, for example:

**Standard ERT:**
```
Run ERT inversion on examples/data/ERT/DAS/20171105_1418.Data
Use rho_sat=541, porosity=0.37, n=1.24
```

**Time-Lapse ERT:**
```
Run time-lapse on files:
2022-03-26_0030.ohm, 2022-04-26_0030.ohm,
2022-05-26_0030.ohm, 2022-06-26_0030.ohm
Temporal regularization: 10
```

**Data Fusion:**
```
Use seismic at data/Seismic/srtfieldline2.dat with
threshold 1000 m/s to constrain ERT inversion at
data/ERT/Bert/fielddataline2.dat
```

Click **"🚀 Run Workflow"** and get your results!

---

## 📓 Jupyter Notebook Users

Open any of these notebooks:
- `Ex_Unified_Workflow.ipynb` - Standard ERT
- `Ex_Unified_Workflow_ex2.ipynb` - Time-Lapse
- `Ex_Unified_Workflow_ex3.ipynb` - Data Fusion

Run all cells and modify the `user_request` to try different workflows!

---

## 📚 Full Documentation
- **Web App Guide**: `WEB_APP_GUIDE.md`
- **System Overview**: `UNIFIED_WORKFLOW_SUMMARY.md`
- **API Docs**: `../docs/`

---

**That's it! You're ready to process geophysical data with natural language! 🎉**

