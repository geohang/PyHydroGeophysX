# Example data

Where each dataset in this folder comes from, and the terms it is
redistributed under. Third-party data keep an `acknowledgement.txt` (or a
`LICENSE`) beside the files. Everything below credited to Hang Chen or to
PyHydroGeophysX was collected or created for this project and is distributed
under the repository licence (`LICENSE`, Apache-2.0).

| Path | Contents | Source | Terms |
| --- | --- | --- | --- |
| `Watercontent.npy`, `Porosity.npy`, `top.npy`, `top.txt`, `bot.npy`, `id.txt`, `precip.npy` | Arrays of the MODFLOW model used by the ERT, SRT and time-lapse examples | Hang Chen | Repository licence |
| `modflow/` | MODFLOW 6 model files | Hang Chen | Repository licence |
| `parflow/test2/` | ParFlow model output | Hang Chen | Repository licence |
| `TL_measurements/` | Synthetic time-lapse ERT measurements; `metadata.json` records how they were made | Hang Chen | Repository licence |
| `ERT/Bert/fielddataline2.dat` | Field ERT line | Hang Chen | Repository licence |
| `Seismic/srtfieldline2.dat`, `Seismic/AP_411.sgy`, `Seismic/location.txt` | Field seismic refraction data | Hang Chen | Repository licence |
| `Seismic/synthetic_seismic_data.dat` | Synthetic travel times written by `EX_SRT_forward.py` | PyHydroGeophysX | Repository licence |
| `synthetic/` | Synthetic hydrology models written by `generate_synthetic_examples.py` | PyHydroGeophysX | Repository licence |
| `EM/joint_synthetic_fdem.csv`, `EM/joint_synthetic_tdem.csv`, `EM/synthetic_fdem.csv`, `EM/synthetic_tem_lci/` | Synthetic EM data | PyHydroGeophysX | Repository licence |
| `modflow_informed/` | Structure arrays repackaged from Hang Chen's [Geophysics_informed_models](https://github.com/geohang/Geophysics_informed_models) | Hang Chen | Apache-2.0; `LICENSE`, `provenance.json` |
| `EM/skytem_bhmar_*.csv` | Five SkyTEM soundings | Brodie (2019), Geoscience Australia, https://doi.org/10.26186/5d314c5de8040 | CC BY 4.0; `EM/acknowledgement.txt` |
| `EM/EastRiver_VTEM/` | VTEM airborne EM subset | Zamudio, Minsley and Ball (2021), U.S. Geological Survey, https://doi.org/10.5066/P949ZCZ8 | Public domain (U.S. Government work); `acknowledgement.txt` |
| `Gravity_Magnetics/` | Bushveld gravity and a Britain aeromagnetic subset | NOAA NCEI and British Geological Survey, curated by Fatiando a Terra | CC BY 4.0; `acknowledgement.txt` |
| `MT/NMX20.xml` | USMTArray magnetotelluric transfer functions | IRIS/SAGE EMTF, https://doi.org/10.17611/DP/EMTF/USMTARRAY/SOUTH | CC BY 4.0; `acknowledgement.txt` |
| `climate/` | Daily weather at the Mt. Snodgrass monitoring site | Daymet Version 4 R1, ORNL DAAC, https://doi.org/10.3334/ORNLDAAC/2129 | Cite the dataset; `acknowledgement.txt` |
| `ERT/DAS/` | Time-lapse DAS-1 ERT acquisitions | See `acknowledgement.txt` | See `acknowledgement.txt` |
| `ERT/E4D/` | Time-lapse ERT acquisitions in E4D format | See `acknowledgement.txt` | See `acknowledgement.txt` |
