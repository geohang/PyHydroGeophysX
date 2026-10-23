Citation
========

If PyHydroGeophysX contributes to your research, please cite the software paper.
The numerical engines underneath do the actual computation, so please also cite
the ones your analysis ran.

Cite the software
-----------------

- Chen, H., Niu, Q., and Wu, Y. (2026). *PyHydroGeophysX: An Extensible
  Open-Source Platform for Integrating Hydrological Models with Geophysical
  Measurements*. SoftwareX, 34, 102710.
  https://doi.org/10.1016/j.softx.2026.102710

Cite this one for any use of the package.

- Chen, H. (2026). *A Generalizable Automated Geophysical Agent Workflow for
  Accessible Subsurface Hydrology Analysis*. Big Data and Earth System, 100042.

Cite this one as well if you used the :doc:`AI-assisted workflows
<agents/index>`.

- Chen, H., Niu, Q., Mendieta, A., Bradford, J., and McNamara, J. (2023).
  Geophysics-informed hydrologic modeling of a mountain headwater catchment for
  studying hydrological partitioning in the critical zone. *Water Resources
  Research*, 59(12), e2023WR035280. https://doi.org/10.1029/2023WR035280

Cite this one for work that uses geophysical observations to inform a
hydrological model, which is the last step of the
:doc:`model-data loop <tutorials/hydro_geophysical_interaction>`.

BibTeX entries for the first two references above are in the
`project README <https://github.com/geohang/PyHydroGeophysX#citation>`_.

Cite GeoSAGE
------------

If your analysis uses the optional :doc:`GeoSAGE assistant <agents/geosage>`
for joint gravity–magnetic inversion, pseudo-geological modeling or geological
interpretation, please also cite:

- Fang, Z., Yang, P., Liu, Y., Feng, D., & Chen, H. (2026). GeoSAGE: A Multi-Agent
  Workflow for Geological Reasoning From Joint Gravity and Magnetic Inversion
  Models. *Computers & Geosciences*, 106282.

The source code and workflow documentation are available at
`ZhengyangFang/GeoSAGE <https://github.com/ZhengyangFang/GeoSAGE>`_.

Cite the engines you used
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - If your analysis used
     - Please also cite
   * - ERT or seismic refraction forward modelling and inversion
     - Rücker, C., Günther, T., and Wagner, F. M. (2017). pyGIMLi.
       *Computers and Geosciences*, 109, 106-123.
   * - Differentiable time-lapse ERT (``engine="adtlert"``)
     - Yang, P., Fang, Z., Liu, Y., Su, X., Feng, D., and Chen, H. (2026).
       ADTLERT. arXiv:2608.14661.
   * - ERT data processing and quality control
     - Blanchy, G., Saneiyan, S., Boyd, J., McLachlan, P., and Binley, A.
       (2020). ResIPy. *Computers and Geosciences*, 137, 104423.
   * - ERT reciprocal errors and the reciprocal error model (the ERT page's
       Reciprocal errors tab, or data errors taken from the reciprocal error
       model)
     - Slater, L., Binley, A. M., Daily, W., and Johnson, R. (2000).
       Cross-hole electrical imaging of a controlled saline tracer injection.
       *Journal of Applied Geophysics*, 44(2-3), 85-102.
       https://doi.org/10.1016/S0926-9851(00)00002-1; Koestel, J., Kemna, A.,
       Javaux, M., Binley, A., and Vereecken, H. (2008). Quantitative imaging
       of solute transport in an unsaturated and undisturbed soil monolith
       with 3-D ERT and TDR. *Water Resources Research*, 44(12).
       https://doi.org/10.1029/2007WR006755
   * - R2 or R3t inversions (``engine="r2"`` or ``engine="r3t"``, single
       surveys or time-lapse)
     - Binley, A. and Slater, L. (2020). *Resistivity and Induced
       Polarization: Theory and Applications to the Near-Surface Earth*.
       Cambridge University Press. https://doi.org/10.1017/9781108685955;
       Binley, A. (2015). Tools and Techniques: Electrical Methods. In
       *Treatise on Geophysics* (2nd ed.), 233-259. Elsevier.
       https://doi.org/10.1016/B978-0-444-53802-4.00192-5; Binley, A. and
       Kemna, A. (2005). DC Resistivity and Induced Polarization Methods. In
       *Hydrogeophysics*, 129-156. Springer.
       https://doi.org/10.1007/1-4020-3102-5_5
   * - A time-lapse series inverted by R2 or R3t (their difference inversion)
     - LaBrecque, D. J. and Yang, X. (2001). Difference inversion of ERT data:
       a fast inversion method for 3-D in situ monitoring. *Journal of
       Environmental and Engineering Geophysics*, 6(2), 83-89.
       https://doi.org/10.4133/JEEG6.2.83
   * - E4D inversions (``engine="e4d"``, single surveys or time-lapse),
       3-D meshes built by the 3D Mesh Builder's E4D engine, E4D mesh
       configurations, or E4D meshes imported for inversion
     - Johnson, T. C., Versteeg, R. J., Ward, A., Day-Lewis, F. D., and Revil,
       A. (2010). Improved hydrogeophysical characterization and monitoring
       through parallel modeling and inversion of time-domain resistivity and
       induced-polarization data. *Geophysics*, 75(4), WA27-WA41.
       https://doi.org/10.1190/1.3475513; Johnson, T. C., Robinson, J. R.,
       White, S. K., Zue, Y., and Jaysaval, P. (2020). *E4D User Guide*.
       Pacific Northwest National Laboratory.
       https://e4d-userguide.pnnl.gov
   * - A mesh built with TetGen (the E4D engine's mesher)
     - Si, H. (2015). TetGen, a Delaunay-based quality tetrahedral mesh
       generator. *ACM Transactions on Mathematical Software*, 41(2), 1-36.
       https://doi.org/10.1145/2629697
   * - A mesh built with Gmsh (the 3D Mesh Builder's Gmsh engine, with zones
       cut in by its OpenCASCADE kernel, or Gmsh standing in for TetGen)
     - Geuzaine, C. and Remacle, J.-F. (2009). Gmsh: a 3-D finite element mesh
       generator with built-in pre- and post-processing facilities.
       *International Journal for Numerical Methods in Engineering*, 79(11),
       1309-1331. https://doi.org/10.1002/nme.2579
   * - TDEM or FDEM forward modelling and inversion
     - Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., and Oldenburg,
       D. W. (2015). SimPEG. *Computers and Geosciences*, 85, 142-154.
   * - Laterally constrained inversion of a TDEM or FDEM line
       (``inversion.em1d_lci``) and its depth of investigation
     - Auken, E. and Christiansen, A. V. (2004). Layered and laterally
       constrained 2D inversion of resistivity data. *Geophysics*, 69(3),
       752-761. https://doi.org/10.1190/1.1759461; for the depth of
       investigation, Christiansen, A. V. and Auken, E. (2012). A global
       measure for depth of investigation. *Geophysics*, 77(4), WB171-WB177.
       https://doi.org/10.1190/geo2011-0393.1 (BibTeX in
       :ref:`em-lci-references`)
   * - MT time-series processing (``mt.process_mt``)
     - Egbert, G. D. and Booker, J. R. (1986). Robust estimation of
       geomagnetic transfer functions. *Geophysical Journal International*,
       87(1), 173-194. https://doi.org/10.1111/j.1365-246X.1986.tb04552.x;
       Egbert, G. D. (1997). Robust multiple-station magnetotelluric data
       processing. *Geophysical Journal International*, 130(2), 475-496.
       https://doi.org/10.1111/j.1365-246X.1997.tb05663.x; with a remote
       reference, Gamble, T. D., Goubau, W. M., and Clarke, J. (1979).
       Magnetotellurics with a remote magnetic reference. *Geophysics*, 44(1),
       53-68. https://doi.org/10.1190/1.1440923; with leverage weights, Chave,
       A. D. and Thomson, D. J. (2004). Bounded influence magnetotelluric
       response function estimation. *Geophysical Journal International*,
       157(3), 988-1006. https://doi.org/10.1111/j.1365-246X.2004.02203.x;
       with jackknife errors, Chave, A. D. and Thomson, D. J. (1989). Some
       comments on magnetotelluric response function estimation. *Journal of
       Geophysical Research*, 94(B10), 14215-14225.
       https://doi.org/10.1029/JB094iB10p14215; with coherence presorting,
       Egbert, G. D. and Livelybrooks, D. W. (1996). Single station
       magnetotelluric impedance estimation: Coherence weighting and the
       regression M-estimate. *Geophysics*, 61(4), 964-970.
       https://doi.org/10.1190/1.1444045; with multitaper spectra, Thomson,
       D. J. (1982). Spectrum estimation and harmonic analysis. *Proceedings
       of the IEEE*, 70(9), 1055-1096. https://doi.org/10.1109/PROC.1982.12433;
       with multi-window stacking, Ernst, T., Sokolova, E. Yu., Varentsov,
       Iv. M., and Golubev, N. G. (2001). Comparison of two techniques for
       magnetotelluric data processing using synthetic data sets. *Acta
       Geophysica Polonica*, 49(2), 213-243; with noise classes, Chen, H.
       (2020). Study on the processing method of magnetotelluric impedance
       estimation for non-Gaussian noise. Undergraduate thesis, Central South
       University, and McCulloch, J. H. (1986). Simple consistent estimators
       of stable distribution parameters. *Communications in Statistics -
       Simulation and Computation*, 15(4), 1109-1136.
       https://doi.org/10.1080/03610918608812563
   * - MT 1D inversion (``mt.occam1d``), with or without a TEM sounding for
       the static shift
     - Constable, S. C., Parker, R. L., and Constable, C. G. (1987). Occam's
       inversion: A practical algorithm for generating smooth models from
       electromagnetic sounding data. *Geophysics*, 52(3), 289-300.
       https://doi.org/10.1190/1.1442303; Wait, J. R. (1954). On the
       relation between telluric currents and the Earth's magnetic field.
       *Geophysics*, 19(2), 281-289. https://doi.org/10.1190/1.1437994; with
       TEM, Meju, M. A. (1996). Joint inversion of TEM and distorted MT
       soundings: Some effective practical considerations. *Geophysics*,
       61(1), 56-65. https://doi.org/10.1190/1.1443956, and Sternberg, B. K.,
       Washburne, J. C., and Pellerin, L. (1988). Correction for the static
       shift in magnetotellurics using transient electromagnetic soundings.
       *Geophysics*, 53(11), 1459-1468. https://doi.org/10.1190/1.1442426
   * - MT 1D inversion for azimuthal anisotropy (``mt.occam1d_anisotropic``)
     - Pek, J. and Santos, F. A. M. (2002). Magnetotelluric impedances and
       parametric sensitivities for 1-D anisotropic layered media. *Computers
       & Geosciences*, 28(8), 939-950.
       https://doi.org/10.1016/S0098-3004(02)00014-6, with Occam's inversion
       (Constable et al. 1987, above)
   * - MT 2D profile inversion (``mt.invert_profile``)
     - SimPEG (Cockett et al. 2015, above)
   * - MT phase tensor, skew and strike (``mt.phase_tensor``)
     - Caldwell, T. G., Bibby, H. M., and Brown, C. (2004). The
       magnetotelluric phase tensor. *Geophysical Journal International*,
       158(2), 457-469. https://doi.org/10.1111/j.1365-246X.2004.02281.x
   * - EMTF XML files (``mt.read_emtf_xml``, ``mt.write_emtf_xml``)
     - Kelbert, A. (2020). EMTF XML: New data interchange format and
       conversion tools for electromagnetic transfer functions. *Geophysics*,
       85(1), F1-F17. https://doi.org/10.1190/geo2018-0679.1
   * - The example MT site NMX20 (``examples/data/MT``)
     - Schultz, A., Pellerin, L., Bedrosian, P., Kelbert, A., and Crosbie, J.
       (2020-2023). USMTArray South Magnetotelluric Transfer Functions.
       Seismological Facility for the Advancement of Geoscience.
       https://doi.org/10.17611/DP/EMTF/USMTARRAY/SOUTH; Incorporated
       Research Institutions for Seismology (2011). Data Services Products:
       EMTF, The Magnetotelluric Transfer Functions.
       https://doi.org/10.17611/DP/EMTF.1 (CC BY 4.0)
   * - Shallow seismic air-wave statics, CMP stacking and velocity analysis
       (``data_processing.seismic_shallow``)
     - Maeda, N. (1985). A method for reading and checking phase time in
       auto-processing system of seismic wave data. *Zisin (Journal of the
       Seismological Society of Japan, 2nd ser.)*, 38(3), 365-379.
       https://doi.org/10.4294/zisin1948.38.3_365; Mayne, W. H. (1962).
       Common reflection point horizontal data stacking techniques.
       *Geophysics*, 27(6), 927-938. https://doi.org/10.1190/1.1439118; Dix,
       C. H. (1955). Seismic velocities from surface measurements.
       *Geophysics*, 20(1), 68-86. https://doi.org/10.1190/1.1438126;
       Neidell, N. S. and Taner, M. T. (1971). Semblance and other coherency
       measures for multichannel data. *Geophysics*, 36(3), 482-497.
       https://doi.org/10.1190/1.1440186; Steeples, D. W. and Miller, R. D.
       (1998). Avoiding pitfalls in shallow seismic reflection surveys.
       *Geophysics*, 63(4), 1213-1224. https://doi.org/10.1190/1.1444422
   * - GPR reading and processing (``data_processing.gpr``, the GPR page): GPR
       processing steps (the standard sequence described by GPRPy, readgssi, ImpDAR and RGPR)
     - Plattner, A. M. (2020). GPRPy: Open-source ground-penetrating radar
       processing and visualization software. *The Leading Edge*, 39(5),
       332-337. https://doi.org/10.1190/tle39050332.1; Nesbitt, I., Simon,
       F.-X., Paulin, T., Hoffmann, F., and teshaw (2018-2022). readgssi: an
       open-source tool to read and plot GSSI ground-penetrating radar data.
       Zenodo. https://doi.org/10.5281/zenodo.1439119; Lilien, D. A., Hills,
       B. H., Driscol, J., Jacobel, R., and Christianson, K. (2020). ImpDAR: an
       open-source impulse radar processor. *Annals of Glaciology*, 61(81),
       114-123. https://doi.org/10.1017/aog.2020.44; Huber, E. and Hans, G.
       (2018). RGPR - An open-source package to process and visualize GPR
       data. *17th International Conference on Ground Penetrating Radar*,
       1-4. https://doi.org/10.1109/ICGPR.2018.8441658
   * - GPR F-K migration (``gpr.migrate``)
     - Stolt, R. H. (1978). Migration by Fourier transform. *Geophysics*,
       43(1), 23-48. https://doi.org/10.1190/1.1440826
   * - Permittivity from water content and porosity by CRIM (``gpr.crim_permittivity``,
       the GPR page's example line)
     - Birchak, J. R., Gardner, C. G., Hipp, J. E., and Victor, J. M. (1974).
       High dielectric constant microwave probes for sensing soil moisture.
       *Proceedings of the IEEE*, 62(1), 93-98.
       https://doi.org/10.1109/PROC.1974.9388; Roth, K., Schulin, R., Flühler,
       H., and Attinger, W. (1990). Calibration of time domain reflectometry
       for water content measurement using a composite dielectric approach.
       *Water Resources Research*, 26(10), 2267-2273.
       https://doi.org/10.1029/WR026i010p02267
   * - MODFLOW, read or written through FloPy
     - Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T.,
       Starn, J. J., and Fienen, M. N. (2016). *Groundwater*, 54(5), 733-739.
       https://doi.org/10.1111/gwat.12413; Langevin, C. D., Hughes, J. D.,
       Banta, E. R., Provost, A. M., Niswonger, R. G., and Panday, S. (2017).
       *MODFLOW 6 Modular Hydrologic Model*. U.S. Geological Survey Software.
       https://doi.org/10.5066/F76Q1VQV; Langevin, C. D., Hughes, J. D.,
       Banta, E. R., Niswonger, R. G., Panday, S., and Provost, A. M. (2017).
       *Documentation for the MODFLOW 6 Groundwater Flow Model*. U.S.
       Geological Survey Techniques and Methods 6-A55.
       https://doi.org/10.3133/tm6A55
   * - ParFlow outputs or written ParFlow inputs
     - Kollet, S. J. and
       Maxwell, R. M. (2006). Integrated surface-groundwater flow modeling: A
       free-surface overland flow boundary condition in a parallel groundwater
       flow model. *Advances in Water Resources*, 29(7), 945-958.
       https://doi.org/10.1016/j.advwatres.2005.08.006; Maxwell, R. M.
       (2013). A terrain-following grid transform and preconditioner for
       parallel, large-scale, integrated hydrologic modeling. *Advances in
       Water Resources*, 53, 109-117.
       https://doi.org/10.1016/j.advwatres.2012.10.001.
   * - ATS outputs (``ATSSaturation``, ``ATSWaterContent``, ...)
     - The code, as ATS asks in all works: Coon, E. T., Berndt, M., Jan, A.,
       Svyatsky, D., Atchley, A. L., Kikinzon, E., Harp, D. R., Manzini, G.,
       Shelef, E., Lipnikov, K., Garimella, R., Xu, C., Moulton, J. D., Karra,
       S., Painter, S. L., Jafarov, E., and Molins, S. (2020). *Advanced
       Terrestrial Simulator*, version 1.0. U.S. Department of Energy.
       https://doi.org/10.11578/dc.20190911.1; and for watershed hydrology,
       Coon, E. T., Moulton, J. D., Kikinzon, E., Berndt, M., Manzini, G.,
       Garimella, R., Lipnikov, K., and Painter, S. L. (2020). Coupling surface
       flow and subsurface flow in complex soil structures using mimetic
       finite differences. *Advances in Water Resources*, 144, 103701.
       https://doi.org/10.1016/j.advwatres.2020.103701
   * - PFLOTRAN outputs (``PFLOTRANSaturation``, ``PFLOTRANWaterContent``,
       ...)
     - The references PFLOTRAN asks users to cite: Hammond, G. E., Lichtner,
       P. C., and Mills, R. T. (2014). Evaluating the performance of parallel
       subsurface simulators: An illustrative example with PFLOTRAN. *Water
       Resources Research*, 50(1), 208-228.
       https://doi.org/10.1002/2012WR013483; Lichtner, P. C., Hammond, G. E.,
       Lu, C., Karra, S., Bisht, G., Andre, B., Mills, R. T., Kumar, J., and
       Frederick, J. M. (2020). *PFLOTRAN User Manual*.
       http://documentation.pflotran.org; and the PFLOTRAN web page (same
       authors, 2020), http://www.pflotran.org
   * - A petrophysical relationship
     - The reference for the model you used, listed with it in
       :doc:`the petrophysics API <api/petrophysics>`.
   * - The depth-of-investigation index (``compute_depth_of_investigation``)
     - Oldenburg, D. W. and Li, Y. (1999). Estimating depth of investigation
       in DC resistivity and IP surveys. *Geophysics*, 64(2), 403-416.
       https://doi.org/10.1190/1.1444545
   * - The linearised posterior standard deviation
       (``uncertainty.regularized_posterior_std``) and the water content's
       +/- drawn from it
     - Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model
       Parameter Estimation*. SIAM. https://doi.org/10.1137/1.9780898717921;
       Aster, R. C., Borchers, B. and Thurber, C. H. (2018). *Parameter
       Estimation and Inverse Problems* (3rd ed.). Elsevier.
       https://doi.org/10.1016/C2015-0-02458-3 (BibTeX in
       :ref:`posterior-references`)
   * - Ensemble Kalman updating, or ES-MDA
     - Evensen, G. (2003). *Ocean Dynamics*, 53(4), 343-367.
       https://doi.org/10.1007/s10236-003-0036-9; for ES-MDA, Emerick, A. A.
       and Reynolds, A. C. (2013). Ensemble smoother with multiple data
       assimilation. *Computers and Geosciences*, 55, 3-15.
       https://doi.org/10.1016/j.cageo.2012.03.011

.. _method-references:

Method references
-----------------

The README carries the references for the core packages and engines an analysis
runs on. The papers behind individual methods, and supporting tools such as the
Gmsh mesher, are kept here with their BibTeX; the table above says which of
them to cite for which part of an analysis.

.. _ert-reciprocal-references:

ERT reciprocal errors
~~~~~~~~~~~~~~~~~~~~~

The ERT page's Reciprocal errors tab, and data errors taken from the reciprocal
error model.

.. code-block:: bibtex

   @article{slater2000crosshole,
     author  = {Slater, L. and Binley, A. M. and Daily, W. and Johnson, R.},
     title   = {Cross-hole electrical imaging of a controlled saline tracer injection},
     journal = {Journal of Applied Geophysics},
     volume  = {44},
     number  = {2-3},
     pages   = {85--102},
     year    = {2000},
     doi     = {10.1016/S0926-9851(00)00002-1}
   }

   @article{koestel2008quantitative,
     author  = {Koestel, Johannes and Kemna, Andreas and Javaux, Mathieu and
                Binley, Andrew and Vereecken, Harry},
     title   = {Quantitative imaging of solute transport in an unsaturated and
                undisturbed soil monolith with {3-D} {ERT} and {TDR}},
     journal = {Water Resources Research},
     volume  = {44},
     number  = {12},
     year    = {2008},
     doi     = {10.1029/2007WR006755}
   }

.. _gmsh-references:

3D meshes built with Gmsh
~~~~~~~~~~~~~~~~~~~~~~~~~

The 3D Mesh Builder's Gmsh engine and its zones, or Gmsh standing in for
TetGen.

.. code-block:: bibtex

   @article{geuzaine2009gmsh,
     author  = {Geuzaine, Christophe and Remacle, Jean-Fran{\c{c}}ois},
     title   = {{Gmsh}: A 3-{D} finite element mesh generator with built-in
                pre- and post-processing facilities},
     journal = {International Journal for Numerical Methods in Engineering},
     volume  = {79},
     number  = {11},
     pages   = {1309--1331},
     year    = {2009},
     doi     = {10.1002/nme.2579}
   }

.. _seismic-shallow-references:

Shallow seismic processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``data_processing.seismic_shallow``, air-wave onsets are picked with Maeda's
(1985) AIC; traces are stacked by common midpoint (Mayne 1962) after NMO with
Dix (1955) interval velocities; velocity analysis uses semblance (Neidell &
Taner 1971); the trace QC, mutes and the flat-event test follow the pitfalls
listed by Steeples & Miller (1998).

.. code-block:: bibtex

   @article{maeda1985method,
     author  = {Maeda, Naoki},
     title   = {A method for reading and checking phase time in auto-processing
                system of seismic wave data},
     journal = {Zisin (Journal of the Seismological Society of Japan. 2nd ser.)},
     volume  = {38},
     number  = {3},
     pages   = {365--379},
     year    = {1985},
     doi     = {10.4294/zisin1948.38.3_365}
   }

   @article{dix1955seismic,
     author  = {Dix, C. Hewitt},
     title   = {Seismic velocities from surface measurements},
     journal = {Geophysics},
     volume  = {20},
     number  = {1},
     pages   = {68--86},
     year    = {1955},
     doi     = {10.1190/1.1438126}
   }

   @article{mayne1962common,
     author  = {Mayne, W. Harry},
     title   = {Common reflection point horizontal data stacking techniques},
     journal = {Geophysics},
     volume  = {27},
     number  = {6},
     pages   = {927--938},
     year    = {1962},
     doi     = {10.1190/1.1439118}
   }

   @article{neidell1971semblance,
     author  = {Neidell, N. S. and Taner, M. Turhan},
     title   = {Semblance and other coherency measures for multichannel data},
     journal = {Geophysics},
     volume  = {36},
     number  = {3},
     pages   = {482--497},
     year    = {1971},
     doi     = {10.1190/1.1440186}
   }

   @article{steeples1998avoiding,
     author  = {Steeples, Don W. and Miller, Richard D.},
     title   = {Avoiding pitfalls in shallow seismic reflection surveys},
     journal = {Geophysics},
     volume  = {63},
     number  = {4},
     pages   = {1213--1224},
     year    = {1998},
     doi     = {10.1190/1.1444422}
   }

.. _gpr-references:

Ground-penetrating radar
~~~~~~~~~~~~~~~~~~~~~~~~

In ``data_processing.gpr`` and the studio's GPR page, the processing steps are
the standard ones also used by GPRPy, readgssi, ImpDAR and RGPR; the readers and
steps are written independently from the file-format descriptions and these
papers. F-K migration follows Stolt (1978); the example line turns water content
into permittivity with CRIM (Birchak et al. 1974; Roth et al. 1990).

.. code-block:: bibtex

   @article{plattner2020gprpy,
     author  = {Plattner, Alain M.},
     title   = {{GPRPy}: Open-source ground-penetrating radar processing and
                visualization software},
     journal = {The Leading Edge},
     volume  = {39},
     number  = {5},
     pages   = {332--337},
     year    = {2020},
     doi     = {10.1190/tle39050332.1}
   }

   @misc{nesbitt2018readgssi,
     author       = {Nesbitt, Ian and Simon, Fran{\c{c}}ois-Xavier and Paulin, Thomas
                     and Hoffmann, Felippe and teshaw},
     title        = {readgssi: an open-source tool to read and plot {GSSI}
                     ground-penetrating radar data},
     howpublished = {Zenodo},
     year         = {2018},
     doi          = {10.5281/zenodo.1439119}
   }

   @article{lilien2020impdar,
     author  = {Lilien, David A. and Hills, Benjamin H. and Driscol, Joshua and
                Jacobel, Robert and Christianson, Knut},
     title   = {{ImpDAR}: an open-source impulse radar processor},
     journal = {Annals of Glaciology},
     volume  = {61},
     number  = {81},
     pages   = {114--123},
     year    = {2020},
     doi     = {10.1017/aog.2020.44}
   }

   @inproceedings{huber2018rgpr,
     author    = {Huber, Emanuel and Hans, Guillaume},
     title     = {{RGPR} -- An open-source package to process and visualize {GPR} data},
     booktitle = {2018 17th International Conference on Ground Penetrating Radar (GPR)},
     pages     = {1--4},
     year      = {2018},
     doi       = {10.1109/ICGPR.2018.8441658}
   }

   @article{stolt1978migration,
     author  = {Stolt, R. H.},
     title   = {Migration by {Fourier} transform},
     journal = {Geophysics},
     volume  = {43},
     number  = {1},
     pages   = {23--48},
     year    = {1978},
     doi     = {10.1190/1.1440826}
   }

   @article{birchak1974high,
     author  = {Birchak, J. R. and Gardner, C. G. and Hipp, J. E. and Victor, J. M.},
     title   = {High dielectric constant microwave probes for sensing soil moisture},
     journal = {Proceedings of the IEEE},
     volume  = {62},
     number  = {1},
     pages   = {93--98},
     year    = {1974},
     doi     = {10.1109/PROC.1974.9388}
   }

   @article{roth1990calibration,
     author  = {Roth, Kurt and Schulin, Rainer and Fl{\"u}hler, Hannes and
                Attinger, Werner},
     title   = {Calibration of time domain reflectometry for water content
                measurement using a composite dielectric approach},
     journal = {Water Resources Research},
     volume  = {26},
     number  = {10},
     pages   = {2267--2273},
     year    = {1990},
     doi     = {10.1029/WR026i010p02267}
   }

.. _mt-references:

Magnetotellurics references
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the references below for the processing and inversion methods used in
:doc:`the MT/AMT workflows <methods/mt>`. The table above maps each method to
the papers to cite.

The processing follows EMTF
(Egbert & Booker 1986; Egbert 1997) with a remote reference (Gamble et al. 1979)
and bounded-influence weights (Chave & Thomson 2004), with optional jackknife
errors (Chave & Thomson 1989), coherence presorting (Egbert & Livelybrooks 1996),
multitaper spectra (Thomson 1982), multi-window stacking (Ernst et al. 2001) and
noise classes (Chen 2020) judged by McCulloch's (1986) stable index; the 1D inversion is Occam's
(Constable et al. 1987) on Wait's (1954) recursion, jointly with TEM after Meju
(1996) for the static shift (Sternberg et al. 1988), and for azimuthal anisotropy
after Pek & Santos (2002); the phase tensor is Caldwell
et al.'s (2004); EMTF XML is Kelbert's (2020) format; the 2D inversion runs on
SimPEG (above). The example site NMX20 is from the USMTArray (IRIS SPUD, CC BY 4.0).

.. code-block:: bibtex

   @article{egbert1986robust,
     author  = {Egbert, Gary D. and Booker, John R.},
     title   = {Robust estimation of geomagnetic transfer functions},
     journal = {Geophysical Journal International},
     volume  = {87},
     number  = {1},
     pages   = {173--194},
     year    = {1986},
     doi     = {10.1111/j.1365-246X.1986.tb04552.x}
   }

   @article{egbert1997robust,
     author  = {Egbert, Gary D.},
     title   = {Robust multiple-station magnetotelluric data processing},
     journal = {Geophysical Journal International},
     volume  = {130},
     number  = {2},
     pages   = {475--496},
     year    = {1997},
     doi     = {10.1111/j.1365-246X.1997.tb05663.x}
   }

   @article{gamble1979remote,
     author  = {Gamble, T. D. and Goubau, W. M. and Clarke, J.},
     title   = {Magnetotellurics with a remote magnetic reference},
     journal = {Geophysics},
     volume  = {44},
     number  = {1},
     pages   = {53--68},
     year    = {1979},
     doi     = {10.1190/1.1440923}
   }

   @article{chave2004bounded,
     author  = {Chave, Alan D. and Thomson, David J.},
     title   = {Bounded influence magnetotelluric response function estimation},
     journal = {Geophysical Journal International},
     volume  = {157},
     number  = {3},
     pages   = {988--1006},
     year    = {2004},
     doi     = {10.1111/j.1365-246X.2004.02203.x}
   }

   @article{chave1989comments,
     author  = {Chave, Alan D. and Thomson, David J.},
     title   = {Some comments on magnetotelluric response function estimation},
     journal = {Journal of Geophysical Research: Solid Earth},
     volume  = {94},
     number  = {B10},
     pages   = {14215--14225},
     year    = {1989},
     doi     = {10.1029/JB094iB10p14215}
   }

   @article{egbert1996single,
     author  = {Egbert, Gary D. and Livelybrooks, Dean W.},
     title   = {Single station magnetotelluric impedance estimation: Coherence
                weighting and the regression M-estimate},
     journal = {Geophysics},
     volume  = {61},
     number  = {4},
     pages   = {964--970},
     year    = {1996},
     doi     = {10.1190/1.1444045}
   }

   @article{thomson1982spectrum,
     author  = {Thomson, David J.},
     title   = {Spectrum estimation and harmonic analysis},
     journal = {Proceedings of the IEEE},
     volume  = {70},
     number  = {9},
     pages   = {1055--1096},
     year    = {1982},
     doi     = {10.1109/PROC.1982.12433}
   }

   @article{ernst2001comparison,
     author  = {Ernst, T. and Sokolova, E. Yu. and Varentsov, Iv. M. and Golubev, N. G.},
     title   = {Comparison of two techniques for magnetotelluric data processing
                using synthetic data sets},
     journal = {Acta Geophysica Polonica},
     volume  = {49},
     number  = {2},
     pages   = {213--243},
     year    = {2001}
   }

   @thesis{chen2020nongaussian,
     author  = {Chen, Hang},
     title   = {Study on the processing method of magnetotelluric impedance
                estimation for non-Gaussian noise},
     type    = {Undergraduate thesis},
     school  = {Central South University},
     year    = {2020}
   }

   @article{mcculloch1986simple,
     author  = {McCulloch, J. Huston},
     title   = {Simple consistent estimators of stable distribution parameters},
     journal = {Communications in Statistics - Simulation and Computation},
     volume  = {15},
     number  = {4},
     pages   = {1109--1136},
     year    = {1986},
     doi     = {10.1080/03610918608812563}
   }

   @article{constable1987occam,
     author  = {Constable, Steven C. and Parker, Robert L. and Constable, Catherine G.},
     title   = {Occam's inversion: A practical algorithm for generating smooth models
                from electromagnetic sounding data},
     journal = {Geophysics},
     volume  = {52},
     number  = {3},
     pages   = {289--300},
     year    = {1987},
     doi     = {10.1190/1.1442303}
   }

   @article{wait1954relation,
     author  = {Wait, James R.},
     title   = {On the relation between telluric currents and the Earth's magnetic field},
     journal = {Geophysics},
     volume  = {19},
     number  = {2},
     pages   = {281--289},
     year    = {1954},
     doi     = {10.1190/1.1437994}
   }

   @article{pek2002magnetotelluric,
     author  = {Pek, Josef and Santos, Fernando A. M.},
     title   = {Magnetotelluric impedances and parametric sensitivities for 1-{D}
                anisotropic layered media},
     journal = {Computers \& Geosciences},
     volume  = {28},
     number  = {8},
     pages   = {939--950},
     year    = {2002},
     doi     = {10.1016/S0098-3004(02)00014-6}
   }

   @article{meju1996joint,
     author  = {Meju, Maxwell A.},
     title   = {Joint inversion of {TEM} and distorted {MT} soundings: Some effective
                practical considerations},
     journal = {Geophysics},
     volume  = {61},
     number  = {1},
     pages   = {56--65},
     year    = {1996},
     doi     = {10.1190/1.1443956}
   }

   @article{sternberg1988static,
     author  = {Sternberg, Ben K. and Washburne, James C. and Pellerin, Louise},
     title   = {Correction for the static shift in magnetotellurics using transient
                electromagnetic soundings},
     journal = {Geophysics},
     volume  = {53},
     number  = {11},
     pages   = {1459--1468},
     year    = {1988},
     doi     = {10.1190/1.1442426}
   }

   @article{caldwell2004phase,
     author  = {Caldwell, T. Grant and Bibby, Hugh M. and Brown, Colin},
     title   = {The magnetotelluric phase tensor},
     journal = {Geophysical Journal International},
     volume  = {158},
     number  = {2},
     pages   = {457--469},
     year    = {2004},
     doi     = {10.1111/j.1365-246X.2004.02281.x}
   }

   @article{kelbert2020emtf,
     author  = {Kelbert, Anna},
     title   = {{EMTF XML}: New data interchange format and conversion tools for
                electromagnetic transfer functions},
     journal = {Geophysics},
     volume  = {85},
     number  = {1},
     pages   = {F1--F17},
     year    = {2020},
     doi     = {10.1190/geo2018-0679.1}
   }

   @misc{schultz2020usmtarray,
     author    = {Schultz, A. and Pellerin, L. and Bedrosian, P. and Kelbert, A. and
                  Crosbie, J.},
     title     = {{USMTArray} South Magnetotelluric Transfer Functions},
     publisher = {Seismological Facility for the Advancement of Geoscience},
     year      = {2020},
     note      = {2020--2023},
     doi       = {10.17611/DP/EMTF/USMTARRAY/SOUTH}
   }

   @misc{iris2011emtf,
     author    = {{Incorporated Research Institutions for Seismology}},
     title     = {Data Services Products: {EMTF}, The Magnetotelluric Transfer
                  Functions},
     publisher = {Seismological Facility for the Advancement of Geoscience},
     year      = {2011},
     doi       = {10.17611/DP/EMTF.1}
   }


.. _em-lci-references:

Laterally constrained EM inversion references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``inversion.em1d_lci`` inverts the soundings of a TDEM or FDEM line as one
system, with lateral constraints between neighbouring soundings, after the
laterally constrained inversion of Auken & Christiansen (2004). Its depth of
investigation is the cumulated sensitivity of Christiansen & Auken (2012).

.. code-block:: bibtex

   @article{auken2004layered,
     author  = {Auken, Esben and Christiansen, Anders Vest},
     title   = {Layered and laterally constrained 2D inversion of resistivity data},
     journal = {Geophysics},
     volume  = {69},
     number  = {3},
     pages   = {752--761},
     year    = {2004},
     doi     = {10.1190/1.1759461}
   }

   @article{christiansen2012global,
     author  = {Christiansen, Anders Vest and Auken, Esben},
     title   = {A global measure for depth of investigation},
     journal = {Geophysics},
     volume  = {77},
     number  = {4},
     pages   = {WB171--WB177},
     year    = {2012},
     doi     = {10.1190/geo2011-0393.1}
   }


.. _posterior-references:

Linearised posterior uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``uncertainty.regularized_posterior_std`` gives each cell's standard deviation
from the linearised posterior covariance of a regularised Gauss-Newton model,
and the water content's +/- draws the resistivity from it (see
:ref:`water-content-uncertainty`).

.. code-block:: bibtex

   @book{tarantola2005inverse,
     author    = {Tarantola, Albert},
     title     = {Inverse Problem Theory and Methods for Model Parameter Estimation},
     publisher = {Society for Industrial and Applied Mathematics},
     year      = {2005},
     doi       = {10.1137/1.9780898717921}
   }

   @book{aster2018parameter,
     author    = {Aster, Richard C. and Borchers, Brian and Thurber, Clifford H.},
     title     = {Parameter Estimation and Inverse Problems},
     edition   = {3},
     publisher = {Elsevier},
     year      = {2018},
     doi       = {10.1016/C2015-0-02458-3}
   }


.. _pore-fluid-references:

Pore-fluid properties and grain contacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the equations, units and calibration limits, see :doc:`methods/petrophysics`.

.. rubric:: Batzle-Wang (brine density and elastic modulus)

.. code-block:: bibtex

   @article{batzle1992fluids,
     author = {Batzle, Michael and Wang, Zhijing},
     title = {Seismic properties of pore fluids},
     journal = {Geophysics},
     volume = {57},
     number = {11},
     pages = {1396--1408},
     year = {1992},
     doi = {10.1190/1.1443207}
   }

.. rubric:: Waxman-Thomas (surface conductivity)

.. code-block:: bibtex

   @article{waxman1974conductivities,
     author = {Waxman, Monroe H. and Thomas, E. C.},
     title = {Electrical Conductivities in Shaly Sands: I. The Relation Between Hydrocarbon Saturation and Resistivity Index; II. The Temperature Coefficient of Electrical Conductivity},
     journal = {Journal of Petroleum Technology},
     volume = {26},
     number = {2},
     pages = {213--225},
     year = {1974},
     doi = {10.2118/4094-PA}
   }

.. rubric:: ORNL surface-water survey (limiting equivalent ionic conductivities)

.. code-block:: bibtex

   @techreport{kanciruk1986conductance,
     author = {Kanciruk, Paul and Gentry, Merilyn and McCord, Raymond and Hook, Les and Eilers, Joseph and Best, Mary D.},
     title = {National Surface Water Survey: Eastern Lake Survey---Phase I, Data Base Dictionary},
     institution = {Oak Ridge National Laboratory},
     number = {ORNL/TM-10153},
     year = {1986},
     note = {Table 8, CONCAL, page 30},
     url = {https://archive.epa.gov/emap/archive-emap/web/pdf/elsi_dd-2.pdf}
   }

.. rubric:: Metrohm conductometry (Kohlrausch square-root correction)

.. code-block:: bibtex

   @manual{metrohmconductometry,
     author = {Bruttel, Peter},
     title = {Conductometry---Conductivity Measurement},
     organization = {Metrohm},
     number = {8.109.5021},
     note = {Revised by Lucia Meier, Sabrina Gschwind and Iris Kalkman; square-root law, page 14},
     url = {https://www.metrohm.com/content/dam/metrohm/shared/documents/monographs/81095021EN.pdf}
   }

.. rubric:: Girault electrochemistry (calcium chloride conductivity benchmark)

.. code-block:: bibtex

   @book{girault2004electrochemistry,
     author = {Girault, Hubert H.},
     title = {Analytical and Physical Electrochemistry},
     publisher = {EPFL Press},
     year = {2004},
     doi = {10.55430/6101APEHG}
   }

.. rubric:: Rock Physics Handbook (grain contacts and fluid substitution)

.. code-block:: bibtex

   @book{mavko2009handbook,
     author = {Mavko, Gary and Mukerji, Tapan and Dvorkin, Jack},
     title = {The Rock Physics Handbook: Tools for Seismic Analysis of Porous Media},
     publisher = {Cambridge University Press},
     edition = {2},
     year = {2009},
     doi = {10.1017/CBO9780511626753}
   }
