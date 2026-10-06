Citation
========

If PyHydroGeophysX contributes to your research, please cite the software paper.
The numerical engines underneath do the actual computation, so please also cite
the ones your analysis ran.

Cite the software
-----------------

- Chen, H., Niu, Q., and Wu, Y. (2026). *PyHydroGeophysX: An Extensible
  Open-Source Platform for Integrating Hydrological Models with Geophysical
  Measurements*. SoftwareX, in press.
  https://doi.org/10.2139/ssrn.6238293

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

BibTeX entries for the three references above are in the
`project README <https://github.com/geohang/PyHydroGeophysX#citation>`_.

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
       E4D-style 3-D meshes (the 3D Mesh Builder's E4D engine), E4D mesh
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
       157(3), 988-1006. https://doi.org/10.1111/j.1365-246X.2004.02203.x
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
     - The four papers ParFlow asks users to cite: Ashby, S. F. and Falgout,
       R. D. (1996). A parallel multigrid preconditioned conjugate gradient
       algorithm for groundwater flow simulations. *Nuclear Science and
       Engineering*, 124(1), 145-159. https://doi.org/10.13182/NSE96-A24230;
       Jones, J. E. and Woodward, C. S. (2001). Newton-Krylov-multigrid solvers
       for large-scale, highly heterogeneous, variably saturated flow problems.
       *Advances in Water Resources*, 24(7), 763-774.
       https://doi.org/10.1016/S0309-1708(00)00075-0; Kollet, S. J. and
       Maxwell, R. M. (2006). Integrated surface-groundwater flow modeling: A
       free-surface overland flow boundary condition in a parallel groundwater
       flow model. *Advances in Water Resources*, 29(7), 945-958.
       https://doi.org/10.1016/j.advwatres.2005.08.006; Maxwell, R. M.
       (2013). A terrain-following grid transform and preconditioner for
       parallel, large-scale, integrated hydrologic modeling. *Advances in
       Water Resources*, 53, 109-117.
       https://doi.org/10.1016/j.advwatres.2012.10.001. Cite the ParFlow
       release you ran as well: https://doi.org/10.5281/zenodo.4816884
       resolves to the latest one.
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
   * - Ensemble Kalman updating, or ES-MDA
     - Evensen, G. (2003). *Ocean Dynamics*, 53(4), 343-367.
       https://doi.org/10.1007/s10236-003-0036-9; for ES-MDA, Emerick, A. A.
       and Reynolds, A. C. (2013). Ensemble smoother with multiple data
       assimilation. *Computers and Geosciences*, 55, 3-15.
       https://doi.org/10.1016/j.cageo.2012.03.011
