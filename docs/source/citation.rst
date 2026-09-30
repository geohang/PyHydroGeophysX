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
   * - E4D-style 3-D meshes (the 3D Mesh Builder's E4D engine), E4D mesh
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
