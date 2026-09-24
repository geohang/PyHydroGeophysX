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
   * - TDEM or FDEM forward modelling and inversion
     - Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., and Oldenburg,
       D. W. (2015). SimPEG. *Computers and Geosciences*, 85, 142-154.
   * - MODFLOW, read or written through FloPy
     - Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T.,
       Starn, J. J., and Fienen, M. N. (2016). *Groundwater*, 54(5), 733-739.
   * - ParFlow outputs or written ParFlow inputs
     - Kollet, S. J. and Maxwell, R. M. (2006). *Advances in Water Resources*,
       29(7), 945-958; Maxwell, R. M. (2013). *Advances in Water Resources*,
       53, 109-117.
   * - A petrophysical relationship
     - The reference for the model you used, listed with it in
       :doc:`the petrophysics API <api/petrophysics>`.
   * - Ensemble Kalman updating
     - Evensen, G. (2003). *Ocean Dynamics*, 53(4), 343-367.
