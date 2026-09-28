PyHydroGeophysX.petrophysics package
====================================

Submodules
----------

PyHydroGeophysX.petrophysics.monte\_carlo module
------------------------------------------------

.. automodule:: PyHydroGeophysX.petrophysics.monte_carlo
   :members:
   :show-inheritance:
   :undoc-members:

PyHydroGeophysX.petrophysics.resistivity\_models module
-------------------------------------------------------

.. automodule:: PyHydroGeophysX.petrophysics.resistivity_models
   :members:
   :show-inheritance:
   :undoc-members:

PyHydroGeophysX.petrophysics.velocity\_models module
----------------------------------------------------

Empirical velocity conversion can be inverted with the same model and known
porosity. For example::

    import numpy as np
    from PyHydroGeophysX.petrophysics import (
        water_content_to_velocity, velocity_to_water_content,
    )

    theta = np.array([0.1, 0.2])
    velocity = water_content_to_velocity(theta, porosity=0.3, model="wyllie")
    recovered = velocity_to_water_content(velocity, porosity=0.3, model="wyllie")
    np.testing.assert_allclose(recovered, theta)

``PetrophysicalCoupling.compare_inversions_to_hydro`` uses the same
``velocity_model`` as its forward helper. Select ``"linear"``, ``"wyllie"`` or
``"raymer"`` directly, or use ``velocity_model="empirical"`` with
``empirical_model``. DEM and Hertz-Mindlin, including the forward helper's
default, require a calibrated ``velocity_inverse(velocity, porosity)`` callable
in ``petro_params``. They no longer silently use a linear inverse. Empirical
velocities outside the physical saturation range or at the forward clipping
limits raise ``ValueError`` instead of producing misleading clipped estimates.
Without a usable velocity inverse, or when no cell converts, the SRT entry is
``{"error": reason}`` and a warning is issued; the ERT and EM comparisons are
still returned. Cells whose velocity lies outside the model's range are NaN and
counted in ``stats['n_masked']``.

.. automodule:: PyHydroGeophysX.petrophysics.velocity_models
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: PyHydroGeophysX.petrophysics
   :members:
   :show-inheritance:
   :undoc-members:
