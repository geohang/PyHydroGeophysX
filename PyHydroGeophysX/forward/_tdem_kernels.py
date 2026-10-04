# rte_hankel and rte_dsigma_hankel are numba versions of _rTE_forward and
# _rTE_gradient from geoana (geoana/kernels/tranverse_electric_reflections.py)
# and are distributed under geoana's licence, below. PyHydroGeophysX's changes
# to them are under the package's Apache-2.0 licence.
#
# MIT License
#
# Copyright (c) 2017 SimPEG Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Compiled layered-earth kernels for the TDEM forward and Jacobian, which need numba.

Imported on first use by :mod:`PyHydroGeophysX.forward.tdem_forward`, so that
importing the forward does not import numba, and a machine without numba keeps
SimPEG's own path.
"""

import numba
import numpy as np
from scipy.constants import mu_0


@numba.njit(cache=True, nogil=True)
def rte_dsigma_hankel(frequencies, lamb, sigma, mu, thicknesses,
                      weights):  # pragma: no cover - compiled
    """``d rTE / d sigma`` of a layered earth, through the Hankel filter.

    The conductivity term of geoana's ``_rTE_gradient`` (the reference NumPy
    version of the kernel SimPEG calls), without the thickness and permeability
    gradients a conductivity inversion discards, and summed over the
    wavenumbers with ``weights`` (n_lambda, n_receiver) as it is computed, so
    the (n_layer, n_frequency, n_lambda) gradient is never stored: allocating
    and reading back several megabytes per sounding is what kept a parallel
    line pass from using its cores. Arguments as in geoana: ``sigma`` and ``mu``
    complex, shaped (n_layer, n_frequency), the first layer the top one and the
    last a half-space. Returns (n_layer, n_frequency, n_receiver).
    """
    n_layer = thicknesses.size + 1
    n_rx = weights.shape[1]
    out = np.zeros((n_layer, frequencies.size, n_rx), dtype=np.complex128)
    grad = np.empty(n_layer, dtype=np.complex128)
    u = np.empty(n_layer, dtype=np.complex128)
    Y = np.empty(n_layer, dtype=np.complex128)
    th = np.empty(n_layer, dtype=np.complex128)
    Yh = np.empty(n_layer, dtype=np.complex128)
    for i in range(frequencies.size):
        omega = 2.0 * np.pi * frequencies[i]
        for j in range(lamb.size):
            l2 = lamb[j] * lamb[j]
            for k in range(n_layer):
                u[k] = np.sqrt(l2 + 1j * omega * mu[k, i] * sigma[k, i])
                Y[k] = u[k] / (1j * omega * mu[k, i])
            # tanh as (1 - e^-2z) / (1 + e^-2z): Re(z) >= 0 here, and numba's
            # complex tanh overflows to NaN for a thick layer at a large
            # wavenumber, where the value is simply 1.
            for k in range(n_layer - 1):
                e = np.exp(-2.0 * u[k] * thicknesses[k])
                th[k] = (1.0 - e) / (1.0 + e)
            Yh[n_layer - 1] = Y[n_layer - 1]
            for k in range(n_layer - 2, -1, -1):
                Yh[k] = Y[k] * (Yh[k + 1] + Y[k] * th[k]) / (Y[k] + Yh[k + 1] * th[k])
            Y0 = lamb[j] / (1j * omega * mu_0)
            gyh0 = -2.0 * Y0 / ((Y0 + Yh[0]) * (Y0 + Yh[0]))
            for k in range(n_layer - 1):
                den = Y[k] + Yh[k + 1] * th[k]
                bot = den * den
                gy = gyh0 * th[k] * (2.0 * th[k] * Y[k] * Yh[k + 1] + Y[k] * Y[k]
                                     + Yh[k + 1] * Yh[k + 1]) / bot
                gtanh = gyh0 * (Y[k] * Y[k] * Y[k] - Y[k] * Yh[k + 1] * Yh[k + 1]) / bot
                gyh0 = gyh0 * -(th[k] * th[k] - 1.0) * Y[k] * Y[k] / bot
                gu = gtanh * thicknesses[k] * (1.0 - th[k] * th[k])
                gu += gy / (1j * omega * mu[k, i])
                grad[k] = gu * -0.5 / u[k] * (-1j * omega * mu[k, i])
            last = n_layer - 1
            gu = gyh0 / (1j * omega * mu[last, i])
            grad[last] = gu * -0.5 / u[last] * (-1j * omega * mu[last, i])
            for r in range(n_rx):
                weight = weights[j, r]
                if weight != 0.0:
                    for k in range(n_layer):
                        out[k, i, r] += grad[k] * weight
    return out


@numba.njit(cache=True, nogil=True)
def rte_hankel(frequencies, lamb, sigma, mu, thicknesses, weights):  # pragma: no cover - compiled
    """The TE reflection coefficient of a layered earth, through the Hankel filter.

    geoana's ``_rTE_forward`` recursion, summed over the wavenumbers with
    ``weights`` (n_lambda, n_receiver) as it goes, as :func:`rte_dsigma_hankel`
    does for the gradient. Returns (n_frequency, n_receiver).
    """
    n_layer = thicknesses.size + 1
    n_rx = weights.shape[1]
    out = np.zeros((frequencies.size, n_rx), dtype=np.complex128)
    Y = np.empty(n_layer, dtype=np.complex128)
    th = np.empty(n_layer, dtype=np.complex128)
    for i in range(frequencies.size):
        omega = 2.0 * np.pi * frequencies[i]
        for j in range(lamb.size):
            l2 = lamb[j] * lamb[j]
            for k in range(n_layer):
                u = np.sqrt(l2 + 1j * omega * mu[k, i] * sigma[k, i])
                Y[k] = u / (1j * omega * mu[k, i])
                if k < n_layer - 1:
                    e = np.exp(-2.0 * u * thicknesses[k])
                    th[k] = (1.0 - e) / (1.0 + e)
            Yh = Y[n_layer - 1]
            for k in range(n_layer - 2, -1, -1):
                Yh = Y[k] * (Yh + Y[k] * th[k]) / (Y[k] + Yh * th[k])
            Y0 = lamb[j] / (1j * omega * mu_0)
            te = (Y0 - Yh) / (Y0 + Yh)
            for r in range(n_rx):
                weight = weights[j, r]
                if weight != 0.0:
                    out[i, r] += te * weight
    return out
