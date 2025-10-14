"""
Plotting utilities for PyHydroGeophysX.
"""
import pygimli as pg
import numpy as np
import matplotlib.pyplot as plt

def drawFirstPicks(ax, data, tt=None, plotva=False, **kwargs):
    """Plot first arrivals as lines.
    
    Parameters
    ----------
    ax : matplotlib.axes
        axis to draw the lines in
    data : :gimliapi:`GIMLI::DataContainer`
        data containing shots ("s"), geophones ("g") and traveltimes ("t")
    tt : array, optional
        traveltimes to use instead of data("t")
    plotva : bool, optional
        plot apparent velocity instead of traveltimes
    
    Return
    ------
    ax : matplotlib.axes
        the modified axis
    """
    # Extract coordinates
    px = pg.x(data)
    gx = np.array([px[int(g)] for g in data("g")])
    sx = np.array([px[int(s)] for s in data("s")])
    
    # Get traveltimes
    if tt is None:
        tt = np.array(data("t"))
    if plotva:
        tt = np.absolute(gx - sx) / tt
    
    # Find unique source positions    
    uns = np.unique(sx)
    
    # Override kwargs with clean, minimalist style
    kwargs['color'] = 'black'
    kwargs['linestyle'] = '--'
    kwargs['linewidth'] = 0.9
    kwargs['marker'] = None  # No markers on the lines
    
    # Plot for each source
    for i, si in enumerate(uns):
        ti = tt[sx == si]
        gi = gx[sx == si]
        ii = gi.argsort()
        
        # Plot line
        ax.plot(gi[ii], ti[ii], **kwargs)
        
        # Add source marker as black square at top
        ax.plot(si, 0.0, 's', color='black', markersize=4, 
                markeredgecolor='black', markeredgewidth=0.5)
    
    # Clean grid style
    ax.grid(True, linestyle='-', linewidth=0.2, color='lightgray')
    
    # Set proper axis labels with units
    if plotva:
        ax.set_ylabel("Apparent velocity (m s$^{-1}$)")
    else:
        ax.set_ylabel("Travel time (s)")
    
    ax.set_xlabel("Distance (m)")
    

    

    
    # Invert y-axis for traveltimes
    ax.invert_yaxis()

    return ax