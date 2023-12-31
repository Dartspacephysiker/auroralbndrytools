|DOI| 

Overview
========

Auroral boundary tools

A set of tools for for creating a continuous representation of auroral boundaries and the motion of the auroral boundary from a set of points, possibly for multiple different times. These are based on the "analytic boundary velocity" equations derived by SMH here: https://essopenarchive.org/doi/full/10.22541/essoar.169447428.84472457/v1


Installation
------------

Using pip ::

    pip install --editable "auroralbndrytools @ git+https://github.com/Dartspacephysiker/auroralbndrytools.git@main"


Dependencies:

- numpy
- pandas
- scipy (scipy.interpolate, scipy.signal, and scipy.misc)
- apexpy (magnetic coordinate conversion)


..
   Quick Start
   -----------
   .. code-block:: python

       >>> # initialize by supplying a set of external conditions:
       >>> from pyswipe import SWIPE
       >>> m = SWIPE(350, # Solar wind velocity in km/s 
		     -4, # IMF By (GSM) in nT
		     -3, # IMF Bz (GSM) in nT, 
		     20, # dipole tilt angle in degrees 
		     80) # F107_index
       >>> # make summary plot:
       >>> m.plot_potential()

   .. image:: docs/static/example_plot.png
       :alt: Ionospheric potential (color) and electric field (pins)
    

References
----------
Derivation of equations for boundary normal velocity by SMH in appendix here: https://essopenarchive.org/doi/full/10.22541/essoar.169447428.84472457/v1

Equations and auroralbndrytools also used by Gasparini et al (in preparation)


.. |DOI| image:: https://zenodo.org/badge/728155056.svg
        :target: https://zenodo.org/badge/latestdoi/728155056
