#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Space for meteorological functions, ex:
    - shear stuffs
    - stability stuffs
    - calculation of diagnostics from primitive variables

"""

power_law = lambda z, zref, uref, alpha: uref*(z/zref)**alpha