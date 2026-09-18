"""Offline analysis for characterization runs.

Deliberately pure standard library - no numpy, no scipy, no matplotlib.

That is not minimalism for its own sake.  Testing happens away from a network,
and an analysis step that fails with "pip install numpy" while the operator is
standing in a car park with no way to install it is an analysis step that does
not exist.  Everything here runs on a stock Python 3 on the Jetson or on the
laptop, today, offline.

The fitting is small enough that this costs little: three-parameter least
squares is twenty lines of normal equations, and the plots are hand-written SVG
that any browser opens.
"""
