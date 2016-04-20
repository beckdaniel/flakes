import sys
try:
    import gpy
except ImportError:
    sys.stderr.write("Warning: GPy wrappers unavailable because either GPy or paramz is not installed.\n")
