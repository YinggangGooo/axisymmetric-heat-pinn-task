"""Reference interpolation and error reporting for the thermal solver."""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator


def interpolate_reference(points, temperatures, query):
    return CloughTocher2DInterpolator(np.asarray(points), np.asarray(temperatures))(query)


def relative_errors(prediction, reference, radii):
    prediction, reference, radii = map(np.asarray, (prediction, reference, radii))
    return np.abs(prediction - reference) / 80.0 * 100.0
