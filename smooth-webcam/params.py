"""
Global Parameter Registry

All tunable parameters live here. UI sliders write to this module,
processing modules read from it. Import and use directly:

    from params import params
    val = params["max_features"]
"""

params = {
    "grid_points": 500,
    "decay_iters": 4,
    "downsample": 4,
    "rbf_smoothing": 0,
    "point_opacity": 50,
}
