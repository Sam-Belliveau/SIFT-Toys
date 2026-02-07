"""
Global Parameter Registry

All tunable parameters live here. UI sliders write to this module,
processing modules read from it. Import and use directly:

    from params import params
    val = params["max_features"]
"""

params = {
    "max_features": 500,
    "filter_rc_ms": 500,
    "downsample": 4,
    "rbf_epsilon": 1.0,
    "point_opacity": 50,
}
