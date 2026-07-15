# verl dynamic-resource smoke compatibility shims
try:
    import transformers.configuration_utils as _verl_configuration_utils
except Exception:
    _verl_configuration_utils = None

if _verl_configuration_utils is not None and not hasattr(_verl_configuration_utils, "ALLOWED_LAYER_TYPES"):
    _verl_configuration_utils.ALLOWED_LAYER_TYPES = [
        "full_attention",
        "sliding_attention",
        "chunked_attention",
    ]
