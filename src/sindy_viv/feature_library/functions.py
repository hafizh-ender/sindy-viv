"""General feature-library preparation for SINDy workflows."""

import pysindy as ps

from sindy_viv.utils import initiate_object

def prepare_library(library_config):
    """
    Instantiate an unfitted library from a module/class_name/kwargs config.
    """
    library = initiate_object(**library_config)
    if isinstance(library, ps.TensoredLibrary):
        # Preserve one outer input group per factor when PySINDy multiplies libraries.
        return ps.GeneralizedLibrary([library])
    return library
