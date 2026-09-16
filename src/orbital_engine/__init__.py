from .registry import register_force_model, get_force_model, all_force_models, mask_for
# Importing a force-model module registers it. Anything that imports this package can then
# enable "j2" by name without importing geopotential itself.
from . import geopotential
