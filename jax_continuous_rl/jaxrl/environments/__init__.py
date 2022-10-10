# import inspect

# from jaxrl.environments import point_mass as my_point_mass

# NEW_DOMAINS = {name: module for name, module in locals().items()
#                if inspect.ismodule(module) and hasattr(module, 'SUITE')}


# from dm_control import suite
# suite._DOMAINS.update(NEW_DOMAINS)