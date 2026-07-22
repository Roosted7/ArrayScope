"""First-party in-process operation packs.

Each pack module self-guards on its optional backend and registers its
:class:`~arrayscope.operations.plugins.PluginOperationSpec` objects via
:func:`arrayscope.operations.registry.register_pack_operation`.  A pack whose
backend is not installed contributes nothing, so ``import arrayscope`` never
eagerly imports the backend.
"""
