"""Pipeline phase modules (plan / execute / validate / replan / compile).

Each phase file imports shared helpers from :mod:`brain_os.pipeline_runtime` only,
never from :mod:`brain_os.pipeline`, to avoid circular imports with the orchestrator.
"""
