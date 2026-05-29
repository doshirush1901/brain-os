# Brain OS pantheon

``src/brain_os/pantheon.py`` is **generated** by ``scripts/export_brain_os.py`` with the
starter roster (see ``EXPECTED_PANTHEON_AGENT_COUNT`` in that script).

When adding a custom agent:

1. Copy ``agents/_template_agent.py`` → ``agents/yours.py``
2. Add import + entry to ``_AGENT_CLASSES`` in ``pantheon.py`` (or re-export and patch)
3. Add ``prompts/yours_system.txt``
