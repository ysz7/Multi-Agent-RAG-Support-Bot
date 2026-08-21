"""Import-time compatibility shims for the evaluation extra.

`ragas` 0.4.3 imports `langchain_community.chat_models.vertexai`, a module that
langchain-community removed in 0.4. The import is unconditional and at module
level, so `import ragas` raises `ModuleNotFoundError` before any of our code
runs — even though nothing here touches Vertex AI.

Pinning langchain-community below 0.4 is not an option: it requires
`langchain-core<1.0`, and LangGraph 1.x requires `langchain-core>=1.0`. So the
missing module is stubbed instead. The stub is only ever imported by ragas'
provider-detection code, which compares classes it will never match.

Confined to `evals/` on purpose: the runtime install has neither ragas nor
langchain-community in it.
"""

from __future__ import annotations

import importlib.util
import sys
import types

_STUBBED = "langchain_community.chat_models.vertexai"


def patch_langchain_community() -> bool:
    """Stub the module ragas expects. Returns True if a stub was installed."""
    if _STUBBED in sys.modules:
        return False
    if importlib.util.find_spec("langchain_community") is None:
        return False
    try:
        if importlib.util.find_spec(_STUBBED) is not None:
            return False  # a version that still ships it: leave it alone
    except (ImportError, AttributeError, ValueError):
        pass  # parent package refuses to resolve the child; stub it below

    module = types.ModuleType(_STUBBED)

    class ChatVertexAI:  # noqa: D401 - placeholder, never instantiated
        """Placeholder for a model class this project never uses."""

    module.ChatVertexAI = ChatVertexAI
    sys.modules[_STUBBED] = module
    return True
