"""Test setup.

The real service needs a GPU, torch and unsloth. These tests only exercise the
HTTP layer, so heavy imports are replaced with light stand-ins before the app
is imported.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for name in ("torch", "unsloth"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name == "unsloth":
            stub.FastLanguageModel = object
        sys.modules[name] = stub
