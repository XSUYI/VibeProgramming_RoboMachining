from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
import vibeik.resources as res

print("OPENAI_API_KEY:", "set" if os.getenv("OPENAI_API_KEY") else "missing")

# Build the index the same way your app does
base_dir = Path("RobotResources")
idx = res.build_resource_index(base_dir)

print("Available tools (display names):", list(idx.display_tool_names.values()))
print("Available tools (keys):", list(idx.tools.keys()))

# Wrap the LLM function to confirm it runs
_called = {"hit": False}
_orig = res._llm_pick_candidate

def wrapped(query, candidates):
    _called["hit"] = True
    print("[DEBUG] _llm_pick_candidate called")
    print("  query:", query)
    print("  candidates:", candidates)
    return _orig(query, candidates)

res._llm_pick_candidate = wrapped

# Trigger the matching
print("match_tool('8mm drilling') ->", idx.match_tool("8mm drilling"))
print("LLM called?", _called["hit"])
