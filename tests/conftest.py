import os

# Use the rule-based lyrics scorer so tests never download the HuggingFace
# emotion model or require torch/transformers.
os.environ.setdefault("MOOD_READER_FAST", "1")
