"""Regression harness: measure what the pipeline does and what it costs, per machine, and write it as one results file.

  PYTHONPATH=tools python3 -m regress run            (just regress)
  PYTHONPATH=tools python3 -m regress idle|fingerprint|show FILE

See benchmarks/SCORECARD.md for what is measured and why, and issue #174 for the story it belongs to.
"""
