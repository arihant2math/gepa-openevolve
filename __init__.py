"""
Generic OpenEvolve → GEPA adapter package.

This folder contains a GEPA adapter that can wrap any OpenEvolve
evaluator.  The key design goals are:

1. Require **no task-specific dataset loader**.  The batch passed to
   `evaluate()` may be a dummy list (e.g., `[None]`).  The underlying
   OpenEvolve evaluator is expected to discover traces / benchmarks
   internally, exactly as the reference Cant-Be-Late evaluator does.

2. Read all tunable parameters (LLM model, temperature, parallelism,
   etc.) from an `openevolve/config.yaml` file, falling back to sensible
   defaults when fields are missing.  This keeps runtime behaviour
   controllable by editing YAML alone.

To use:

```python
from gepa.adapters.generic_openevolve_adapter import GenericOpenEvolveAdapter
adapter = GenericOpenEvolveAdapter(
    evaluation_file="openevolve/examples/ADRS/cant-be-late/evaluator.py"
)

# dummy datasets because adapter ignores them
train = [None]
val   = [None]

from gepa import optimize
result = optimize(
    seed_candidate={"program": INITIAL_CODE},
    trainset=train,
    valset=val,
    adapter=adapter,
)
```
"""

from .generic_openevolve_adapter import GenericOpenEvolveAdapter  # noqa: F401
