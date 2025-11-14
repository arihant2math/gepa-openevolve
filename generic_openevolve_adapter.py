from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Optional
from pathlib import Path

from gepa import EvaluationBatch, GEPAAdapter

class GenericOpenEvolveAdapter(GEPAAdapter[Any, Any, Any]):
    def __init__(self, path: Path, failed_score=-1e6):
    
        self.failed_score = failed_score
