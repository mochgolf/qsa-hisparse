"""Site hook for the W8 checkpoint observer (see ../observer.py).

Put this directory first on PYTHONPATH in both arms. With
QSA_EVIDENCE_OBSERVER_DIR set it installs the observer's import hook; it
imports neither torch nor SGLang. It then runs the sitecustomize module it
shadows (Ubuntu's apport hook in the validation interpreter), so the
interpreter starts as it would without this directory.
"""

import importlib.machinery
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

if os.environ.get("QSA_EVIDENCE_OBSERVER_DIR"):
    _spec = importlib.util.spec_from_file_location(
        "qsa_evidence_observer", os.path.join(os.path.dirname(_here), "observer.py")
    )
    _observer = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _observer
    _spec.loader.exec_module(_observer)
    _observer.install()

_shadowed = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.abspath(p or ".") != _here]
)
if _shadowed is not None:
    _shadowed.loader.exec_module(importlib.util.module_from_spec(_shadowed))
