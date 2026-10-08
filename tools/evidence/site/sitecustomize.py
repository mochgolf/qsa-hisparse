"""Site hook for the evidence observers (see ../observer.py, ../vit_observer.py).

Put this directory first on PYTHONPATH in both arms. With
QSA_EVIDENCE_OBSERVER_DIR set it installs the checkpoint observer's import
hook, with QSA_EVIDENCE_VIT_DIR the ViT encode observer's (Track I); it
imports neither torch nor SGLang. It then runs the sitecustomize module it
shadows (Ubuntu's apport hook in the validation interpreter), so the
interpreter starts as it would without this directory.
"""

import importlib.machinery
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.realpath(__file__))


def _install(name):
    spec = importlib.util.spec_from_file_location(
        f"qsa_evidence_{name}", os.path.join(os.path.dirname(_here), f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.install()


if os.environ.get("QSA_EVIDENCE_OBSERVER_DIR"):
    _install("observer")
if os.environ.get("QSA_EVIDENCE_VIT_DIR"):
    _install("vit_observer")

_shadowed = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.realpath(p or ".") != _here]
)
if _shadowed is not None:
    _shadowed.loader.exec_module(importlib.util.module_from_spec(_shadowed))
