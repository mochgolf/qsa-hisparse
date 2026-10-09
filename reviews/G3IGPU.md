One nonblocking documentation finding: [window3-results.md:6](/docs/window3-results.md:6) cites `d5/plugin/results.json`, which does not exist. The actual artifact is [d5/results.json](qwen:results/plugin-window3-20261008/d5/results.json), as written by `d5_check.py`; it supports the reported D5 pass.

No GPU acceptance gaps found. The reviewed source tree, frozen fixture SHA, restore hashes, ViT behavior, token comparisons and text controls check out within the GPU gate’s scope.

G3-I (GPU): cleared