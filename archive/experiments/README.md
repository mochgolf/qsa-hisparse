# Experiment harnesses

These are frozen copies of the harnesses used to produce the committed reduced
results. They are evidence and starting points for reproduction, rather than a
portable benchmark package.

Run CPU-only checks before any GPU or service command. Replace the original
workspace, model, NUMA, artifact, and endpoint values with values for the target
host. Service drivers assume an idle baseline and a separate free test endpoint;
read the adjacent contract before running one.

The useful entry points are:

- `assessment/replay_lru.py`: offline hot-cache replay
- `microbench/microbench.py`: transfer and cache-policy microbenchmarks
- `integration/integration_spike.py`: storage and decode integration checks
- `profile/profile_driver.py`: bounded component attribution
- `fastpath/b1_driver.py`: B1 fast-path service comparison
- `fastpath/bgt1_service_driver.py`: B2/B8 fast-path comparison
- `production/scaling_driver.py`: B1/B2/B4/B8 256K scaling run
