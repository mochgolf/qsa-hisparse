# G6-GPU response

Review: `reviews/G6GPU.md` (not cleared; 2 medium, 2 low).

1. **Medium — G2-3 cold `prompt_tokens_details`.** Accepted. F returns
   `null`, P `{"cached_tokens": 0}`: upstream #43005 (`11cd2701a8`, in
   v0.5.22: "preserve zero cached tokens when cache reporting is enabled";
   production's base 35f3c96ff4 has `_details_if_cached`, v0.5.22 does not).
   Not plugin behavior. The results table now reports G2-3 as passing with
   this upstream response change; DEVIATIONS.md has a D8 addendum pending
   the owner's acceptance. `run_g23.py` now requires each arm's exact cold
   form (`COLD_DETAILS`: fork `None`, plugin `{"cached_tokens": 0}`), with a
   test that the swapped forms fail; the window's smoke reports satisfy it.
2. **Medium — latency attribution overstated.** Accepted. The conclusion now
   reads "likely explanation", limits the plugin statement to "no
   substantial regression in the sampled cache path and hooks", and lists
   what the evidence does not cover (CPU-timed contract check, profiled vs
   unprofiled, interpreter and plugin changing together). STATUS aligned.
3. **Low — skip count.** Accepted. "1 skip (SM121)"; `pytest_outcomes.py`
   counts only file/reason skip entries in its summary (re-run: "91 outcomes
   + 1 skipped per arm (F); 4 plugin-only tests", PASS).
4. **Low — profiling statistics.** Accepted. Sample groups labelled
   (before / after / all ten unprofiled medians: 0.2002/0.2218,
   0.2183/0.2402, 0.2095/0.2310 s); the scheduling span is the first
   profiled request, the other figures are per-request averages.

Evidence tests: `tests/evidence` 120 passed.
