# Historical documents

Superseded material, retained for provenance. **None of this describes the current codebase.**
Do not use any of it as a specification, and do not restore APIs from it.

| File | What it was | Why it is here |
|---|---|---|
| `antigravity-dod-protocol.md` | Architectural protocol written to steer AI assistants during the DOD migration | Describes an architecture that was never built |
| `dod-migration-blueprint.md` | Prompt/blueprint for the OOP → DOD migration (v1.3.0) | The migration is complete; its code samples target a pre-vectorisation API |
| `coe_from_rv.py` | First functional prototype for state-vector reconstruction (v0.1.0) | Superseded by `src/orbital_engine/frames.py`; untouched since Nov 2025 |

## Specific claims that are false today

`antigravity-dod-protocol.md` references:

- `state_buffer.py` and `topology.py` — neither module exists; arena state lives in `simulator.py`
- `local_curr` / `local_next` / `global_curr` / `global_next` — actual names are `local_states` and
  `global_states`, and there is no double-buffering
- `body_system_map` / `system_parent_map` — actual names are `body_sys_map` and `parent_indices`
- 64-bit bit-packed generational indices (`UID = (generation << 32) | index`) — never implemented;
  slot allocation is a plain free list
- `types.py` — renamed `custom_types.py` in v1.3.0 to avoid shadowing the standard library

`dod-migration-blueprint.md` additionally assumes a NamedTuple COE object with attribute access
(`coe.theta`, `coe.u`, `coe.lambda_true`, `coe._replace(...)`). The COE representation is now a plain
`(N, 6)` float64 array, and `rv_to_coe` returns `(ndarray, success_mask)`.

---

Note: `antigravity-dod-protocol.md` was previously `.antigravity.md` in the repository root, where it
may have been read automatically by the Antigravity IDE. It was renamed on the move because
`.gitignore` matches that basename at any depth, which would have left it untracked here.
