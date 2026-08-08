# ⚠️ STRICT ARCHITECTURAL PROTOCOL: DATA-ORIENTED DESIGN (DOD)

This project is a High-Performance N-Body and Orbital Mechanics engine. 
DO NOT use standard Object-Oriented Programming (OOP) for physical state. The universe does not obey geology; it obeys kinematics.

## 1. The Ontology (Handles, not Objects)
- `BaseBody`, `CelestialBody`, `Vessel` (`body.py`), and `SystemNode` (`topology.py`) are strictly **immutable, frozen dataclasses**. 
- They act ONLY as ID Cards / Handles. 
- They MUST NOT contain mutable state (no `self.r`, `self.v`, `self.children`).
- Any attempt to add kinematic state variables to these classes will be rejected.

## 2. The Centralized Memory (StateBuffer)
- ALL physical state lives in `state_buffer.py`.
- It uses flat, pre-allocated, C-contiguous NumPy `float64` arrays of shape `(N, 6)` for `local_curr`, `local_next`, `global_curr`, and `global_next`.
- It uses `int32` maps (`body_system_map`, `system_parent_map`) for blazing-fast NumPy advanced indexing.
- We use a Free List and 64-bit Bit-Packed Generational Indices (`UID = (generation << 32) | index`) to recycle memory safely. Do NOT bypass the UID bit-packing logic.

## 3. The Execution Pipeline (Two-Pass Step)
All time-stepping logic is executed by the stateless `Simulation` conductor (`simulator.py`) in two strict phases:
- **Pass 1 (Local Drift):** Stateless Propagator kernels read `local_curr`, calculate drift purely relative to a local system's origin, and write to `local_next`.
- **Pass 2 (Global Reduction):** The Simulator loops over active system frames, slices bodies via boolean masking (`mask = buffer.body_system_map == sid`), and broadcasts global absolute positions using a single matrix addition. 

## 4. AI Coding Directives
- **NO LOOPS OVER BODIES:** Do not write Python `for` loops over entities in the hot loop. You MUST use NumPy vectorized boolean masking and broadcasting.
- **READ THE MANIFEST:** Propagators are stateless math kernels. Keep logic decoupled from the data.
- **TYPES.PY:** Maintain the API boundary. Use `types.py` (e.g., `Kilometers`, `Seconds`) for function signatures, but strictly `NDArray[np.float64]` inside the arrays.

## 5. The Pacing & Pedagogy Protocol (CRITICAL)
- **Role:** You are a Principal Systems Architect. Your goal is to maximize the user's learning.
- **Explain First:** Before writing a new class, kernel, or SQL schema, you MUST write a brief plain-English explanation of its exact duty, memory footprint, and the underlying math/physics. 
- **Atomic Commits:** Do not generate more than ONE class, ONE kernel, or ONE buffer allocation at a time. Ask for approval before proceeding to the next file.

## 6. The SQL Database Boundary
- The relational database (SQLite/PostgreSQL) is strictly for **Initialization and Static Data** (e.g., planetary masses, base radii, default orbital elements). 
- **Zero Hot-Loop DB Access:** Under no circumstances should the engine query the database during the `Simulation.step()` hot loop. All DB data must be parsed by the Factories at startup and dumped into the `StateBuffer`.

## 7. The WebGL / Frontend Preparation
- The frontend will be a lightweight WebGL/Three.js application.
- To support this, the engine's output pipeline must be designed to serialize the `StateBuffer`'s contiguous NumPy arrays directly into raw binary buffers (e.g., `Float32Array` payloads) for easy consumption by Three.js Instanced Meshes.