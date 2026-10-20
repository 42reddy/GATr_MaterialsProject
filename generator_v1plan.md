# Generator Fine-tuning Loop — v1 Plan

## Context
The GATr-based encoder predicts formation energies well but the generative side underperforms, likely due to small model size from compute constraints. Rather than scaling the in-house generator, the plan is to bootstrap from a strong open-source generative model (DiffCSP-family), use the existing encoder + an ML potential (CHGNet) as a filtering funnel, and feed the surviving structures back to fine-tune the generator — building a self-improving loop targeted at a specific use case (e.g., Na-ion battery electrolyte candidates).

## Refined Loop

1. **Conditional generation (the open question)**
   - DiffCSP does not natively condition on application/chemistry. Two practical routes:
     a. Use a composition-conditioned variant (e.g., DiffCSP++ / composition-constrained sampling) and restrict sampling to chemical spaces relevant to Na-ion electrolytes (Na + {O, S, P, Cl, ...}).
     b. Fine-tune the base DiffCSP on a curated subset pulled from Materials Project / OQMD restricted to the target chemistries, then sample unconditionally from the fine-tuned model.
   - Route (b) is simpler to implement first and sidesteps needing true conditional architecture support.

2. **Generate candidates**
   - Start small (~1–2k) to validate the full pipeline mechanics and tune thresholds before committing to 10k — relaxation cost adds up fast even with ML potentials.

3. **Cheap pre-filter before the encoder**
   - Add a geometry/validity sanity check (bond-length plausibility, density range, no overlapping atoms) using e.g. pymatgen, before spending encoder/CHGNet compute on obviously broken structures.

4. **Formation-energy filter (GATr encoder)**
   - Define the pass threshold relative to the convex hull / known competing phases for that chemical space, not an absolute energy cutoff — "unphysical" is relative to local thermodynamic context.

5. **CHGNet relaxation**
   - Define explicit pass criteria up front: post-relaxation energy/force convergence, and structural similarity pre- vs post-relaxation (large displacement = the generator proposed something CHGNet doesn't agree is a local minimum → reject).

6. **De-duplication / diversity check**
   - Use structure matching (pymatgen StructureMatcher) to drop near-duplicates before they're added back — avoids wasted compute and mode collapse in the generator.

7. **Feed back for fine-tuning**
   - Add passing structures to the **training** set (not the test set — keep a fixed held-out test set for honest evaluation across loop iterations).
   - Maintain a replay buffer of original training data when fine-tuning, to avoid catastrophic forgetting / narrowing of the generator's distribution.

8. **Echo-chamber guard**
   - Because the filter (your own encoder) and the thing being trained (your own generator) are coupled, errors can reinforce themselves. Periodically spot-check a small sample with DFT (or at least cross-check against an independent ML potential/encoder) to catch drift.

9. **Track metrics per iteration**
   - % surviving each stage, diversity/novelty vs. known materials (e.g., MP), and — where possible — downstream validation rate (DFT or experiment), to know whether the loop is actually improving the generator or just converging on its own biases.

## Pros
- Avoids retraining a large generator from scratch — leverages a strong pretrained model.
- The encoder-then-relax funnel is a sensible cheap→expensive triage, minimizing wasted DFT-like compute.
- Creates a flywheel that can specialize the generator toward a target application (Na-ion electrolytes) over time.

## Key Risks / Holdbacks (and mitigations above)
- **Conditional generation is unsolved** — pick route (b) fine-tuning first; it's the lower-effort path to a working pipeline.
- **Formation energy alone is a weak "unphysical" filter** — pair with a geometry pre-filter and hull-relative thresholds.
- **Self-reinforcing bias / model collapse** — replay buffer + periodic independent validation (DFT spot-check).
- **Test-set contamination** — only add to training data; keep test set frozen.
- **Compute cost at 10k scale** — pilot at ~1–2k first, then scale once thresholds and pass-rates are understood.
