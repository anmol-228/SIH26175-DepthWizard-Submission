# FINAL CLAIM / EVIDENCE MATRIX — DepthWizard

> **Checkpoint note:** model results in this document were produced with earlier or experiment-specific checkpoints (the frozen B0/iter2 baseline or the named research checkpoint), **not with the v3 runner default**. Percentage reductions and RMSE figures against a model-derived or oracle-fitted reference describe agreement with that reference and are not deployable accuracy claims.

> **Public-snapshot note:** this file comes from the team's private development repository. Files it names that are not in this snapshot (working logs, raw metric outputs, private handoff files, numbered audit documents, preregistration and pre-run freeze records) remain in that private record; nothing in this snapshot needs them to run.

## Purpose

This document freezes the bounded claims supported by the current repository state for evidence and release review.

It distinguishes:

* implemented capabilities;
* demonstrated experimental evidence;
* unresolved validation gaps; and
* claims that must **not** be made.

This document does not replace the underlying evidence artifacts or historical Phase-1/Phase-2 records.

---

# 1. Claim Matrix

| Claim                                        | Evidence                                                                       | Status                      | Boundary                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Working non-stubbed end-to-end pipeline      | DW-TASK2/3 evidence                                                            | **CONFIRMED**               | Demonstrates real pipeline execution; does not establish field or deployment accuracy                                                               |
| P-14 reference-DSM UI                        | P0-A2 / #62                                                                    | **CONFIRMED**               | Reference-DSM comparison UI is implemented; UI capability does not imply accuracy validation                                                        |
| P0-A3 image-upload UI                        | #64 / `a9943b7`                                                                | **CONFIRMED**               | Image-upload UI and server plumbing are merged; uploaded-image end-to-end execution remains separately bounded                                      |
| P0-A5 hilly-relief reference-elevation gate  | #67 / A5 evidence                                                              | **CONFIRMED WITH BOUNDARY** | Gate can refuse metric calibration when supplied reference-point relief exceeds the provisional threshold; it does not solve hilly-terrain accuracy |
| Sparse/desert evaluation evidence exists     | #65 / Issue #48 evidence                                                       | **CONFIRMED WITH BOUNDARY** | Evidence exists for the tested sparse/desert experiment; this does not establish general sparse/desert accuracy                                     |
| Canonical adapted sparse-10 result           | `finetuned_calibration_report_iter2_partial_unfreeze.json` / `result_5501.txt` | **CONFIRMED**               | Global sparse-10 result of **5.500616560872978 m RMSE**; remains separate from the native-0.09m experiment                                          |
| Native-0.09m sparse-10 result                | `result_4914.txt` / `investigation_A1_original.json`                           | **CONFIRMED**               | Separate native-resolution/GSD sparse-10 result of **4.91391568540946 m RMSE**; not final, field, or deployment accuracy                            |
| Genuine satellite-domain accuracy validation | Phase-2 evidence                                                               | **NOT VALIDATED**           | Satellite probes or satellite-input experiments do not establish quantitative satellite-domain accuracy validation                                  |
| Hilly-terrain accuracy solved                | Phase-2 hilly-terrain evidence                                                 | **NOT SUPPORTED**           | Hilly-terrain accuracy remains a known severe failure; A5 is a refusal/safety mechanism, not an accuracy fix                                        |
| Oracle result is deployable accuracy         | `result_4914.txt` / `investigation_A1_original.json`                           | **NOT SUPPORTED**           | Full per-tile oracle calibration gives **4.403400056893794 m RMSE** as a non-deployable upper bound                                                 |
| External mentor/jury demo rights             | `RIGHTS_FOR_EXTERNAL_DEMO = UNVERIFIED`                                        | **UNCONFIRMED**             | No recorded approval decision has been identified; rights must be explicitly confirmed before external-demo claims are made                         |

---

# 2. Core Results and Required Separation

## 2.1 5.500616560872978 m RMSE

This is the **historical canonical adapted sparse-10 result**.

The preserved evaluation evidence shows a global sparse-calibration result of:

**5.500616560872978 m RMSE**

with:

* **MAE:** 3.80322027223354 m
* **Correlation:** 0.8757490941961965
* **Bias:** -0.8565458709261182 m
* **10 calibration points per tile**

The preserved reproduction path documents:

```text
EVAL_CKPT_NAME=da2_small_finetuned_iter2_partial_unfreeze.pt
EVAL_OUT_NAME=finetuned_calibration_report_rerun_nirmalya.json
python scripts/07_calibrate_and_eval.py
```

The current repository also preserves the result in:

```text
outputs/metrics/finetuned_calibration_report_iter2_partial_unfreeze.json
```

and the corresponding result artifact:

```text
result_5501.txt
```

Required boundaries:

* Keep this value unchanged.
* Treat it as the canonical adapted sparse-10 result.
* Do not replace it with the native-0.09m experiment.
* Do not average it with the 4.9139 m result.
* Do not describe it as accuracy achieved on every experiment.
* Do not describe it as field-validated or deployment accuracy.

## 2.2 4.91391568540946 m RMSE

This is a **separate native-0.09m / GSD sparse-10 experiment**.

The preserved evidence reports:

* **GSD:** 0.09 m
* **Global sparse-10 RMSE:** 4.91391568540946 m
* **MAE:** 3.445559943853313 m
* **Correlation:** 0.8979771449203373
* **Bias:** 0.6633125707140248 m

Evidence is preserved in:

```text
result_4914.txt
investigation_A1_original.json
```

The repository also contains a recipient-side reproduction at:

**4.913933638861965 m RMSE**

This reproduces the same separate A1/native-0.09m regime and does not replace the historical 4.91391568540946 m value.

Required boundaries:

* It is a legitimate separate experimental result.
* It is not the final system accuracy.
* It is not field or deployment accuracy.
* It must remain analytically separate from the 5.5006 m canonical result.
* It must not be averaged with the 5.5006 m result.

## 2.3 4.403400056893794 m RMSE

This is the **full per-tile oracle calibration result**.

The preserved native-0.09m evidence reports:

**4.403400056893794 m RMSE**

for full per-tile oracle calibration.

Required boundaries:

* It represents a non-deployable upper bound.
* It uses information unavailable in the intended deployment setting.
* It must not be presented as deployment accuracy.
* It must not be presented as field accuracy.
* It must not be used as the headline system accuracy.

---

# 3. Mandatory Result Separation

The following results belong to different experimental regimes:

| Result       | Regime                                 | Use                         |
| ------------ | -------------------------------------- | --------------------------- |
| **5.5006 m** | Historical canonical adapted sparse-10 | Canonical historical result |
| **4.9139 m** | Native-0.09m / GSD sparse-10           | Separate experiment         |
| **4.4034 m** | Full per-tile oracle calibration       | Non-deployable upper bound  |

### Mandatory rule

**5.5006 m and 4.9139 m must never be averaged, merged, or presented as one result.**

Likewise:

**4.4034 m must never be presented as deployment or field accuracy.**

---

# 4. P0-A5 Hilly-Relief Reference-Elevation Gate

P0-A5 adds a failure-aware reference-elevation relief gate.

The preserved A5 evidence demonstrates two real invocations of the actual pipeline entry point:

1. A well-spread reference set with **25.92 m** relief passed the provisional **87.5 m** threshold and entered the metric branch.
2. A well-spread reference set with **200.0 m** relief exceeded the threshold and was refused metric calibration, falling back to the relative branch.

The refusal is non-erroring and prevents a metric artifact from being emitted for the triggering case.

## Threshold

The current provisional threshold is:

**87.5 m**

The preserved analysis reports:

* known-good samples: **7.774–44.355 m**
* known-failure samples: **130.650–266.360 m**
* frozen midpoint threshold: **87.50271606445312 m**
* code constant: **87.5 m**

The threshold remains **provisional** and is not a validated general statistical error-rate separator.

## A5 boundary

A5 does **not** establish that hilly-terrain accuracy has been solved.

The following boundaries remain mandatory:

* **Hilly refusal ≠ hilly accuracy solved.**
* The gate observes only the elevation spread of the reference points actually supplied.
* A genuinely hilly scene whose supplied reference points under-sample the elevation range may not trigger this particular gate.
* The threshold is based on a small preserved evidence set and has not been independently validated as a general error-rate separator.
* The underlying hilly-terrain accuracy failure remains disclosed.
* A5 is a failure-aware refusal mechanism, not a correction to the underlying model/calibration weakness.

---

# 5. Hilly-Terrain Status

Current repository evidence indicates that genuine hilly terrain remains a severe failure mode.

The current verdict therefore remains:

**CONDITIONAL GO — Low-Medium confidence**

Do not claim:

* that hilly terrain is solved;
* that the system has reliable hilly-terrain accuracy;
* that the A5 gate makes all hilly scenes safe;
* that the 87.5 m threshold catches every hilly scene.

The A5 gate improves failure handling for cases where the supplied reference-point elevations expose sufficiently large relief, but it does not eliminate the underlying hilly-terrain limitation.

---

# 6. Sparse / Desert Boundary

The repository contains preserved sparse/desert evaluation evidence associated with Issue #48 / P-18.

This supports the bounded claim that:

**A sparse/desert evaluation experiment and corresponding evidence exist.**

It does **not** support the stronger claims that:

* sparse/desert accuracy is generally validated;
* sparse/desert performance is solved across unseen scenes;
* the tested result represents a universally established sparse/desert RMSE;
* the system is generally validated for all sparse/desert environments.

Therefore:

**Sparse/desert evidence exists ≠ sparse/desert accuracy validated.**

---

# 7. Satellite-Domain Boundary

The current repository does not establish genuine quantitative satellite-domain accuracy validation.

A satellite probe, satellite-input test, or interaction with satellite imagery is not sufficient by itself to establish domain accuracy.

Therefore:

**Satellite probe ≠ satellite-domain accuracy validation.**

Do not claim that DepthWizard is satellite-validated unless independent quantitative satellite-domain evaluation evidence is available.

Current status:

**NOT VALIDATED**

---

# 8. Oracle Boundary

The **4.4034 m** result comes from full per-tile oracle calibration.

Oracle calibration uses information that is not available in the intended deployment setting.

Therefore:

**Oracle ≠ deployable.**

The 4.4034 m result must remain explicitly labeled:

> **Non-deployable oracle upper bound**

It must not be presented as:

* deployment accuracy;
* field accuracy;
* final system accuracy; or
* independently validated real-world accuracy.

---

# 9. P0-A3 Image Upload Boundary

P0-A3 / #64 is **merged into `main`** and provides the image-upload UI/server plumbing.

The implementing commit is:

`a9943b7 — P0-A3: image upload UI (server plumbing fixed, pipeline execution untested pending checkpoint) (#64)`

This commit remains the evidence reference for the upload UI/server-plumbing claim. It is not the current repository HEAD.

Therefore the safe claim is:

**Image-upload UI/server plumbing is implemented and merged.**

However, the merge itself does not independently establish complete uploaded-image end-to-end execution under the final checkpoint/runtime conditions.

Do not convert:

**upload UI implemented**

into:

**uploaded-image end-to-end accuracy validated**

without corresponding runtime evidence.

---

# 10. Working End-to-End Pipeline Boundary

The preserved DW-TASK2/3 evidence demonstrates a real, non-stubbed end-to-end pipeline execution.

This supports:

**DepthWizard has a working non-stubbed end-to-end pipeline.**

However, that evidence does not independently establish field or deployment accuracy because the relevant controls/reference information were not independent field truth.

Therefore:

**Pipeline execution ≠ field/deployment validation.**

---

# 11. External Demo Rights — P0-D1

The current repository evidence does not identify a recorded P0-D1 rights decision.

The viewer verification evidence explicitly records:

`RIGHTS_FOR_EXTERNAL_DEMO = UNVERIFIED`

Searches for P0-D1, rights decisions, external-demo wording, mentor/jury display, and the rights status did not identify a separate recorded approval decision.

Therefore the current status is:

**EXTERNAL DEMO RIGHTS — UNCONFIRMED**

Do not claim that external mentor/jury display rights have been confirmed.

Before the final external-demo freeze, the responsible team member must explicitly record the decision and its scope.

The decision should establish, at minimum:

* whether the relevant imagery/data may be displayed externally;
* whether derived visualizations/results may be displayed;
* whether mentor/jury presentation is permitted;
* any attribution requirements;
* any restrictions on redistribution or publication.

Until that decision is recorded:

**Rights status must remain UNCONFIRMED.**

---

# 12. Final Mandatory Claim Boundaries

The following rules are mandatory for the evidence freeze:

* **Hilly refusal ≠ hilly accuracy solved.**
* **Sparse/desert evidence exists ≠ sparse/desert accuracy validated.**
* **Satellite probe ≠ satellite-domain accuracy validation.**
* **Oracle ≠ deployable.**
* **5.5006 m and 4.9139 m are separate experimental regimes.**
* **5.5006 m and 4.9139 m must not be averaged.**
* **4.4034 m oracle RMSE must not be presented as deployment or field accuracy.**
* **A merged UI feature does not by itself establish end-to-end accuracy validation.**
* **Pipeline execution does not by itself establish field/deployment accuracy.**
* **External demo rights must not be described as confirmed until the P0-D1 decision is explicitly recorded.**

---

# 13. Safe Claims for External Presentation

The following claims are supported by the current evidence:

## We CAN say

* DepthWizard has a working non-stubbed end-to-end pipeline.
* The system supports metric output when the calibration gate passes.
* The system supports relative output when metric calibration is not requested or is refused.
* The system can refuse metric calibration when supplied reference-point geometry indicates unsafe relief.
* The canonical adapted sparse-10 result is **5.5006 m RMSE**.
* A separate native-0.09m sparse-10 experiment produced approximately **4.9139 m RMSE**.
* A failure-aware hilly-relief gate has been implemented.
* Sparse/desert evaluation evidence exists for the tested experiment.
* Reference-DSM comparison UI is implemented.
* Image-upload UI/server plumbing is implemented.

## We MUST NOT say

* "DepthWizard has 4.4034 m deployment accuracy."
* "Our final accuracy is 4.9139 m."
* "We achieved 5.5006 m on every experiment."
* "We solved hilly terrain."
* "DepthWizard is satellite validated."
* "The system has validated sparse/desert accuracy in general."
* "The browser runs at X FPS" unless FPS was actually measured on the stated machine/browser/configuration.
* "Adaptation made performance worse."
* "The metric output is field validated."
* "External mentor/jury display rights are confirmed" while P0-D1 remains unrecorded.

---

# 14. Evidence Freeze State

## Confirmed

* **#62** P-14 reference-DSM UI is landed.
* **#64** P0-A3 image-upload UI/server plumbing is merged.
* **#65** sparse/desert evaluation evidence is preserved.
* **#67** P0-A5 hilly-relief reference-elevation gate is present in the current release history.
* Non-stubbed end-to-end pipeline evidence exists.
* The **5.5006 m** canonical adapted sparse-10 result is preserved.
* The separate **4.9139 m** native-0.09m sparse-10 result is preserved.
* The **4.4034 m** oracle result is preserved as a non-deployable upper bound.
* The canonical sparse-10 evaluation path is documented through `scripts/07_calibrate_and_eval.py`, the iter-2 partial-unfreeze checkpoint, the test manifest, and the preserved 10-point sparse-calibration result.

## Remains bounded / unresolved

* Hilly-terrain accuracy remains a known severe failure.
* Genuine satellite-domain accuracy validation remains incomplete.
* General sparse/desert accuracy remains unvalidated beyond the preserved evidence.
* The A5 **87.5 m** threshold remains provisional.
* Uploaded-image end-to-end pipeline execution remains separately bounded.
* Live 3D FPS remains unverified unless a valid measurement is added.
* External mentor/jury demo rights remain **UNCONFIRMED** until the P0-D1 decision is explicitly recorded.

---

# 15. Current Overall Verdict

**CONDITIONAL GO — Low-Medium confidence**

The current evidence supports presenting DepthWizard as a functioning prototype with demonstrated pipeline mechanics and several implemented safeguards/features.

It does **not** support presenting the system as:

* field validated;
* satellite-domain validated;
* universally accurate across terrain types;
* solved for hilly terrain; or
* approved for external mentor/jury display until the P0-D1 rights decision is explicitly recorded.

The evidence freeze must preserve these boundaries.
