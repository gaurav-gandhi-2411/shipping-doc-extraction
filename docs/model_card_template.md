---
# TEMPLATE: nothing here is published. Fill every {{placeholder}} from the run artifacts named in
# the section, never from memory. Delete this comment block when filling.
license: apache-2.0  # chosen by GG; same licence as the base model (see Licence below)
base_model: Qwen/Qwen3.5-4B  # pinned revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - lora
  - document-extraction
  - invoices
  - air-waybills
---

# {{model_name}}

LoRA adapter on `Qwen/Qwen3.5-4B` (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) that
extracts structured JSON from page images of commercial invoices and air waybills. Adapter code
SHA: `{{code_sha}}`; adapter sha256: `{{adapter_sha256}}` (from `final/manifest.json`).

## Training data

- Documents: {{n_docs}} documents ({{n_pages}} page images): commercial invoices and air
  waybills from {{n_suppliers}} suppliers, each page annotated with a field-level JSON target
  (header fields and line items).
- Splits: {{n_train_docs}} train / {{n_dev_docs}} dev; {{n_test_docs}} further documents have no
  labels. Folds are leave-supplier-out ({{n_folds}} folds); the adapter released here was trained
  on {{released_adapter_scope}} (one of: fold K, all {{n_docs}} documents).
- **The data is confidential and is not distributed**, and neither are its labels, supplier
  names, identifiers or any extracted value. The model may have memorised some of it; do not
  assume its outputs are free of training-set content.
- Training recipe: `configs/finetune_qwen35_4b.yaml` (seed 42), {{epochs}} epochs,
  {{optimizer_steps}} optimizer steps, LoRA rank {{lora_rank}}, {{trainable_params}} trainable
  parameters.

## Intended use

Research and evaluation of image-only extraction of shipping documents into the task schema, with
a human review step: the calibrated review flag below marks fields that should be checked.

## Out-of-scope use

- Fully automated processing of financial, customs or legal records without human review.
- Document types, languages, layouts or scan qualities outside the training distribution
  ({{covered_document_types}}).
- Any use that requires the base model's licence terms to be waived; see Licence below.
- Extracting personal data from documents you have no right to process.

## How to load

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

BASE = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
processor = AutoProcessor.from_pretrained(BASE, revision=REVISION)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE, revision=REVISION, torch_dtype=torch.float16  # fp16 on a T4 (no bf16); bf16 on an L4
)
model = PeftModel.from_pretrained(base, "{{adapter_repo_or_path}}")
model = model.merge_and_unload()  # optional: merge the adapter into the base weights
```

Inference settings used for every number below: image only, keyed output format, prompt
`{{prompt_version}}`, greedy decoding, thinking off, `max_pixels` {{max_pixels}},
`max_new_tokens` {{max_new_tokens}}, and the post-processing of `{{postproc_commit}}`. Reproduce
with the repository at `<PINNED_SHA>` (see "Reproduce from images" in the README).

## Evaluation protocol

Leave-supplier-out: a document's supplier never appears in the training set of the fold that
predicts it, so every score is out of fold. Scorer: the official task scorer (not redistributed).
Confidence intervals are {{ci_method}} (document-level bootstrap, {{n_boot}} resamples, seed 42).
The baseline is the zero-shot base model with the same prompt, decoding and post-processing,
compared on exactly the same documents (paired). Both are always reported.

## Metrics

Source: `{{metrics_artifact}}` at commit `{{metrics_commit}}`. n = {{n_eval_docs}} documents from
{{n_eval_suppliers}} held-out suppliers.

| System | Overall | Header | Line items | 95% CI |
|---|---|---|---|---|
| Zero-shot base (baseline) | {{zs_overall}} | {{zs_header}} | {{zs_rows}} | {{zs_ci}} |
| This adapter | {{ft_overall}} | {{ft_header}} | {{ft_rows}} | {{ft_ci}} |
| Paired difference | {{delta_overall}} | {{delta_header}} | {{delta_rows}} | {{delta_ci}} |

Also: JSON-valid pages {{json_valid_rate}}, false-fill rate {{false_fill_rate}}. Per-field and
per-document-type breakdowns: {{per_field_artifact}}. State plainly if the difference to the
baseline is within the noise floor ({{noise_floor}}).

## Calibration and review flag

Each emitted field carries a confidence from the token logprobs of its value. A calibrator
({{calibrator_type}}, fitted on {{calibrator_fit_data}}) maps it to a probability of being
correct and a field is flagged for review below the threshold {{review_threshold}}.

| Quantity | Value | 95% CI |
|---|---|---|
| Expected calibration error | {{ece}} | {{ece_ci}} |
| Fields flagged for review | {{flag_rate}} | {{flag_rate_ci}} |
| Errors caught by the flag (recall) | {{flag_recall}} | {{flag_recall_ci}} |
| Accuracy of unflagged fields | {{unflagged_accuracy}} | {{unflagged_ci}} |

Source: `{{calibration_artifact}}` at `{{calibration_commit}}`. The calibrator was fitted on
{{calibrator_scope}}; its numbers are out of fold only if stated here: {{calibration_oof_note}}.

## Limitations

- Small, supplier-clustered evaluation set: confidence intervals are wide and a supplier that
  differs from the training suppliers can fail in ways these numbers do not show.
- Image-only input at a fixed resolution cap; very small print, stamps and handwriting are the
  expected failure cases ({{known_failure_modes}}).
- Dates, amounts and identifiers can be wrong while looking plausible; the review flag lowers but
  does not remove this risk ({{flag_recall}} of errors caught, see above).
- Possible memorisation of confidential training documents (see Training data).
- Not evaluated for fairness, robustness to adversarial documents or prompt injection inside
  documents.

## Environmental impact and compute

- Hardware: {{gpu_types}} (Google Colab T4 and/or L4).
- Fine-tuning: {{train_gpu_hours}} GPU hours; OOF inference: {{infer_gpu_hours}} GPU hours;
  zero-shot baseline: {{zs_gpu_hours}} GPU hours; compute units used: {{compute_units}}.
- Estimates are labelled ESTIMATE where not measured; carbon: {{carbon_estimate_or_not_measured}}.

## Licence and citations

- Adapter licence: Apache-2.0. GG chose to release the adapter weights under the same licence as
  the base model.
- Base model: `Qwen/Qwen3.5-4B`, licence {{base_model_license}} (read from the model card at fetch
  time, not from the pinned revision; its terms apply to the merged weights).
- Cite the base model as published by its authors: {{base_model_citation}}.
- Libraries: `transformers`, `peft` (versions pinned in `uv.lock`).

```bibtex
@misc{ {{citation_key}},
  title  = { {{model_name}} },
  author = { {{authors}} },
  year   = { {{year}} },
  note   = { Code at commit {{code_sha}} }
}
```
