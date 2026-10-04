# Archive

Superseded notebooks of the earlier 1,260-token path, the spike and the
resolution sweep. Kept as a record; **not runnable from this repository** (they were
pinned to commits of the private development history that do not exist here).

- `01_spike.ipynb`: the VLM spike: six configs on 40 documents (model and output-format choice), 1,260 tokens
- `02_zeroshot500.ipynb`: zero-shot over the 500 labelled documents at 1,260 tokens; superseded by 02n_zeroshot500_native
- `03_finetune.ipynb`: LoRA fine-tune at 1,260 tokens (fold 0 was trained from it); superseded by 03n_finetune_native
- `04_predict_test.ipynb`: v0 safety submission on the 200 test documents, 1,260 tokens, no OCR, no rules; superseded
- `04b_predict_test_v1.ipynb`: v1 = 04 plus OCR and the rules R1 to R3, 1,260 tokens; superseded by 04c_predict_test_native
- `04c_predict_test_v2.ipynb`: fine-tuned model plus OCR, rules and review flags at 1,260 tokens; superseded by 04c_predict_test_native (MODEL = ft)
- `05_oof_infer.ipynb`: out-of-fold inference of a 1,260-token fold adapter; superseded by 05n_oof_infer_native
- `05b_dev_final.ipynb`: the final adapter on the 100 dev documents at 1,260 tokens; superseded by 08_dev_final
- `06_res_sweep.ipynb`: the resolution sweep on 40 dev documents that chose the native setting (reports/res_sweep.md); pinned to the private history
- `07_hdrhint_ab.ipynb`: continuation-page header-hint A/B at 1,260 tokens; its native successor (07n) was cancelled before it was built
