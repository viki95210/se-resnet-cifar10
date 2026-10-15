# Report assets - SE-ResNet (24CS3062)

- `docs/` - your IEEE report PDF and the original SE paper (arXiv 1709.01507v4)
- `figures/from_report/` - Fig. 1-3 exactly as embedded in the report (extracted from the PDF)
- `figures/regenerated/` - clean re-plots from the training logs (test acc, post-decay zoom, loss, train-vs-test, ablation first-19-epochs, params vs r)
- `figures/from_notebook/01_main_results/` - plots straight from the notebook that are safe to use
- `figures/from_notebook/02_caveated_see_report/` - usable only with the caveats written in the report (r=16 ablation checkpoint, invalid ablation plot, time plot with the 1317 s outlier epoch)
- `figures/from_notebook/03_not_used_as_evidence/` - channel-visualisation plots; the hook captured SE *outputs*, not gate values, so do not present them as attention weights
- `results/` - the 4 CSVs from the notebook + `key_numbers.md`
- `logs/` - per-epoch history (loss/acc/time) parsed from the notebook output for all 6 runs
- Model checkpoints (.pth, ~45 MB each) are NOT included - put them in a GitHub Release or Git LFS.
