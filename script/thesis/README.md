# Thesis-specific scripts

These scripts prepare local figures, numerical tables and copyable LaTeX blocks.
General fitting, evaluation and diagnostic entry points remain in `script/`.
Run the commands from the repository root with the project Python environment.

| Script | Purpose |
| --- | --- |
| `results_plan_data.py` | Prepare physical summaries, derivative/coefficients tables and structural IVP data in `img/04_results/data`. May evaluate existing models; never fits coefficients. |
| `generate_results_plan.py` | Render Chapter 4 figures/tables and complete LaTeX environments in `img/04_results`. |
| `generate_methodology_plan.py` | Render Chapter 3 histories/window illustrations and tables in `img/03_methodology`. |
| `generate_single_structural_results_figures.py` | Export the three figures used in the marked single-cylinder results section from saved fits and cached IVP traces. No fitting or integration. |
| `manuscript_assets.py` | Shared local paths, numerical-input import, table/figure snippets, gallery and independent review document. |

The default output parent is this repository's **`img/`**. The chapter generators
accept `--output-root` to relocate it. They never edit manuscript text, insert
chapter/appendix material, or write to the sibling thesis directory. Main versus
appendix is only a suggested grouping in exported files. Paraphrasing and actual
manuscript placement are deferred.

Each chapter folder contains standalone PDF/SVG/450-dpi PNG figures, a combined
PDF, `data/`, a manifest and gallery, and:

- `figures_all.tex`, `figures_main.tex`, `figures_appendix.tex`: complete
  `figure` environments with standalone-PDF paths and native widths.
- `tables_all.tex`: complete `table` environments. Tandem metrics are split by
  cylinder/spacing, so the blocks can be placed independently without longtable.
- `snippets/*.tex`: one figure or table block per file.
- `table_template.tex`: empty table for later editing.
- `preview.tex`: an independent local review document; compile from the repo root.

Snippets require `graphicx`/`booktabs` and reference `img/<chapter>/...`.
Copy the corresponding chapter assets with the snippets when integrating them
into another document later. Changing combined-PDF page order does not break
the standalone-PDF references.

```powershell
# Reuse the existing numerical preparation run for the review examples.
.venv/Scripts/python.exe -B script/thesis/generate_methodology_plan.py --run-dir state/results_plan_runs/20261030-061231Z --examples
.venv/Scripts/python.exe -B script/thesis/generate_results_plan.py --run-dir state/results_plan_runs/20261030-061231Z --examples

# Rebuild the lift-derivative CSV through the existing preparation API.
.venv/Scripts/python.exe -B script/thesis/results_plan_data.py --stage derivatives
```

The renderers require `--run-dir` for existing caches/statuses. `--examples`
renders 4 methodology and 21 results assets, covering every documented group;
omit it for the full 59/58 sets. `--reuse-figures` refreshes local tables/snippets
without rendering or editing the manuscript.

Data preparation stages are `physical`, `derivatives`, `coefficients`, `ivp` and
`joint`. Prepare coefficients before joint. With no `--run-dir`, data preparation
uses `img/04_results/data/run`. Existing numerical caches are reused; invalidate
the affected caches deliberately when changing numerical rather than visual
settings. No fits are launched by these scripts.

The previous output cleanup preserved a numerical-input ZIP. To import its
recognized CSV/JSON/NPZ inputs into the new local chapter layout:

```powershell
.venv/Scripts/python.exe -B script/thesis/results_plan_data.py --stage import --input-archive state/manuscript_regeneration_inputs/20261101T121349Z/regeneration_inputs.zip
```

The importer records hashes and refuses to overwrite different files. It does
not restore old PNG/SVG/PDF figures or manuscript chapters. Local reference
traces/tables are under `img/04_results/data/reference`. The standalone
single-cylinder renderer now defaults to `img/04_results/single_structural`
and reads these local reference traces; its independent formatting is outside
the chapter settings reference.

`generate_methodology_plan.py` imports the sibling `generate_results_plan.py`
for shared plotting/export helpers. Both direct invocation and
`python -m script.thesis.generate_methodology_plan --help` are supported.
The numerical package in `src/sindy_viv` has no dependency on these scripts.

[MANUSCRIPT_PLOTTING_SETTINGS.md](../../MANUSCRIPT_PLOTTING_SETTINGS.md) documents
all figure groups, embeds real rendered examples, and shows empty table previews.
