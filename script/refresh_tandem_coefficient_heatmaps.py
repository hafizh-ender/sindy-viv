"""Split existing combined tandem coefficient reports without refitting models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sindy_viv.case_bryan.evaluate_sindy.single_suite import _nrmse_heatmap, _write_json


def refresh_combined_heatmaps(root: Path, manifest_path: Path) -> list[dict]:
    """Replace two-target reports with verified per-cylinder PNG/CSV/JSON reports.

    Read full-precision saved values; do not load or alter model coefficients.
    Only exact combined-output filenames below the resolved root are removed,
    after both replacements exist and their CSV values match the original.
    Preserve each original JSON payload in the migration manifest.
    """
    root = root.resolve()
    sources = sorted(set(root.rglob('hddot_coefficients_summary.json')) |
                     set(root.rglob('hddot_tensor_coefficients_summary.json')))
    migrated = []
    for source in sources:
        source = source.resolve()
        if not source.is_relative_to(root):
            raise ValueError('Combined coefficient report escaped the selected root.')
        original_bytes = source.read_bytes()
        payload = json.loads(original_bytes)
        targets = [r'\ddot{h}_0', r'\ddot{h}_1']
        if payload.get('targets') != targets:
            continue
        tensor = source.name == 'hddot_tensor_coefficients_summary.json'
        old_stem = source.name.removesuffix('_summary.json')
        old_csv = source.with_name(old_stem + '_heatmap.csv')
        original = pd.read_csv(old_csv, index_col=0, converters={0:str})
        records = payload['records']
        columns = [row['LpD'] if tensor else row['VR'] for row in records]
        replacements = []
        for cylinder,target in enumerate(targets):
            prefix = target + r':\ '
            keys = [key for key in records[0]['values'] if key.startswith(prefix)]
            terms = [key[len(prefix):] for key in keys]
            if not terms or any(set(key for key in row['values'] if key.startswith(prefix)) != set(keys)
                                for row in records):
                raise ValueError(f'Inconsistent target terms in {source}')
            rows = [dict(**{key:value for key,value in row.items() if key!='values'},
                         values={term:row['values'][old] for term,old in zip(terms,keys)})
                    for row in records]
            stem = f'hddot{cylinder}_' + ('tensor_coefficients' if tensor else 'coefficients')
            path = source.with_name(stem + '_heatmap.png')
            labels = [' '.join(factor for factor in term.split() if factor!='1') or '1'
                      for term in terms] if tensor else terms
            _nrmse_heatmap(rows,terms,columns,path,weights=True,
                           xlabel=r'$L/D$' if tensor else r'$V_R$',row_labels=labels)
            new_payload = dict(payload,targets=[target],target=target,records=rows)
            if tensor:
                new_payload['features'] = terms
            _write_json(source.with_name(stem + '_summary.json'),new_payload)
            replacement = pd.read_csv(path.with_suffix('.csv'),index_col=0,converters={0:str})
            if list(replacement.index)!=terms or list(replacement.columns)!=list(original.columns):
                raise ValueError('Replacement coefficient axes do not match the original report.')
            np.testing.assert_allclose(replacement.to_numpy(),original.loc[keys].to_numpy(),
                                       rtol=1e-14,atol=0,equal_nan=True)
            replacements.extend([str(path),str(path.with_suffix('.csv')),
                                 str(source.with_name(stem + '_summary.json'))])
        if source.read_bytes()!=original_bytes:
            raise ValueError('Combined source report changed during plot regeneration.')
        obsolete = [source,old_csv,source.with_name(old_stem + '_heatmap.png')]
        for path in obsolete:
            if not path.resolve().is_relative_to(root):
                raise ValueError('Refusing to remove a report outside the selected root.')
        entry = dict(original_summary=str(source),original_sha256=hashlib.sha256(original_bytes).hexdigest(),
                     original_payload=payload,replacement_files=replacements,
                     removed_files=[str(path) for path in obsolete if path.is_file()])
        migrated.append(entry)
        _write_json(manifest_path,migrated)
        for path in obsolete:
            path.unlink(missing_ok=True)
    _write_json(manifest_path,migrated)
    return migrated


def main() -> None:
    """Reformat only saved tandem coefficient plots and record all replaced artifacts."""
    parser = argparse.ArgumentParser(
        description='Split saved combined tandem coefficient maps into hddot0/hddot1 reports.',
    )
    parser.add_argument(
        '--results_root',
        type=Path,
        default=Path('results/oscillating-cylinder-tandem-viv'),
        help='Dataset subtree containing saved combined coefficient summaries.',
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        default=Path('state/tandem_coefficient_split.json'),
        help='Migration manifest, including original JSON payloads and replacement paths.',
    )
    args = parser.parse_args()
    result = refresh_combined_heatmaps(args.results_root,args.manifest)
    print(f'Split {len(result)} combined coefficient reports into separate cylinder heatmaps.')


if __name__ == '__main__':
    main()
