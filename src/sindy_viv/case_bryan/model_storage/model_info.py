"""Read saved model metadata and validate independent evaluation selections."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from sindy_viv.case_bryan.model_storage.paths import resolve_model_directory
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.feature_library import prepare_library


@dataclass
class ModelInfo:
    """A training.json, or equivalent metadata pointing to a manual model.csv."""

    directory: Path
    metadata: dict
    source: Path

    @classmethod
    def read(cls, path):
        path = Path(path).resolve()
        if path.is_dir():
            path /= 'training.json'
        metadata = json.loads(path.read_text(encoding='utf-8-sig'))
        directory = Path(metadata.get('model_directory', '.'))
        if not directory.is_absolute():
            directory = path.parent/directory
        directory = directory.resolve()
        if not (directory/'model.csv').is_file():
            raise FileNotFoundError(f'No model.csv in {directory}')
        if metadata.get('time_coordinate') != 'tau=t*/VR':
            raise ValueError('Model metadata must specify time_coordinate="tau=t*/VR".')
        get_case_layout(metadata['dynamics_case'], metadata['cylinder_type'])
        mode = metadata['parameterization']
        if mode not in ('individual', 'vr', 'lpd', 'vr_lpd'):
            raise ValueError(f'Unknown model parameterization: {mode}')
        if metadata['cylinder_type'] == 'single' and mode in ('lpd', 'vr_lpd'):
            raise ValueError('Single-cylinder models cannot parameterize L/D.')
        if not metadata.get('trajectories'):
            raise ValueError('Model metadata must list its training trajectories (VR and LpD).')
        configs = metadata['configurations']
        prepare_library(configs['state_feature_library'])
        if mode != 'individual':
            prepare_library(configs['parameter_feature_library'])
        return cls(directory, metadata, path)

    def validate_selection(self, vr, lpd):
        """Allow extrapolation only along this model's parameterized dimensions."""
        resolve_model_directory(self.directory, self.metadata['cylinder_type'],
            self.metadata['parameterization'], vr, lpd, model_dir=self.directory)
        # Manual model-info files have no adjacent training.json for the shared resolver.
        trajectories = self.metadata['trajectories']
        mode = self.metadata['parameterization']
        if mode in ('individual', 'lpd') and {row['VR'] for row in trajectories} != {vr}:
            raise ValueError(f'Model has a fixed VR different from {vr}: {self.directory}')
        if lpd is not None and mode in ('individual', 'vr') and {row['LpD'] for row in trajectories} != {lpd}:
            raise ValueError(f'Model has a fixed L/D different from {lpd}: {self.directory}')

    def manifest(self):
        return dict(model_directory=str(self.directory), metadata_source=str(self.source),
                    model_sha256=hashlib.sha256((self.directory/'model.csv').read_bytes()).hexdigest(),
                    training=self.metadata)

