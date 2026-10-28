"""Physical-case directories shared by raw-data analysis producers and readers."""

from pathlib import Path

from sindy_viv.case_bryan.model_storage.paths import number_text


def analysis_directory(results_root, case_name, cylinder_type, re, mstar, *, lpd=None, vr=None) -> Path:
    """Locate a raw-analysis aggregate or individual physical-case directory."""
    folder = Path(results_root)/case_name/'analysis'
    if cylinder_type == 'single':
        folder = folder/f'Re = {re}'/f'Mstar = {mstar}'
    elif cylinder_type != 'two_tandem':
        raise ValueError('cylinder_type must be single or two_tandem.')
    if lpd is not None:
        folder /= f'LpD = {float(lpd)}'
    if vr is not None:
        folder /= f'VR = {number_text(vr)}'
    return folder
