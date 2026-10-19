"""Canonical state and measured-control names for Bryan dynamics Cases 1-4."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sindy_viv._typing import CylinderType, DynamicsCase


@dataclass(frozen=True)
class CaseLayout:
    """Ordered channel names; lift-rate states follow all measured base states."""

    cylinder_ids: tuple[int, ...]
    base_names: tuple[str, ...]
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    lift_names: tuple[str, ...]
    rate_names: tuple[str, ...]

    @property
    def response_names(self) -> tuple[str, ...]:
        """Return state names followed by measured controls, excluding parameters."""
        return self.state_names + self.input_names

    @property
    def measured_names(self) -> tuple[str, ...]:
        """Return the measured channels required by this case, in loader order."""
        return self.base_names + self.input_names


def get_case_layout(case: DynamicsCase, cylinder_type: CylinderType) -> CaseLayout:
    """Describe Cases 1-4 without changing channels between fitting and evaluation.

    Case 1 models h, hdot and CL.
    Case 2 also models CLdot.
    Case 3 models h and hdot with prescribed CL.
    Case 4 models CL and CLdot with prescribed h, hdot and hddot.
    All rates use tau=t*/VR; this function never rescales data.

    Parameters:
        case (int): Physical dynamics case, one of 1, 2, 3, or 4.
        cylinder_type (str): Cylinder configuration: single or two_tandem.
    """
    if cylinder_type == "single":
        cylinders = (0,)
    elif cylinder_type == "two_tandem":
        cylinders = (0, 1)
    else:
        raise ValueError("cylinder_type must be 'single' or 'two_tandem'.")

    lift = tuple(f'C_{{L{i}}}' for i in cylinders)
    rates = tuple(rf'\dot{{C}}_{{L{i}}}' for i in cylinders)

    if isinstance(case, (bool, np.bool_)):
        raise ValueError("dynamics_case must be 1, 2, 3, or 4.")
    match case:
        case 1 | 2:
            base = tuple(name for i in cylinders for name in (f'h_{i}', rf'\dot{{h}}_{i}', f'C_{{L{i}}}'))
            controls = ()
            rates = rates if case == 2 else ()
        case 3:
            base = tuple(name for i in cylinders for name in (f'h_{i}', rf'\dot{{h}}_{i}'))
            controls, rates = lift, ()
        case 4:
            base = lift
            controls = tuple(name for i in cylinders for name in (f'h_{i}', rf'\dot{{h}}_{i}', rf'\ddot{{h}}_{i}'))
        case _:
            raise ValueError("dynamics_case must be 1, 2, 3, or 4.")

    return CaseLayout(cylinders, base, base + rates, controls, lift, rates)
