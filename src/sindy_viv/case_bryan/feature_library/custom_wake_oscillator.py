"""Configurable wake-oscillator factories returning explicitly selected polynomial features.

The response library acts on states followed by measured controls. VR and L/D
are excluded here and handled by the parameter library during parametric fits.
Case 4 therefore supports measured acceleration coupling. Cases 2 and 5 do not
contain acceleration, so that option is rejected rather than inferred.

Restoring options: linear, cubic. Damping options: linear, van_der_pol,
rayleigh. Each may be a string or list. Van der Pol expands into CLdot and
CL^2*CLdot; Rayleigh expands into CLdot and CLdot^3, with independent fitted
coefficients. Coupling options are
displacement, velocity, acceleration (Case 4 only), nonlinear (all response
polynomials), and nonlinear_motion (only polynomials containing h or hdot,
without acceleration), through nonlinear_degree. Empty lists disable a term family.

The lift-rate identity and, for Case 2, structural linear terms are always
included. Tandem libraries contain both cylinders' terms; every target can use
either cylinder's features. inter_cylinder_degree adds products of matching
channels across cylinders through the requested total degree (zero disables).
"""

from __future__ import annotations

from collections.abc import Sequence
from sindy_viv._typing import CylinderType, DynamicsCase
from typing import Any

from itertools import combinations_with_replacement
from numbers import Integral

from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout

# Retain the old lookup name solely for existing CustomLibrary model pickles.
from ..compatibility.wake_callables import _WakeFunction
from sindy_viv.feature_library.selected_polynomial_library import SelectedPolynomialLibrary


def _options(values: Sequence[Any], allowed: set[str], name: str) -> tuple[str, ...]:
    """Normalize one option or an option list, rejecting unknown values and removing duplicates.

    Parameters:
        values (Sequence[Any]): One named term option or an ordered sequence of term options.
        allowed (set[str]): Recognized string options for this term family.
        name (str): Argument or parameter label used in validation messages.
    """
    if isinstance(values, str):
        values = (values,)
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError(f"{name} must be a string or list from {sorted(allowed)}.") from None
    if any(not isinstance(value, str) or value not in allowed for value in values):
        raise ValueError(f"{name} must select from {sorted(allowed)}.")
    return tuple(dict.fromkeys(values))


def _make_library(
    names: Sequence[str],
    cylinder_type: CylinderType,
    case: DynamicsCase,
    restoring: str | Sequence[str],
    damping: str | Sequence[str],
    coupling: str | Sequence[str],
    nonlinear_degree: int,
    inter_cylinder_degree: int,
    include_bias: bool,
) -> SelectedPolynomialLibrary:
    """Select ordered physical/polynomial monomials for the canonical state/control layout.

    Parameters:
        names (Sequence[str]): Ordered variable names identifying array columns.

        cylinder_type (str): Cylinder configuration: single or two_tandem.
        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.
        restoring (str or Sequence[str]): Selected restoring terms: linear, cubic, or a sequence of these.
        damping (str or Sequence[str]): Selected damping terms: linear, van_der_pol, rayleigh, or a sequence of these.
        coupling (str or Sequence[str]): Selected motion or nonlinear coupling families; acceleration is available only in Case 4.
        nonlinear_degree (int): Maximum total degree for the selected nonlinear coupling terms.
        inter_cylinder_degree (int): Maximum total degree of matching-channel cross-cylinder products; zero disables them.
        include_bias (bool): Whether to include a constant feature equal to one.
    """
    restoring = _options(restoring, {'linear', 'cubic'}, 'restoring')
    damping = _options(damping, {'linear', 'van_der_pol', 'rayleigh'}, 'damping')
    coupling = _options(coupling, {'displacement', 'velocity', 'acceleration', 'nonlinear', 'nonlinear_motion'}, 'coupling')

    if 'acceleration' in coupling and case != 4:
        raise ValueError(f"Acceleration coupling requires Case 4's measured acceleration controls; Case {case} has none.")

    if isinstance(nonlinear_degree, bool) or not isinstance(nonlinear_degree, Integral) or nonlinear_degree < 1:
        raise ValueError("nonlinear_degree must be a positive integer.")

    if (isinstance(inter_cylinder_degree, bool) or not isinstance(inter_cylinder_degree, Integral)
            or inter_cylinder_degree < 0 or inter_cylinder_degree == 1):
        raise ValueError("inter_cylinder_degree must be zero or an integer of at least two.")

    if cylinder_type == 'single' and inter_cylinder_degree:
        raise ValueError("Inter-cylinder products require cylinder_type='two_tandem'.")

    if not isinstance(include_bias, bool):
        raise ValueError("include_bias must be a boolean.")

    terms = {}

    def add_term(*factors: tuple[str, int]) -> None:
        """Resolve named factors to one exponent vector, preserving first-insertion order.

        Parameters:
            factors (tuple[str, int]): Variable/exponent pairs forming one monomial. Default: no extra arguments
        """
        powers = [0] * len(names)

        for variable, exponent in factors:
            powers[names.index(variable)] += exponent

        terms[tuple(powers)] = None

    cylinders = (0,) if cylinder_type == 'single' else (0, 1)
    channels = []
    for cylinder in cylinders:
        lift = f'C_{{L{cylinder}}}'
        rate = rf'\dot{{C}}_{{L{cylinder}}}'
        displacement, velocity, acceleration = f'h_{cylinder}', rf'\dot{{h}}_{cylinder}', rf'\ddot{{h}}_{cylinder}'

        add_term((rate, 1))

        if case == 2:

            # Structural equations still need h, hdot and lift even when wake terms are restricted.
            for variable in (displacement, velocity, lift):
                add_term((variable, 1))

        for option in restoring:
            add_term((lift, 1 if option == 'linear' else 3))

        for option in damping:
            add_term((rate, 1))

            if option == 'van_der_pol':
                add_term((lift, 2), (rate, 1))

            elif option == 'rayleigh':
                add_term((rate, 3))

        for option, variable in (('displacement', displacement), ('velocity', velocity), ('acceleration', acceleration)):
            if option in coupling:
                add_term((variable, 1))

        channels.append([lift, rate, displacement, velocity] + ([acceleration] if case == 4 else []))

    if 'nonlinear' in coupling:
        for degree in range(1, nonlinear_degree + 1):
            for indices in combinations_with_replacement(range(len(names)), degree):
                add_term(*[(names[index], 1) for index in indices])

    if 'nonlinear_motion' in coupling:

        # Every coupling monomial contains h or hdot; pure wake terms and acceleration stay excluded.
        motion_names = [name for cylinder in cylinders for name in (f'h_{cylinder}', rf'\dot{{h}}_{cylinder}')]
        candidates = [name for name in names if name in motion_names or name.startswith(('C_', r'\dot{C}'))]
        for degree in range(1, nonlinear_degree + 1):
            for factors in combinations_with_replacement(candidates, degree):
                if any(name in motion_names for name in factors):
                    add_term(*[(name, 1) for name in factors])

    if inter_cylinder_degree:
        for first, second in zip(*channels):
            for degree in range(2, inter_cylinder_degree + 1):
                for first_power in range(1, degree):
                    add_term((first, first_power), (second, degree - first_power))

    return SelectedPolynomialLibrary(input_names=tuple(names), term_powers=tuple(terms), include_bias=include_bias)

class WakeOscillatorCase2Library:
    """Return selected monomials for motion/lift states in the canonical Case 2 layout."""

    def __new__(
        cls,
        cylinder_type: CylinderType = 'single',
        restoring: str | Sequence[str] = 'linear',
        damping: str | Sequence[str] = 'van_der_pol',
        coupling: str | Sequence[str] = ('displacement', 'velocity'),
        nonlinear_degree: int = 3,
        inter_cylinder_degree: int = 0,
        include_bias: bool = False,
    ) -> SelectedPolynomialLibrary:
        """Build a selected polynomial library using the canonical case response layout.

        Parameters:
            cylinder_type (str): Cylinder configuration: single or two_tandem. Default: 'single'
            restoring (str or Sequence[str]): Selected restoring terms: linear, cubic, or a sequence of these. Default: 'linear'
            damping (str or Sequence[str]): Selected damping terms: linear, van_der_pol, rayleigh, or a sequence of these. Default: 'van_der_pol'
            coupling (str or Sequence[str]): Selected motion or nonlinear coupling families; acceleration is available only in Case 4. Default: ('displacement', 'velocity')
            nonlinear_degree (int): Maximum total degree for the selected nonlinear coupling terms. Default: 3
            inter_cylinder_degree (int): Maximum total degree of matching-channel cross-cylinder products; zero disables them. Default: 0
            include_bias (bool): Whether to include a constant feature equal to one. Default: False
        """
        names = list(get_case_layout(2, cylinder_type).response_names)
        return _make_library(names, cylinder_type, 2, restoring, damping, coupling,
                             nonlinear_degree, inter_cylinder_degree, include_bias)

class WakeOscillatorCase4Library:
    """Return selected monomials for lift states followed by measured motion controls."""

    def __new__(
        cls,
        cylinder_type: CylinderType = 'single',
        restoring: str | Sequence[str] = 'linear',
        damping: str | Sequence[str] = 'van_der_pol',
        coupling: str | Sequence[str] = ('acceleration',),
        nonlinear_degree: int = 3,
        inter_cylinder_degree: int = 0,
        include_bias: bool = False,
    ) -> SelectedPolynomialLibrary:
        """Build a selected polynomial library using the canonical case response layout.

        Parameters:
            cylinder_type (str): Cylinder configuration: single or two_tandem. Default: 'single'
            restoring (str or Sequence[str]): Selected restoring terms: linear, cubic, or a sequence of these. Default: 'linear'
            damping (str or Sequence[str]): Selected damping terms: linear, van_der_pol, rayleigh, or a sequence of these. Default: 'van_der_pol'
            coupling (str or Sequence[str]): Selected motion or nonlinear coupling families; acceleration is available only in Case 4. Default: ('acceleration',)
            nonlinear_degree (int): Maximum total degree for the selected nonlinear coupling terms. Default: 3
            inter_cylinder_degree (int): Maximum total degree of matching-channel cross-cylinder products; zero disables them. Default: 0
            include_bias (bool): Whether to include a constant feature equal to one. Default: False
        """
        names = list(get_case_layout(4, cylinder_type).response_names)
        return _make_library(names, cylinder_type, 4, restoring, damping, coupling,
                             nonlinear_degree, inter_cylinder_degree, include_bias)


class WakeOscillatorCase5Library:
    """Return Case 5 wake terms with displacement/velocity coupling and no acceleration input."""

    def __new__(
        cls,
        cylinder_type: CylinderType = 'single',
        restoring: str | Sequence[str] = 'linear',
        damping: str | Sequence[str] = 'van_der_pol',
        coupling: str | Sequence[str] = ('displacement', 'velocity'),
        nonlinear_degree: int = 3,
        inter_cylinder_degree: int = 0,
        include_bias: bool = False,
    ) -> SelectedPolynomialLibrary:
        """Build a selected polynomial library using the canonical case response layout.

        Parameters:
            cylinder_type (str): Cylinder configuration: single or two_tandem. Default: 'single'
            restoring (str or Sequence[str]): Selected restoring terms: linear, cubic, or a sequence of these. Default: 'linear'
            damping (str or Sequence[str]): Selected damping terms: linear, van_der_pol, rayleigh, or a sequence of these. Default: 'van_der_pol'
            coupling (str or Sequence[str]): Selected motion or nonlinear coupling families; acceleration is available only in Case 4. Default: ('displacement', 'velocity')
            nonlinear_degree (int): Maximum total degree for the selected nonlinear coupling terms. Default: 3
            inter_cylinder_degree (int): Maximum total degree of matching-channel cross-cylinder products; zero disables them. Default: 0
            include_bias (bool): Whether to include a constant feature equal to one. Default: False
        """
        names = list(get_case_layout(5, cylinder_type).response_names)
        return _make_library(names, cylinder_type, 5, restoring, damping, coupling,
                             nonlinear_degree, inter_cylinder_degree, include_bias)
