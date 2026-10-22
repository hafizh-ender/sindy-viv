
from __future__ import annotations

from collections.abc import Sequence
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.optimizers import BaseOptimizer
from sindy_viv._typing import CylinderType, DynamicsCase
from typing import Any
import numpy as np
import pysindy as ps

from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.utils import initiate_object
from sindy_viv.feature_library.functions import fit_library_metadata

def prepare_constrained_sr3(
    feature_names: Sequence[str] | None,
    state_names: Sequence[str],
    *,
    case: DynamicsCase,
    cylinder_type: CylinderType,
    mstar: float,
    vr: float | None = None,
    parameter_names: Sequence[str] = (),
    input_names: Sequence[str] = (),
    optimizer_kwargs: dict[str, Any] | None = None,
) -> ps.ConstrainedSR3:
    """Construct ConstrainedSR3 with complete, fixed physical equations.

    Enforce h' = hdot and hdot' = -4*pi**2*h + 2*VR**2*CL/(pi*mstar)
    for each cylinder.
    Cases 2, 4 and 5 enforce CL' = CLdot; cases 4/5 have no structural state equations.
    Case 3 uses measured lift controls in the structural forcing term.
    Other equations remain free.
    Parametric libraries need linear state terms and a parameter bias.
    VR-varying models also need V_R^2; L/D-only models use the fixed vr argument.
    L/D does not enter the structural equation.

    Names must come from the fitted pointwise library in its actual order.
    mstar is the physical mass ratio, rather than the rounded data label.
    The returned optimizer exposes constraint_lhs and constraint_rhs.

    Parameters:
        feature_names (Sequence[str] or None): Expanded feature labels in the fitted library column order; a nonempty sequence is required.
        state_names (Sequence[str]): Ordered names of the fitted state columns.

        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.
        cylinder_type (str): Cylinder configuration: single or two_tandem.

        mstar (float): Positive finite physical mass ratio used in structural constraints.
        vr (float or None): Fixed reduced velocity; None is allowed when VR is included as a parameter or the case has no structural equation. Default: None
        parameter_names (Sequence[str]): Ordered names of the constant parameter columns. Default: ()

        input_names (Sequence[str]): Ordered measured-control names, followed by parameter names when applicable. Default: ()
        optimizer_kwargs (dict[str, Any] or None): Keyword settings forwarded to ConstrainedSR3. Default: None
    """
    layout = get_case_layout(case, cylinder_type)
    cylinder_ids = layout.cylinder_ids

    if not np.isscalar(mstar) or not np.isfinite(mstar) or mstar <= 0:
        raise ValueError("mstar must be a finite positive physical mass ratio.")

    parameter_names = list(parameter_names)
    if parameter_names not in ([], ['V_R'], ['L/D'], ['L/D', 'V_R']):
        raise ValueError("parameter_names must be [], ['V_R'], ['L/D'], or ['L/D', 'V_R'].")

    if case not in (4, 5) and 'V_R' not in parameter_names and (vr is None or not np.isscalar(vr) or not np.isfinite(vr) or vr <= 0):
        raise ValueError("Individual constraints require a finite positive vr.")

    feature_names, state_names = list(feature_names), list(state_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be nonempty and unique for physical constraints.")

    if not state_names or len(set(state_names)) != len(state_names):
        raise ValueError("State names must be nonempty and unique for physical constraints.")

    kwargs = dict(optimizer_kwargs or {})
    if kwargs.get('constraint_lhs') is not None or kwargs.get('constraint_rhs') is not None:
        raise ValueError("Physical constraints are generated from feature names; omit constraint_lhs and constraint_rhs.")

    if kwargs.get('inequality_constraints', False) or kwargs.get('equality_constraints') is False:
        raise ValueError("Bryan physical relations require equality constraints only.")

    if kwargs.get('unbias', False):
        raise ValueError("Physical constraints require unbias=False.")

    # PySINDy normalizes the library without rescaling Cw=d, changing physical units.
    if kwargs.get('normalize_columns', False):
        raise ValueError("Physical constraints require normalize_columns=False.")

    order = kwargs.get('constraint_order', 'feature')
    if order not in ('target', 'feature'):
        raise ValueError("constraint_order must be 'target' or 'feature'.")

    n_features, n_targets = len(feature_names), len(state_names)
    fixed_equations = {}
    prefix = '1 ' if parameter_names else ''
    lift_scale = 2 / (np.pi * mstar)

    # L/D-only fits retain a fixed VR, so its square belongs in the forcing coefficient.
    if case not in (4, 5) and 'V_R' not in parameter_names:
        lift_scale *= vr**2

    for cylinder_id in cylinder_ids:

        # Construct the physical equations for each cylinder.
        displacement = rf'h_{cylinder_id}'
        velocity = rf'\dot{{h}}_{cylinder_id}'
        lift = rf'C_{{L{cylinder_id}}}'

        if case not in (4, 5) and (lift not in list(state_names) + list(input_names) or velocity not in state_names):
            raise ValueError(f"Physical constraints require state {velocity!r} and state/input {lift!r}.")

        lift_feature = f'V_R^2 {lift}' if 'V_R' in parameter_names else prefix + lift
        equations = {} if case in (4, 5) else {
            displacement: {prefix + velocity: 1.0},
            velocity: {prefix + displacement: -4 * np.pi**2, lift_feature: lift_scale},
        }

        if case in (2, 4, 5):
            lift_rate = rf'\dot{{C}}_{{L{cylinder_id}}}'

            if lift_rate not in state_names:
                raise ValueError(f"Case {case} requires the rate state {lift_rate!r}.")

            equations[lift] = {prefix + lift_rate: 1.0}

        # Up to this point, equations are expressed in dictionary form with symbolic names.
        # Example for case 2, single cylinder, first-order polynomial library, no parameters:
        # equations = {
        #    'h_0': { '\dot{h}_0': 1.0 },
        #    '\dot{h}_0': { 'h_0': -4*pi**2, 'C_{L0}': 2/(pi*mstar)*VR^2 },
        #    'C_{L0}': { '\dot{C}_{L0}': 1.0 }  # Case 2 only
        # }

        # Convert symbolic names to feature indices and coefficients for the optimizer.
        for target_name, terms in equations.items():
            if target_name not in state_names:
                raise ValueError(f"Physical constraints require state {target_name!r}.")

            coefficients = np.zeros(n_features, dtype=np.float64)

            for feature_name, value in terms.items():
                if feature_name not in feature_names:
                    raise ValueError(
                        f"Physical constraints require library feature {feature_name!r}. "
                        "Include linear state terms and, for parametric fits, a parameter bias and V_R^2."
                    )
                coefficients[feature_names.index(feature_name)] = value

            fixed_equations[state_names.index(target_name)] = coefficients

        # Up to this point, fixed_equations is a dictionary mapping target indices to coefficient arrays.
        # Example for case 2, single cylinder, first-order polynomial library, no parameters:
        # fixed_equations = {
        #     index('h_0'): {
        #         index('1'): 0.0,
        #         index('h_0'): 0.0,
        #         index('\dot{h}_0'): 1.0,
        #         index('C_{L0}'): 0.0,
        #         index('\dot{C}_{L0}'): 0.0
        #     },
        #     index('\dot{h}_0'): {
        #         index('1'): 0.0,
        #         index('h_0'): -4*pi**2,
        #         index('\dot{h}_0'): 0.0,
        #         index('C_{L0}'): 2/(pi*mstar)*VR^2,
        #         index('\dot{C}_{L0}'): 0.0
        #     },
        #     index('C_{L0}'): {
        #         index('1'): 0.0,
        #         index('h_0'): 0.0,
        #         index('\dot{h}_0'): 0.0,
        #         index('C_{L0}'): 0.0,
        #         index('\dot{C}_{L0}'): 1.0
        #     },

    # Pin every coefficient in each known equation, including all unwanted terms to zero.
    # LHS has one nonzero per row but uses dense storage required by this ConstrainedSR3 path.
    # RHS is a vector of the known values for each constraint.
    lhs = np.zeros((len(fixed_equations) * n_features, n_features * n_targets), dtype=np.float64)
    rhs = np.empty(len(lhs), dtype=np.float64)

    # lhs: (3 * 5, 5 * 4) = (15, 20) for case 2, single cylinder, first-order polynomial library, no parameters
    # rhs: (15,) for case 2, single cylinder, first-order polynomial library, no parameters

    feature_indices = np.arange(n_features)
    for equation_index, (target_index, coefficients) in enumerate(fixed_equations.items()):

        # 0, (index(h_0), {index('1'): 0.0, index('h_0'): 0.0, index('\dot{h}_0'): 1.0, index('C_{L0}'): 0.0, index('\dot{C}_{L0}'): 0.0})
        # 1, (index(\dot{h}_0), {index('1'): 0.0, index('h_0'): -4*pi**2, index('\dot{h}_0'): 0.0, index('C_{L0}'): 2/(pi*mstar)*VR^2, index('\dot{C}_{L0}'): 0.0})
        # 2, (index(C_{L0}), {index('1'): 0.0, index('h_0'): 0.0, index('\dot{h}_0'): 0.0, index('C_{L0}'): 0.0, index('\dot{C}_{L0}'): 1.0})

        rows = equation_index * n_features + feature_indices

        # rows = 0 * n_features + [0, 1, 2, 3, 4] = [0, 1, 2, 3, 4]
        # rows = 1 * n_features + [0, 1, 2, 3, 4] = [5, 6, 7, 8, 9]
        # rows = 2 * n_features + [0, 1, 2, 3, 4] = [10, 11, 12, 13, 14]

        if order == 'target':
            columns = target_index * n_features + feature_indices

            # columns = 0 * n_features + [0, 1, 2, 3, 4] = [0, 1, 2, 3, 4]
            # columns = 1 * n_features + [0, 1, 2, 3, 4] = [5, 6, 7, 8, 9]
            # columns = 2 * n_features + [0, 1, 2, 3, 4] = [10, 11, 12, 13, 14]
        else:
            columns = feature_indices * n_targets + target_index

        lhs[rows, columns] = 1.0

        # lhs[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]] = 1.0
        # lhs[[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]] = 1.0
        # lhs[[10, 11, 12, 13, 14], [10, 11, 12, 13, 14]] = 1.0

        rhs[rows] = coefficients

        # rhs[[0, 1, 2, 3, 4]] = [0.0, 0.0, 1.0, 0.0, 0.0]
        # rhs[[5, 6, 7, 8, 9]] = [0.0, -4*pi**2, 0.0, 2/(pi*mstar)*VR^2, 0.0]
        # rhs[[10, 11, 12, 13, 14]] = [0.0, 0.0, 0.0, 0.0, 1.0]

    # Up to this point, lhs is a 2D array with one row per constraint and one column per coefficient in the flattened coefficient array.
    # Example for case 2, single cylinder, first-order polynomial library, no parameters:
    # In target order, lhs has identity(15) in its first 15 columns; feature order interleaves columns.
    # rhs = [
    #     0.0,                      # Coefficients for h_0 equation
    #     0.0,
    #     1.0,
    #     0.0,
    #     0.0,
    #     0.0,                      # Coefficients for \dot{h}_0 equation
    #     -4*pi**2,
    #     0.0,
    #     2/(pi*mstar)*VR^2,
    #     0.0,
    #     0.0,                      # Coefficients for C_{L0} equation
    #     0.0,
    #     0.0,
    #     0.0,
    #     1.0
    # ]

    # Exempt fixed equations from the sparse proximal step so coef_ also satisfies Cw=d.
    weights = np.asarray(kwargs.get('reg_weight_lam', 0.005), dtype=np.float64)
    if weights.ndim not in (0, 2) or (weights.ndim == 2 and weights.shape != (n_targets, n_features)):
        raise ValueError("reg_weight_lam must be a scalar or an array shaped (n_targets, n_features).")

    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("reg_weight_lam must contain finite nonnegative values.")

    weights = np.broadcast_to(weights, (n_targets, n_features)).copy()
    weights[list(fixed_equations.keys()), :] = 0.0
    regularizer = kwargs.get('regularizer', 'l0').lower()
    if regularizer not in ('l0', 'l1', 'l2', 'weighted_l0', 'weighted_l1', 'weighted_l2'):
        raise ValueError("Unsupported ConstrainedSR3 regularizer.")

    kwargs.update(
        constraint_lhs=lhs,
        constraint_rhs=rhs,
        constraint_order=order,
        equality_constraints=True,
        inequality_constraints=False,
        unbias=False,
        normalize_columns=False,
        regularizer=regularizer if regularizer.startswith('weighted_') else 'weighted_' + regularizer,
        reg_weight_lam=weights,
    )

    if not hasattr(ps, 'ConstrainedSR3'):
        raise ImportError("ConstrainedSR3 requires PySINDy's cvxpy extra. Install the project dependencies with uv sync.")

    return ps.ConstrainedSR3(**kwargs)

def prepare_optimizer(
    optimizer_config: dict[str, Any],
    feature_library: BaseFeatureLibrary,
    state_data: np.ndarray | Sequence[np.ndarray],
    state_names: Sequence[str],
    *,
    case: DynamicsCase,
    cylinder_type: CylinderType,
    mstar: float,
    vr: float | None = None,
    input_data: np.ndarray | Sequence[np.ndarray] | None = None,
    parameter_names: Sequence[str] = (),
    input_names: Sequence[str] | None = None,
) -> BaseOptimizer:
    """Instantiate an ordinary optimizer or construct physical ConstrainedSR3 constraints.

    state_data/input_data are arrays or trajectory lists; the first trajectory
    establishes feature names. input_names orders measured controls followed by
    parameter_names. Recognized shape-only libraries fit metadata on one row;
    unknown/reciprocal libraries retain full fitting/validation. The library may
    be fitted/mutated. Returns an optimizer; ordinary optimizers bypass metadata
    fitting. Invalid dimensions or incompatible physical settings raise ValueError.

    Parameters:
        optimizer_config (dict[str, Any]): Optimizer object descriptor with constructor settings.
        feature_library (BaseFeatureLibrary): Feature-library instance used to construct the regression basis.
        state_data (numpy.ndarray or Sequence[numpy.ndarray]): Prepared state samples used to establish feature metadata.
        state_names (Sequence[str]): Ordered names of the fitted state columns.

        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.
        cylinder_type (str): Cylinder configuration: single or two_tandem.

        mstar (float): Positive finite physical mass ratio used in structural constraints.
        vr (float or None): Fixed reduced velocity; None is allowed when VR is included as a parameter or the case has no structural equation. Default: None
        input_data (numpy.ndarray, Sequence[numpy.ndarray], or None): Measured control samples aligned with the state rows, or None. Default: None
        parameter_names (Sequence[str]): Ordered names of the constant parameter columns. Default: ()

        input_names (Sequence[str] or None): Ordered measured-control names, followed by parameter names when applicable. Default: None
    """
    if optimizer_config['class_name'] != 'ConstrainedSR3':
        return initiate_object(**optimizer_config)

    parameter_names = list(parameter_names)
    input_names = list(parameter_names if input_names is None else input_names)
    if isinstance(state_data, (list, tuple)) and not state_data:
        raise ValueError("state_data must contain at least one trajectory.")

    samples = state_data[0] if isinstance(state_data, (list, tuple)) else state_data
    samples = np.asarray(samples)

    if samples.ndim != 2 or samples.shape[1] != len(state_names) or not len(samples):
        raise ValueError("state_data must contain nonempty arrays with one column per state name.")

    inputs = None
    if input_names:
        if input_data is None:
            raise ValueError("Constraints with controls or parameters require input_data.")

        if isinstance(input_data, (list, tuple)) and not input_data:
            raise ValueError("input_data must contain at least one trajectory.")

        inputs = input_data[0] if isinstance(input_data, (list, tuple)) else input_data
        inputs = np.asarray(inputs)

        if inputs.shape != (len(samples), len(input_names)):
            raise ValueError("input_data must align with state_data and input_names.")

    elif input_data is not None:
        raise ValueError("input_data requires input_names or parameter_names.")

    # Resolve library ordering before constructing the coefficient constraint columns.
    fit_library_metadata(feature_library, samples, inputs)
    feature_names = feature_library.get_feature_names(list(state_names) + input_names)

    return prepare_constrained_sr3(
        feature_names,
        state_names,
        case=case,
        cylinder_type=cylinder_type,
        mstar=mstar, vr=vr,
        parameter_names=parameter_names,
        input_names=input_names,
        optimizer_kwargs=optimizer_config.get('kwargs', {}),
    )
