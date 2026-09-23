# Check single cylinder VIV data
from sindy_viv.case_bryan.input import *

re = 150
mstar = 5.1
vrs = [3, 4, 5, 6, 7, 8]

times = {vr: load_data_single_cylinder(re, mstar, vr)[0] for vr in vrs}
datas = {vr: load_data_single_cylinder(re, mstar, vr)[1] for vr in vrs}

# Check variable names
from sindy_viv.case_bryan.constants import SINGLE_CYLINDER_STATE_VARIABLES

print(f"Single cylinder VIV variable names: {SINGLE_CYLINDER_STATE_VARIABLES}")

# Perform derivative on single cylinder VIV data
from sindy_viv.differentiation import perform_derivative
from sindy_viv.utils import add_dot_text, initiate_object
from sindy_viv.case_bryan.prepare_sindy.input import prepare_input_single_parametric

# Perform derivative on single cylinder VIV data
module = 'sindy_viv.differentiation'
class_name = 'SavitzkyGolayDerivative'
kwargs = {'window_length': 100, 'polyorder': 3, 'deriv': 1}

from pysindy import FiniteDifference

differentiator = initiate_object(module=module, class_name=class_name, kwargs=kwargs)

t_s, X_s, X_dot_s, chosen_state_variables, U_s, chosen_input_variables = prepare_input_single_parametric(
    times=times,
    full_state_variables=datas,
    full_state_variable_names=SINGLE_CYLINDER_STATE_VARIABLES,
    differentiators=differentiator,
    input_variables=vrs,
    individual_case=1,
)

chosen_overall_variable_names = chosen_state_variables + chosen_input_variables

# Define a combined state and input variable dummy data
XU = np.concat([X_s[0], U_s[0]], axis=-1)

# Define feature library for SINDy
from sindy_viv.utils import initiate_object

from pysindy.differentiation.finite_difference import FiniteDifference

feature_library_kwargs = {
    'module': 'pysindy',
    'class_name': 'ParameterizedLibrary',
    'kwargs': {
        'feature_library': {
            'module': 'pysindy.feature_library.polynomial_library',
            'class_name': 'PolynomialLibrary',
            'kwargs': {
                'degree': 3,
                'include_interaction': True,
                'include_bias': False
            }
        },
        'parameter_library': {
            'module': 'pysindy.feature_library.polynomial_library',
            'class_name': 'PolynomialLibrary',
            'kwargs': {
                'degree': 1,
                'include_interaction': False,
                'include_bias': False
            }
        },
        'num_features': X_s[0].shape[1],
        'num_parameters': U_s[0].shape[1]
    }
}

feature_library = initiate_object(**feature_library_kwargs)

# print(f"Feature library: {feature_library}")

# Fit SINDy model to derivative data
feature_library.fit(x=XU)

# Check fitted features
fitted_features = feature_library.get_feature_names(chosen_overall_variable_names)

print(f"Fitted features: {fitted_features}")

# Define optimizer for SINDy
optimizer_kwargs = {
    'module': 'pysindy.optimizers.stlsq',
    'class_name': 'STLSQ',
    'kwargs': {
        'threshold': 1e-1,
        'alpha': 5e-2,
        'normalize_columns': False,
        'verbose': True
    }
}

optimizer = initiate_object(**optimizer_kwargs)

print(f"Optimizer: {optimizer}")

# Perform SINDy
from sindy_viv.sindy.functions import perform_sindy

sindy_model = perform_sindy(
    time=t_s,
    state_variable_data=X_s,
    feature_library=feature_library,
    optimizer=optimizer,
    state_variable_differentiated_data=X_dot_s,
    differentiation_method=None,
    feature_names=chosen_overall_variable_names,
    input_variable_data=U_s
)