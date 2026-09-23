# Check single cylinder VIV data
from sindy_viv.case_bryan.input import *

re = 150
mstar = 5.1
vr = 3

single_vr7_time, single_vr7_data = load_data_single_cylinder(re, mstar, vr)

print(f"Single cylinder VIV data for Re={re}, m*={mstar}, Vr={vr}:")
print(f"Time data shape: {single_vr7_time.shape}")
print(f"Data shape: {single_vr7_data.shape}")

# Check variable names
from sindy_viv.case_bryan.constants import SINGLE_CYLINDER_STATE_VARIABLES

print(f"Single cylinder VIV variable names: {SINGLE_CYLINDER_STATE_VARIABLES}")

# Perform derivative on single cylinder VIV data
from sindy_viv.differentiation import perform_derivative
from sindy_viv.utils import add_dot_text, initiate_object
from sindy_viv.case_bryan.prepare_sindy.input import prepare_input_single_case_1, prepare_input_single_case_2

# Perform derivative on single cylinder VIV data
module = 'sindy_viv.differentiation'
class_name = 'SavitzkyGolayDerivative'
kwargs = {'window_length': 100, 'polyorder': 3, 'deriv': 1}

differentiator = initiate_object(module=module, class_name=class_name, kwargs=kwargs)

t, X, X_dot, chosen_state_variables = prepare_input_single_case_2(
    time=single_vr7_time,
    full_state_variable_data=single_vr7_data,
    full_state_variable_names=SINGLE_CYLINDER_STATE_VARIABLES,
    differentiator=differentiator
)

print(f"Prepared data for Re={re}, m*={mstar}, Vr={vr}:")
print(f"Time data shape: {t.shape}")
print(f"State variable data shape: {X.shape}")
print(f"State variable derivative data shape: {X_dot.shape}")
print(f"Chosen state variables: {chosen_state_variables}")

# Define feature library for SINDy
from sindy_viv.utils import initiate_object

feature_library_kwargs = {
    'module': 'pysindy.feature_library.polynomial_library',
    'class_name': 'PolynomialLibrary',
    'kwargs': {
        'degree': 3,
        'include_interaction': True,
        'include_bias': False
    }
}

feature_library = initiate_object(**feature_library_kwargs)

print(f"Feature library: {feature_library}")

# Fit SINDy model to derivative data
feature_library.fit(x=X)

# Check fitted features
fitted_features = feature_library.get_feature_names(chosen_state_variables)

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
    time=t,
    state_variable_data=X,
    feature_library=feature_library,
    optimizer=optimizer,
    state_variable_differentiated_data=X_dot,
    differentiation_method=None,
    feature_names=chosen_state_variables,
    input_variable_data=None
)