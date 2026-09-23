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

# Perform derivative on single cylinder VIV data
module = 'pysindy'
class_name = 'FiniteDifference'
kwargs = {'order': 2}

differentiator = initiate_object(module=module, class_name=class_name, kwargs=kwargs)

derivative_data = perform_derivative(single_vr7_time, single_vr7_data, differentiator=differentiator)

print(f"Derivative data shape: {derivative_data.shape}")