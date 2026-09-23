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
module = 'sindy_viv.differentiation'
class_name = 'SavitzkyGolayDerivative'
kwargs = {'window_length': 100, 'polyorder': 3, 'deriv': 2}

differentiator = initiate_object(module=module, class_name=class_name, kwargs=kwargs)

derivative_data = perform_derivative(single_vr7_time, single_vr7_data, differentiator=differentiator)

single_cylinder_state_variable_names_with_dot = add_dot_text(SINGLE_CYLINDER_STATE_VARIABLES)
single_cylinder_state_variable_names_with_dot = add_dot_text(single_cylinder_state_variable_names_with_dot)

present_in_derivative = [var for var in SINGLE_CYLINDER_STATE_VARIABLES if var in single_cylinder_state_variable_names_with_dot]
print(f"Variables present in derivative data: {present_in_derivative}")

print(f"Derivative data shape: {derivative_data.shape}")

# Check figure
import matplotlib.pyplot as plt

from sindy_viv.case_bryan.plot import plot_data, plot_compare, plot_compare_prior_check

fig, axs = plot_data(
    time=single_vr7_time,
    data=single_vr7_data,
    state_variable_names=SINGLE_CYLINDER_STATE_VARIABLES,
    parameter_values_dict={'V_R': vr},
    xlim=(100, 200),
    ylim_mean_time_start=100,
    ylim_mean_time_end=200
)

fig2, axs2 = plot_compare_prior_check(
    time=single_vr7_time,
    data_true=single_vr7_data,
    data_pred=derivative_data,
    true_state_variable_names=SINGLE_CYLINDER_STATE_VARIABLES,
    pred_state_variable_names=single_cylinder_state_variable_names_with_dot,
    parameter_values_dict={'V_R': vr},
    xlim=(100, 200),
    ylim_mean_time_start=100,
    ylim_mean_time_end=200,
    ylim_source='true'
)

plt.show()