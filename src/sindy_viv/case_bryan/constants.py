import numpy as np

""" 
Single cylinder VIV data constants
"""
SINGLE_CYLINDER_DATA_DIR = "./data/oscillating-cylinder-viv"

SINGLE_CYLINDER_CASE_NAME = "oscillating-cylinder-viv"

SINGLE_CYLINDER_STATE_VARIABLES = (
    "h_0",
    "\\dot{h}_0",
    "\\ddot{h}_0",
    "C_{L0}",
    "C_{D0}",
)

SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES = {
    "h_0": "Normalized displacement at object 0.csv",
    "\\dot{h}_0": "Normalized velocity at object 0.csv",
    "\\ddot{h}_0": "Normalized acceleration at object 0.csv",
    "C_{L0}": "Lift coefficient at object 0.csv",
    "C_{D0}": "Drag coefficient at object 0.csv"
}

SINGLE_CYLINDER_MSTAR = 5.1
SINGLE_CYLINDER_MSTAR_TRUE = 2 / (np.pi * 0.125)
SINGLE_CYLINDER_RE = 150

""" 
Two tandem cylinder VIV data constants
"""
TWO_TANDEM_CYLINDERS_DATA_DIR = "./data/oscillating-cylinder-tandem-viv"

TWO_TANDEM_CYLINDERS_CASE_NAME = "oscillating-cylinder-tandem-viv"

TWO_TANDEM_CYLINDERS_STATE_VARIABLES = (
    "h_0",
    "\\dot{h}_0",
    "\\ddot{h}_0",
    "C_{L0}",
    "C_{D0}",
    "h_1",
    "\\dot{h}_1",
    "\\ddot{h}_1",
    "C_{L1}",
    "C_{D1}"
)

TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES = {
    "h_0": "Normalized displacement at object 0.csv",
    "\\dot{h}_0": "Normalized velocity at object 0.csv",
    "\\ddot{h}_0": "Normalized acceleration at object 0.csv",
    "C_{L0}": "Lift coefficient at object 0.csv",
    "C_{D0}": "Drag coefficient at object 0.csv",
    
    "h_1": "Normalized displacement at object 1.csv",
    "\\dot{h}_1": "Normalized velocity at object 1.csv",
    "\\ddot{h}_1": "Normalized acceleration at object 1.csv",
    "C_{L1}": "Lift coefficient at object 1.csv",
    "C_{D1}": "Drag coefficient at object 1.csv"
}

TWO_TANDEM_CYLINDERS_MSTAR = 2.54648
TWO_TANDEM_CYLINDERS_MSTAR_TRUE = 2 / (np.pi * 0.25)
TWO_TANDEM_CYLINDERS_RE = 200