from sindy_viv.case_bryan.input import *

# Check single cylinder VIV data
re = 150
mstar = 5.1
vr = 3

single_vr7_time, single_vr7_data = load_data_single_cylinder(re, mstar, vr)

print(f"Single cylinder VIV data for Re={re}, m*={mstar}, Vr={vr}:")
print(f"Time data shape: {single_vr7_time.shape}")
print(f"Data shape: {single_vr7_data.shape}")

# Check two tandem cylinders VIV data
print("")

lpd = 2.0
vr = 5
re = 200
mstar = 2.54648

two_tandem_vr5_time, two_tandem_vr5_data = load_data_two_tandem_cylinders(re, mstar, vr, lpd)

print(f"Two tandem cylinders VIV data for Re={re}, m*={mstar}, Vr={vr}, L/D={lpd}:")
print(f"Time data shape: {two_tandem_vr5_time.shape}")
print(f"Data shape: {two_tandem_vr5_data.shape}")