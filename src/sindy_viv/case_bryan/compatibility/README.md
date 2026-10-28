# Historical DataFrame and pickle interfaces

These adapters preserve earlier imports and serialized wake-library callables.
Maintained numerical workflows use `case_bryan.input.load_data_*`, canonical
trajectory preparation and explicit `SelectedPolynomialLibrary` terms instead.

| Module | Objects/functions | Relationship to current code |
| --- | --- | --- |
| `dataframe_loading.py` | `load_raw_data_single_cylinder`, `load_raw_data_two_tandem_cylinders` | Return ordered channel-to-DataFrame mappings with recorded t* timestamps. |
| `dataframe_conversion.py` | `convert_to_array` | Check identical timestamp columns and convert the mapping to aligned time/value arrays. |
| `wake_callables.py` | `_WakeFunction`, `_full_input_lambda` | Reconstruct the callable arity/naming behavior required by historical CustomLibrary pickles. |

The raw loader names are re-exported by `case_bryan.input`, and conversion is
also available through `case_bryan.preprocessing`. `feature_library.legacy`
retains the previous pickle import path by re-exporting the wake callables.
These private names therefore remain part of the serialized compatibility
contract even though new fitting does not construct them.

## Read an older DataFrame workflow

```python
from sindy_viv.case_bryan.compatibility.dataframe_loading import load_raw_data_single_cylinder
from sindy_viv.case_bryan.compatibility.dataframe_conversion import convert_to_array

tables = load_raw_data_single_cylinder(
    re=150, mstar=5.1, vr=3, variables=["h_0", "C_{L0}"]
)
tstar, values = convert_to_array(tables)
```

The converter preserves mapping column order and the reference timestamp dtype;
it does not change fluid time to tau. The raw loader honors dtype for time and
signals, so float64 is the normal choice. In contrast, processed
`load_data_single_cylinder`/`load_data_two_tandem_cylinders` return tau directly
and keep float64 time independently of state dtype. This distinction matters
when an old DataFrame notebook is connected to current preparation functions.

Unpickling invokes `_WakeFunction` through its recorded import path and exposes
the historical callable signature through `__code__`. New restricted libraries
store explicit exponent vectors; their implementation and examples are in the
[Bryan feature-library guide](../feature_library/README.md).
