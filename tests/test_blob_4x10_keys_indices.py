"""
Run the SDP certification model for config/blob_4x10.yaml and dump
all keys/indices stored by variable_elements.py.

Marker tags emitted on stdout (for grep filtering):
    [KI-NEURON]  one line per (layer, neuron) entry of Equivalent_Neurons_Index
    [KI-WEIGHT]  one line per weight entry (weights / weights_front / weights_back)
    [KI-BETA]    one line per Equivalent_Betas_Index entry
    [KI-ELEM]    one line per ElementsinConstraintsObjectives entry

Usage:
    conda activate certif
    python tests/test_blob_4x10_keys_indices.py
"""

import os
import sys
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from solve.sdp_solve.handler.variable_elements import (
    _get_layer_neuron_from_key_,
    _get_linear_indices_from_key,
    _get_quad_indices_from_key,
    ElementsinConstraintsObjectives,
    Equivalent_Neurons_Index,
    Equivalent_Betas_Index,
)
from solve.sdp_solve.handler.variables_call import VariablesCall


def _dump_equivalent_neurons(eq: Equivalent_Neurons_Index, tag: str = ""):
    print(f"\n=== [KI-DUMP] Equivalent_Neurons_Index {tag} ===")
    for key, info in eq.equivalent_neurons.items():
        layer, neuron = _get_layer_neuron_from_key_(key, eq.M)
        constant = info.get("constant", 0.0)
        print(
            f"[KI-NEURON] tag={tag} key={key} layer={layer} neuron={neuron} "
            f"constant={constant}"
        )
        for sub_name in ("weights", "weights_front", "weights_back"):
            sub = info.get(sub_name, None)
            if sub is None:
                continue
            for sub_key in sub.keys():
                value = sub[sub_key]
                i, num_matrix = _get_linear_indices_from_key(sub_key)
                print(
                    f"[KI-WEIGHT] tag={tag} parent_key={key} layer={layer} "
                    f"neuron={neuron} group={sub_name} sub_key={sub_key} "
                    f"i={i} num_matrix={num_matrix} value={value}"
                )


def _dump_equivalent_betas(eq: Equivalent_Betas_Index, tag: str = ""):
    print(f"\n=== [KI-DUMP] Equivalent_Betas_Index {tag} ===")
    for class_label, d in eq.equivalent_indexes_betas.items():
        for sub_key in d.keys():
            value = d[sub_key]
            i, num_matrix = _get_linear_indices_from_key(sub_key)
            print(
                f"[KI-BETA] tag={tag} class_label={class_label} "
                f"sub_key={sub_key} i={i} num_matrix={num_matrix} value={value}"
            )


def _dump_elements(elements: ElementsinConstraintsObjectives, tag: str = ""):
    nb_index = elements.nb_index
    for key in elements.elements.keys():
        value = elements.elements[key]
        i, j, num_matrix = _get_quad_indices_from_key(key, nb_index)
        print(
            f"[KI-ELEM] tag={tag} nb_index={nb_index} key={key} "
            f"i={i} j={j} num_matrix={num_matrix} value={value}"
        )


# ----------------------------------------------------------------------
# Monkey-patch VariablesCall.__init__ so each constructed instance dumps
# its keys/indices once it is fully initialized.
# ----------------------------------------------------------------------
_orig_variables_call_init = VariablesCall.__init__
_dump_counter = {"n": 0}


def _patched_variables_call_init(self, *args, **kwargs):
    _orig_variables_call_init(self, *args, **kwargs)
    _dump_counter["n"] += 1
    tag = f"VC#{_dump_counter['n']}/{type(self).__name__}"
    print(
        f"\n############ [KI] VariablesCall init complete -> dumping ({tag}) "
        f"############"
    )
    _dump_equivalent_neurons(self.equivalent_neurons, tag=tag)
    if getattr(self, "BETAS", False):
        _dump_equivalent_betas(self.equivalent_indexes_betas, tag=tag)


VariablesCall.__init__ = _patched_variables_call_init


# ----------------------------------------------------------------------
# Monkey-patch ElementsinConstraintsObjectives so we can dump the
# `elements` dict on demand. We hook `decode_key_vec` to dump right
# before the solver consumes the elements.
# ----------------------------------------------------------------------
_orig_decode_key_vec = ElementsinConstraintsObjectives.decode_key_vec
_elem_counter = {"n": 0}


def _patched_decode_key_vec(self):
    _elem_counter["n"] += 1
    tag = f"ELEM#{_elem_counter['n']}"
    _dump_elements(self, tag=tag)
    return _orig_decode_key_vec(self)


ElementsinConstraintsObjectives.decode_key_vec = _patched_decode_key_vec


# ----------------------------------------------------------------------
# Run the certification problem on blob_4x10.yaml
# ----------------------------------------------------------------------
from certification_problem import Certification_Problem


def main():
    yaml_file = "blob_4x10.yaml"
    print(f"[KI] Loading certification problem from {yaml_file}")
    certif_problem = Certification_Problem.load_from_yaml(yaml_file)

    launch_date = datetime.datetime.now().strftime("%Y_%m_%d_%Hh%M_%Ss")
    title_run_full = launch_date + "_test_keys_indices"
    print(f"[KI] Running solve with title_run = {title_run_full}")
    certif_problem.solve(title_run_full)
    print("[KI] Done.")


if __name__ == "__main__":
    main()
