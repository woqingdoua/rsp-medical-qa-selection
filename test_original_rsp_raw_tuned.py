from derive_original_rsp_lambda_scores import LAMBDAS, derive, method_name
from make_original_rsp_raw_tuned_manifest import BASELINES, BUDGET, DATASETS, SEEDS
from select_original_rsp_lambda import choose_candidate


def test_lambda_score_derivation_and_ineligible_sentinel():
    rows = [
        {
            "uid": "eligible",
            "rsp_prefilter_eligible": True,
            "rsp_flip": 0.25,
            "rsp_drop": 0.4,
        },
        {
            "uid": "ineligible",
            "rsp_prefilter_eligible": False,
            "rsp_flip": None,
            "rsp_drop": None,
        },
    ]
    output = derive(rows)
    assert output[0][method_name(0.0)] == 0.25
    assert output[0][method_name(0.5)] == 0.45
    assert output[0][method_name(2.0)] == 1.05
    assert output[1][method_name(1.0)] == -1e30


def test_development_selection_tie_breaks_are_frozen():
    rows = [
        {"lambda": 0.0, "dev_accuracy_mean": 0.7, "dev_macro_f1_mean": 0.6},
        {"lambda": 0.5, "dev_accuracy_mean": 0.7, "dev_macro_f1_mean": 0.65},
        {"lambda": 1.0, "dev_accuracy_mean": 0.7, "dev_macro_f1_mean": 0.65},
    ]
    assert choose_candidate(rows)["lambda"] == 0.5


def test_protocol_grid_size():
    expected = len(DATASETS) * len(SEEDS) * (len(BASELINES) + len(LAMBDAS))
    assert expected == 96
    assert BUDGET == 128
