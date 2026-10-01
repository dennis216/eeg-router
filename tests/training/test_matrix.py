from bfa.training.matrix import build_native_matrix, build_unified_matrix


def test_mandatory_matrix_counts_and_unique_ids() -> None:
    unified = build_unified_matrix(
        models=["singlem", "cbramod", "tcn_gat"],
        model_seeds=[17, 42, 3407],
        split_seeds=[17, 42, 3407, 2026, 777],
    )
    native = build_native_matrix(
        models=["singlem", "cbramod", "tcn_gat"],
        model_seeds=[17, 42, 3407],
        split_seed=17,
    )
    assert len(unified) == 45
    assert len(native) == 9
    assert len({run.run_id for run in unified + native}) == 54


def test_content_addressed_identity_changes_with_split() -> None:
    runs = build_unified_matrix(models=["singlem"], model_seeds=[17], split_seeds=[17, 42])
    assert runs[0].run_id != runs[1].run_id
