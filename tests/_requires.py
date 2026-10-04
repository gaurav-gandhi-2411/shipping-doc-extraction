"""Which tests need the evaluators' package (``assignment/``) and how they skip without it.

The public repository does not contain ``assignment/`` (the official scorer, ``schema.json``, the
task text): it belongs to the evaluators. A test that reaches the scorer or the schema through the
code under test (``coerce`` reads the date rules of ``schema.json``, ``predict`` validates against
it, ``eval`` loads ``score.py``) cannot run without it, and on a fresh clone it must SKIP with a
reason, not fail. ``tests/conftest.py`` skips every test listed in ``NEEDS_ASSIGNMENT`` when
``missing_inputs()`` is not empty.

In the private repository the inputs exist, so nothing is skipped there and every listed test
still runs and must pass. ``tests/test_private_inputs.py`` pins that: in the private tree it
fails when an input is missing, so a skip can never hide a real failure there. A test that needs
the inputs and is not listed here fails on a fresh clone (the public clone check catches it); add
its name here, or give its module its own ``skipif`` marker.

``NEEDS_ASSIGNMENT`` maps a test module (file stem) to the names of its test functions (parameter
ids dropped). Generated once from a run of the suite without ``assignment/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from shipdoc import paths

SCORER_ENV = "SHIPDOC_SCORER_PATH"  # the same override ``shipdoc.eval.load_scorer`` honours
SKIP_REASON = (
    "needs the evaluators' package (assignment/schema.json and assignment/score.py), "
    "which is not part of this repository: see the README"
)

NEEDS_ASSIGNMENT: dict[str, tuple[str, ...]] = {
    "test_batching": (
        "test_batch_size_one_is_the_unbatched_path_byte_for_byte",
        "test_batched_run_gives_the_same_documents_in_the_same_order",
        "test_manifest_records_the_experiment",
        "test_oom_falls_back_to_smaller_batches_and_loses_no_page",
        "test_pages_are_sorted_into_batches_but_documents_leave_in_input_order",
        "test_require_same_batch_size_between_dev_and_test_runs",
        "test_resume_of_a_pre_manifest_run_is_batch_size_one",
        "test_resume_reuses_the_stored_batch_size_and_never_silently_changes_it",
    ),
    "test_bench": (
        "test_spike_cli_uses_the_bench_choice_and_records_it_in_the_manifest",
        "test_without_gold_the_bench_cannot_use_step_two_and_still_runs",
    ),
    "test_coerce": (
        "test_merge_shards_rewrites_shard_traces_that_still_hold_numbers",
        "test_resume_over_a_trace_written_with_raw_numbers_is_rewritten_as_strings",
        "test_spike_runner_writes_plain_number_strings_identical_to_the_string_path",
        "test_the_number_backend_really_emits_numbers",
    ),
    "test_confidence_v2": ("test_header_columns_from_the_row_page_with_page1_fallback_and_none",),
    "test_determinism_replay": (
        "test_a_batch_size_one_run_from_before_the_stamp_is_still_replayable",
        "test_a_call_whose_members_disagree_in_the_trace_is_refused",
        "test_a_nondeterministic_backend_fails_even_with_the_composition_replicated",
        "test_a_shuffled_replay_fails",
        "test_a_trace_without_the_composition_record_raises_a_clear_error",
        "test_an_old_determinism_report_without_the_replication_flag_is_not_accepted",
        "test_composition_dependent_backend_passes_when_the_composition_is_replicated",
        "test_every_page_records_the_call_that_decoded_it",
        "test_replay_plan_covers_every_call_of_the_check_docs_in_member_order",
        "test_replay_with_resume_does_not_duplicate_documents",
        "test_the_old_second_pass_would_have_failed_this_backend",
    ),
    "test_devfinal": (
        "test_a_guard_difference_falls_back_to_batch_1_and_says_so",
        "test_a_resumed_run_keeps_the_stored_batch_decision_and_redoes_nothing",
        "test_a_short_merge_is_refused_before_the_guard",
        "test_adapters_that_are_not_the_final_adapter_are_refused_before_any_merge",
        "test_banner_names_the_seen_layout_caveat",
        "test_compare_refuses_a_dev_document_in_the_shape_learner_input",
        "test_compare_refuses_an_unfinished_or_foreign_or_wrong_run",
        "test_compare_refuses_predictions_outside_the_dev_ids",
        "test_compare_scores_ft_vs_zs_on_the_dev_docs_under_three_arms",
        "test_compare_stops_when_the_dev_ocr_is_missing_for_a_waybill",
        "test_infer_refuses_documents_that_are_not_the_dev_documents",
        "test_infer_runs_verify_merge_guard_infer_in_order_on_exactly_the_dev_ids",
        "test_main_prints_a_refusal_and_exits_1",
        "test_plan_counts_from_the_labels_and_prints_no_id",
        "test_plumbing_check_uses_the_zs_run_as_both_arms_and_every_delta_is_zero",
        "test_reports_hold_aggregates_only",
        "test_run_arms_fails_closed_when_rules_off_does_not_reproduce_the_predictions",
        "test_the_batch_size_defaults_to_the_02_run_and_a_manual_one_is_warned",
        "test_the_rules_run_on_the_dev_docs_and_both_arms_get_the_same_ones",
    ),
    "test_devfinal_native": (
        "test_a_missing_folder_a_missing_manifest_and_a_hashless_manifest_are_refused",
        "test_a_native_adapter_passes_the_resolution_gate_and_prints_its_row",
        "test_a_native_run_passes_and_a_1260_run_is_refused",
        "test_an_adapter_trained_at_1260_is_refused_by_the_gate_before_verification",
        "test_an_adapter_without_a_readable_training_resolution_is_refused",
        "test_cli_compare_refuses_mixed_resolutions_and_a_run_without_the_hash",
        "test_cli_infer_refuses_a_1260_adapter_and_a_1260_zero_shot_run_before_any_model",
        "test_cli_plan_estimate_verify_infer_compare_at_native",
        "test_the_estimate_is_labelled_scales_from_the_native_pace_and_names_08",
        "test_the_estimate_refuses_a_1260_zero_shot_run",
        "test_the_plumbing_check_still_works_and_is_stamped_at_native",
    ),
    "test_flags": (
        "test_a_behaviour_change_of_the_feature_builder_is_caught_by_the_probe",
        "test_flags_accept_above_tau_review_below_none_means_review_and_nulls_are_not_scored",
        "test_flags_refuse_an_emitted_field_without_a_probability",
        "test_the_calibration_path_still_refuses_test_ids_and_the_probe_is_stable",
        "test_training_and_inference_paths_build_identical_features_and_probabilities",
    ),
    "test_freeze_calibrator": (
        "test_fine_tuned_arm_artifact_records_the_agreement_columns",
        "test_frozen_field_model_equals_a_fresh_fit_of_the_same_model",
        "test_per_type_structure_gives_every_type_a_model_even_without_training_rows",
        "test_stored_tau_is_the_pooled_oof_tau_and_nested_estimates_sit_beside_it",
        "test_the_document_oof_must_match_the_calibration_csv",
        "test_the_novelty_reference_and_document_model_are_stored",
    ),
    "test_g4": (
        "test_paired_delta_is_ft_minus_zs_and_verdicts_use_it",
        "test_rules_effect_sums_splits_and_reports_rules_on_minus_off",
        "test_verdict_over_null_clause_counts_ft_against_zs",
    ),
    "test_g4_fold": (
        "test_cause_summary_counts_units_slots_and_waybill_cells",
        "test_default_report_path_is_per_fold",
        "test_honest_shapes_exclude_the_held_out_fold",
        "test_native_ft_with_native_zs_passes",
        "test_over_null_counting_follows_the_arm",
        "test_plumbing_check_does_not_need_an_oof_section_but_oof_mode_requires_the_dir",
        "test_plumbing_check_uses_the_zs_run_as_both_arms_and_writes_nothing",
        "test_refuses_a_mixed_resolution_pair_and_names_both",
        "test_refuses_a_zero_shot_run_that_lacks_fold_docs",
        "test_refuses_bad_oof_sections",
        "test_refuses_fold_flag_contradicting_the_manifest",
        "test_refuses_incomplete_runs",
        "test_refuses_missing_gold",
        "test_refuses_missing_ocr_pages_for_a_waybill",
        "test_refuses_missing_trace_file_and_missing_trace_doc",
        "test_refuses_traces_that_already_carry_rule_records",
        "test_refuses_when_all_rules_off_does_not_reproduce_the_saved_predictions",
        "test_refuses_when_r3_learner_overlaps_the_held_out_suppliers",
        "test_refuses_without_oof_section_and_names_plumbing",
        "test_refuses_wrong_doc_ids_in_the_oof_run",
        "test_report_holds_no_gold_value_and_no_doc_id",
        "test_report_sections_verdicts_and_sign_convention",
        "test_rules_are_applied_to_both_arms_not_just_one",
        "test_warns_when_the_zs_folder_is_not_the_one_the_manifest_names",
    ),
    "test_g4pooled": (
        "test_all_native_folds_with_a_native_zs_run_pass",
        "test_an_extra_over_null_in_ft_fails_c3_and_selects_zs",
        "test_driver_selects_ft_on_a_clean_win_and_writes_the_report",
        "test_oof_run_dir_is_required_without_plumbing",
        "test_plumbing_check_is_not_a_result_picks_zs_and_writes_nothing",
        "test_r3_warning_and_r3_off_decision_when_r3_breaks_cells_on_fold_1",
        "test_refuses_a_manifest_without_the_fields_to_compare",
        "test_refuses_a_total_other_than_500_docs",
        "test_refuses_a_zero_shot_run_that_lacks_fold_docs",
        "test_refuses_a_zs_run_without_a_config_hash",
        "test_refuses_an_oof_run_holding_another_folds_docs",
        "test_refuses_fewer_duplicate_or_swapped_folds",
        "test_refuses_missing_ocr_pages_for_a_waybill",
        "test_refuses_native_folds_against_a_1260_zs_run_and_the_reverse",
        "test_refuses_runs_that_differ_in_code_sha_batch_config_or_share_an_adapter",
        "test_refuses_traces_that_already_carry_rule_records",
        "test_refuses_when_all_rules_off_does_not_reproduce_the_saved_predictions",
        "test_refuses_when_r3_learner_overlaps_a_held_out_supplier",
        "test_refuses_without_oof_section_unverified_adapter_or_incomplete_run",
        "test_report_holds_no_gold_value_and_no_doc_id",
        "test_sign_convention_ft_minus_zs_via_compare_models",
        "test_the_same_rules_hit_both_arms_in_every_fold",
    ),
    "test_gate": (
        "test_decide_g_from_pair_reads_g_as_the_oof_slot",
        "test_gate_predictions_fails_if_coerce_would_change_the_output",
        "test_gated_output_validates_against_the_repo_output_schema",
    ),
    "test_gate_eval": (
        "test_exploratory_run_is_stamped_and_prints_no_decision",
        "test_plumbing_check_needs_no_oof_run_but_a_real_run_does",
        "test_plumbing_check_writes_nothing_and_g_equals_zs",
        "test_refuses_a_duplicate_doc_in_a_run",
        "test_refuses_a_mixed_resolution_between_ft_and_zs",
        "test_refuses_a_repeated_fold",
        "test_refuses_a_run_count_that_differs_from_the_folds",
        "test_refuses_an_unverified_adapter",
        "test_refuses_fewer_than_three_folds_without_exploratory",
        "test_refuses_runs_with_different_resolution_across_folds",
        "test_report_holds_no_gold_value_and_no_doc_id",
        "test_requires_logprobs_when_not_relaxed",
        "test_three_fold_run_decides_with_the_primary_rule_first",
        "test_three_folds_with_the_flag_are_stamped_and_not_decided",
        "test_validate_runs_refuses_a_missing_fold_and_an_unrequested_fold",
    ),
    "test_nativerun": (
        "test_a_1310720_adapter_is_refused_by_the_native_config_and_2196480_is_accepted",
        "test_a_native_adapter_and_a_native_zero_shot_run_are_refused_by_the_1260_config",
    ),
    "test_notebook_dev_final": ("test_banner_prints_the_seen_layout_number_and_the_files",),
    "test_notebook_dev_final_native": (
        "test_banner_prints_the_native_seen_layout_number_and_the_files",
    ),
    "test_notebook_hdrhint": (
        "test_pin_is_the_raising_placeholder_and_clone_asserts_head_equals_pin",
    ),
    "test_notebook_oof": ("test_banner_prints_the_interim_verdict_and_the_files",),
    "test_notebook_predict_v1": ("test_the_pin_is_a_placeholder_that_raises_until_gg_pins",),
    "test_notebook_predict_v2": (
        "test_a_failed_guard_falls_back_to_batch_1_and_a_resume_keeps_the_decision",
        "test_a_failed_smoke_gate_on_the_merged_model_stops_before_the_test_run",
        "test_determinism_pass_merges_again_and_needs_the_first_pass_decision",
        "test_ft_pipeline_verifies_merges_guards_smokes_runs_and_records_everything",
    ),
    "test_notebook_ressweep": (
        "test_pin_placeholder_blocks_the_notebook_until_the_pin_commit_fills_it",
    ),
    "test_notebook_spike": (
        "test_parameters_cell_contract",
        "test_summary_reads_real_mock_spike_metrics",
    ),
    "test_notebook_zeroshot": (
        "test_pin_placeholder_blocks_the_notebook_until_the_pin_commit_fills_it",
    ),
    "test_notebook_zs_native": (
        "test_pin_placeholder_blocks_the_notebook_until_the_pin_commit_fills_it",
    ),
    "test_oof": (
        "test_a_batched_decode_that_differs_falls_back_to_batch_1",
        "test_a_merge_that_merged_the_wrong_number_of_modules_is_refused",
        "test_a_missing_manifest_and_a_foreign_zero_shot_run_are_refused",
        "test_an_unfinished_zero_shot_run_is_refused",
        "test_an_unreachable_training_sha_is_refused_and_a_different_pin_only_warns",
        "test_batch_size_comes_from_the_zero_shot_run_and_is_refused_when_absent",
        "test_compare_needs_a_complete_oof_run",
        "test_compare_refuses_missing_documents_and_formats_aggregates_only",
        "test_every_refusal_path_fails_closed_before_any_run",
        "test_full_pipeline_verify_merge_guard_infer_score_compare",
        "test_missing_bench_result_stops_the_run_before_the_merge",
        "test_over_null_counts_header_rows_and_false_fills",
        "test_resume_after_a_kill_gives_the_same_files",
        "test_the_ci_and_the_paired_delta_use_the_documented_bootstrap",
        "test_the_guard_is_skipped_at_batch_1_and_runs_before_the_full_inference",
        "test_the_regression_verdict_names_the_failed_clause",
    ),
    "test_postrules": (
        "test_production_with_all_rules_off_reproduces_the_input_predictions",
        "test_run_spike_rule_cfg_is_opt_in_and_adds_a_value_free_rules_record",
    ),
    "test_predict": (
        "test_assemble_never_writes_into_the_repo_tree",
        "test_batch_contract_mismatch_refuses",
        "test_determinism_passes_for_a_deterministic_backend_and_fails_for_a_flaky_one",
        "test_resume_after_a_kill_is_byte_identical",
        "test_run_refuses_when_the_decision_no_longer_matches_the_dev_run",
        "test_second_pass_uses_its_own_run_folder_with_the_same_batch_size",
        "test_smoke_gate_blocks_the_test_run_when_it_fails",
        "test_smoke_gate_error_is_a_failed_gate_and_a_passed_one_is_not_repeated",
        "test_smoke_gate_missing_status_or_other_code_blocks_the_run",
        "test_stored_dev_bench_is_reused_when_it_applies_else_the_bench_reruns",
    ),
    "test_run_config_hash": ("test_freezing_records_the_run_config_hash_additively",),
    "test_schema_repair": ("test_spike_writer_repairs_traces_predictions_and_manifest",),
    "test_shards": (
        "test_batched_shards_merge_to_the_same_documents_as_an_unsharded_batched_run",
        "test_merge_refuses_another_doc_list_or_config",
        "test_merge_refuses_incomplete_missing_duplicate_and_miscounted_shards",
        "test_merge_refuses_shards_that_are_not_the_same_experiment",
        "test_merge_shards_cli",
        "test_merged_two_shards_are_byte_identical_to_an_unsharded_run",
        "test_resuming_a_shard_continues_that_shard_only",
        "test_shards_cover_every_doc_once_and_balance_pages",
    ),
}


ROOT = Path(__file__).resolve().parents[1]


def assignment_inputs(root: Path = ROOT) -> dict[str, list[Path]]:
    """The evaluators' files these tests read, by the name used in messages.

    Two locations each: where ``shipdoc.paths`` / the scorer override point (what the code under
    test reads) and ``<repo>/assignment/`` (what several tests hard-code as ``SCHEMA``). Both must
    exist: with only one of them (``$SHIPDOC_ASSIGNMENT_DIR`` set, no ``assignment/`` folder) a
    test using the other would fail, which is how ``test_reuse_v0`` failed in the public tree.
    """
    return {
        "assignment/schema.json": [
            paths.assignment_dir() / "schema.json",
            root / "assignment" / "schema.json",
        ],
        "assignment/score.py": [
            Path(os.environ.get(SCORER_ENV) or paths.scorer_path()),
            root / "assignment" / "score.py",
        ],
    }


def missing_inputs(root: Path = ROOT) -> list[str]:
    """Names of the evaluators' files that do not exist (empty in the private repository)."""
    return [
        name
        for name, candidates in assignment_inputs(root).items()
        if not all(p.is_file() for p in candidates)
    ]


def needs_assignment(module: str, test: str) -> bool:
    """True when ``module`` (file stem) lists the test function ``test``."""
    return test in NEEDS_ASSIGNMENT.get(module, ())
