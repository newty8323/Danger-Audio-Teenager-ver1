from scripts.export_mobile_asr import budget_report


def test_budget_report_includes_other_on_device_models():
    report = budget_report(40_000_000, other_model_mb=38.0, total_budget_mb=100.0)
    assert report["total_model_mb"] == 78.0
    assert report["within_budget"] is True
    assert report["remaining_mb"] == 22.0


def test_budget_report_rejects_whisper_base_sized_export():
    report = budget_report(80_000_000, other_model_mb=38.0, total_budget_mb=100.0)
    assert report["within_budget"] is False

