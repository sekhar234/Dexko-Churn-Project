from src.dashboard.contracts import DASHBOARD_PAGES, REQUIRED_TABLES


def test_phase11_dashboard_pages():
    assert DASHBOARD_PAGES == [
        "Data Upload",
        "Executive Overview",
        "Customer Workbench",
        "Customer 360",
        "Category Intelligence",
        "Risk Movement",
        "Model Health",
        "Pipeline Health",
    ]


def test_phase11_required_tables():
    required = set(REQUIRED_TABLES)
    assert "category_scores" in required
    assert "customer_rollup" in required
    assert "reason_long" in required
    assert "drift_metrics" in required
    assert "dashboard_metadata" in required
    assert "pipeline_stage_status" in required


def test_streamlit_app_compiles():
    from pathlib import Path
    import ast

    app = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    ast.parse(app.read_text(encoding="utf-8"))


def test_dashboard_visual_assets_compile_and_exist():
    from pathlib import Path
    import ast

    root = Path(__file__).resolve().parents[1]
    for relative in [
        "dashboard/app.py",
        "src/dashboard/ui.py",
        "src/dashboard/upload.py",
        "src/dashboard/upload_ui.py",
    ]:
        path = root / relative
        ast.parse(path.read_text(encoding="utf-8"))

    assert (root / "dashboard" / "styles.css").exists()
    assert (root / ".streamlit" / "config.toml").exists()
