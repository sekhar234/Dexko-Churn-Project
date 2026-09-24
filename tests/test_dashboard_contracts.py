from src.dashboard.contracts import DASHBOARD_PAGES, REQUIRED_TABLES


def test_phase11_dashboard_pages():
    assert DASHBOARD_PAGES == [
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
