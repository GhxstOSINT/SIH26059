from datetime import date

import pytest

from ml.evaluate_independent_corridors import calendar_coverage


def test_calendar_coverage_counts_missing_windows():
    route = {"summary": {"scored_days": 1}, "days": [
        {"date": "2024-06-03", "sample_points": 10, "common_valid_points": 5},
        {"date": "2024-06-05", "sample_points": 10, "common_valid_points": 0},
    ]}
    result = calendar_coverage(route, date(2024, 6, 1), date(2024, 6, 5))
    assert result["scheduled_target_days"] == 3
    assert result["missing_three_day_windows"] == 1
    assert result["calendar_coverage_fraction"] == pytest.approx(5 / 30)


def test_calendar_coverage_rejects_duplicate_days():
    route = {"summary": {"scored_days": 2}, "days": [
        {"date": "2024-06-03", "sample_points": 10, "common_valid_points": 5},
        {"date": "2024-06-03", "sample_points": 10, "common_valid_points": 5},
    ]}
    with pytest.raises(ValueError, match="Duplicate"):
        calendar_coverage(route, date(2024, 6, 1), date(2024, 6, 5))
