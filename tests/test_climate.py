from evaluador_lotes_mini.climate.nasa_power import MonthlyClimate, analyze_climate


def test_climate_produces_campaign_quadrants() -> None:
    rows = [
        MonthlyClimate(year, month, 40 + (year % 5) * 10, 15 + (year % 4), 20)
        for year in range(1991, 2022)
        for month in range(1, 13)
    ]
    result = analyze_climate(rows, 1991, 2021)
    assert len(result["monthly_profile"]) == 12
    assert result["campaigns"]["gruesa"]["years"]
    assert sum(
        item["count"] for item in result["campaigns"]["fina"]["quadrant_summary"].values()
    ) == len(result["campaigns"]["fina"]["years"])
    assert result["methodology"]["normal_period"] == "1991-2020"
    assert "water_balance_mm" in result["campaigns"]["fina"]["years"][0]


def test_incomplete_campaign_is_not_classified() -> None:
    rows = [
        MonthlyClimate(2020, month, 50, 20, 30)
        for month in (4, 5, 6, 7, 8)
    ]
    result = analyze_climate(rows, 2020, 2020)
    assert result["campaigns"]["fina"]["years"] == []
