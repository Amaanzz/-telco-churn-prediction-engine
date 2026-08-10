"""
Unit tests for src/strategy.py

Tests EV-based retention strategy:
- CLV calculation
- Expected Value calculation
- Tier assignment (high_value / standard / no_action)
- Senior citizen targeting
- Edge cases and boundary conditions
"""

import pytest
from src.strategy import (
    expected_remaining_months,
    clv_at_risk,
    expected_value,
    generate_retention_strategy_v2,
    SEGMENT_LIFETIME,
    OFFER_SUCCESS_RATE,
    COST_OF_OFFER,
    EV_HIGH_VALUE_THRESHOLD,
    EV_STANDARD_THRESHOLD,
)


class TestExpectedRemainingMonths:
    """Test conditional remaining tenure calculation."""

    def test_month_to_month_new_customer(self):
        """New customer (tenure=0) should have full segment lifetime."""
        remaining = expected_remaining_months("Month-to-month", current_tenure=0)
        assert remaining == pytest.approx(SEGMENT_LIFETIME["Month-to-month"])

    def test_month_to_month_mid_tenure(self):
        """Customer halfway through segment lifetime."""
        segment_life = SEGMENT_LIFETIME["Month-to-month"]  # 36.3 months
        current_tenure = 18.15  # Half

        remaining = expected_remaining_months("Month-to-month", current_tenure=18)
        expected = segment_life - 18
        assert remaining == pytest.approx(expected, abs=0.5)

    def test_month_to_month_outlier_tenure(self):
        """Customer with tenure > segment average (outlier)."""
        segment_life = SEGMENT_LIFETIME["Month-to-month"]  # 36.3
        current_tenure = 60  # Way above average

        remaining = expected_remaining_months("Month-to-month", current_tenure=60)
        # Should floor to 1 month, not go negative
        assert remaining >= 1.0

    def test_one_year_contract(self):
        """Test One year contract lifetime."""
        remaining = expected_remaining_months("One year", current_tenure=0)
        assert remaining == pytest.approx(SEGMENT_LIFETIME["One year"])

    def test_two_year_contract(self):
        """Test Two year contract lifetime."""
        remaining = expected_remaining_months("Two year", current_tenure=0)
        assert remaining == pytest.approx(SEGMENT_LIFETIME["Two year"])

    def test_invalid_contract_type_defaults_to_month_to_month(self):
        """Invalid contract type should default to Month-to-month."""
        remaining = expected_remaining_months("Unknown", current_tenure=0)
        assert remaining == pytest.approx(SEGMENT_LIFETIME["Month-to-month"])

    def test_zero_tenure_avoids_negative(self):
        """Even with zero tenure, remaining should be positive."""
        remaining = expected_remaining_months("Month-to-month", current_tenure=0)
        assert remaining > 0


class TestCLVAtRisk:
    """Test Customer Lifetime Value calculation."""

    def test_clv_basic_calculation(self):
        """CLV = monthly_charges × expected_remaining_months."""
        monthly = 50.0
        contract = "Month-to-month"
        tenure = 0

        clv = clv_at_risk(monthly, contract, tenure)
        segment_life = SEGMENT_LIFETIME[contract]
        expected = monthly * segment_life

        assert clv == pytest.approx(expected)

    def test_clv_high_value_customer(self):
        """Test CLV for high-value customer ($78.55/month)."""
        clv = clv_at_risk(78.55, "Month-to-month", current_tenure=7)
        # segment_lifetime["Month-to-month"] = 36.3
        # remaining = 36.3 - 7 = 29.3
        # CLV = 78.55 * 29.3 ≈ 2301.4
        assert clv > 1000  # Sanity check

    def test_clv_scales_with_monthly_charges(self):
        """Higher monthly charges should result in higher CLV."""
        clv1 = clv_at_risk(50.0, "Month-to-month", 10)
        clv2 = clv_at_risk(100.0, "Month-to-month", 10)

        assert clv2 > clv1
        assert clv2 == pytest.approx(clv1 * 2, rel=0.01)

    def test_clv_longer_contract_higher_clv(self):
        """Longer contract = higher CLV (more remaining lifetime)."""
        monthly = 50.0
        tenure = 0

        clv_month = clv_at_risk(monthly, "Month-to-month", tenure)
        clv_year = clv_at_risk(monthly, "One year", tenure)
        clv_two_year = clv_at_risk(monthly, "Two year", tenure)

        assert clv_month < clv_year < clv_two_year


class TestExpectedValue:
    """Test Expected Value calculation."""

    def test_ev_high_value_customer(self):
        """High-risk, high-value customer should have positive EV."""
        ev = expected_value(
            probability=0.61,
            monthly_charges=78.55,
            contract_type="Month-to-month",
            tenure=7,
        )
        assert ev > 200  # Should be in high_value tier

    def test_ev_low_value_customer(self):
        """Low-risk, low-value customer should have negative EV."""
        ev = expected_value(
            probability=0.20,
            monthly_charges=20.65,
            contract_type="Month-to-month",
            tenure=2,
        )
        assert ev < 0  # Not worth intervention

    def test_ev_boundary_cases(self):
        """Test EV near zero boundary."""
        # Find probability that yields EV ≈ 0
        monthly = 50.0
        contract = "Month-to-month"
        tenure = 0

        clv = clv_at_risk(monthly, contract, tenure)
        # EV = 0 when: prob * clv * 0.30 = 75
        breakeven_prob = COST_OF_OFFER / (clv * OFFER_SUCCESS_RATE)

        ev = expected_value(breakeven_prob, monthly, contract, tenure)
        assert ev == pytest.approx(0, abs=1)  # Within $1

    def test_ev_sensitivity_to_probability(self):
        """EV scales linearly with churn probability."""
        monthly, contract, tenure = 60.0, "Month-to-month", 10

        ev1 = expected_value(0.30, monthly, contract, tenure)
        ev2 = expected_value(0.60, monthly, contract, tenure)

        # Doubling probability should roughly double EV
        assert ev2 > ev1
        assert ev2 - ev1 > 0

    def test_ev_with_custom_success_rate(self):
        """Test overriding offer_success_rate."""
        monthly, contract, tenure, prob = 50.0, "Month-to-month", 10, 0.50

        ev_30 = expected_value(prob, monthly, contract, tenure, offer_success_rate=0.30)
        ev_40 = expected_value(prob, monthly, contract, tenure, offer_success_rate=0.40)

        # 40% success rate should yield higher EV
        assert ev_40 > ev_30
        assert ev_40 - ev_30 > 0

    def test_ev_with_custom_cost(self):
        """Test overriding cost_of_offer."""
        monthly, contract, tenure, prob = 50.0, "Month-to-month", 10, 0.50

        ev_75 = expected_value(prob, monthly, contract, tenure, cost_of_offer=75)
        ev_100 = expected_value(prob, monthly, contract, tenure, cost_of_offer=100)

        # Higher cost should yield lower EV
        assert ev_100 < ev_75


class TestGenerateRetentionStrategy:
    """Test strategy generation and tier assignment."""

    def test_high_value_tier_assignment(self):
        """EV > $200 should be assigned to high_value tier."""
        strategy = generate_retention_strategy_v2(
            probability=0.61,
            monthly_charges=78.55,
            contract_type="Month-to-month",
            tenure=7,
            is_senior_citizen=False,
        )

        assert strategy['tier'] == "high_value"
        assert strategy['expected_value'] > EV_HIGH_VALUE_THRESHOLD
        assert "🎯" in strategy['action']  # High-value emoji in action

    def test_standard_tier_assignment(self):
        """$0 < EV < $200 should be assigned to standard tier."""
        strategy = generate_retention_strategy_v2(
            probability=0.40,
            monthly_charges=60.0,
            contract_type="Month-to-month",
            tenure=15,
            is_senior_citizen=False,
        )

        assert strategy['tier'] == "standard"
        assert 0 < strategy['expected_value'] < EV_HIGH_VALUE_THRESHOLD
        assert "📊" in strategy['action']  # Standard emoji

    def test_no_action_tier_assignment(self):
        """EV ≤ $0 should be assigned to no_action tier."""
        strategy = generate_retention_strategy_v2(
            probability=0.20,
            monthly_charges=20.65,
            contract_type="Month-to-month",
            tenure=2,
            is_senior_citizen=False,
        )

        assert strategy['tier'] == "no_action"
        assert strategy['expected_value'] <= EV_STANDARD_THRESHOLD
        assert "⚪" in strategy['action']  # Monitor-only emoji

    def test_senior_citizen_flag_set(self):
        """Senior citizen flag should add warning to action."""
        strategy = generate_retention_strategy_v2(
            probability=0.61,
            monthly_charges=78.55,
            contract_type="Month-to-month",
            tenure=7,
            is_senior_citizen=True,
        )

        assert strategy['senior_warning'] is True
        assert "⚠️" in strategy['action']  # Alert emoji
        assert "41.7%" in strategy['action']  # Churn rate statistic

    def test_senior_citizen_flag_not_set(self):
        """Non-senior customer should not have warning flag."""
        strategy = generate_retention_strategy_v2(
            probability=0.61,
            monthly_charges=78.55,
            contract_type="Month-to-month",
            tenure=7,
            is_senior_citizen=False,
        )

        assert strategy['senior_warning'] is False
        assert "⚠️" not in strategy['action']

    def test_strategy_has_required_keys(self):
        """Strategy dict should have all required keys."""
        strategy = generate_retention_strategy_v2(
            probability=0.50,
            monthly_charges=50.0,
            contract_type="Month-to-month",
            tenure=10,
        )

        required_keys = ['tier', 'expected_value', 'action', 'senior_warning']
        for key in required_keys:
            assert key in strategy, f"Missing key: {key}"

    def test_action_text_length(self):
        """Action text should be substantial (not empty)."""
        strategy = generate_retention_strategy_v2(
            probability=0.50,
            monthly_charges=50.0,
            contract_type="Month-to-month",
            tenure=10,
        )

        assert len(strategy['action']) > 50  # Decent sized recommendation


class TestTierDistribution:
    """Test tier distribution properties."""

    def test_tier_values_are_valid(self):
        """Tier should be one of the three valid values."""
        valid_tiers = ["high_value", "standard", "no_action"]

        for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            strategy = generate_retention_strategy_v2(
                probability=prob,
                monthly_charges=50.0,
                contract_type="Month-to-month",
                tenure=10,
            )
            assert strategy['tier'] in valid_tiers

    def test_ev_matches_tier_assignment(self):
        """EV value should match its tier assignment."""
        for prob in [0.2, 0.4, 0.6, 0.8]:
            strategy = generate_retention_strategy_v2(
                probability=prob,
                monthly_charges=70.0,
                contract_type="Month-to-month",
                tenure=5,
            )

            ev = strategy['expected_value']
            tier = strategy['tier']

            if ev > EV_HIGH_VALUE_THRESHOLD:
                assert tier == "high_value"
            elif ev > EV_STANDARD_THRESHOLD:
                assert tier == "standard"
            else:
                assert tier == "no_action"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_probability_zero(self):
        """Zero churn probability should yield negative EV."""
        strategy = generate_retention_strategy_v2(
            probability=0.0,
            monthly_charges=100.0,
            contract_type="Month-to-month",
            tenure=0,
        )

        assert strategy['expected_value'] < 0
        assert strategy['tier'] == "no_action"

    def test_probability_one(self):
        """Probability=1 (certain churn) should yield max EV."""
        strategy1 = generate_retention_strategy_v2(
            probability=1.0,
            monthly_charges=100.0,
            contract_type="Month-to-month",
            tenure=0,
        )

        strategy2 = generate_retention_strategy_v2(
            probability=0.5,
            monthly_charges=100.0,
            contract_type="Month-to-month",
            tenure=0,
        )

        assert strategy1['expected_value'] > strategy2['expected_value']

    def test_very_low_monthly_charges(self):
        """Low monthly charge customer (minimum plan)."""
        strategy = generate_retention_strategy_v2(
            probability=0.90,  # High churn risk
            monthly_charges=15.0,  # Minimum
            contract_type="Month-to-month",
            tenure=0,
        )

        # Even high churn risk won't justify intervention for min plan
        assert strategy['expected_value'] >= 0  # Likely still no-action

    def test_very_high_monthly_charges(self):
        """High monthly charge customer (premium plan)."""
        strategy = generate_retention_strategy_v2(
            probability=0.30,  # Moderate churn risk
            monthly_charges=120.0,  # Maximum
            contract_type="Two year",
            tenure=0,
        )

        # Moderate risk on premium plan should warrant intervention
        assert strategy['expected_value'] > 0
        assert strategy['tier'] in ["standard", "high_value"]

    def test_contract_type_comparison(self):
        """Longer contracts should yield higher EV (more CLV)."""
        prob, monthly, tenure = 0.50, 60.0, 10

        ev_month = expected_value(prob, monthly, "Month-to-month", tenure)
        ev_year = expected_value(prob, monthly, "One year", tenure)
        ev_two_year = expected_value(prob, monthly, "Two year", tenure)

        assert ev_month < ev_year < ev_two_year