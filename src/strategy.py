"""
Strategy Module v2 — Expected Value Based Retention Logic

This module replaces the old probability-threshold-based strategy with
an economically defensible Expected Value (EV) engine.

Key business assumptions:
- offer_success_rate = 0.30 (30% of offers successfully prevent churn)
- cost_of_offer = $75 ($50 discount + $25 outreach labor)
- segment_lifetime (Kaplan-Meier restricted mean, 72-month horizon):
  * Month-to-month: 36.3 months
  * One year: 66.4 months
  * Two year: 71.5 months

Formula: EV = P(churn) × CLV_at_risk × offer_success_rate − cost_of_offer
"""

from typing import Dict, Any

# ===== BUSINESS PARAMETERS =====
# These are the core economic assumptions. Document changes in code review.

SEGMENT_LIFETIME = {
    "Month-to-month": 36.3,
    "One year": 66.4,
    "Two year": 71.5,
}
"""
Expected remaining lifetime (months) per contract type.
Source: Kaplan-Meier survival curves fitted on training data (Phase 2).
Horizon: 72 months (6 years), avoiding extrapolation beyond observed tenure range.
"""

OFFER_SUCCESS_RATE = 0.30
"""
Probability that a retention offer successfully prevents churn.
Justification: Conservative industry benchmark. 
Sensitivity: Model tested with [0.15, 0.20, 0.25, 0.30, 0.35, 0.40].
At 0.30, tier distribution: high_value=17.5%, standard=20.7%, no_action=61.8%.
"""

COST_OF_OFFER = 75
"""
Direct cost of making a retention offer.
Breakdown: $50 (discount/incentive) + $25 (outreach labor cost).
Justification: Consistent with Phase 1 false-positive cost definition.
"""

# EV thresholds for tier assignment
EV_HIGH_VALUE_THRESHOLD = 200
"""
EV > $200 → high-value intervention (senior agent outreach + significant offer).
Justification: Above $200 expected value justifies dedicated agent time (~$50/call).
"""

EV_STANDARD_THRESHOLD = 0
"""
$0 < EV ≤ $200 → standard intervention (automated offer).
EV = 0 is the break-even point (expected benefit = expected cost).
"""


# ===== HELPER FUNCTIONS =====

def expected_remaining_months(contract_type: str, current_tenure: int, floor: float = 1.0) -> float:
    """
    Conditional expected remaining tenure for this customer's segment.

    Handles tenure outliers: if a Month-to-month customer has tenure=60 months
    (above segment average of 36.3), this returns floored 1 month, not negative.

    This is a simplified version; the notebook uses survival-curve integration
    for more precision. This version uses naive segment average − current tenure.

    Args:
        contract_type: One of ['Month-to-month', 'One year', 'Two year']
        current_tenure: Customer's current tenure in months
        floor: Minimum remaining months (avoid zero or negative CLV)

    Returns:
        Remaining months, guaranteed >= floor
    """
    total_expected = SEGMENT_LIFETIME.get(contract_type, 36.3)  # Default to Month-to-month
    return max(total_expected - current_tenure, floor)


def clv_at_risk(monthly_charges: float, contract_type: str, current_tenure: int) -> float:
    """
    Customer Lifetime Value at risk (the revenue we'll lose if they churn).

    CLV = monthly_charges × expected_remaining_months

    Uses survival-informed lifetime, not naive 12 months.

    Args:
        monthly_charges: Customer's monthly subscription cost
        contract_type: Contract type (determines remaining lifetime)
        current_tenure: Customer's tenure in months

    Returns:
        CLV in dollars
    """
    remaining_months = expected_remaining_months(contract_type, current_tenure)
    return monthly_charges * remaining_months


def expected_value(
        probability: float,
        monthly_charges: float,
        contract_type: str,
        tenure: int,
        offer_success_rate: float = OFFER_SUCCESS_RATE,
        cost_of_offer: float = COST_OF_OFFER,
) -> float:
    """
    Expected Value of making a retention offer to this customer.

    Formula:
        EV = P(churn) × CLV_at_risk × offer_success_rate − cost_of_offer

    Interpretation:
        EV > 0: Expected value of intervention is positive → make offer
        EV < 0: Expected value is negative → don't offer (too costly)

    Args:
        probability: P(churn) from the classifier [0, 1]
        monthly_charges: Customer's monthly cost
        contract_type: Contract type
        tenure: Customer's tenure in months
        offer_success_rate: (Optional) Override the default 30% success rate
        cost_of_offer: (Optional) Override the default $75 cost

    Returns:
        Expected value in dollars
    """
    clv = clv_at_risk(monthly_charges, contract_type, tenure)
    return probability * clv * offer_success_rate - cost_of_offer


# ===== STRATEGY GENERATION =====

def generate_retention_strategy_v2(
        probability: float,
        monthly_charges: float,
        contract_type: str,
        tenure: int,
        is_senior_citizen: bool = False,
        offer_success_rate: float = OFFER_SUCCESS_RATE,
        cost_of_offer: float = COST_OF_OFFER,
) -> Dict[str, Any]:
    """
    Generate a retention strategy based on Expected Value tiers.

    This replaces the old probability-threshold approach (e.g., "if prob > 0.75: high risk").
    Now: EV > $200 → high-value tier → senior agent outreach.

    Args:
        probability: P(churn) from classifier
        monthly_charges: Customer's monthly cost
        contract_type: Contract type
        tenure: Customer's tenure in months
        is_senior_citizen: Boolean flag for senior citizen targeting
        offer_success_rate: Override default success rate if needed
        cost_of_offer: Override default offer cost if needed

    Returns:
        Dictionary with keys:
            - tier: "high_value", "standard", or "no_action"
            - expected_value: Computed EV (rounded to 2 decimals)
            - action: Human-readable recommendation
            - senior_warning: Boolean; True if customer is senior (high churn risk)
    """
    ev = expected_value(
        probability=probability,
        monthly_charges=monthly_charges,
        contract_type=contract_type,
        tenure=tenure,
        offer_success_rate=offer_success_rate,
        cost_of_offer=cost_of_offer,
    )

    # Tier assignment logic (EV-based, not probability-based)
    if ev > EV_HIGH_VALUE_THRESHOLD:
        tier = "high_value"
        action = (
            f"🎯 **High-Value Intervention** | "
            f"Expected value: ${ev:.2f}\n\n"
            f"**Recommended action:**\n"
            f"• Assign senior agent for personal outreach\n"
            f"• Offer significant discount ($50+) + upgrade incentive\n"
            f"• Emphasize contract upgrade path (Month-to-month → One year)\n"
            f"• Response time: 24 hours"
        )
    elif ev > EV_STANDARD_THRESHOLD:
        tier = "standard"
        action = (
            f"📊 **Standard Intervention** | "
            f"Expected value: ${ev:.2f}\n\n"
            f"**Recommended action:**\n"
            f"• Send automated retention offer (email/SMS)\n"
            f"• Offer modest discount ($25–35) or service upgrade\n"
            f"• Include option to contact support for negotiation\n"
            f"• Response time: 2–3 days"
        )
    else:
        tier = "no_action"
        action = (
            f"⚪ **Monitor Only** | "
            f"Expected value: ${ev:.2f}\n\n"
            f"**Recommended action:**\n"
            f"• No proactive intervention — expected cost exceeds expected benefit\n"
            f"• Monitor churn risk via quarterly reviews\n"
            f"• Engage only if customer initiates support request"
        )

    # Senior citizen warning
    senior_warning = is_senior_citizen
    if is_senior_citizen:
        action += (
            f"\n\n⚠️ **Senior Citizen Alert** | "
            f"Churn risk is 41.7% (vs. 23.7% for non-seniors)\n"
            f"• Prioritize empathetic communication\n"
            f"• Consider bundled offers (simpler service packages)"
        )

    return {
        "tier": tier,
        "expected_value": round(ev, 2),
        "action": action,
        "senior_warning": senior_warning,
    }


# ===== LEGACY SUPPORT =====
# Keep old function for backward compatibility (deprecated)

def generate_retention_strategy(probability: float, monthly_charges: float, tenure: int) -> Dict[str, Any]:
    """
    DEPRECATED: Old probability-threshold-based strategy.

    Use generate_retention_strategy_v2() instead for EV-based logic.
    This function is kept for backward compatibility only.
    """
    # Map old strategy to new EV-based one
    # Note: This is approximate and doesn't use full EV calculation
    ev_approx = probability * monthly_charges * 12 * 0.30 - 75

    if probability >= 0.75:
        risk_level = "Critical"
    elif probability >= 0.50:
        risk_level = "High"
    elif probability >= 0.25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    action = {
        "Critical": "🔥 Urgent intervention required — assign to senior agent immediately",
        "High": "⚠️ High risk — send retention offer with significant incentive",
        "Medium": "📊 Moderate risk — automated retention offer recommended",
        "Low": "✅ Low risk — continue standard engagement",
    }[risk_level]

    return {
        "risk_level": risk_level,
        "action": action,
    }