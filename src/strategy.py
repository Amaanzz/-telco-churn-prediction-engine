from typing import Dict


def generate_retention_strategy(probability: float, monthly_charges: float, tenure: int) -> Dict[str, str]:
    """
    Translates churn probability into a structured business strategy.
    Considers customer lifetime value proxy (tenure & monthly charges).
    """
    is_high_value = monthly_charges > 70.0 and tenure > 12

    if probability >= 0.75:
        risk_level = "Critical"
        if is_high_value:
            action = "Immediate priority outreach by Senior Retention Agent. Offer up to 20% discount on 1-year contract extension."
        else:
            action = "Send automated free-upgrade offer or complimentary add-on to regain engagement."

    elif probability >= 0.50:
        risk_level = "High"
        if is_high_value:
            action = "Proactive check-in call to assess service satisfaction. Offer loyalty rewards."
        else:
            action = "Targeted email campaign highlighting unused features and offering flexible payment plans."

    elif probability >= 0.30:
        risk_level = "Medium"
        action = "Monitor usage patterns. Subscribe customer to standard promotional newsletter to maintain top-of-mind awareness."

    else:
        risk_level = "Low"
        action = "No intervention required. Maintain standard customer journey communications."

    return {
        "risk_level": risk_level,
        "action": action
    }