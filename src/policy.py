"""Business rules for the email decision (shared by the API)."""
import numpy as np

POP = 100_000


def break_even_costs(g_mens, g_womens, value):
    """Highest cost per email at which each email still pays for itself on average."""
    return g_mens * value, g_womens * value


def simple_recommendation(cost, g_mens, g_womens, value):
    be_m, be_w = break_even_costs(g_mens, g_womens, value)
    profit_men_all = POP * (g_mens * value - cost)
    profit_women_all = POP * (g_womens * value - cost)

    if cost >= be_m:
        action = "send_nothing"
        message = (
            f"At ${cost:.2f} per email, even the better (men's) email costs more than it "
            f"earns on average (break-even ${be_m:.2f}). Send nothing to everyone; "
            "use only a small, measured A/B test."
        )
    else:
        action = "mens_email_to_everyone"
        message = (
            f"Send the men's email to everyone. Estimated profit about "
            f"${profit_men_all:,.0f} per {POP:,} customers (from randomized averages)."
        )

    return {
        "action": action,
        "message": message,
        "cost_per_email": cost,
        "profit_per_extra_buyer": value,
        "break_even_cost_mens": be_m,
        "break_even_cost_womens": be_w,
        "womens_email_pays_off": cost < be_w,
        "near_break_even": bool(0.75 * be_m <= cost < be_m),
        "estimated_profit_per_100k_mens_to_all": profit_men_all,
        "estimated_profit_per_100k_womens_to_all": profit_women_all,
    }


def shrink_scores(up_m, up_w, cfg):
    """Pull raw scores toward the average effect of each email (see Step 10)."""
    lam = cfg["lambda"]
    s_m = cfg["global_uplift"]["mens"] + lam * (up_m - cfg["score_centre"]["mens"])
    s_w = cfg["global_uplift"]["womens"] + lam * (up_w - cfg["score_centre"]["womens"])
    return s_m, s_w


def experimental_plan(s_m, s_w, threshold):
    """Better email per customer, or no email if the score is below break-even."""
    best = np.maximum(s_m, s_w)
    return np.where(
        best > threshold,
        np.where(s_m >= s_w, "Men's email", "Women's email"),
        "Do not email",
    )