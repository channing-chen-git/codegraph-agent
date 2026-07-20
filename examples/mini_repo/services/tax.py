TAX_RATE = {
    "US": 0.07,
    "EU": 0.2,
    "CN": 0.06,
}


def apply_tax(amount, region):
    """Apply region-specific tax to a numeric amount."""
    rate = TAX_RATE.get(region, 0.0)
    return amount * (1 + rate)
