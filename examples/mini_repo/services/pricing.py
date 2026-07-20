from services.tax import apply_tax


def calculate_total(items, user_region):
    """Calculate final cart price after item subtotal and regional tax."""
    subtotal = sum(item["price"] * item.get("quantity", 1) for item in items)
    return apply_tax(subtotal, user_region)


def discount_total(items, user_region, coupon):
    total = calculate_total(items, user_region)
    if coupon == "VIP10":
        return total * 0.9
    return total
