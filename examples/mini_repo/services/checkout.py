from services.pricing import calculate_total, discount_total


class CheckoutService:
    def quote(self, cart, user):
        return calculate_total(cart["items"], user.region)

    def quote_with_coupon(self, cart, user, coupon):
        return discount_total(cart["items"], user.region, coupon)
