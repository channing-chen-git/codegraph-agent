#include "cart.h"
#include <vector>

double subtotal(const std::vector<CartItem>& items) {
    double total = 0.0;
    for (const auto& item : items) {
        total += item.price * item.quantity;
    }
    return total;
}

double quote_cart(const std::vector<CartItem>& items, double tax_rate) {
    return subtotal(items) * (1.0 + tax_rate);
}
