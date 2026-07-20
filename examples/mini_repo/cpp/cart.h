#pragma once

struct CartItem {
    double price;
    int quantity;
};

double subtotal(const std::vector<CartItem>& items);
double quote_cart(const std::vector<CartItem>& items, double tax_rate);
