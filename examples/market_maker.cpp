/**
 * AGentic_C Example: Market Making Strategy
 * File: examples/market_maker.cpp
 *
 * This is a simplified market making strategy that continuously
 * quotes bid and ask prices around a fair value estimate and
 * manages inventory risk. It demonstrates the kind of code that
 * AGentic_C is designed to optimise for HFT workloads.
 *
 * HOT functions are annotated with [[hft::hot]] — these are the
 * functions that must execute within tight nanosecond budgets.
 * COLD functions handle setup, logging, and risk reports.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <array>
#include <atomic>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

// ──────────────────────────────────────────────────────────────────────────────
// Constants and Configuration
// ──────────────────────────────────────────────────────────────────────────────

static constexpr double BASE_SPREAD     = 0.0002;   // 2 bps base spread
static constexpr double MAX_INVENTORY   = 500.0;    // maximum net position
static constexpr double RISK_LIMIT      = 1000.0;   // max gross exposure
static constexpr double TICK_SIZE       = 0.01;     // minimum price increment
static constexpr int    ORDER_LEVELS    = 3;        // quote levels each side
static constexpr int    BOOK_DEPTH      = 10;       // order book depth tracked

// ──────────────────────────────────────────────────────────────────────────────
// Data Structures
// ──────────────────────────────────────────────────────────────────────────────

struct PriceLevel {
    double price;
    double qty;
    uint64_t timestamp_ns;
};

struct OrderBook {
    std::array<PriceLevel, BOOK_DEPTH> bids;
    std::array<PriceLevel, BOOK_DEPTH> asks;
    int bid_count;
    int ask_count;
    uint64_t last_update_ns;
};

struct Quote {
    double bid_price;
    double ask_price;
    double bid_qty;
    double ask_qty;
    uint64_t valid_until_ns;
};

struct Fill {
    double price;
    double qty;
    int    side;          // +1 = buy fill, -1 = sell fill
    uint64_t fill_time_ns;
};

struct MarketState {
    double fair_value;
    double volatility;
    double imbalance;     // order book imbalance -1 to +1
    double inventory;
    double pnl;
    uint64_t tick_count;
};

// ──────────────────────────────────────────────────────────────────────────────
// Global state (in real HFT these would be cache-line padded and NUMA-local)
// ──────────────────────────────────────────────────────────────────────────────

static OrderBook  g_book   = {};
static MarketState g_state = {};
static Quote       g_quote = {};

// ──────────────────────────────────────────────────────────────────────────────
// HOT FUNCTIONS — must complete within their latency budgets
// ──────────────────────────────────────────────────────────────────────────────

/**
 * on_book_update — called on every order book change from the exchange feed.
 * Budget: 200 ns.  Must update fair value and trigger requote if needed.
 * LAP: heap allocation inside (new), printf inside — AGentic_C will flag these.
 */
[[hft::hot]]
void on_book_update(const double* bid_prices, const double* bid_qtys,
                    const double* ask_prices, const double* ask_qtys,
                    int depth, uint64_t recv_ns)
{
    // Update order book — direct array write, fast path
    for (int i = 0; i < depth && i < BOOK_DEPTH; ++i) {
        g_book.bids[i] = { bid_prices[i], bid_qtys[i], recv_ns };
        g_book.asks[i] = { ask_prices[i], ask_qtys[i], recv_ns };
    }
    g_book.bid_count = depth;
    g_book.ask_count = depth;
    g_book.last_update_ns = recv_ns;

    // Compute mid price
    double best_bid = g_book.bids[0].price;
    double best_ask = g_book.asks[0].price;
    double mid = (best_bid + best_ask) * 0.5;

    // Compute order book imbalance (buy pressure vs sell pressure)
    double total_bid_qty = 0.0;
    double total_ask_qty = 0.0;
    for (int i = 0; i < depth && i < BOOK_DEPTH; ++i) {
        total_bid_qty += g_book.bids[i].qty;
        total_ask_qty += g_book.asks[i].qty;
    }
    double imbalance = (total_bid_qty - total_ask_qty)
                       / (total_bid_qty + total_ask_qty + 1e-9);

    // Fair value tilts toward the pressure side
    g_state.fair_value = mid + imbalance * (best_ask - best_bid) * 0.1;
    g_state.imbalance  = imbalance;

    // LAP-001: heap allocation — should not appear in hot path
    std::string* debug_str = new std::string("book_update");
    printf("[tick %llu] mid=%.4f fv=%.4f imb=%.3f %s\n",
           (unsigned long long)g_state.tick_count,
           mid, g_state.fair_value, imbalance, debug_str->c_str());
    delete debug_str;

    ++g_state.tick_count;
}

// ──────────────────────────────────────────────────────────────────────────────

/**
 * compute_quotes — determines bid/ask prices and sizes to send.
 * Budget: 150 ns.  Pure arithmetic, should be fully inlineable and vectorisable.
 * LAP: virtual call via std::function — will be detected by Fixer Agent.
 */
[[hft::hot]]
Quote compute_quotes(double fair_value,
                     double volatility,
                     double inventory,
                     std::function<double(double)> skew_fn)
{
    // Spread widens with volatility and narrows when inventory is flat
    double spread = BASE_SPREAD + volatility * 0.5;

    // Inventory skew: shift quotes to attract offsetting flow
    double inv_ratio = inventory / MAX_INVENTORY;
    double skew = skew_fn(inv_ratio);   // LAP-006: indirect call via std::function

    double bid = fair_value - spread * 0.5 - skew;
    double ask = fair_value + spread * 0.5 - skew;

    // Round to tick grid
    bid = std::floor(bid / TICK_SIZE) * TICK_SIZE;
    ask = std::ceil(ask  / TICK_SIZE) * TICK_SIZE;

    // Size scales down when inventory is large (exposure management)
    double base_qty = 100.0;
    double bid_qty  = base_qty * (1.0 - std::max(0.0,  inv_ratio));
    double ask_qty  = base_qty * (1.0 - std::max(0.0, -inv_ratio));

    return Quote{ bid, ask, bid_qty, ask_qty, 0 };
}

// ──────────────────────────────────────────────────────────────────────────────

/**
 * on_fill — called when one of our quotes is hit by the market.
 * Budget: 100 ns.  Updates inventory and PnL, triggers risk check.
 */
[[hft::hot]]
void on_fill(const Fill& fill)
{
    // Update position
    double delta = fill.qty * fill.side;
    g_state.inventory += delta;

    // Mark-to-market PnL update
    double entry_cost = fill.price * fill.qty * fill.side;
    g_state.pnl -= entry_cost;  // will close out at fair value

    // Inline risk gate — no function call overhead
    double gross_exposure = std::abs(g_state.inventory) * g_book.bids[0].price;
    if (gross_exposure > RISK_LIMIT) {
        // LAP-005: exception in hot path — hard to optimise, must fix at source
        throw std::runtime_error("Risk limit breached on fill");
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/**
 * check_stale_quotes — checks if our current quotes are too old.
 * Budget: 80 ns.  Simple timestamp comparison, very tight budget.
 */
[[hft::hot]]
bool check_stale_quotes(uint64_t now_ns, uint64_t max_age_ns)
{
    uint64_t age = now_ns - g_book.last_update_ns;
    if (age > max_age_ns) {
        return true;
    }

    // Check if our fair value moved enough to need a requote
    double spread = g_book.asks[0].price - g_book.bids[0].price;
    double fv_move = std::abs(g_state.fair_value - g_quote.bid_price - spread * 0.5);
    return fv_move > TICK_SIZE * 2.0;
}

// ──────────────────────────────────────────────────────────────────────────────

/**
 * update_volatility — exponential moving average of bid-ask spread as vol proxy.
 * Budget: 120 ns.  Math intensive, good candidate for SIMD vectorisation.
 */
[[hft::hot]]
void update_volatility(uint64_t now_ns)
{
    static double prev_mid  = 0.0;
    static uint64_t prev_ns = 0;

    double best_bid = g_book.bids[0].price;
    double best_ask = g_book.asks[0].price;
    double mid = (best_bid + best_ask) * 0.5;

    if (prev_ns > 0 && now_ns > prev_ns) {
        double dt_sec  = (now_ns - prev_ns) * 1e-9;
        double ret     = (mid - prev_mid) / (prev_mid + 1e-9);
        double sq_ret  = ret * ret;
        double lambda  = std::exp(-dt_sec / 60.0);  // 1-minute half-life
        g_state.volatility = lambda * g_state.volatility + (1.0 - lambda) * sq_ret;
        g_state.volatility = std::sqrt(g_state.volatility * (1.0 / dt_sec));
    }

    prev_mid = mid;
    prev_ns  = now_ns;
}

// ──────────────────────────────────────────────────────────────────────────────
// COLD FUNCTIONS — logging, reporting, startup (no tight latency requirement)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * print_status — human readable state dump, called every second.
 */
[[hft::cold]]
void print_status()
{
    printf("=== Market Maker Status ===\n");
    printf("  Fair Value : %.4f\n", g_state.fair_value);
    printf("  Inventory  : %.2f\n", g_state.inventory);
    printf("  PnL        : %.2f\n", g_state.pnl);
    printf("  Volatility : %.6f\n", g_state.volatility);
    printf("  Imbalance  : %.3f\n", g_state.imbalance);
    printf("  Ticks      : %llu\n", (unsigned long long)g_state.tick_count);
    printf("  Quote Bid  : %.4f x %.0f\n", g_quote.bid_price, g_quote.bid_qty);
    printf("  Quote Ask  : %.4f x %.0f\n", g_quote.ask_price, g_quote.ask_qty);
}

/**
 * generate_risk_report — builds and returns a detailed risk summary string.
 * Called once per minute by a monitoring thread.
 */
[[hft::cold]]
std::string generate_risk_report()
{
    std::string report;
    report += "=== Risk Report ===\n";
    report += "Inventory  : " + std::to_string(g_state.inventory) + "\n";
    report += "PnL        : " + std::to_string(g_state.pnl) + "\n";
    report += "Volatility : " + std::to_string(g_state.volatility) + "\n";
    report += "Tick count : " + std::to_string(g_state.tick_count) + "\n";
    report += "Max inv    : " + std::to_string(MAX_INVENTORY) + "\n";
    report += "Risk limit : " + std::to_string(RISK_LIMIT) + "\n";
    return report;
}

/**
 * initialise — set up initial state, called once at startup.
 */
[[hft::cold]]
void initialise(double start_fair_value)
{
    memset(&g_book,  0, sizeof(g_book));
    memset(&g_state, 0, sizeof(g_state));
    memset(&g_quote, 0, sizeof(g_quote));
    g_state.fair_value = start_fair_value;
    printf("[init] Market maker ready. Fair value = %.4f\n", start_fair_value);
}

// ──────────────────────────────────────────────────────────────────────────────
// MAIN — demo entry point
// ──────────────────────────────────────────────────────────────────────────────

int main()
{
    initialise(100.00);

    // Simulated skew function (normally a lookup table in production)
    auto skew_fn = [](double inv_ratio) -> double {
        return inv_ratio * BASE_SPREAD * 2.0;
    };

    // Simulate 5 book update ticks
    for (int tick = 0; tick < 5; ++tick) {
        double base  = 100.00 + tick * 0.01;
        double bids[BOOK_DEPTH] = {}, asks[BOOK_DEPTH] = {};
        double bqty[BOOK_DEPTH] = {}, aqty[BOOK_DEPTH] = {};
        for (int i = 0; i < BOOK_DEPTH; ++i) {
            bids[i] = base - (i + 1) * TICK_SIZE;
            asks[i] = base + (i + 1) * TICK_SIZE;
            bqty[i] = 100.0 * (BOOK_DEPTH - i);
            aqty[i] = 100.0 * (BOOK_DEPTH - i);
        }
        uint64_t now_ns = 1700000000000000000ULL + (uint64_t)tick * 1000000ULL;
        on_book_update(bids, bqty, asks, aqty, BOOK_DEPTH, now_ns);
        update_volatility(now_ns);

        g_quote = compute_quotes(
            g_state.fair_value,
            g_state.volatility,
            g_state.inventory,
            skew_fn
        );

        if (check_stale_quotes(now_ns + 5000, 1000000ULL)) {
            printf("[requote needed at tick %d]\n", tick);
        }
    }

    // Simulate a fill
    Fill fill { 100.02, 50.0, +1, 1700000000005000000ULL };
    on_fill(fill);

    print_status();
    printf("\n%s\n", generate_risk_report().c_str());
    return 0;
}
