// =============================================================================
// AGentic_C — Example HFT Strategy (with deliberate anti-patterns for demo)
// =============================================================================
//
// This file demonstrates a realistic (simplified) HFT trading strategy.
// Several functions intentionally contain HFT anti-patterns (LAP-001 to LAP-010)
// so that the AGentic_C compiler pipeline can detect and fix them.
//
// Functions:
//   on_market_data    → HOT  LAP-001 (heap), LAP-005 (I/O)
//   evaluate_signal   → HOT  LAP-010 (branches), LAP-002 (virtual)
//   check_risk        → HOT  CLEAN (good example)
//   submit_order      → HOT  LAP-006 (indirect call), LAP-003 (exception)
//   load_config       → COLD (setup)
//   log_trade         → COLD (logging)
//
// Run through AGentic_C:
//   python src/pipeline.py examples/hft_strategy.cpp
//
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constants and types
// ---------------------------------------------------------------------------

static const int    MAX_POSITION  = 10000;
static const int    MAX_ORDER_QTY = 500;
static const double TICK_SIZE     = 0.01;
static const int    EMA_WINDOW    = 20;

struct Tick {
    double  price;
    int     size;
    uint8_t side;        // 0 = bid, 1 = ask
    long    timestamp_ns;
};

struct Order {
    double  price;
    int     qty;
    uint8_t side;
    char    symbol[8];
};

static Order   g_order;
static int     g_socket_fd   = -1;
static int     g_position     = 0;
static double  g_ema_fast     = 0.0;
static double  g_ema_slow     = 0.0;
static double  g_mid_price    = 0.0;

// ---------------------------------------------------------------------------
// HOT PATH FUNCTIONS
// ---------------------------------------------------------------------------

// LAP-001: heap allocation (new OrderEntry) on every tick
// LAP-005: printf is a system call — kernel context switch on hot path
[[hft::hot]]
double on_market_data(const Tick& tick) {
    // ❌ LAP-001: dynamic allocation per tick — causes 100ns-10µs latency spikes
    double* entry = new double(tick.price);

    // ❌ LAP-005: printf = syscall, costs 1-10µs kernel context switch
    printf("Tick received: price=%.2f size=%d\n", tick.price, tick.size);

    // Update mid price and EMAs
    g_mid_price = tick.price;

    double alpha_fast = 2.0 / (EMA_WINDOW + 1);
    double alpha_slow = 2.0 / (EMA_WINDOW * 4 + 1);

    g_ema_fast = alpha_fast * (*entry) + (1.0 - alpha_fast) * g_ema_fast;
    g_ema_slow = alpha_slow * (*entry) + (1.0 - alpha_slow) * g_ema_slow;

    delete entry;   // heap deallocation — also slow
    return g_mid_price;
}

// LAP-002: virtual dispatch (mock interface)
// LAP-010: branch-heavy logic with nested if-else chains
struct SignalBase {
    virtual double compute(double fast, double slow) { return 0.0; }
};

struct MomentumSignal : public SignalBase {
    virtual double compute(double fast, double slow) override {
        return fast - slow;
    }
};

[[hft::hot]]
int evaluate_signal(double fast_ema, double slow_ema) {
    // ❌ LAP-002: virtual dispatch through pointer — vtable lookup + cache miss
    SignalBase* sig = new MomentumSignal();   // also LAP-001

    double signal_val = sig->compute(fast_ema, slow_ema);
    delete sig;

    // ❌ LAP-010: deeply nested if-else — unpredictable branches → misprediction
    if (signal_val > 0.05) {
        if (signal_val > 0.15) {
            if (signal_val > 0.30) {
                return 3;  // strong buy
            } else {
                return 2;  // moderate buy
            }
        } else {
            return 1;  // weak buy
        }
    } else if (signal_val < -0.05) {
        if (signal_val < -0.15) {
            if (signal_val < -0.30) {
                return -3;  // strong sell
            } else {
                return -2;  // moderate sell
            }
        } else {
            return -1;  // weak sell
        }
    }
    return 0;  // neutral
}

// CLEAN HOT PATH — this is the gold standard HFT function
// No anti-patterns, uses inline and noexcept correctly
[[hft::hot]]
inline bool check_risk(int qty, int current_position) noexcept {
    // Branchless: no heap, no calls, no exceptions
    int new_position = current_position + qty;
    bool qty_ok      = (qty > 0 && qty <= MAX_ORDER_QTY);
    bool pos_ok      = (new_position >= -MAX_POSITION && new_position <= MAX_POSITION);
    return qty_ok && pos_ok;
}

// LAP-003: exception on hot path
// LAP-006: std::function (indirect, non-inlinable call)
[[hft::hot]]
bool submit_order(int qty, double price, uint8_t side) {
    // ❌ LAP-003: throw on error path — exception unwind is catastrophic latency
    if (qty <= 0 || price <= 0.0) {
        throw std::invalid_argument("Invalid order parameters");
    }

    // ❌ LAP-006: std::function prevents inlining, adds indirect dispatch
    std::function<bool()> send_fn = [&]() {
        g_order.price = price;
        g_order.qty   = qty;
        g_order.side  = side;
        // Simulate kernel-bypass send
        return (g_socket_fd >= 0);
    };

    if (!check_risk(qty, g_position)) {
        return false;
    }

    bool sent = send_fn();
    if (sent) {
        g_position += (side == 0) ? qty : -qty;
    }
    return sent;
}

// ---------------------------------------------------------------------------
// COLD PATH FUNCTIONS
// ---------------------------------------------------------------------------

[[hft::cold]]
void load_config(const char* config_path) {
    // Cold path — file I/O, string parsing is acceptable here
    printf("Loading config from: %s\n", config_path);

    // Simulate config loading
    g_socket_fd   = 0;   // would be real fd in production
    g_position    = 0;
    g_ema_fast    = 0.0;
    g_ema_slow    = 0.0;
    g_mid_price   = 0.0;

    printf("Config loaded. socket_fd=%d initial_position=%d\n",
           g_socket_fd, g_position);
}

[[hft::cold]]
void log_trade(int qty, double price, uint8_t side, bool success) {
    // Cold path — logging is not on the critical path
    const char* side_str = (side == 0) ? "BUY" : "SELL";
    printf("[TRADE] %s qty=%d price=%.4f success=%s position=%d\n",
           side_str, qty, price, success ? "YES" : "NO", g_position);
}

// ---------------------------------------------------------------------------
// Main entry point for standalone testing
// ---------------------------------------------------------------------------

int main() {
    printf("=== AGentic_C HFT Strategy Demo ===\n");

    // Cold path setup
    load_config("config/hft.yaml");

    // Simulate a market data event loop
    Tick tick = { 100.05, 1000, 0, 1713000000000LL };

    double price = on_market_data(tick);
    int    signal= evaluate_signal(g_ema_fast, g_ema_slow);
    bool   ok    = check_risk(100, g_position);

    if (ok && signal > 0) {
        bool sent = submit_order(100, price, 0);
        log_trade(100, price, 0, sent);
    }

    printf("=== Done. Position: %d ===\n", g_position);
    return 0;
}
