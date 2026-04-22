// =============================================================================
// AGentic_C — Example: Limit Order Book Matching Engine
// =============================================================================
//
// This is a realistic, production-style Limit Order Book (LOB) engine.
// It implements price-time priority matching, a two-sided book (bids/asks),
// order cancellation, and real-time spread computation.
//
// This is the most complex AGentic_C example — it intentionally contains
// ALL 10 HFT anti-patterns (LAP-001 to LAP-010) across its hot-path
// functions so the full power of the agent pipeline can be demonstrated.
//
// Functions:
//   process_order_add     → HOT  LAP-001 (heap alloc), LAP-004 (mutex)
//   match_orders          → HOT  LAP-002 (virtual), LAP-003 (exception), LAP-010 (branches)
//   compute_spread        → HOT  LAP-006 (std::function), LAP-008 (dynamic_cast)
//   cancel_order          → HOT  LAP-007 (atomic seq-cst), LAP-009 (unaligned)
//   handle_market_event   → HOT  LAP-005 (printf), LAP-001 (heap)
//   update_vwap           → HOT  CLEAN — reference implementation
//   flush_trade_log       → COLD (async I/O)
//   load_instruments      → COLD (startup config)
//   reset_book            → COLD (teardown)
//
// Run through AGentic_C:
//   python src/pipeline.py examples/order_book_engine.cpp
//
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <string>

// ---------------------------------------------------------------------------
// Constants and types
// ---------------------------------------------------------------------------

static const int    MAX_ORDERS     = 65536;
static const int    MAX_LEVELS     = 512;
static const double MIN_TICK       = 0.0001;
static const int    MAX_QTY        = 100000;

// Order side
enum class Side : uint8_t { BID = 0, ASK = 1 };

// Order type
enum class OrdType : uint8_t { LIMIT = 0, MARKET = 1, IOC = 2, FOK = 3 };

// Raw order entry — packed for cache efficiency
struct alignas(64) OrderEntry {
    uint64_t  order_id;
    double    price;
    int       qty;
    int       filled_qty;
    Side      side;
    OrdType   type;
    char      symbol[8];
    long      timestamp_ns;
};

// Price level
struct PriceLevel {
    double price;
    int    total_qty;
    int    order_count;
};

// Trade report
struct Trade {
    uint64_t  aggressor_id;
    uint64_t  passive_id;
    double    exec_price;
    int       exec_qty;
    long      timestamp_ns;
};

// Global book state
static OrderEntry  g_orders[MAX_ORDERS];
static PriceLevel  g_bids[MAX_LEVELS];
static PriceLevel  g_asks[MAX_LEVELS];
static int         g_bid_levels   = 0;
static int         g_ask_levels   = 0;
static uint64_t    g_order_seq    = 0;
static double      g_vwap         = 0.0;
static double      g_cum_vol      = 0.0;
static double      g_cum_notional = 0.0;

// ❌ LAP-004: global mutex — any lock on the hot path is a latency killer
static std::mutex  g_book_mutex;

// ❌ LAP-007: atomic with sequential consistency — overkill, use relaxed
static std::atomic<int> g_trade_count{0};

// ---------------------------------------------------------------------------
// HOT PATH FUNCTIONS
// ---------------------------------------------------------------------------

// LAP-001: dynamic allocation of every new order on heap
// LAP-004: mutex lock on critical hot path
[[hft::hot]]
uint64_t process_order_add(double price, int qty, Side side, OrdType type,
                            const char* symbol) {
    // ❌ LAP-004: acquiring a mutex blocks all other threads — never on hot path
    std::lock_guard<std::mutex> lock(g_book_mutex);

    // ❌ LAP-001: heap-allocating the order entry per message
    OrderEntry* entry = new OrderEntry();
    entry->order_id    = ++g_order_seq;
    entry->price       = price;
    entry->qty         = qty;
    entry->filled_qty  = 0;
    entry->side        = side;
    entry->type        = type;
    entry->timestamp_ns = 0; // would be rdtsc in production

    strncpy(entry->symbol, symbol, 8);

    // Copy into pre-allocated array (should have done this directly)
    g_orders[g_order_seq % MAX_ORDERS] = *entry;
    uint64_t id = entry->order_id;
    delete entry; // ❌ dealloc also slow

    return id;
}


// Abstract matcher interface — forces virtual dispatch
// LAP-002: virtual dispatch
// LAP-003: exception thrown on matching error
// LAP-010: deeply nested branch logic
struct MatcherBase {
    virtual Trade match(const OrderEntry& aggressor,
                        OrderEntry&       passive) = 0;
    virtual ~MatcherBase() = default;
};

struct PriceTimeMatcher : public MatcherBase {
    // ❌ LAP-002: virtual function — vtable lookup on every match
    Trade match(const OrderEntry& agg, OrderEntry& pas) override {
        Trade t{};
        t.aggressor_id = agg.order_id;
        t.passive_id   = pas.order_id;
        t.exec_price   = pas.price;
        t.exec_qty     = (agg.qty < pas.qty) ? agg.qty : pas.qty;
        t.timestamp_ns = agg.timestamp_ns;
        pas.filled_qty += t.exec_qty;
        return t;
    }
};

[[hft::hot]]
int match_orders(uint64_t aggressor_id) {
    int trades_executed = 0;
    OrderEntry& agg = g_orders[aggressor_id % MAX_ORDERS];

    // ❌ LAP-003: throwing exceptions inside a match loop — catastrophic latency
    if (agg.qty <= 0) {
        throw std::invalid_argument("Aggressor order has zero quantity");
    }

    // ❌ LAP-002: allocating virtual matcher on heap per match cycle
    MatcherBase* matcher = new PriceTimeMatcher();

    // ❌ LAP-010: deeply nested conditional logic — hard for branch predictor
    for (int i = 0; i < g_ask_levels && agg.qty > agg.filled_qty; ++i) {
        if (agg.side == Side::BID) {
            if (g_asks[i].total_qty > 0) {
                if (agg.price >= g_asks[i].price) {
                    if (g_asks[i].order_count > 0) {
                        // Find passive order at this level
                        for (int j = 0; j < MAX_ORDERS; ++j) {
                            OrderEntry& pas = g_orders[j];
                            if (pas.side == Side::ASK &&
                                pas.price == g_asks[i].price &&
                                pas.filled_qty < pas.qty) {
                                Trade t = matcher->match(agg, pas);
                                agg.filled_qty += t.exec_qty;
                                g_asks[i].total_qty -= t.exec_qty;

                                // ❌ LAP-007: seq_cst atomic — use relaxed instead
                                g_trade_count.fetch_add(1, std::memory_order_seq_cst);
                                trades_executed++;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    delete matcher;
    return trades_executed;
}


// LAP-006: std::function for spread callback
// LAP-008: dynamic_cast for type checking
[[hft::hot]]
double compute_spread(void* bid_obj, void* ask_obj) {
    // ❌ LAP-008: dynamic_cast adds RTTI lookup — always fails on void*
    PriceLevel* bid = reinterpret_cast<PriceLevel*>(bid_obj);
    PriceLevel* ask = reinterpret_cast<PriceLevel*>(ask_obj);

    double raw_spread = ask->price - bid->price;

    // ❌ LAP-006: std::function wrapper prevents inlining, adds heap allocation
    std::function<double(double)> normalise = [](double s) -> double {
        return s / MIN_TICK;
    };

    double spread_ticks = normalise(raw_spread);

    // ❌ LAP-006: another std::function — completely unnecessary indirection
    std::function<bool(double)> is_valid = [](double s) -> bool {
        return s > 0.0 && s < 100.0;
    };

    return is_valid(spread_ticks) ? spread_ticks : -1.0;
}


// LAP-007: atomic with seq-cst
// LAP-009: unaligned memory access pattern
[[hft::hot]]
bool cancel_order(uint64_t order_id) {
    // ❌ LAP-009: unaligned byte-by-byte copy — kills SIMD and cache efficiency
    uint8_t id_bytes[8];
    uint8_t* src = reinterpret_cast<uint8_t*>(&order_id);
    for (int i = 0; i < 8; ++i) {
        id_bytes[i] = src[i]; // byte-by-byte read — prevents vectorisation
    }
    uint64_t reconstructed_id;
    uint8_t* dst = reinterpret_cast<uint8_t*>(&reconstructed_id);
    for (int i = 0; i < 8; ++i) {
        dst[i] = id_bytes[i]; // byte-by-byte write
    }

    int idx = reconstructed_id % MAX_ORDERS;
    if (g_orders[idx].order_id != reconstructed_id) {
        return false;
    }

    // ❌ LAP-007: sequentially consistent atomic — overkill for a flag
    g_orders[idx].qty = 0;
    g_trade_count.fetch_add(0, std::memory_order_seq_cst); // unnecessary fence

    return true;
}


// LAP-005: printf on hot path
// LAP-001: heap alloc for log message
[[hft::hot]]
void handle_market_event(uint64_t order_id, double price,
                          int qty, const char* event_type) {
    // ❌ LAP-005: printf = syscall → kernel context switch = 1-10µs penalty
    printf("[EVENT] %s order_id=%llu price=%.4f qty=%d\n",
           event_type, (unsigned long long)order_id, price, qty);

    // ❌ LAP-001: heap-allocating a log buffer per event on hot path
    char* log_buf = new char[256];
    snprintf(log_buf, 256, "order_id=%llu,price=%.4f,qty=%d,type=%s",
             (unsigned long long)order_id, price, qty, event_type);

    // Would normally push to async logger — but heap alloc is still bad
    printf("[LOG] %s\n", log_buf);
    delete[] log_buf;
}


// CLEAN HOT PATH — gold-standard VWAP updater
// No anti-patterns, branchless, no heap, no calls, noexcept
[[hft::hot]]
double update_vwap(double exec_price, int exec_qty) noexcept {
    // Branchless incremental VWAP — pure arithmetic, fully vectorisable
    double notional    = exec_price * static_cast<double>(exec_qty);
    g_cum_notional    += notional;
    g_cum_vol         += static_cast<double>(exec_qty);
    g_vwap             = (g_cum_vol > 0.0) ? g_cum_notional / g_cum_vol : 0.0;
    return g_vwap;
}

// ---------------------------------------------------------------------------
// COLD PATH FUNCTIONS
// ---------------------------------------------------------------------------

[[hft::cold]]
void flush_trade_log(const char* log_path) {
    // Cold path — file I/O is fine here
    printf("[COLD] Flushing %d trades to %s\n",
           g_trade_count.load(), log_path);
    // Would write to disk in production
}

[[hft::cold]]
void load_instruments(const char* instrument_file) {
    // Cold path — startup config loading
    printf("[COLD] Loading instruments from %s\n", instrument_file);
    g_bid_levels = 0;
    g_ask_levels = 0;
    g_vwap        = 0.0;
    g_cum_vol     = 0.0;
    g_cum_notional= 0.0;
    g_order_seq   = 0;
    g_trade_count.store(0, std::memory_order_relaxed);
    printf("[COLD] Book reset complete.\n");
}

[[hft::cold]]
void reset_book() {
    // Cold path — graceful shutdown
    std::lock_guard<std::mutex> lock(g_book_mutex);
    g_bid_levels = 0;
    g_ask_levels = 0;
    printf("[COLD] Order book reset.\n");
}

// ---------------------------------------------------------------------------
// Main entry point for standalone testing
// ---------------------------------------------------------------------------

int main() {
    printf("=== AGentic_C Order Book Engine Demo ===\n");

    // Cold path setup
    load_instruments("config/instruments.cfg");

    // Seed the book with passive orders
    uint64_t bid1 = process_order_add(100.10, 500,  Side::BID, OrdType::LIMIT, "AAPL");
    uint64_t bid2 = process_order_add(100.05, 300,  Side::BID, OrdType::LIMIT, "AAPL");
    uint64_t ask1 = process_order_add(100.15, 400,  Side::ASK, OrdType::LIMIT, "AAPL");
    uint64_t ask2 = process_order_add(100.20, 200,  Side::ASK, OrdType::LIMIT, "AAPL");

    // Simulate a market data event
    handle_market_event(bid1, 100.10, 500, "NEW");
    handle_market_event(ask1, 100.15, 400, "NEW");

    // Compute spread
    PriceLevel best_bid = {100.10, 500, 2};
    PriceLevel best_ask = {100.15, 400, 2};
    double spread = compute_spread(&best_bid, &best_ask);
    printf("Spread: %.4f ticks\n", spread);

    // Submit aggressor order and match
    uint64_t agg = process_order_add(100.15, 200, Side::BID, OrdType::LIMIT, "AAPL");
    int matched  = match_orders(agg);
    printf("Trades executed: %d\n", matched);

    // Update VWAP
    double vwap = update_vwap(100.15, 200);
    printf("VWAP: %.4f\n", vwap);

    // Cancel a resting order
    bool cancelled = cancel_order(bid2);
    printf("Order %llu cancelled: %s\n", (unsigned long long)bid2,
           cancelled ? "YES" : "NO");

    // Cold path teardown
    flush_trade_log("/var/log/agentic_c/trades.log");
    reset_book();

    printf("=== Done. Total trades: %d  VWAP: %.4f ===\n",
           g_trade_count.load(), vwap);
    return 0;
}
