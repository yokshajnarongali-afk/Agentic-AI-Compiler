/**
 * AGentic_C  —  Enterprise Multi-Strategy HFT Engine
 * ===================================================
 * Production-grade latency-arbitrage system featuring:
 *   · VWAP / Mean-Reversion / Order-Book-Imbalance signals
 *   · Lock-free SPSC ring buffer (LAP-004 target)
 *   · Cache-line-padded atomic counters (LAP-009 target)
 *   · Deliberate heap allocations in hot paths (LAP-001 target)
 *   · Sequential std::atomic (LAP-007 target)
 *   · Branch-heavy risk logic (LAP-010 target)
 *   · Virtual dispatch in hot path (LAP-002 target)
 *
 * Annotations: [[hft::hot]] / [[hft::cold]]
 * Build: clang++ -std=c++17 -O0 enterprise_hft_engine.cpp
 */

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
//  GLOBAL CONSTANTS
// ============================================================
static constexpr std::size_t CACHE_LINE   = 64;
static constexpr std::size_t RING_SZ      = 65536;   // must be power of 2
static constexpr std::size_t BOOK_DEPTH   = 10;
static constexpr double      TICK_SIZE    = 0.01;
static constexpr int         MAX_POSITION = 5000;
static constexpr double      MAX_NOTIONAL = 2'000'000.0;

// ============================================================
//  TIMESTAMP UTILITY
// ============================================================
[[hft::hot]]
inline uint64_t now_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

// ============================================================
//  MARKET DATA STRUCTURES
// ============================================================

// LAP-009: packed struct forces unaligned reads on hot path
#pragma pack(push, 1)
struct Tick {
    char     venue;         // 1
    double   bid;           // 8  (unaligned at offset 1)
    double   ask;           // 8
    int32_t  bid_qty;       // 4
    int32_t  ask_qty;       // 4
    uint64_t recv_ns;       // 8
    uint32_t seq;           // 4
};
#pragma pack(pop)

struct Level {
    double  price  = 0.0;
    int32_t qty    = 0;
};

struct BookSnapshot {
    std::array<Level, BOOK_DEPTH> bids;
    std::array<Level, BOOK_DEPTH> asks;
    uint64_t ts_ns = 0;
};

// ============================================================
//  LOCK-FREE SPSC RING BUFFER  (LAP-004 target)
// ============================================================
template<typename T, std::size_t N>
class SpscRing {
    static_assert((N & (N-1)) == 0, "N must be power of 2");

    alignas(CACHE_LINE) std::atomic<std::size_t> head_{0};
    alignas(CACHE_LINE) std::atomic<std::size_t> tail_{0};
    std::array<T, N> buf_;

public:
    [[hft::hot]]
    bool push(const T& v) {
        std::size_t t = tail_.load(std::memory_order_relaxed);
        std::size_t nt = (t + 1) & (N - 1);
        if (nt == head_.load(std::memory_order_acquire)) return false;
        buf_[t] = v;
        tail_.store(nt, std::memory_order_release);
        return true;
    }

    [[hft::hot]]
    bool pop(T& v) {
        std::size_t h = head_.load(std::memory_order_relaxed);
        if (h == tail_.load(std::memory_order_acquire)) return false;
        v = buf_[h];
        head_.store((h + 1) & (N - 1), std::memory_order_release);
        return true;
    }
};

// ============================================================
//  CACHE-ALIGNED METRICS (LAP-009 fix target)
// ============================================================
struct alignas(CACHE_LINE) PerfCounters {
    std::atomic<uint64_t> ticks_rcvd   {0};
    char _p1[CACHE_LINE - sizeof(uint64_t)];

    std::atomic<uint64_t> orders_sent  {0};
    char _p2[CACHE_LINE - sizeof(uint64_t)];

    std::atomic<uint64_t> fills_rcvd   {0};
    char _p3[CACHE_LINE - sizeof(uint64_t)];

    std::atomic<uint64_t> rejects      {0};
};

// ============================================================
//  VWAP CALCULATOR
// ============================================================
class VwapCalc {
    double   sum_pv_  = 0.0;
    double   sum_v_   = 0.0;
    int      window_  = 0;
    static constexpr int WINDOW = 500;

public:
    [[hft::hot]]
    double update(double price, int32_t vol) {
        // LAP-001: new inside hot path (intentional anti-pattern)
        double* tmp = new double[4];
        tmp[0] = price;
        tmp[1] = static_cast<double>(vol);
        tmp[2] = tmp[0] * tmp[1];
        tmp[3] = sum_pv_ + tmp[2];
        sum_pv_ = tmp[3];
        sum_v_ += tmp[1];
        delete[] tmp;

        if (++window_ >= WINDOW) {
            sum_pv_ *= 0.98;   // exponential decay
            sum_v_  *= 0.98;
            window_  = 0;
        }
        return (sum_v_ > 0.0) ? (sum_pv_ / sum_v_) : price;
    }

    [[hft::hot]]
    double get() const { return (sum_v_ > 0.0) ? (sum_pv_ / sum_v_) : 0.0; }
};

// ============================================================
//  MEAN-REVERSION SIGNAL
// ============================================================
class MeanRevSignal {
    double mu_   = 0.0;
    double var_  = 1.0;
    double alpha_;

public:
    explicit MeanRevSignal(double alpha = 0.002) : alpha_(alpha) {}

    [[hft::hot]]
    double update(double price) {
        double delta = price - mu_;
        mu_  += alpha_ * delta;
        var_  = (1.0 - alpha_) * (var_ + alpha_ * delta * delta);
        double sd = std::sqrt(var_);
        return (sd < 1e-8) ? 0.0 : (price - mu_) / sd;
    }
};

// ============================================================
//  ORDER-BOOK IMBALANCE DETECTOR
// ============================================================
class ObiDetector {
    static constexpr int HIST = 64;
    double history_[HIST] = {};
    int    idx_ = 0;

public:
    [[hft::hot]]
    double update(const BookSnapshot& snap) {
        double bid_sz = 0.0, ask_sz = 0.0;

        // SIMD-vectorisable loop (LAP-010 — loop body is branch-free)
        for (std::size_t i = 0; i < BOOK_DEPTH; ++i) {
            double w = 1.0 / static_cast<double>(i + 1);
            bid_sz += snap.bids[i].qty * w;
            ask_sz += snap.asks[i].qty * w;
        }
        double total = bid_sz + ask_sz;
        double imb   = (total > 0.0) ? (bid_sz - ask_sz) / total : 0.0;
        history_[idx_ & (HIST - 1)] = imb;
        ++idx_;

        // compute rolling mean over history
        double sum = 0.0;
        for (int i = 0; i < HIST; ++i) sum += history_[i];
        return sum / HIST;
    }
};

// ============================================================
//  ALPHA COMBINER — blends three signals into one score
// ============================================================
struct AlphaWeights { double vwap, mr, obi; };

[[hft::hot]]
double combine_alpha(double vwap_dev, double mr_z, double obi_mean,
                     const AlphaWeights& w) {
    return w.vwap * vwap_dev + w.mr * mr_z + w.obi * obi_mean;
}

// ============================================================
//  ORDER TYPES & EXECUTION REQUEST
// ============================================================
enum class Side   : uint8_t { BUY = 0, SELL = 1 };
enum class OrdType: uint8_t { LIMIT, MARKET, IOC, CANCEL };

struct ExecRequest {
    uint64_t  req_id;
    OrdType   type;
    Side      side;
    double    price;
    int32_t   qty;
    char      symbol[8];
    uint64_t  ts_ns;
};

// ============================================================
//  RISK MANAGER
// ============================================================
class RiskManager {
    double  net_pos_   = 0.0;
    double  gross_exp_ = 0.0;
    int32_t ord_count_ = 0;
    static constexpr int MAX_ORDS_PER_SEC = 10000;

public:
    [[hft::hot]]
    bool check(const ExecRequest& req) {
        double notional = req.price * req.qty;

        // LAP-010: nested branch-heavy logic on hot path
        if (notional > MAX_NOTIONAL) return false;
        if (gross_exp_ + notional > MAX_NOTIONAL * 3) return false;
        if (std::abs(net_pos_) > MAX_POSITION) return false;

        if (req.side == Side::BUY) {
            if (net_pos_ + req.qty > MAX_POSITION) return false;
        } else {
            if (net_pos_ - req.qty < -MAX_POSITION) return false;
        }

        if (++ord_count_ > MAX_ORDS_PER_SEC) {
            ord_count_ = 0;
            return false;
        }

        // Commit
        gross_exp_ += notional;
        net_pos_   += (req.side == Side::BUY) ? req.qty : -req.qty;
        return true;
    }

    [[hft::hot]]
    void on_fill(Side s, int32_t qty, double price) {
        double n = price * qty;
        gross_exp_ = std::max(0.0, gross_exp_ - n);
        net_pos_  += (s == Side::BUY) ? qty : -qty;
    }

    [[hft::cold]]
    void print_state() const {
        printf("[Risk] net_pos=%.0f  gross_exp=%.2f\n", net_pos_, gross_exp_);
    }
};

// ============================================================
//  FIX-LITE MESSAGE BUILDER  (cold path)
// ============================================================
class FixBuilder {
    char buf_[512] = {};

public:
    [[hft::cold]]
    const char* build_new_order(uint64_t id, Side side,
                                double price, int32_t qty,
                                const char* sym) {
        std::snprintf(buf_, sizeof(buf_),
            "8=FIX.4.2\x01"
            "35=D\x01"
            "11=%llu\x01"
            "55=%s\x01"
            "54=%d\x01"
            "44=%.4f\x01"
            "38=%d\x01"
            "10=000\x01",
            static_cast<unsigned long long>(id),
            sym,
            static_cast<int>(side),
            price,
            static_cast<int>(qty));
        return buf_;
    }

    [[hft::cold]]
    const char* build_cancel(uint64_t orig_id) {
        std::snprintf(buf_, sizeof(buf_),
            "8=FIX.4.2\x01" "35=F\x01" "41=%llu\x01" "10=000\x01",
            static_cast<unsigned long long>(orig_id));
        return buf_;
    }
};

// ============================================================
//  EXECUTION ENGINE
// ============================================================
class ExecEngine {
    SpscRing<ExecRequest, RING_SZ> queue_;
    RiskManager&                   risk_;
    PerfCounters&                  perf_;
    FixBuilder                     fix_;
    uint64_t                       next_id_ = 1;

public:
    ExecEngine(RiskManager& r, PerfCounters& p) : risk_(r), perf_(p) {}

    [[hft::hot]]
    bool submit(Side side, double price, int32_t qty, const char* sym) {
        ExecRequest req;
        req.req_id = next_id_++;
        req.type   = OrdType::LIMIT;
        req.side   = side;
        req.price  = price;
        req.qty    = qty;
        req.ts_ns  = now_ns();
        std::strncpy(req.symbol, sym, sizeof(req.symbol) - 1);
        req.symbol[sizeof(req.symbol) - 1] = '\0';

        if (!risk_.check(req)) {
            // LAP-007: seq_cst on hot path
            perf_.rejects.fetch_add(1, std::memory_order_seq_cst);
            return false;
        }

        bool ok = queue_.push(req);
        if (ok) {
            perf_.orders_sent.fetch_add(1, std::memory_order_seq_cst);
        }
        return ok;
    }

    [[hft::hot]]
    bool drain_one() {
        ExecRequest req;
        if (!queue_.pop(req)) return false;
        risk_.on_fill(req.side, req.qty, req.price);
        perf_.fills_rcvd.fetch_add(1, std::memory_order_seq_cst);
        return true;
    }

    [[hft::cold]]
    void send_via_fix(const ExecRequest& req) {
        const char* msg = fix_.build_new_order(
            req.req_id, req.side, req.price, req.qty, req.symbol);
        // In production: write(sock_fd, msg, strlen(msg))
        (void)msg;
    }
};

// ============================================================
//  STRATEGY CORE
// ============================================================
class MultiStrategyCore {
    VwapCalc       vwap_;
    MeanRevSignal  mr_;
    ObiDetector    obi_;
    AlphaWeights   weights_ {0.4, 0.35, 0.25};

public:
    struct Decision {
        bool   act   = false;
        Side   side  = Side::BUY;
        double price = 0.0;
        int    qty   = 0;
    };

    [[hft::hot]]
    Decision on_tick(const Tick& t, const BookSnapshot& snap) {
        double vwap    = vwap_.update(t.bid, t.bid_qty);
        double mr_z    = mr_.update((t.bid + t.ask) * 0.5);
        double obi     = obi_.update(snap);
        double vwap_dev= (vwap > 0.0) ? (t.bid - vwap) / vwap : 0.0;

        double alpha = combine_alpha(vwap_dev, mr_z, obi, weights_);

        Decision d;
        if (alpha > 0.30) {
            d.act   = true;
            d.side  = Side::BUY;
            d.price = t.ask;
            d.qty   = static_cast<int>(50.0 * alpha);
        } else if (alpha < -0.30) {
            d.act   = true;
            d.side  = Side::SELL;
            d.price = t.bid;
            d.qty   = static_cast<int>(-50.0 * alpha);
        }
        d.qty = std::max(1, std::min(d.qty, 500));
        return d;
    }
};

// ============================================================
//  TOP-LEVEL TRADING SYSTEM
// ============================================================
class TradingSystem {
    PerfCounters      perf_;
    RiskManager       risk_;
    ExecEngine        exec_;
    MultiStrategyCore strategy_;

public:
    [[hft::cold]]
    TradingSystem() : exec_(risk_, perf_) {
        std::cout << "[System] TradingSystem initialised.\n";
    }

    [[hft::hot]]
    void on_market_data(const Tick& t, const BookSnapshot& snap) {
        perf_.ticks_rcvd.fetch_add(1, std::memory_order_relaxed);

        auto dec = strategy_.on_tick(t, snap);
        if (!dec.act) return;

        exec_.submit(dec.side, dec.price, dec.qty, "AAPL");
        exec_.drain_one();
    }

    [[hft::cold]]
    void print_stats() const {
        printf("[Stats] ticks=%-8llu  orders=%-8llu  fills=%-8llu  rejects=%llu\n",
               static_cast<unsigned long long>(perf_.ticks_rcvd.load()),
               static_cast<unsigned long long>(perf_.orders_sent.load()),
               static_cast<unsigned long long>(perf_.fills_rcvd.load()),
               static_cast<unsigned long long>(perf_.rejects.load()));
        risk_.print_state();
    }
};

// ============================================================
//  SYNTHETIC MARKET DATA GENERATOR  (cold — simulation only)
// ============================================================
[[hft::cold]]
static void fill_book(BookSnapshot& snap, double mid, double spread) {
    for (std::size_t i = 0; i < BOOK_DEPTH; ++i) {
        snap.bids[i] = { mid - spread * (i + 1), 100 + (int)(i * 25) };
        snap.asks[i] = { mid + spread * (i + 1), 100 + (int)(i * 25) };
    }
    snap.ts_ns = now_ns();
}

[[hft::cold]]
static Tick make_tick(double mid, double spread, uint32_t seq) {
    Tick t;
    t.venue   = 'N';
    t.bid     = mid - spread * 0.5;
    t.ask     = mid + spread * 0.5;
    t.bid_qty = 500;
    t.ask_qty = 500;
    t.recv_ns = now_ns();
    t.seq     = seq;
    return t;
}

// ============================================================
//  SIMULATION HARNESS
// ============================================================
[[hft::cold]]
static void run_simulation(int n_ticks) {
    TradingSystem sys;

    double      mid    = 150.0;
    double      spread = 0.02;
    BookSnapshot snap;
    fill_book(snap, mid, spread);

    printf("[Sim] Running %d ticks...\n", n_ticks);
    uint64_t t0 = now_ns();

    for (int i = 0; i < n_ticks; ++i) {
        // Synthetic Brownian price walk
        double move = ((i % 7) - 3) * TICK_SIZE * 0.5;
        mid += move;
        if (mid < 1.0) mid = 1.0;

        Tick t = make_tick(mid, spread, static_cast<uint32_t>(i));
        fill_book(snap, mid, spread);

        sys.on_market_data(t, snap);
    }

    uint64_t elapsed = now_ns() - t0;
    sys.print_stats();
    printf("[Sim] Done. Wall time: %llu ms  (%.1f ns/tick)\n",
           static_cast<unsigned long long>(elapsed / 1'000'000),
           static_cast<double>(elapsed) / n_ticks);
}



// ============================================================
//  KALMAN FILTER — price / velocity tracker
// ============================================================
class KalmanFilter {
    double x_ = 0.0;   // state: price estimate
    double v_ = 0.0;   // state: velocity estimate
    double p_  = 1.0;  // error covariance
    double q_  = 1e-4; // process noise
    double r_  = 0.01; // measurement noise

public:
    [[hft::hot]]
    double update(double meas) {
        // Predict
        double x_pred = x_ + v_;
        double p_pred = p_ + q_;

        // Update
        double k  = p_pred / (p_pred + r_);
        double inn = meas - x_pred;
        x_ = x_pred + k * inn;
        v_ = v_ + 0.1 * inn;
        p_ = (1.0 - k) * p_pred;
        return x_;
    }

    [[hft::hot]] double velocity() const { return v_; }
    [[hft::hot]] double estimate() const { return x_; }
};

// ============================================================
//  TWAP SCHEDULER — breaks large orders into slices
// ============================================================
class TwapScheduler {
    int    total_qty_   = 0;
    int    slices_      = 0;
    int    slice_sz_    = 0;
    int    done_slices_ = 0;
    uint64_t slice_interval_ns_ = 0;
    uint64_t next_slice_ns_     = 0;
    Side   side_ = Side::BUY;
    double limit_price_ = 0.0;

public:
    [[hft::cold]]
    void start(int qty, int n_slices, uint64_t duration_ns, Side s, double px) {
        total_qty_        = qty;
        slices_           = n_slices;
        slice_sz_         = qty / n_slices;
        done_slices_      = 0;
        slice_interval_ns_= duration_ns / n_slices;
        next_slice_ns_    = now_ns() + slice_interval_ns_;
        side_             = s;
        limit_price_      = px;
    }

    [[hft::hot]]
    bool get_slice(Side& s, double& px, int& qty) {
        if (done_slices_ >= slices_) return false;
        if (now_ns() < next_slice_ns_)  return false;

        s   = side_;
        px  = limit_price_;
        qty = (done_slices_ == slices_ - 1)
              ? (total_qty_ - done_slices_ * slice_sz_)  // last slice: remainder
              : slice_sz_;

        ++done_slices_;
        next_slice_ns_ += slice_interval_ns_;
        return true;
    }

    [[hft::cold]]
    bool is_complete() const { return done_slices_ >= slices_; }
};

// ============================================================
//  CORRELATED PAIRS MONITOR
// ============================================================
class PairsMonitor {
    static constexpr int  N = 256;
    double  a_[N] = {};
    double  b_[N] = {};
    int     idx_  = 0;

    [[hft::hot]]
    double mean(const double* arr, int n) const {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += arr[i];
        return s / n;
    }

public:
    [[hft::hot]]
    void push(double price_a, double price_b) {
        a_[idx_ & (N-1)] = price_a;
        b_[idx_ & (N-1)] = price_b;
        ++idx_;
    }

    [[hft::hot]]
    double correlation() const {
        int n = std::min(idx_, N);
        if (n < 2) return 0.0;
        double ma = mean(a_, n);
        double mb = mean(b_, n);
        double cov = 0.0, va = 0.0, vb = 0.0;
        for (int i = 0; i < n; ++i) {
            double da = a_[i] - ma;
            double db = b_[i] - mb;
            cov += da * db;
            va  += da * da;
            vb  += db * db;
        }
        double denom = std::sqrt(va * vb);
        return (denom < 1e-12) ? 0.0 : cov / denom;
    }

    [[hft::hot]]
    double spread_zscore() const {
        int n = std::min(idx_, N);
        if (n < 2) return 0.0;
        double spreads[N];
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            spreads[i] = a_[i] - b_[i];
            sum += spreads[i];
        }
        double mu  = sum / n;
        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = spreads[i] - mu;
            var += d * d;
        }
        double sd = std::sqrt(var / n);
        int last = (idx_ - 1) & (N - 1);
        return (sd < 1e-8) ? 0.0 : ((a_[last] - b_[last]) - mu) / sd;
    }
};

// ============================================================
//  MARKET SESSION MANAGER  (state machine)
// ============================================================
class SessionManager {
public:
    enum class Phase { PRE_OPEN, OPEN, AUCTION, CLOSE, HALTED };

private:
    Phase    phase_     = Phase::PRE_OPEN;
    uint64_t phase_start_ns_ = 0;
    int      halt_count_ = 0;

public:
    [[hft::cold]]
    void transition(Phase next) {
        printf("[Session] %s → %s\n", phase_name(phase_), phase_name(next));
        phase_       = next;
        phase_start_ns_ = now_ns();
    }

    [[hft::hot]]
    bool trading_allowed() const {
        return phase_ == Phase::OPEN;
    }

    [[hft::hot]]
    bool in_auction() const {
        return phase_ == Phase::AUCTION;
    }

    [[hft::cold]]
    void on_halt() {
        ++halt_count_;
        transition(Phase::HALTED);
    }

    [[hft::cold]]
    static const char* phase_name(Phase p) {
        switch (p) {
            case Phase::PRE_OPEN: return "PRE_OPEN";
            case Phase::OPEN:     return "OPEN";
            case Phase::AUCTION:  return "AUCTION";
            case Phase::CLOSE:    return "CLOSE";
            case Phase::HALTED:   return "HALTED";
            default:              return "UNKNOWN";
        }
    }

    [[hft::cold]]
    int halt_count() const { return halt_count_; }
};

// ============================================================
//  LATENCY HISTOGRAM  (nanosecond precision)
// ============================================================
class LatencyHistogram {
    static constexpr int BUCKETS = 128;
    // Each bucket covers 100 ns; bucket[i] = [i*100, (i+1)*100) ns
    uint64_t counts_[BUCKETS] = {};
    uint64_t overflow_        = 0;
    uint64_t total_           = 0;
    uint64_t sum_             = 0;

public:
    [[hft::hot]]
    void record(uint64_t latency_ns) {
        ++total_;
        sum_ += latency_ns;
        int bucket = static_cast<int>(latency_ns / 100);
        if (bucket < BUCKETS) ++counts_[bucket];
        else                   ++overflow_;
    }

    [[hft::cold]]
    void print() const {
        if (total_ == 0) { printf("[Hist] No samples.\n"); return; }
        printf("[Hist] samples=%llu  mean=%.0f ns\n",
               static_cast<unsigned long long>(total_),
               static_cast<double>(sum_) / total_);
        // Print non-zero buckets
        for (int i = 0; i < BUCKETS; ++i) {
            if (counts_[i] > 0) {
                printf("  [%4d-%4d ns]: %llu\n", i*100, (i+1)*100,
                       static_cast<unsigned long long>(counts_[i]));
            }
        }
        if (overflow_ > 0)
            printf("  [>12800 ns  ]: %llu\n",
                   static_cast<unsigned long long>(overflow_));
    }

    [[hft::hot]]
    uint64_t mean_ns() const {
        return (total_ > 0) ? (sum_ / total_) : 0;
    }
};

// ============================================================
//  EXTENDED SIMULATION HARNESS (uses all new components)
// ============================================================
[[hft::cold]]
static void run_extended_simulation(int n_ticks) {
    TradingSystem    sys;
    KalmanFilter     kf;
    TwapScheduler    twap;
    PairsMonitor     pairs;
    SessionManager   session;
    LatencyHistogram hist;

    session.transition(SessionManager::Phase::OPEN);

    // Start a 10,000-share TWAP over 60 seconds
    twap.start(10000, 20, 60'000'000'000ULL, Side::BUY, 150.50);

    double mid_a = 150.0, mid_b = 148.5;

    printf("[ExtSim] Running %d ticks with Kalman + TWAP + Pairs...\n", n_ticks);

    for (int i = 0; i < n_ticks; ++i) {
        uint64_t t0 = now_ns();

        double move_a = ((i % 11) - 5) * TICK_SIZE * 0.3;
        double move_b = ((i % 13) - 6) * TICK_SIZE * 0.3;
        mid_a += move_a;
        mid_b += move_b;
        if (mid_a < 1.0) mid_a = 1.0;
        if (mid_b < 1.0) mid_b = 1.0;

        // Kalman update
        double filtered_a = kf.update(mid_a);

        // Pairs update
        pairs.push(mid_a, mid_b);

        // Build synthetic tick + book
        Tick t = make_tick(filtered_a, 0.02, static_cast<uint32_t>(i));
        BookSnapshot snap;
        fill_book(snap, filtered_a, 0.02);

        if (session.trading_allowed()) {
            sys.on_market_data(t, snap);

            // TWAP slice
            Side s; double px; int qty;
            if (twap.get_slice(s, px, qty)) {
                // In production: submit directly to exchange
                (void)s; (void)px; (void)qty;
            }
        }

        // Record per-tick latency
        hist.record(now_ns() - t0);

        // Halt simulation at midpoint for stress test
        if (i == n_ticks / 2) {
            session.on_halt();
            session.transition(SessionManager::Phase::OPEN);
        }
    }

    sys.print_stats();
    hist.print();
    printf("[PairCorr] correlation=%.4f  spread_z=%.4f\n",
           pairs.correlation(), pairs.spread_zscore());
    printf("[Kalman]   estimate=%.4f  velocity=%.6f\n",
           kf.estimate(), kf.velocity());
}

// ============================================================
//  MAIN  (runs both basic + extended simulations)
// ============================================================
int main() {
    printf("=== AGentic_C Enterprise HFT Engine — Full Suite ===\n\n");
    printf("--- Phase 1: Basic Multi-Strategy Simulation ---\n");
    run_simulation(200'000);
    printf("\n--- Phase 2: Extended (Kalman + TWAP + Pairs + Histogram) ---\n");
    run_extended_simulation(200'000);
    printf("\n=== Done ===\n");
    return 0;
}
