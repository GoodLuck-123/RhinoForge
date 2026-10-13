// Standalone host coverage for exact expert selection and tie handling.
#include "rpu_lingbot_v2_moe_select_ref.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <set>
#include <cstring>
#include <string>

using lingbot_v2_moe_ref::router_select;

static int g_fail = 0, g_pass = 0;

static void check(bool cond, const char* what) {
    if (cond) { ++g_pass; }
    else { ++g_fail; std::printf("  FAIL: %s\n", what); }
}

// logit whose sigmoid is `p` — lets tests state intent in score space.
static float logit_of(float p) { return std::log(p / (1.0f - p)); }

// Every test asserts the count invariant + distinctness + weight-sum identity.
static void assert_wellformed(const char* name, const int32_t* idx, const float* w,
                              int k, int E, float scaling) {
    std::set<int32_t> uniq(idx, idx + k);
    check((int)uniq.size() == k, (std::string(name) + ": indices distinct").c_str());
    for (int j = 0; j < k; ++j)
        check(idx[j] >= 0 && idx[j] < E, (std::string(name) + ": index in range").c_str());
    float s = 0.0f;
    for (int j = 0; j < k; ++j) s += w[j];
    // Normalised then scaled => weights must sum to routed_scaling.
    check(std::fabs(s - scaling) < 1e-3f, (std::string(name) + ": weights sum to scaling").c_str());
    for (int j = 0; j < k; ++j)
        check(w[j] >= 0.0f && std::isfinite(w[j]), (std::string(name) + ": weight finite/nonneg").c_str());
}

int main() {
    const int E = 32, k = 4;
    const float SC = 4.0f;   // routed_scaling_factor
    int32_t idx[8]; float w[8];

    std::printf("=== LingBot2 MoE router-select — adversarial suite ===\n");
    std::printf("tie rule under test: LOWEST INDEX WINS (strict > ascending scan)\n\n");

    // ── (a) exact ties: all 32 scores exactly equal ──────────────────────────
    {
        std::vector<float> lg(E, logit_of(0.5f));
        int n = router_select(lg.data(), nullptr, E, k, SC, idx, w);
        std::printf("(a) all-32-equal          -> n=%d idx=[%d,%d,%d,%d]\n",
                    n, idx[0], idx[1], idx[2], idx[3]);
        check(n == k, "(a) exactly k selected");
        // Lowest-index-wins => the first k indices, in order.
        check(idx[0]==0 && idx[1]==1 && idx[2]==2 && idx[3]==3, "(a) lowest-index-wins => [0,1,2,3]");
        // All equal => equal weights => each == scaling/k.
        for (int j = 0; j < k; ++j)
            check(std::fabs(w[j] - SC/k) < 1e-5f, "(a) equal weights = scaling/k");
        assert_wellformed("(a)", idx, w, k, E, SC);
    }

    // ── (b) near-ties at 1 ulp ───────────────────────────────────────────────
    {
        std::vector<float> lg(E, -10.0f);
        float base = 0.6f;
        float up1  = std::nextafterf(base, 1.0f);          // +1 ulp
        float up2  = std::nextafterf(up1,  1.0f);          // +2 ulp
        lg[9]  = logit_of(up2);      // strictly largest
        lg[4]  = logit_of(up1);      // 1 ulp below
        lg[27] = logit_of(base);     // 2 ulp below
        lg[15] = logit_of(std::nextafterf(base, 0.0f));    // 3 ulp below -> 4th
        lg[2]  = logit_of(std::nextafterf(std::nextafterf(base,0.0f),0.0f)); // 5th, excluded
        int n = router_select(lg.data(), nullptr, E, k, SC, idx, w);
        std::printf("(b) 1-ulp ladder          -> n=%d idx=[%d,%d,%d,%d]\n",
                    n, idx[0], idx[1], idx[2], idx[3]);
        check(n == k, "(b) exactly k selected");
        check(idx[0]==9 && idx[1]==4 && idx[2]==27 && idx[3]==15,
              "(b) 1-ulp ordering resolved exactly in fp32");
        check(std::set<int32_t>(idx,idx+k).count(2) == 0, "(b) 5th (1 ulp below 4th) excluded");
        assert_wellformed("(b)", idx, w, k, E, SC);
    }

    // ── (c) 5-way tie exactly at the 4th position ────────────────────────────
    {
        std::vector<float> lg(E, -10.0f);
        lg[10] = logit_of(0.9f); lg[20] = logit_of(0.8f); lg[30] = logit_of(0.7f);
        for (int e : {2, 5, 9, 17, 25}) lg[e] = logit_of(0.6f);   // 5 tied for slot 4
        int n = router_select(lg.data(), nullptr, E, k, SC, idx, w);
        std::printf("(c) 5-way tie at 4th slot -> n=%d idx=[%d,%d,%d,%d]\n",
                    n, idx[0], idx[1], idx[2], idx[3]);
        check(n == k, "(c) exactly k selected (mask would have given 7)");
        check(idx[0]==10 && idx[1]==20 && idx[2]==30, "(c) three clear winners first");
        check(idx[3]==2, "(c) lowest index (2) wins the 5-way tie");
        assert_wellformed("(c)", idx, w, k, E, SC);
    }

    // ── (d) 8-way tie spanning the whole top-k boundary ──────────────────────
    {
        std::vector<float> lg(E, -10.0f);
        for (int e : {31, 1, 19, 7, 23, 3, 11, 29}) lg[e] = logit_of(0.55f);
        int n = router_select(lg.data(), nullptr, E, k, SC, idx, w);
        std::printf("(d) 8-way tie across top-k-> n=%d idx=[%d,%d,%d,%d]\n",
                    n, idx[0], idx[1], idx[2], idx[3]);
        check(n == k, "(d) exactly k selected");
        check(idx[0]==1 && idx[1]==3 && idx[2]==7 && idx[3]==11,
              "(d) four lowest tied indices win, ascending");
        assert_wellformed("(d)", idx, w, k, E, SC);
    }

    // ── (e) correction bias affects SELECTION ONLY, never the weights ────────
    {
        std::vector<float> lg(E, -10.0f);
        lg[0]=logit_of(0.90f); lg[1]=logit_of(0.80f);
        lg[2]=logit_of(0.70f); lg[3]=logit_of(0.60f); lg[4]=logit_of(0.50f);
        // Unbiased top-4 = {0,1,2,3}. Bias promotes 4 over 3.
        std::vector<float> bias(E, 0.0f); bias[4] = 0.5f;
        int32_t bi[8]; float bw[8];
        router_select(lg.data(), bias.data(), E, k, SC, bi, bw);
        std::printf("(e) bias promotes e=4     -> idx=[%d,%d,%d,%d]\n", bi[0],bi[1],bi[2],bi[3]);
        check(bi[0]==4, "(e) biased score selects e=4 first");
        check(std::set<int32_t>(bi,bi+k).count(3) == 0, "(e) e=3 displaced by bias");
        // The weight for e=4 must come from the UNBIASED score 0.50, NOT 1.00.
        // Selected unbiased scores: {0.50, 0.90, 0.80, 0.70}, sum 2.90.
        check(std::fabs(bw[0] - (0.50f/2.90f)*SC) < 1e-4f,
              "(e) weight gathered from UNBIASED score (bias excluded)");
        assert_wellformed("(e)", bi, bw, k, E, SC);
    }

    // ── (f) zero bias == null bias (the real checkpoint's measured case) ─────
    {
        std::vector<float> lg(E); for (int e=0;e<E;++e) lg[e] = 0.1f*(float)((e*7)%13) - 0.6f;
        std::vector<float> zero(E, 0.0f);
        int32_t ai[8], bi2[8]; float aw[8], bw2[8];
        router_select(lg.data(), nullptr,      E, k, SC, ai,  aw);
        router_select(lg.data(), zero.data(),  E, k, SC, bi2, bw2);
        check(std::memcmp(ai, bi2, k*sizeof(int32_t)) == 0, "(f) null bias == zero bias (indices)");
        check(std::memcmp(aw, bw2, k*sizeof(float))   == 0, "(f) null bias == zero bias (weights)");
        std::printf("(f) zero-bias equivalence -> idx=[%d,%d,%d,%d]\n", ai[0],ai[1],ai[2],ai[3]);
    }

    // ── (g) monotonicity: with bias==0, topk(logits) == topk(sigmoid(logits)) ─
    //     The directive's free conformance assert. sigmoid is strictly monotonic,
    //     so ranking by logit and by sigmoid must agree exactly in fp32.
    {
        int flips = 0;
        unsigned seed = 12345u;
        auto rnd = [&seed]() {   // deterministic LCG — no <random> variance
            seed = seed * 1103515245u + 12345u;
            return ((float)((seed >> 16) & 0x7FFF) / 32767.0f) * 4.0f - 2.0f;
        };
        for (int trial = 0; trial < 1800; ++trial) {
            std::vector<float> lg(E); for (int e=0;e<E;++e) lg[e] = rnd();
            int32_t si[8]; float sw[8];
            router_select(lg.data(), nullptr, E, k, SC, si, sw);
            // Rank by RAW LOGIT with the identical strict/lowest-index rule.
            std::vector<bool> tk(E,false); int li[8];
            for (int j=0;j<k;++j){ int b=-1;
                for(int e=0;e<E;++e){ if(tk[e]) continue; if(b<0||lg[e]>lg[b]) b=e; }
                tk[b]=true; li[j]=b; }
            for (int j=0;j<k;++j) if (li[j]!=si[j]) { ++flips; break; }
        }
        std::printf("(g) topk(logit)==topk(sigmoid), 1800 trials -> flips=%d\n", flips);
        check(flips == 0, "(g) sigmoid monotonicity holds in fp32 (0 flips)");
    }

    std::printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
