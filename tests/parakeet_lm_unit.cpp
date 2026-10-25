// parakeet_lm_unit.cpp - unit test for the Parakeet n-gram language-model
// fusion (src/arch/parakeet/lm.h): ARPA loading, backoff, and which pieces
// a fused step scores.

#include "arch/parakeet/lm.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using transcribe::parakeet::apply_lm;
using transcribe::parakeet::LmContext;
using transcribe::parakeet::LmFusion;
using transcribe::parakeet::NgramLm;

constexpr float kLn10 = 2.302585093f;

bool near(float a, float b) {
    return std::fabs(a - b) < 1e-4f;
}

// Pieces 1..3, a bigram model: after 1, 2 is likely; 3 is only a unigram.
const char * kArpa =
    "\\data\\\n"
    "ngram 1=4\n"
    "ngram 2=3\n"
    "\n"
    "\\1-grams:\n"
    "-99\t<s>\t-0.30103\n"
    "-0.5\t1\t-0.2\n"
    "-0.7\t2\t-0.1\n"
    "-1.5\t3\n"
    "\n"
    "\\2-grams:\n"
    "-0.4\t<s> 1\n"
    "-0.05\t1 2\n"
    "-0.9\t2 1\n"
    "\n"
    "\\end\\\n";

std::string write_temp(const char * text) {
    std::string path =
        std::string(std::getenv("TMPDIR") != nullptr ? std::getenv("TMPDIR") : "/tmp") + "/parakeet_lm_unit.arpa";
    FILE * f = std::fopen(path.c_str(), "wb");
    std::fputs(text, f);
    std::fclose(f);
    return path;
}

void test_backoff() {
    NgramLm lm;
    CHECK(lm.load(write_temp(kArpa)));
    CHECK(lm.order() == 2);
    const int32_t start[] = { NgramLm::kStart };
    const int32_t one[]   = { 1 };
    const int32_t two[]   = { 2 };
    // Listed bigrams.
    CHECK(near(lm.log_prob(start, 1, 1), -0.4f * kLn10));
    CHECK(near(lm.log_prob(one, 1, 2), -0.05f * kLn10));
    // Unlisted: the context's backoff plus the unigram.
    CHECK(near(lm.log_prob(one, 1, 3), (-0.2f - 1.5f) * kLn10));
    CHECK(near(lm.log_prob(two, 1, 2), (-0.1f - 0.7f) * kLn10));
    // A piece the model never saw: below its rarest unigram.
    CHECK(lm.log_prob(one, 1, 7) < -1.5f * kLn10);
    CHECK(!NgramLm().load("/nonexistent/lm.arpa"));
}

void test_fusion() {
    NgramLm lm;
    CHECK(lm.load(write_temp(kArpa)));
    LmContext ctx;
    ctx.push(1);
    const LmFusion f{ &lm, 1.0f, &ctx };
    CHECK(f.active());
    // Blank (0) untouched; 1 and 2 tie acoustically, the model prefers 2
    // after 1; 3 is far below the best and pushed out of reach.
    std::vector<float> logits = { 5.0f, 2.0f, 2.0f, -30.0f };
    std::vector<float> fused  = logits;
    apply_lm(f, fused.data(), 4, /*blank_id=*/0);
    CHECK(fused[0] == 5.0f);
    CHECK(fused[2] > fused[1]);
    CHECK(near(fused[2], 2.0f - 0.05f * kLn10));
    CHECK(near(fused[1], 2.0f - (0.2f + 0.5f) * kLn10));
    CHECK(std::isinf(fused[3]) && fused[3] < 0);
    // An empty context is <s>.
    LmContext          fresh;
    const LmFusion     g{ &lm, 1.0f, &fresh };
    std::vector<float> again = { 0.0f, 0.0f, 0.0f, 0.0f };
    apply_lm(g, again.data(), 4, 0);
    CHECK(near(again[1], -0.4f * kLn10));
    CHECK(near(again[3], (-0.30103f - 1.5f) * kLn10));
    // The context keeps the newest pieces.
    LmContext full;
    for (int i = 0; i < LmContext::kMax + 3; ++i) {
        full.push(i);
    }
    CHECK(full.n == LmContext::kMax && full.ids[LmContext::kMax - 1] == LmContext::kMax + 2 && full.ids[0] == 3);
}

}  // namespace

int main() {
    test_backoff();
    test_fusion();
    if (g_failures > 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("parakeet_lm_unit: ok\n");
    return 0;
}
