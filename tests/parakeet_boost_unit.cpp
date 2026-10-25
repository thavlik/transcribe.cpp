// parakeet_boost_unit.cpp - unit test for the Parakeet vocabulary boost
// (src/arch/parakeet/boost.h): trie matching over SentencePiece pieces and
// the greedy pick rules, on a synthetic vocabulary and hand-set logits.

#include "arch/parakeet/boost.h"
#include "transcribe-tokenizer.h"

#include <cstdio>
#include <cstdlib>
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

using transcribe::parakeet::PhraseBoost;

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// Piece ids of the synthetic vocabulary below.
enum Id : int {
    kUnk = 0,
    kLisW,      // "▁lis"
    kIn,        // "in"
    kOp,        // "op"
    kRil,       // "ril"
    kLW,        // "▁l"
    kIs,        // "is"
    kLikeW,     // "▁like"
    kHydroW,    // "▁hydro"
    kCh,        // "ch"
    kLor,       // "lor"
    kOth,       // "oth"
    kI,         // "i"
    kA,         // "a"
    kIa,        // "ia"
    kZ,         // "z"
    kIde,       // "ide"
    kTheW,      // "▁the"
    kAtrialW,   // "▁atrial"
    kFibW,      // "▁fib"
    kR,         // "r"
    kIllation,  // "illation"
    kMark,      // "▁"
    kLisCap,    // "Lis"
    kPeriod,    // "."
    kS,         // "s"
    kRils,      // "rils"
    kNTokens,
    kBlank = kNTokens,
    kNClasses,
};

transcribe::Tokenizer make_tokenizer() {
    const std::string        m      = "\xE2\x96\x81";  // ▁
    std::vector<std::string> tokens = {
        "<unk>",      m + "lis", "in",  "op",       "ril", m + "l", "is", m + "like", m + "hydro",
        "ch",         "lor",     "oth", "i",        "a",   "ia",    "z",  "ide",      m + "the",
        m + "atrial", m + "fib", "r",   "illation", m,     "Lis",   ".",  "s",        "rils",
    };
    transcribe::Tokenizer   tok;
    const transcribe_status st = tok.load_decode_only_raw_bytes(std::move(tokens));
    CHECK(st == TRANSCRIBE_OK);
    CHECK(tok.n_tokens() == kNTokens);
    return tok;
}

PhraseBoost make_boost(const transcribe::Tokenizer & tok, float weight = 2.0f) {
    PhraseBoost b;
    b.build(tok, kNClasses, kBlank,
            { "Lisinopril", "  hydrochlorothiazide ", "atrial   fibrillation", "lisinopril", "" }, weight);
    return b;
}

// Joint logits: every class at -10 except the ones set.
struct Logits {
    std::vector<float> v = std::vector<float>(kNClasses, -10.0f);

    Logits & set(int id, float value) {
        v[static_cast<size_t>(id)] = value;
        return *this;
    }
};

int32_t walk(const PhraseBoost & b, std::initializer_list<int> ids, int32_t cursor = PhraseBoost::kRoot) {
    for (const int id : ids) {
        cursor = b.advance(cursor, id);
    }
    return cursor;
}

void test_build() {
    const transcribe::Tokenizer tok = make_tokenizer();
    const PhraseBoost           b   = make_boost(tok);
    CHECK(!b.empty());
    CHECK(b.n_terms() == 3);  // duplicates, case and spacing collapse; plurals are not terms
    CHECK(b.n_classes() == kNClasses);

    PhraseBoost off;
    off.build(tok, kNClasses, kBlank, { "lisinopril" }, 0.0f);
    CHECK(off.empty());
    off.build(tok, kNClasses, kBlank, { " ", "" }, 2.0f);
    CHECK(off.empty());
}

void test_advance() {
    const transcribe::Tokenizer tok = make_tokenizer();
    const PhraseBoost           b   = make_boost(tok);

    // A term matches under any segmentation and capitalization.
    CHECK(walk(b, { kLisW, kIn, kOp, kRil }) > PhraseBoost::kRoot);
    CHECK(walk(b, { kLW, kIs, kIn, kOp, kRil }) == walk(b, { kLisW, kIn, kOp, kRil }));
    CHECK(walk(b, { kMark, kLisCap, kIn }) == walk(b, { kLisW, kIn }));
    // A bare "▁" is a word boundary.
    CHECK(walk(b, { kMark }) == PhraseBoost::kRoot);

    // Punctuation or a longer word leaves the match; nothing matches again
    // until the next word boundary.
    CHECK(walk(b, { kLisW, kIn, kOp, kRil, kPeriod }) == PhraseBoost::kNone);
    CHECK(walk(b, { kLikeW }) == PhraseBoost::kNone);
    CHECK(walk(b, { kLikeW, kIs }) == PhraseBoost::kNone);
    CHECK(walk(b, { kLikeW, kLisW }) == walk(b, { kLisW }));
    // Matches start at word boundaries only.
    CHECK(walk(b, { kIs }, PhraseBoost::kNone) == PhraseBoost::kNone);

    // Multi-word terms continue across "▁".
    CHECK(walk(b, { kAtrialW, kFibW, kR, kIllation }) > PhraseBoost::kRoot);
    CHECK(walk(b, { kAtrialW, kTheW }) == PhraseBoost::kNone);

    // Specials break a match.
    CHECK(walk(b, { kLisW, kUnk }) == PhraseBoost::kNone);
}

int pick(const PhraseBoost & b, int32_t cursor, const Logits & l) {
    return b.pick(cursor, l.v.data());
}

void test_pick() {
    const transcribe::Tokenizer tok  = make_tokenizer();
    const PhraseBoost           b    = make_boost(tok);
    const int32_t               none = PhraseBoost::kNone;

    // No bonus in play: plain argmax.
    CHECK(pick(b, none, Logits().set(kTheW, 5.0f).set(kIs, 4.0f)) == kTheW);
    CHECK(pick(b, none, Logits().set(kBlank, 6.0f).set(kTheW, 5.0f)) == kBlank);

    // In-word continuation outweighs a near miss: "hydrochloroth" + i, not z.
    const int32_t hydrochloroth = walk(b, { kHydroW, kCh, kLor, kOth });
    CHECK(hydrochloroth > PhraseBoost::kRoot);
    CHECK(pick(b, hydrochloroth, Logits().set(kZ, 5.5f).set(kI, 5.0f).set(kIa, 4.0f).set(kBlank, 1.0f)) == kI);
    // ...but not a clear winner.
    CHECK(pick(b, hydrochloroth, Logits().set(kZ, 8.0f).set(kI, 5.0f).set(kBlank, 1.0f)) == kZ);

    // An in-word continuation can turn a blank step into an emission, by at
    // most kMaxBlankBonus whatever the weight; on an emitting step the full
    // weight counts.
    const int32_t hydrochlorothi = b.advance(hydrochloroth, kI);
    CHECK(pick(b, hydrochlorothi, Logits().set(kBlank, 6.0f).set(kA, 4.5f)) == kA);
    CHECK(pick(b, hydrochlorothi, Logits().set(kBlank, 7.0f).set(kA, 4.5f)) == kBlank);
    const PhraseBoost strong = make_boost(tok, 10.0f);
    CHECK(pick(strong, hydrochlorothi, Logits().set(kBlank, 7.0f).set(kA, 4.5f)) == kA);
    CHECK(pick(strong, hydrochlorothi, Logits().set(kBlank, 9.0f).set(kA, 4.5f)) == kBlank);
    CHECK(pick(strong, hydrochloroth, Logits().set(kZ, 12.0f).set(kI, 3.0f).set(kBlank, 1.0f)) == kI);

    // A word's first characters get no bonus: a short opening piece begins
    // many terms, so the model chooses it alone...
    CHECK(pick(b, none, Logits().set(kLikeW, 5.0f).set(kLisW, 4.9f)) == kLikeW);
    CHECK(pick(b, PhraseBoost::kRoot, Logits().set(kMark, 5.0f).set(kLisCap, 4.0f)) == kMark);
    // ...and a continuation counts from kMinDepth characters: "▁l" + "is"
    // ("lis") is still too short, "▁lis" + "in" ("lisin") is not.
    CHECK(pick(b, walk(b, { kLW }), Logits().set(kIs, 4.5f).set(kIn, 5.0f)) == kIn);
    CHECK(pick(b, walk(b, { kLisW }), Logits().set(kIn, 4.5f).set(kIs, 5.0f)) == kIn);

    // The next word of a multi-word term gets the bonus, but only on a step
    // that emits.
    const int32_t atrial = walk(b, { kAtrialW });
    CHECK(pick(b, atrial, Logits().set(kTheW, 5.0f).set(kFibW, 3.5f)) == kFibW);
    CHECK(pick(b, atrial, Logits().set(kBlank, 6.0f).set(kFibW, 5.0f)) == kBlank);

    // Specials are never boosted.
    CHECK(pick(b, PhraseBoost::kRoot, Logits().set(kTheW, 5.0f).set(kUnk, 4.9f)) == kTheW);

    // A plural continues a term as well as the singular does (no push
    // toward "ril" over "rils"), but is never pushed against blank.
    const int32_t lisinop = walk(b, { kLisW, kIn, kOp });
    CHECK(pick(b, lisinop, Logits().set(kRils, 5.0f).set(kRil, 4.0f)) == kRils);
    CHECK(pick(b, lisinop, Logits().set(kRils, 4.0f).set(kRil, 5.0f)) == kRil);
    CHECK(b.advance(lisinop, kRils) > PhraseBoost::kRoot);
    const int32_t lisinopril = b.advance(lisinop, kRil);
    CHECK(pick(b, lisinopril, Logits().set(kBlank, 6.0f).set(kS, 4.5f)) == kBlank);
}

void test_weight_env() {
    const char * name = "TRANSCRIBE_PARAKEET_BOOST_WEIGHT";
    set_env(name, nullptr);
    CHECK(transcribe::parakeet::phrase_boost_weight() == PhraseBoost::kDefaultWeight);
    set_env(name, "3.5");
    CHECK(transcribe::parakeet::phrase_boost_weight() == 3.5f);
    set_env(name, "0");
    CHECK(transcribe::parakeet::phrase_boost_weight() == 0.0f);
    set_env(name, nullptr);
}

}  // namespace

int main() {
    test_build();
    test_advance();
    test_pick();
    test_weight_env();

    if (g_failures > 0) {
        std::fprintf(stderr, "FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}
