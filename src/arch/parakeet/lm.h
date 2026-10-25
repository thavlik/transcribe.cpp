// arch/parakeet/lm.h - n-gram language-model fusion for the greedy RNN-T
// decoders.
//
// Shallow fusion: an n-gram model over the model's own pieces (ARPA format,
// piece ids as its words, <s> at a stream's start) scores the non-blank
// choices of each greedy step, and weight * ln P(piece | the pieces before
// it) is added to their joint logits before the argmax. Only which piece
// is emitted changes: whether one is (against blank) stays the
// transducer's call, since a language model that dislikes the next words
// would otherwise hold the decoder on blank through them ("milligrams
// orally provided no contraindications" came out as "mg"). Raw logits
// differ from log-probabilities by one constant per step, so the argmax
// needs no normalization. Only pieces within kLmWindow nats of the best
// non-blank one are scored; the rest cannot win.
//
// Each decoder keeps its own context (LmContext: the pieces it emitted,
// newest last), carried across chunks like its predictor state.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::parakeet {

class NgramLm {
  public:
    // The context element standing for <s>.
    static constexpr int32_t kStart = -1;

    // Reads an ARPA file; false (with an error logged) when it cannot.
    bool load(const std::string & path);

    bool empty() const { return order_ == 0; }

    int order() const { return order_; }

    size_t size() const { return n_entries_; }

    // ln P(token | ctx), ctx the pieces before it (oldest first, kStart for
    // <s>), backing off as ARPA does; a piece the model never saw gets
    // floor().
    float log_prob(const int32_t * ctx, int n_ctx, int32_t token) const;

    float floor() const { return floor_; }

  private:
    struct Entry {
        uint64_t key     = 0;  // 0 = empty slot
        float    log_p   = 0.0f;
        float    backoff = 0.0f;
    };

    static uint64_t hash(const int32_t * ids, int n);
    const Entry *   find(uint64_t key) const;
    void            insert(uint64_t key, float log_p, float backoff);

    std::vector<Entry> table_;
    uint64_t           mask_      = 0;
    size_t             n_entries_ = 0;
    int                order_     = 0;
    float              floor_     = -20.0f;
};

// The pieces a decoder emitted, newest last; empty means a stream's start.
struct LmContext {
    static constexpr int kMax = 15;

    int32_t ids[kMax] = {};
    int     n         = 0;

    void push(int32_t id);
};

// One decoder's fusion: the model, its weight and the decoder's context.
struct LmFusion {
    const NgramLm * lm     = nullptr;
    float           weight = 0.0f;
    LmContext *     ctx    = nullptr;

    bool active() const { return lm != nullptr && !lm->empty() && weight > 0.0f && ctx != nullptr; }
};

// Pieces this far (nats) below the best non-blank one are not scored.
constexpr float kLmWindow = 20.0f;

// Adds the fusion's term to the non-blank `logits` (n_cls of them, blank at
// blank_id) within kLmWindow of the best, and pushes the rest out of reach.
void apply_lm(const LmFusion & f, float * logits, int n_cls, int blank_id);

// The process's language model: the ARPA file TRANSCRIBE_PARAKEET_LM names,
// loaded on first use, or nullptr.
const NgramLm * process_lm();

// Its weight: TRANSCRIBE_PARAKEET_LM_WEIGHT, kDefaultLmWeight when unset.
constexpr float kDefaultLmWeight = 0.2f;
float           lm_weight();

}  // namespace transcribe::parakeet
