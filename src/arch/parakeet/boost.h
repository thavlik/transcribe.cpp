// arch/parakeet/boost.h - vocabulary boosting for the greedy transducer
// decoders (transcribe_run_params::vocabulary on TDT / RNN-T heads).
//
// Shallow-fusion contextual biasing. While the text a decoder has emitted
// for the current word(s) spells the beginning of a vocabulary term, the
// tokens that continue the term get a bonus on their joint logit before the
// greedy argmax. Terms are matched on ASCII-lowercased characters over a
// trie, so any tokenization and any capitalization of a term continues the
// match; a match starts only at a word boundary (a SentencePiece "▁" piece).
//
// The bonus (the weight) goes to a piece that continues a match to at least
// kMinDepth characters: the rest of the current word, or the next word of a
// multi-word term. A word's first characters get none. The pieces that open
// words are short (a 1024-piece vocabulary spells "lisinopril" as
// "▁l is in op r il"), and a short prefix begins many terms and many
// ordinary words: boosting from the third letter turned "palpitations" into
// "pulpitations" (toward "pulmonary embolism") in medical dictation.
// Only an in-word continuation can turn a blank step into an emission, so
// a term the model has started is not left with a sub-word missing, and
// only by up to kMaxBlankBonus: a larger push emits pieces before their
// audio, and the decoder then spells them again when the audio arrives
// ("hydrochlochloro..."). The next word of a term changes which token a
// step emits but never whether it emits, so a term is not tacked onto the
// end of a phrase. Token confidences stay on the raw logits.
//
// The match state is one int32 cursor per decoder (kRoot at stream start),
// carried in ParakeetStreamingDecoderState across chunks. An empty boost
// (no terms, or weight <= 0) leaves the decoders on the plain argmax.

#pragma once

#include "transcribe-tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::parakeet {

class PhraseBoost {
  public:
    // At a word boundary: the next word may begin a term.
    static constexpr int32_t kRoot = 0;
    // Inside a word that matches no term: nothing can match until the next
    // word boundary.
    static constexpr int32_t kNone = -1;

    // Default bonus in nats (TRANSCRIBE_PARAKEET_BOOST_WEIGHT overrides it;
    // <= 0 turns boosting off). On medical dictation 2-8 all fixed about
    // one misspelling in six; 8 started to pull ordinary words toward
    // terms that share their first letters.
    static constexpr float kDefaultWeight = 4.0f;

    // Characters of a term a piece must match up to for its bonus.
    static constexpr int kMinDepth = 5;

    // Most an in-word continuation's bonus counts against blank.
    static constexpr float kMaxBlankBonus = 3.0f;

    // Indexes the model's pieces (n_classes joint outputs, blank_id among
    // them) and builds the trie over `terms`. Terms are trimmed, ASCII-
    // lowercased and whitespace-collapsed; empty ones are skipped.
    void build(const Tokenizer &                tok,
               int                              n_classes,
               int                              blank_id,
               const std::vector<std::string> & terms,
               float                            weight);

    void clear();

    bool empty() const { return weight_ <= 0.0f || n_terms_ == 0; }

    int n_classes() const { return n_classes_; }

    size_t n_terms() const { return n_terms_; }

    float weight() const { return weight_; }

    // Token one greedy step emits from `logits` (n_classes() floats) at
    // match state `cursor`. With no bonus in play this is the plain argmax
    // (first maximum).
    int pick(int32_t cursor, const float * logits) const;

    // Match state after `token` (non-blank) is emitted at `cursor`.
    int32_t advance(int32_t cursor, int token) const;

  private:
    enum class Match : uint8_t { None, Start, NextWord, InWord };

    struct Piece {
        std::string text;                // lowercased, without the leading "▁"
        bool        word_start = false;  // the piece began with "▁"
        bool        breaks     = true;   // blank / special: never matches
    };

    struct Step {
        int32_t node;
        Match   match;
    };

    int32_t child(int32_t node, uint8_t c) const;
    int32_t walk(int32_t node, const std::string & text) const;
    Step    step(int32_t cursor, int token) const;
    float   bonus(Step s) const;

    // Trie in CSR form: node n's edges are [edge_begin_[n], edge_begin_[n+1]),
    // sorted by character; depth_[n] is the length of node n's prefix and
    // variant_[n] marks a node only a term's plural reaches.
    std::vector<uint32_t> edge_begin_;
    std::vector<uint16_t> depth_;
    std::vector<bool>     variant_;
    std::vector<uint8_t>  edge_char_;
    std::vector<int32_t>  edge_child_;

    std::vector<Piece> pieces_;  // per joint output class
    size_t             n_terms_   = 0;
    int                n_classes_ = 0;
    int                blank_id_  = -1;
    float              weight_    = 0.0f;
};

// Continuation bonus for this process: TRANSCRIBE_PARAKEET_BOOST_WEIGHT when
// set, PhraseBoost::kDefaultWeight otherwise.
float phrase_boost_weight();

}  // namespace transcribe::parakeet
