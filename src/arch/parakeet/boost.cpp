// arch/parakeet/boost.cpp - vocabulary boosting for the greedy transducer
// decoders. See boost.h for the matching and bonus rules.

#include "boost.h"

#include "transcribe-log.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <utility>

namespace transcribe::parakeet {

namespace {

// SentencePiece word-boundary marker (U+2581) in UTF-8.
constexpr char   kSpMark[]  = "\xE2\x96\x81";
constexpr size_t kSpMarkLen = 3;

char lower_ascii(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Trimmed, ASCII-lowercased, whitespace runs collapsed to one space.
std::string normalize_term(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool pending_space = false;
    for (const char c : s) {
        if (is_space(c)) {
            pending_space = !out.empty();
            continue;
        }
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        out.push_back(lower_ascii(c));
    }
    return out;
}

}  // namespace

void PhraseBoost::clear() {
    edge_begin_.clear();
    depth_.clear();
    variant_.clear();
    edge_char_.clear();
    edge_child_.clear();
    pieces_.clear();
    n_terms_   = 0;
    n_classes_ = 0;
    blank_id_  = -1;
    weight_    = 0.0f;
}

void PhraseBoost::build(const Tokenizer &                tok,
                        int                              n_classes,
                        int                              blank_id,
                        const std::vector<std::string> & terms,
                        float                            weight) {
    clear();
    if (!(weight > 0.0f) || n_classes <= 0 || terms.empty()) {
        return;
    }

    // Trie edges keyed by (parent << 8 | character) while inserting. The
    // terms go in first, then their plurals ("+s", "+es") as variant-only
    // nodes: a plural then continues a match as well as the singular does,
    // instead of losing to it ("troponins" -> "troponin's"), but is never
    // pushed against blank (which would add plurals nobody said).
    std::unordered_map<uint64_t, int32_t> edges;
    std::vector<bool>                     is_term(1, false);
    std::vector<uint16_t>                 depth(1, 0);
    std::vector<bool>                     variant(1, false);
    auto                                  insert = [&](const std::string & text, bool is_variant) {
        int32_t node = kRoot;
        for (const char c : text) {
            const uint64_t key      = (static_cast<uint64_t>(node) << 8) | static_cast<uint8_t>(c);
            const auto [it, is_new] = edges.try_emplace(key, static_cast<int32_t>(is_term.size()));
            if (is_new) {
                is_term.push_back(false);
                depth.push_back(static_cast<uint16_t>(std::min<int>(depth[static_cast<size_t>(node)] + 1, 0xFFFF)));
                variant.push_back(is_variant);
            }
            node = it->second;
        }
        return node;
    };
    std::vector<std::string> normalized;
    for (const std::string & raw : terms) {
        std::string term = normalize_term(raw);
        if (term.empty()) {
            continue;
        }
        const int32_t node = insert(term, false);
        if (!is_term[static_cast<size_t>(node)]) {
            is_term[static_cast<size_t>(node)] = true;
            ++n_terms_;
        }
        normalized.push_back(std::move(term));
    }
    if (n_terms_ == 0) {
        return;
    }
    for (const std::string & term : normalized) {
        insert(term + "s", true);
        insert(term + "es", true);
    }

    // CSR: sorting the keys groups each node's edges, ordered by character.
    std::vector<std::pair<uint64_t, int32_t>> sorted(edges.begin(), edges.end());
    std::sort(sorted.begin(), sorted.end());
    const size_t n_nodes = is_term.size();
    edge_begin_.assign(n_nodes + 1, 0);
    edge_char_.resize(sorted.size());
    edge_child_.resize(sorted.size());
    for (size_t i = 0; i < sorted.size(); ++i) {
        edge_char_[i]  = static_cast<uint8_t>(sorted[i].first & 0xFF);
        edge_child_[i] = sorted[i].second;
        ++edge_begin_[static_cast<size_t>(sorted[i].first >> 8) + 1];
    }
    for (size_t n = 0; n < n_nodes; ++n) {
        edge_begin_[n + 1] += edge_begin_[n];
    }
    depth_   = std::move(depth);
    variant_ = std::move(variant);

    // Pieces as the matcher reads them. Blank, ids past the vocabulary and
    // strippable specials (<unk>, control tags) break any match.
    pieces_.resize(static_cast<size_t>(n_classes));
    for (int id = 0; id < n_classes; ++id) {
        if (id == blank_id || id >= tok.n_tokens() || tok.is_strippable_special(id)) {
            continue;
        }
        Piece &     p = pieces_[static_cast<size_t>(id)];
        std::string s = tok.token(id);
        if (s.compare(0, kSpMarkLen, kSpMark) == 0) {
            p.word_start = true;
            s.erase(0, kSpMarkLen);
        }
        for (size_t pos = s.find(kSpMark); pos != std::string::npos; pos = s.find(kSpMark, pos + 1)) {
            s.replace(pos, kSpMarkLen, " ");
        }
        for (char & c : s) {
            c = lower_ascii(c);
        }
        p.text   = std::move(s);
        p.breaks = false;
    }

    n_classes_ = n_classes;
    blank_id_  = blank_id;
    weight_    = weight;
}

int32_t PhraseBoost::child(int32_t node, uint8_t c) const {
    const auto first = edge_char_.begin() + edge_begin_[static_cast<size_t>(node)];
    const auto last  = edge_char_.begin() + edge_begin_[static_cast<size_t>(node) + 1];
    const auto it    = std::lower_bound(first, last, c);
    return (it != last && *it == c) ? edge_child_[static_cast<size_t>(it - edge_char_.begin())] : kNone;
}

int32_t PhraseBoost::walk(int32_t node, const std::string & text) const {
    for (const char c : text) {
        node = child(node, static_cast<uint8_t>(c));
        if (node < 0) {
            return kNone;
        }
    }
    return node;
}

PhraseBoost::Step PhraseBoost::step(int32_t cursor, int token) const {
    if (token < 0 || token >= n_classes_) {
        return { kNone, Match::None };
    }
    const Piece & p = pieces_[static_cast<size_t>(token)];
    if (p.breaks) {
        return { kNone, Match::None };
    }
    if (p.word_start) {
        // The next word of a multi-word term...
        if (cursor > kRoot) {
            if (const int32_t space = child(cursor, ' '); space >= 0) {
                if (const int32_t n = walk(space, p.text); n >= 0) {
                    return { n, Match::NextWord };
                }
            }
        }
        // ...or the first word of one.
        const int32_t n = walk(kRoot, p.text);
        if (n < 0) {
            return { kNone, Match::None };
        }
        return { n, n == kRoot ? Match::None : Match::Start };
    }
    if (cursor < kRoot) {
        return { kNone, Match::None };
    }
    const int32_t n = walk(cursor, p.text);
    if (n < 0) {
        return { kNone, Match::None };
    }
    if (n == cursor) {
        return { n, Match::None };
    }
    return { n, cursor == kRoot ? Match::Start : Match::InWord };
}

float PhraseBoost::bonus(Step s) const {
    const bool continues = s.match == Match::InWord || s.match == Match::NextWord;
    return (continues && depth_[static_cast<size_t>(s.node)] >= kMinDepth) ? weight_ : 0.0f;
}

int PhraseBoost::pick(int32_t cursor, const float * logits) const {
    // Plain argmax over every class and over the non-blank ones.
    int raw = 0;
    int nb  = -1;
    for (int i = 0; i < n_classes_; ++i) {
        if (logits[i] > logits[raw]) {
            raw = i;
        }
        if (i != blank_id_ && (nb < 0 || logits[i] > logits[nb])) {
            nb = i;
        }
    }
    if (nb < 0) {
        return raw;
    }

    // Only a token within one bonus of the best non-blank logit can win.
    // Ties keep the plain argmax.
    const float floor_v = logits[nb] - weight_;
    int         best    = nb;
    float       best_v  = -std::numeric_limits<float>::infinity();
    int         cont    = -1;  // best in-word continuation against blank
    float       cont_v  = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_classes_; ++i) {
        if (i == blank_id_ || !(logits[i] > floor_v)) {
            continue;
        }
        const Step  st = step(cursor, i);
        const float b  = bonus(st);
        const float v  = logits[i] + b;
        if (v > best_v || (i == nb && v == best_v)) {
            best   = i;
            best_v = v;
        }
        if (st.match == Match::InWord && b > 0.0f && !variant_[static_cast<size_t>(st.node)]) {
            const float vb = logits[i] + std::min(b, kMaxBlankBonus);
            if (vb > cont_v) {
                cont   = i;
                cont_v = vb;
            }
        }
    }
    // A blank step stays blank unless the current word's next piece beats it.
    const int pick = raw != blank_id_ ? best : (cont >= 0 && cont_v > logits[blank_id_]) ? cont : raw;
    if (pick != raw) {
        auto name = [&](int id) {
            return id == blank_id_ ? std::string("<blank>") : "'" + pieces_[id].text + "'";
        };
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "parakeet boost: %s (%.2f) over %s (%.2f)", name(pick).c_str(),
                static_cast<double>(logits[pick]), name(raw).c_str(), static_cast<double>(logits[raw]));
    }
    return pick;
}

int32_t PhraseBoost::advance(int32_t cursor, int token) const {
    return step(cursor, token).node;
}

float phrase_boost_weight() {
    if (const char * v = std::getenv("TRANSCRIBE_PARAKEET_BOOST_WEIGHT"); v != nullptr && v[0] != '\0') {
        return std::strtof(v, nullptr);
    }
    return PhraseBoost::kDefaultWeight;
}

}  // namespace transcribe::parakeet
