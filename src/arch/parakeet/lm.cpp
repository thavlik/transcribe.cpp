// arch/parakeet/lm.cpp - n-gram language-model fusion for the greedy RNN-T
// decoders. See lm.h.

#include "lm.h"

#include "transcribe-log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

namespace transcribe::parakeet {

namespace {

constexpr float kLn10 = 2.302585093f;

uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

}  // namespace

uint64_t NgramLm::hash(const int32_t * ids, int n) {
    uint64_t h = 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(n + 1);
    for (int i = 0; i < n; ++i) {
        h = mix(h ^ (static_cast<uint64_t>(static_cast<uint32_t>(ids[i])) + 0x632be59bd9b4e019ULL));
    }
    return h == 0 ? 1 : h;
}

const NgramLm::Entry * NgramLm::find(uint64_t key) const {
    if (table_.empty()) {
        return nullptr;
    }
    for (uint64_t i = key & mask_;; i = (i + 1) & mask_) {
        const Entry & e = table_[i];
        if (e.key == key) {
            return &e;
        }
        if (e.key == 0) {
            return nullptr;
        }
    }
}

void NgramLm::insert(uint64_t key, float log_p, float backoff) {
    uint64_t i = key & mask_;
    while (table_[i].key != 0 && table_[i].key != key) {
        i = (i + 1) & mask_;
    }
    if (table_[i].key == 0) {
        ++n_entries_;
    }
    table_[i] = Entry{ key, log_p, backoff };
}

bool NgramLm::load(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet lm: cannot open %s", path.c_str());
        return false;
    }
    std::vector<char>    line(1 << 16);
    std::vector<size_t>  counts;
    int                  section = -1;  // n of the "\n-grams:" being read
    size_t               total   = 0;
    float                lowest  = 0.0f;
    bool                 ok      = true;
    std::vector<int32_t> ids;
    while (std::fgets(line.data(), static_cast<int>(line.size()), f) != nullptr) {
        char * s = line.data();
        if (s[0] == '\\') {
            int n = 0;
            if (std::sscanf(s, "\\%d-grams:", &n) == 1) {
                section = n;
            } else if (std::strncmp(s, "\\data\\", 6) == 0) {
                section = 0;
            } else if (std::strncmp(s, "\\end\\", 5) == 0) {
                break;
            }
            continue;
        }
        if (section == 0) {
            int    n = 0;
            size_t c = 0;
            if (std::sscanf(s, "ngram %d=%zu", &n, &c) == 2 && n > 0) {
                counts.resize(std::max<size_t>(counts.size(), static_cast<size_t>(n)));
                counts[static_cast<size_t>(n - 1)] = c;
                total += c;
            }
            continue;
        }
        if (section < 1 || s[0] == '\n' || s[0] == '\r' || s[0] == '\0') {
            continue;
        }
        if (table_.empty()) {
            size_t cap = 1;
            while (static_cast<double>(cap) < 1.7 * static_cast<double>(std::max<size_t>(total, 16))) {
                cap <<= 1;
            }
            table_.assign(cap, Entry{});
            mask_  = cap - 1;
            order_ = static_cast<int>(counts.size());
        }
        // "<log10 p>\t<w1> ... <wn>[\t<log10 backoff>]"
        char *      end   = nullptr;
        const float log_p = std::strtof(s, &end);
        if (end == s) {
            ok = false;
            break;
        }
        s = end;
        ids.clear();
        for (int k = 0; k < section; ++k) {
            while (*s == ' ' || *s == '\t') {
                ++s;
            }
            if (std::strncmp(s, "<s>", 3) == 0) {
                ids.push_back(kStart);
                s += 3;
            } else if (std::strncmp(s, "</s>", 4) == 0) {
                ids.push_back(-2);
                s += 4;
            } else if (std::strncmp(s, "<unk>", 5) == 0) {
                ids.push_back(-3);
                s += 5;
            } else {
                const long v = std::strtol(s, &end, 10);
                if (end == s) {
                    ok = false;
                    break;
                }
                ids.push_back(static_cast<int32_t>(v));
                s = end;
            }
        }
        if (!ok) {
            break;
        }
        const float backoff = std::strtof(s, &end);
        insert(hash(ids.data(), static_cast<int>(ids.size())), log_p * kLn10, end == s ? 0.0f : backoff * kLn10);
        if (section == 1 && log_p > -90.0f) {
            lowest = std::min(lowest, log_p * kLn10);
        }
    }
    std::fclose(f);
    if (!ok || order_ == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet lm: %s is not an ARPA file of piece ids", path.c_str());
        table_.clear();
        order_     = 0;
        n_entries_ = 0;
        return false;
    }
    // A piece the model never saw: a little less likely than its rarest.
    floor_ = lowest - 1.0f;
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "parakeet lm: %s, order %d, %zu n-grams", path.c_str(), order_, n_entries_);
    return true;
}

float NgramLm::log_prob(const int32_t * ctx, int n_ctx, int32_t token) const {
    int32_t   gram[LmContext::kMax + 1];
    const int k_max = std::min(n_ctx, order_ - 1);
    float     bo    = 0.0f;
    for (int k = k_max; k >= 0; --k) {
        std::memcpy(gram, ctx + (n_ctx - k), static_cast<size_t>(k) * sizeof(int32_t));
        gram[k] = token;
        if (const Entry * e = find(hash(gram, k + 1))) {
            return e->log_p + bo;
        }
        if (k > 0) {
            if (const Entry * h = find(hash(gram, k))) {
                bo += h->backoff;
            }
        }
    }
    return floor_ + bo;
}

void LmContext::push(int32_t id) {
    if (n == kMax) {
        std::memmove(ids, ids + 1, sizeof(int32_t) * (kMax - 1));
        --n;
    }
    ids[n++] = id;
}

void apply_lm(const LmFusion & f, float * logits, int n_cls, int blank_id) {
    float best = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n_cls; ++i) {
        if (i != blank_id) {
            best = std::max(best, logits[i]);
        }
    }
    // A stream's start is <s>.
    int32_t         start[1] = { NgramLm::kStart };
    const int32_t * ctx      = f.ctx->n > 0 ? f.ctx->ids : start;
    const int       n_ctx    = f.ctx->n > 0 ? f.ctx->n : 1;
    for (int i = 0; i < n_cls; ++i) {
        if (i == blank_id) {
            continue;
        }
        if (logits[i] < best - kLmWindow) {
            logits[i] = -std::numeric_limits<float>::infinity();
            continue;
        }
        logits[i] += f.weight * f.lm->log_prob(ctx, n_ctx, i);
    }
}

const NgramLm * process_lm() {
    static NgramLm        lm;
    static std::once_flag once;
    static bool           loaded = false;
    std::call_once(once, [] {
        if (const char * p = std::getenv("TRANSCRIBE_PARAKEET_LM"); p != nullptr && p[0] != '\0') {
            loaded = lm.load(p);
        }
    });
    return loaded ? &lm : nullptr;
}

float lm_weight() {
    if (const char * v = std::getenv("TRANSCRIBE_PARAKEET_LM_WEIGHT"); v != nullptr && v[0] != '\0') {
        return std::max(0.0f, std::strtof(v, nullptr));
    }
    return kDefaultLmWeight;
}

}  // namespace transcribe::parakeet
