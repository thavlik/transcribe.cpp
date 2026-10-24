// arch/parakeet/multitalker.cpp - multitalker orchestration for bundle
// GGUFs (multitalker Parakeet + embedded streaming Sortformer).
//
// Mirrors NeMo's SpeakerTaggedASR parallel streaming pipeline
// (speech_to_text_multitalker_streaming_infer.py at the pinned rev):
//
//   1. The embedded Sortformer runs its streaming AOSC/FIFO forward at the
//      multitalker operating point (chunk = ASR chunk cadence
//      att_context_right+1 = 14 diar frames at [70, 13], lc/rc = 0,
//      spkcache 188, fifo 188, update period 188 — the example's config
//      on the v2.1 checkpoint defaults). The accumulated per-chunk preds
//      equal the causal per-chunk supervision the reference loop feeds
//      each ASR instance.
//   2. Active speakers = columns whose activity has exceeded 0.5.
//   3. One cache-aware ASR instance per active speaker, advanced only on
//      the chunks where the reference's cache gating selects that speaker,
//      with that speaker's supervision: kernel mode (masked_asr=False, the
//      default here) drives the layer-0 speaker-kernel injection; masked
//      mode (masked_asr=True, the NeMo example default, selected via
//      TRANSCRIBE_MULTITALKER_MODE=masked) masks mel features.
//   4. Per-speaker results merge into one speaker-tagged result: segments
//      ordered by start time with speaker_id set, words/tokens re-indexed
//      under them, full_text joined in segment order (SegLST rendering),
//      and the transcript-independent speaker_segment rows emitted from
//      the diarizer preds.
//
// Every stage is causal at chunk granularity, so one chunk pipeline serves
// two entry points: transcribe_run (run_multitalker, the whole clip at
// once) and the streaming API (multitalker_stream_*, chunk by chunk as
// audio arrives, with step 4 rebuilt per snapshot). Feeding a clip through
// the stream API and finalizing reproduces the transcribe_run result.
//
// TRANSCRIBE_MULTITALKER_OFFLINE=1 keeps the older offline replay for
// transcribe_run: the validated OFFLINE chunked-attention graph over the
// whole clip (parity/dump tooling; O(T^2) attention buffers).

#include "../sortformer/sortformer.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "parakeet.h"
#include "transcribe-backend.h"
#include "transcribe-debug.h"
#include "transcribe-diarize.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace transcribe::parakeet {

namespace sf = transcribe::sortformer;

namespace {

struct SpeakerDecode {
    int                                           speaker = 0;  // 0-based diar column
    std::vector<transcribe_session::TokenEntry>   tokens;
    std::vector<transcribe_session::WordEntry>    words;
    std::vector<transcribe_session::SegmentEntry> segments;
    std::string                                   full_text;
    std::string                                   raw_text;
};

// One speaker slot's private cache-aware streaming instance (an entry of
// the reference's instance manager). The driver swaps it into the session
// around each emit_streaming_chunk so the single-speaker streaming code
// runs unchanged. The steady-state encoder graph, its scheduler and its
// compute context belong to the instance too: that graph binds the
// instance's cache tensors as leaves, so replaying one graph for another
// speaker would attend over the wrong speaker's caches.
struct SpeakerInstance {
    ParakeetStreamingCaches       caches;
    ParakeetStreamingDecoderState dec;
    ParakeetStreamingGraph        graph;
    ggml_backend_sched_t          sched       = nullptr;
    ggml_context *                compute_ctx = nullptr;
    std::vector<TdtToken>         raw;
    std::vector<int32_t>          fed;  // absolute enc frame per gathered frame
    bool                          started = false;

    // Streaming snapshots: the speaker's result as last built, and the raw
    // token count it was built from (rebuilt only when tokens arrive).
    SpeakerDecode built;
    size_t        built_from = 0;
    bool          has_built  = false;
};

// Swaps an instance into the session for one scope and back on every exit
// path, so the session never keeps another instance's state.
class InstanceScope {
  public:
    InstanceScope(ParakeetSession * pc, SpeakerInstance & inst) : pc_(pc), inst_(inst) { swap(); }

    ~InstanceScope() { swap(); }

    InstanceScope(const InstanceScope &)             = delete;
    InstanceScope & operator=(const InstanceScope &) = delete;

  private:
    void swap() noexcept {
        std::swap(pc_->stream_caches, inst_.caches);
        std::swap(pc_->stream_dec_state, inst_.dec);
        std::swap(pc_->raw_tokens, inst_.raw);
        std::swap(pc_->stream_graph, inst_.graph);
        std::swap(pc_->sched, inst_.sched);
        std::swap(pc_->compute_ctx, inst_.compute_ctx);
    }

    ParakeetSession * pc_;
    SpeakerInstance & inst_;
};

void free_instance(SpeakerInstance & I) noexcept {
    I.graph.reset();
    // Scheduler first, then the graph context (the base-class order).
    transcribe::release_compute_scratch(I.sched, I.compute_ctx);
    if (I.caches.buffer != nullptr) {
        transcribe::safe_buffer_free(I.caches.buffer);
        I.caches.buffer = nullptr;
    }
    if (I.caches.ctx != nullptr) {
        ggml_free(I.caches.ctx);
        I.caches.ctx = nullptr;
    }
    if (I.caches.pos_proj_buf != nullptr) {
        transcribe::safe_buffer_free(I.caches.pos_proj_buf);
        I.caches.pos_proj_buf = nullptr;
    }
    if (I.caches.pos_proj_ctx != nullptr) {
        ggml_free(I.caches.pos_proj_ctx);
        I.caches.pos_proj_ctx = nullptr;
    }
    I.caches.pos_proj.clear();
    I.caches.pos_proj_len = -1;
    I.caches.last_channel.clear();
    I.caches.last_time.clear();
    I.caches.last_k.clear();
    I.caches.last_v.clear();
    I.caches.initialized = false;
    I.started            = false;
}

// The chunk grid shared by the diarizer and every speaker's encoder (NeMo
// setup_streaming_params at the stream's lookahead): chunk 0 consumes
// mel_first fresh mel columns; every later chunk consumes mel_chunk new
// columns with mel_cache history columns prepended (reproducing the
// continuous pre_encode) and drops drop_sub pre-encode frames. At [70, 13]:
// 105 / 112 / 9 / 2, i.e. 14 encoder frames (1.12 s) per chunk.
struct ChunkGrid {
    int att_left  = 0;
    int att_right = 0;
    int chunk_enc = 0;  // encoder (= diarizer) frames per chunk: R + 1
    int sub       = 0;  // mel columns per encoder frame
    int mel_first = 0;
    int mel_chunk = 0;
    int mel_cache = 0;
    int drop_sub  = 0;

    // First NEW mel column of chunk k, its new-column count, its history
    // prepend and its pre-encode drop.
    int64_t new_col(int64_t k) const { return k == 0 ? 0 : mel_first + (k - 1) * static_cast<int64_t>(mel_chunk); }

    int advance(int64_t k) const { return k == 0 ? mel_first : mel_chunk; }

    int history(int64_t k) const { return k == 0 ? 0 : mel_cache; }

    int drop(int64_t k) const { return k == 0 ? 0 : drop_sub; }
};

ChunkGrid make_grid(const ParakeetHParams & hp, int att_left, int att_right) {
    ChunkGrid g;
    g.att_left  = att_left;
    g.att_right = att_right;
    g.chunk_enc = att_right + 1;
    g.sub       = hp.enc_subsampling_factor;
    g.mel_first = hp.enc_stream_sampling_frames_first + g.sub * att_right;
    g.mel_chunk = g.chunk_enc * g.sub;
    g.mel_cache = hp.enc_stream_pre_encode_cache_size;
    g.drop_sub  = hp.enc_stream_drop_extra_pre_encoded;
    return g;
}

// The diarizer at the multitalker operating point on grid g. Reference-exact
// windowing: the NeMo loop feeds the diarizer the ASR streaming buffer's
// chunks (chunk_size_first mel frames first, then pre_encode_cache-frame
// overlapped windows with drop_extra_pre_encoded outputs discarded).
sf::SortformerStreamParams diar_stream_params(const ParakeetModel * pm, const ChunkGrid & g) {
    sf::SortformerStreamParams P = sf::resolve_stream_params(pm->diar->hp, TRANSCRIBE_SORTFORMER_PRESET_DEFAULT);
    P.chunk_len                  = g.chunk_enc;
    P.chunk_left_context         = 0;
    P.chunk_right_context        = 0;
    P.spkcache_len               = 188;
    P.fifo_len                   = 188;
    P.spkcache_update_period     = 188;
    P.feat_first_chunk           = g.mel_first;
    P.feat_cache                 = g.mel_cache;
    P.drop_pre_encode            = g.drop_sub;
    return P;
}

// Supervision mode. Default is the checkpoint's trained speaker-kernel
// path: on ami-ihm-test it scores 19.35% cpWER vs masked's 23.73%
// (and vs the NeMo reference's own 21.39% kernel / 24.00% masked).
// TRANSCRIBE_MULTITALKER_MODE=masked selects mel feature masking
// (the NeMo example default) until the family run-ext grows a knob.
bool multitalker_use_kernel() {
    if (const char * mode = std::getenv("TRANSCRIBE_MULTITALKER_MODE")) {
        return std::strcmp(mode, "masked") != 0;
    }
    return true;
}

// Reference cache_gating: speaker s is active in chunk k iff the max of its
// accumulated preds over the last 2 chunks (incl. current) exceeds 0.5.
// Applies to BOTH supervision modes (the reference instance selection is
// mode-independent).
bool chunk_active(const float * probs, int T_diar, int n_spk, int chunk_enc, int s, int64_t k) {
    const int64_t begin = std::max<int64_t>(0, (k - 1) * chunk_enc);
    const int64_t end   = std::min<int64_t>((k + 1) * chunk_enc, T_diar);
    float         mx    = 0.0f;
    for (int64_t t = begin; t < end; ++t) {
        mx = std::max(mx, probs[static_cast<size_t>(t) * n_spk + s]);
    }
    return mx > 0.5f;
}

// Add every speaker whose preds in [t0, t1) exceed 0.5 to `active`
// (ascending slot order, the order the offline path iterates).
void note_active_speakers(const float * probs, int n_spk, int t0, int t1, std::vector<int> & active) {
    for (int s = 0; s < n_spk; ++s) {
        if (std::find(active.begin(), active.end(), s) != active.end()) {
            continue;
        }
        for (int t = t0; t < t1; ++t) {
            if (probs[static_cast<size_t>(t) * n_spk + s] > 0.5f) {
                active.insert(std::upper_bound(active.begin(), active.end(), s), s);
                break;
            }
        }
    }
}

// Feed one chunk of the grid to speaker instance I (allocating and zeroing
// its caches on its first chunk) and record the absolute encoder frame of
// every frame its decoder consumed.
transcribe_status feed_instance(ParakeetSession * pc,
                                ParakeetModel *   pm,
                                const ChunkGrid & g,
                                SpeakerInstance & I,
                                const float *     chunk_mel,
                                int               n_feed,
                                int               drop,
                                int               advance,
                                const float *     spk,
                                const float *     bg,
                                int               mask_len,
                                int               t_enc0) {
    InstanceScope scope(pc, I);
    if (!I.started) {
        if (const transcribe_status st = init_streaming_caches(pc, pm); st != TRANSCRIBE_OK) {
            return st;
        }
        // The allocator does not zero a fresh cache buffer, and
        // cache_last_time (the depthwise convolutions' left context) is not
        // covered by the unfilled-cache attention mask: NeMo starts every
        // instance from an all-zero cache state.
        zero_streaming_caches(pc);
        reset_streaming_decoder_state(pc, pm);
        pc->stream_caches.att_context_left  = g.att_left;
        pc->stream_caches.att_context_right = g.att_right;
        pc->stream_caches.channel_len       = 0;
        pc->stream_caches.chunk_step        = 0;
        I.started                           = true;
    }
    const int64_t           off_before = pc->stream_dec_state.frame_offset;
    const transcribe_status st =
        emit_streaming_chunk(pc, pm, chunk_mel, n_feed, drop, advance, spk, bg, spk != nullptr ? mask_len : 0);
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    const int n_new = static_cast<int>(pc->stream_dec_state.frame_offset - off_before);
    for (int f = 0; f < n_new; ++f) {
        I.fed.push_back(t_enc0 + f);
    }
    return TRANSCRIBE_OK;
}

// One chunk of the grid as seen by the ASR side: absolute mel window
// [c0, c1) with new columns from c0_new, read from `mel` (row-major
// [n_mels, n_cols], column 0 = absolute mel frame col0).
struct AsrChunk {
    int64_t       k      = 0;
    const float * mel    = nullptr;
    int           n_mels = 0;
    int64_t       n_cols = 0;
    int64_t       col0   = 0;
    int64_t       c0     = 0;
    int64_t       c1     = 0;
    int64_t       c0_new = 0;
};

// Run ASR chunk ch.k for every speaker the cache gating selects, each with
// its own supervision. probs is the diarizer's row-major [T_diar, n_spk].
transcribe_status run_asr_chunk(ParakeetSession *              pc,
                                ParakeetModel *                pm,
                                const ChunkGrid &              g,
                                std::vector<SpeakerInstance> & inst,
                                const std::vector<int> &       active,
                                const AsrChunk &               ch,
                                const float *                  probs,
                                int                            T_diar,
                                int                            n_spk,
                                bool                           use_kernel,
                                std::vector<float> &           chunk_buf) {
    const int n_feed   = static_cast<int>(ch.c1 - ch.c0);
    const int advance  = static_cast<int>(ch.c1 - ch.c0_new);
    const int ce       = g.chunk_enc;
    const int t_enc0   = static_cast<int>(ch.k * ce);
    const int mask_len = std::max(0, std::min(ce, T_diar - t_enc0));

    // TRANSCRIBE_MULTITALKER_REF_BG_COMPAT=1 reproduces the reference
    // harness's background-slot indexing bug
    // (multispk_transcribe_utils.get_active_speakers_info takes
    // range(len(active)) minus the SLOT id, so with active slots like [0,2]
    // speaker 0's background tracks the silent slot 1 instead of slot 2).
    // Verified against the reference's own bg_targets dumps. Parity tooling
    // only — the default is the intended union of the other ACTIVE slots.
    static const bool ref_bg_compat = (std::getenv("TRANSCRIBE_MULTITALKER_REF_BG_COMPAT") != nullptr);

    std::vector<float> spk_vals(static_cast<size_t>(mask_len));
    std::vector<float> bg_vals(static_cast<size_t>(mask_len));

    for (const int s : active) {
        if (!chunk_active(probs, T_diar, n_spk, ce, s, ch.k)) {
            continue;
        }

        // Slice this chunk's mel columns (shared grid; the history prepend
        // reproduces the continuous pre_encode).
        chunk_buf.resize(static_cast<size_t>(ch.n_mels) * static_cast<size_t>(n_feed));
        const int64_t rel = ch.c0 - ch.col0;
        for (int m = 0; m < ch.n_mels; ++m) {
            std::memcpy(chunk_buf.data() + static_cast<size_t>(m) * n_feed,
                        ch.mel + static_cast<size_t>(m) * static_cast<size_t>(ch.n_cols) + static_cast<size_t>(rel),
                        static_cast<size_t>(n_feed) * sizeof(float));
        }

        // Masked mode: reference mask_features semantics —
        //   masked = mel * mask (masked columns -> 0.0, NOT the
        //   log-silence value); cells whose ORIGINAL mel was exactly
        //   0 -> mask_value; the upsampled mask is left-aligned to
        //   the chunk's NEW columns (col j -> chunk frame j/8) and
        //   the short side is left-PADDED WITH 0, so the history
        //   prepend columns are always masked.
        if (!use_kernel) {
            const int hist = static_cast<int>(ch.c0_new - ch.c0);  // 9 (0 for the first chunk)
            for (int c = 0; c < n_feed; ++c) {
                bool keep = false;
                if (c >= hist) {
                    const int f = t_enc0 + (c - hist) / g.sub;
                    keep        = f < T_diar && probs[static_cast<size_t>(f) * n_spk + s] > 0.5f;
                }
                for (int m = 0; m < ch.n_mels; ++m) {
                    float & v = chunk_buf[static_cast<size_t>(m) * n_feed + c];
                    if (v == 0.0f) {
                        v = -16.6355f;
                    } else if (!keep) {
                        v = 0.0f;
                    }
                }
            }
        }

        // Kernel mode: per-chunk supervision over the NEW enc frames: raw
        // sigmoids for the target, binarized union of the other active
        // speakers for the background.
        const float * spk_ptr = nullptr;
        const float * bg_ptr  = nullptr;
        if (use_kernel && mask_len > 0) {
            for (int t = 0; t < mask_len; ++t) {
                const size_t row                 = static_cast<size_t>(t_enc0 + t) * n_spk;
                spk_vals[static_cast<size_t>(t)] = probs[row + s];
                float b                          = 0.0f;
                if (ref_bg_compat) {
                    // The reference compares the POSITION in this chunk's
                    // active list to the SLOT id.
                    int n_act_k = 0;
                    for (const int o : active) {
                        n_act_k += chunk_active(probs, T_diar, n_spk, ce, o, ch.k) ? 1 : 0;
                    }
                    for (int slot = 0; slot < n_act_k; ++slot) {
                        if (slot != s && probs[row + slot] > 0.5f) {
                            b = 1.0f;
                        }
                    }
                } else {
                    for (const int o : active) {
                        if (o != s && probs[row + o] > 0.5f) {
                            b = 1.0f;
                        }
                    }
                }
                bg_vals[static_cast<size_t>(t)] = b;
            }
            spk_ptr = spk_vals.data();
            bg_ptr  = bg_vals.data();
        }

        if (const transcribe_status st =
                feed_instance(pc, pm, g, inst[static_cast<size_t>(s)], chunk_buf.data(), n_feed, g.drop(ch.k), advance,
                              spk_ptr, bg_ptr, mask_len, t_enc0);
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    return TRANSCRIBE_OK;
}

// Result rows only: clear_result() also resets the stream bookkeeping
// (revision, committed text, audio cursors), which a mid-stream snapshot
// rebuild must keep.
void clear_result_rows(ParakeetSession * pc) {
    pc->tokens.clear();
    pc->words.clear();
    pc->segments.clear();
    pc->speaker_segments.clear();
    pc->full_text.clear();
    pc->raw_text.clear();
}

// Build one speaker's tokens/words/segment from its raw tokens, with
// step_at_emit remapped from the speaker's gathered timeline to absolute
// encoder frames through its fed-frame list. Decodes at full timestamp
// resolution; the merged result is elided to the caller's request.
transcribe_status build_speaker(ParakeetSession *             pc,
                                ParakeetModel *               pm,
                                const transcribe_run_params * params,
                                int                           speaker,
                                const std::vector<TdtToken> & raw,
                                const std::vector<int32_t> &  fed,
                                SpeakerDecode &               out) {
    if (fed.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    pc->raw_tokens.assign(raw.begin(), raw.end());
    for (auto & rt : pc->raw_tokens) {
        const size_t f  = std::min(static_cast<size_t>(std::max(rt.step_at_emit, 0)), fed.size() - 1);
        rt.step_at_emit = fed[f];
    }
    transcribe_run_params sp = *params;
    sp.timestamps            = TRANSCRIBE_TIMESTAMPS_AUTO;
    clear_result_rows(pc);
    const transcribe_status st = build_result_from_raw_tokens(pc, pm, &sp);
    pc->raw_tokens.clear();
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    out.speaker   = speaker;
    out.tokens    = std::move(pc->tokens);
    out.words     = std::move(pc->words);
    out.segments  = std::move(pc->segments);
    out.full_text = std::move(pc->full_text);
    out.raw_text  = std::move(pc->raw_text);
    clear_result_rows(pc);
    return TRANSCRIBE_OK;
}

// Step 4: merge per-speaker decodes (in ascending speaker order) into one
// speaker-tagged result on pc, add the diarizer's speaker turns, and elide
// to the caller's requested timestamp granularity.
void merge_speakers(ParakeetSession *                          pc,
                    const ParakeetModel *                      pm,
                    const transcribe_run_params *              params,
                    const std::vector<const SpeakerDecode *> & per_spk,
                    const float *                              probs,
                    int                                        T_diar,
                    int                                        n_spk,
                    double                                     diar_ms_per_frame) {
    clear_result_rows(pc);

    struct SegRef {
        const SpeakerDecode * r;
        int                   seg_idx;
    };

    std::vector<SegRef> order;
    for (const SpeakerDecode * r : per_spk) {
        for (size_t i = 0; i < r->segments.size(); ++i) {
            order.push_back({ r, static_cast<int>(i) });
        }
    }
    std::stable_sort(order.begin(), order.end(), [](const SegRef & a, const SegRef & b) {
        return a.r->segments[static_cast<size_t>(a.seg_idx)].t0_ms <
               b.r->segments[static_cast<size_t>(b.seg_idx)].t0_ms;
    });

    std::string merged_text;
    std::string merged_raw;
    for (const SegRef & ref : order) {
        const SpeakerDecode &                    r   = *ref.r;
        const transcribe_session::SegmentEntry & src = r.segments[static_cast<size_t>(ref.seg_idx)];

        const int new_seg_idx = static_cast<int>(pc->segments.size());
        const int word_base   = static_cast<int>(pc->words.size());
        const int token_base  = static_cast<int>(pc->tokens.size());

        for (int j = 0; j < src.n_tokens; ++j) {
            transcribe_session::TokenEntry te = r.tokens[static_cast<size_t>(src.first_token + j)];
            te.seg_index                      = new_seg_idx;
            te.word_index                     = te.word_index - src.first_word + word_base;
            pc->tokens.push_back(std::move(te));
        }
        for (int j = 0; j < src.n_words; ++j) {
            transcribe_session::WordEntry we = r.words[static_cast<size_t>(src.first_word + j)];
            we.seg_index                     = new_seg_idx;
            we.first_token                   = we.first_token - src.first_token + token_base;
            pc->words.push_back(std::move(we));
        }

        transcribe_session::SegmentEntry seg = src;
        seg.first_word                       = word_base;
        seg.first_token                      = token_base;
        seg.speaker_id                       = r.speaker + 1;  // 1-based
        pc->segments.push_back(std::move(seg));

        if (!src.text.empty()) {
            if (!merged_text.empty()) {
                merged_text += ' ';
            }
            merged_text += src.text;
        }
        // One raw-text contribution per speaker, in first-segment order.
        if (ref.seg_idx == 0 && !r.raw_text.empty()) {
            if (!merged_raw.empty()) {
                merged_raw += ' ';
            }
            merged_raw += r.raw_text;
        }
    }
    pc->full_text = std::move(merged_text);
    pc->raw_text  = std::move(merged_raw);

    // Transcript-independent "who spoke when" rows from the diarizer preds
    // (same emission as the standalone sortformer family), each carrying
    // the speaker's mean activity over the turn as its confidence.
    transcribe::probs_to_segments(probs, T_diar, n_spk, diar_ms_per_frame, pc->speaker_segments);
    for (auto & row : pc->speaker_segments) {
        const int t0 = static_cast<int>(std::llround(static_cast<double>(row.t0_ms) / diar_ms_per_frame));
        const int t1 =
            std::min(T_diar, static_cast<int>(std::llround(static_cast<double>(row.t1_ms) / diar_ms_per_frame)));
        double sum = 0.0;
        for (int t = t0; t < t1; ++t) {
            sum += probs[static_cast<size_t>(t) * n_spk + (row.speaker_id - 1)];
        }
        row.p = t1 > t0 ? static_cast<float>(sum / (t1 - t0)) : NAN;
    }

    // ---- Elide to the caller's requested granularity (same convention
    // as decode_and_populate). ----
    transcribe_timestamp_kind eff = params->timestamps;
    if (eff == TRANSCRIBE_TIMESTAMPS_AUTO) {
        eff = pm->caps.max_timestamp_kind;  // TOKEN for parakeet
    }
    if (eff == TRANSCRIBE_TIMESTAMPS_NONE) {
        pc->tokens.clear();
        pc->words.clear();
        for (auto & s : pc->segments) {
            s.t0_ms       = 0;
            s.t1_ms       = 0;
            s.first_word  = 0;
            s.n_words     = 0;
            s.first_token = 0;
            s.n_tokens    = 0;
        }
    } else if (eff == TRANSCRIBE_TIMESTAMPS_SEGMENT) {
        pc->tokens.clear();
        pc->words.clear();
        for (auto & s : pc->segments) {
            s.first_word  = 0;
            s.n_words     = 0;
            s.first_token = 0;
            s.n_tokens    = 0;
        }
    } else if (eff == TRANSCRIBE_TIMESTAMPS_WORD) {
        pc->tokens.clear();
        for (auto & w : pc->words) {
            w.first_token = 0;
            w.n_tokens    = 0;
        }
        for (auto & s : pc->segments) {
            s.first_token = 0;
            s.n_tokens    = 0;
        }
    }

    pc->result_kind = eff;
    pc->has_result  = true;
}

}  // namespace

// ---- Offline: streaming multitalker passes (bounded per-speaker state) ----
//
// Chunk-faithful mirror of the reference loop (conformer_stream_step +
// instance manager) over a precomputed whole-clip mel: one global chunk
// grid; per active speaker a private streaming cache + decoder-state
// instance that only advances on chunks where cache-gating selects that
// speaker. Per-speaker encoder/decoder working state is O(1) in clip
// length (~5 MB per speaker plus one chunk graph), unlike the offline
// replay's O(T^2) attention buffers. The enclosing one-shot call still
// retains O(T) input, mel, and diarizer-prediction buffers.
static transcribe_status run_streaming_passes(ParakeetSession *             pc,
                                              ParakeetModel *               pm,
                                              const float *                 pcm,
                                              int                           n_samples,
                                              const transcribe_run_params * params,
                                              const std::vector<float> &    probs,
                                              int                           T_diar,
                                              int                           n_spk,
                                              const std::vector<int> &      active,
                                              bool                          use_kernel,
                                              std::vector<SpeakerDecode> &  per_spk,
                                              int64_t &                     mel_us,
                                              int64_t &                     enc_us,
                                              int64_t &                     dec_us) {
    const auto &    hp = pm->hparams;
    const ChunkGrid g  = make_grid(hp, hp.enc_att_context_left, hp.enc_att_context_right);

    // ASR mel over the whole clip (row-major [n_mels, n_frames]; each
    // mel row contiguous over frames). This is O(T), about 170 MB at 55 min;
    // unlike offline attention, no O(T^2) buffer is allocated.
    const int64_t      t_mel_start = ggml_time_us();
    std::vector<float> mel;
    int                n_mels = 0, n_frames = 0;
    if (const transcribe_status st = pm->mel->compute(pcm, static_cast<size_t>(n_samples), mel, n_mels, n_frames);
        st != TRANSCRIBE_OK) {
        return st;
    }
    mel_us += ggml_time_us() - t_mel_start;

    std::vector<SpeakerInstance> inst(static_cast<size_t>(n_spk));
    std::vector<float>           chunk_buf;
    transcribe_status            st = TRANSCRIBE_OK;

    const int64_t t_enc_start = ggml_time_us();
    for (int64_t k = 0; st == TRANSCRIBE_OK; ++k) {
        const int64_t c0_new = g.new_col(k);
        if (c0_new >= n_frames) {
            break;
        }
        const int64_t c0 = c0_new - g.history(k);
        const int64_t c1 = std::min<int64_t>(c0_new + g.advance(k), n_frames);
        if (c1 - c0 <= 0 || c1 - c0_new <= 0) {
            break;
        }
        if (pc->poll_abort()) {
            st = TRANSCRIBE_ERR_ABORTED;
            break;
        }
        AsrChunk ch;
        ch.k      = k;
        ch.mel    = mel.data();
        ch.n_mels = n_mels;
        ch.n_cols = n_frames;
        ch.col0   = 0;
        ch.c0     = c0;
        ch.c1     = c1;
        ch.c0_new = c0_new;
        st        = run_asr_chunk(pc, pm, g, inst, active, ch, probs.data(), T_diar, n_spk, use_kernel, chunk_buf);
    }
    enc_us += ggml_time_us() - t_enc_start;

    const int64_t t_dec_start = ggml_time_us();
    for (const int s : active) {
        SpeakerInstance & I = inst[static_cast<size_t>(s)];
        if (st == TRANSCRIBE_OK && !I.raw.empty()) {
            SpeakerDecode r;
            st = build_speaker(pc, pm, params, s, I.raw, I.fed, r);
            if (st == TRANSCRIBE_OK) {
                per_spk.push_back(std::move(r));
            }
        }
    }
    for (auto & I : inst) {
        free_instance(I);
    }
    dec_us += ggml_time_us() - t_dec_start;
    return st;
}

transcribe_status run_multitalker(ParakeetSession *             pc,
                                  ParakeetModel *               pm,
                                  const float *                 pcm,
                                  int                           n_samples,
                                  const transcribe_run_params * params) {
    if (pm->diar == nullptr || !pm->diar_mel.has_value() || !pm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    transcribe::debug::init();

    const bool use_kernel = multitalker_use_kernel();

    const sf::SortformerEmbedded & diar = *pm->diar;

    // ---- 1. Embedded diarizer streaming forward ----
    std::vector<float> diar_mel;
    int                dm_mels = 0, dm_frames = 0;
    if (const transcribe_status st =
            pm->diar_mel->compute(pcm, static_cast<size_t>(n_samples), diar_mel, dm_mels, dm_frames);
        st != TRANSCRIBE_OK) {
        return st;
    }

    if (pc->sched == nullptr) {
        pc->sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(pm->plan.scheduler_list.size()),
                                           /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
        if (pc->sched == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet multitalker: ggml_backend_sched_new failed");
            return TRANSCRIBE_ERR_BACKEND;
        }
    }

    const ChunkGrid g = make_grid(pm->hparams, pm->hparams.enc_att_context_left, pm->hparams.enc_att_context_right);
    const sf::SortformerStreamParams P = diar_stream_params(pm, g);

    const int64_t t_diar_start = ggml_time_us();
    transcribe::debug::push_name_prefix("mt.diar");
    const transcribe_status dst = sf::run_diar_streaming_core(
        pc->diar_scratch, diar.hp, diar.conformer_hp, diar.conformer, diar.weights, pm->backend.c_str(), pc->sched,
        pc->n_threads, P, diar_mel.data(), dm_mels, dm_frames, pc);
    transcribe::debug::pop_name_prefix();
    if (dst != TRANSCRIBE_OK) {
        return dst;
    }
    const int64_t t_diar_us = ggml_time_us() - t_diar_start;

    const int                  sub    = diar.hp.enc_subsampling_factor > 0 ? diar.hp.enc_subsampling_factor : 8;
    const int                  n_spk  = diar.hp.max_speakers;
    const int                  T_diar = std::min(pc->diar_scratch.stream.total_n, (dm_frames + sub - 1) / sub);
    const std::vector<float> & probs  = pc->diar_scratch.stream.total_preds;  // row-major [total_n, n_spk]

    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_diar, n_spk };
        transcribe::debug::dump_host_f32("mt.diar.probs", probs.data(), static_cast<long long>(T_diar) * n_spk, shape,
                                         2, "diarize");
    }

    // ---- 2. Active speakers ----
    std::vector<int> active;
    note_active_speakers(probs.data(), n_spk, 0, T_diar, active);

    const double diar_ms_per_frame =
        1000.0 * static_cast<double>(diar.hp.frame_hop) / static_cast<double>(diar.hp.fe_sample_rate);

    if (active.empty()) {
        // No speech activity detected anywhere (reference behavior: every
        // chunk skipped). Empty-but-valid result.
        pc->t_mel_us    = 0;
        pc->t_encode_us = t_diar_us;
        pc->t_decode_us = 0;
        pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        pc->has_result  = true;
        return TRANSCRIBE_OK;
    }

    // ---- 3. Per-speaker decode passes ----
    //
    // Default is the bounded-memory streaming pass (chunk-faithful to the
    // reference loop). TRANSCRIBE_MULTITALKER_OFFLINE=1 selects the
    // offline whole-clip replay (O(T^2) attention buffers; parity/dump
    // tooling only).
    std::vector<SpeakerDecode> per_spk;
    per_spk.reserve(active.size());
    int64_t mel_us = 0, enc_us = 0, dec_us = 0;

    const bool offline_replay = (std::getenv("TRANSCRIBE_MULTITALKER_OFFLINE") != nullptr);
    if (!offline_replay) {
        if (const transcribe_status st = run_streaming_passes(pc, pm, pcm, n_samples, params, probs, T_diar, n_spk,
                                                              active, use_kernel, per_spk, mel_us, enc_us, dec_us);
            st != TRANSCRIBE_OK) {
            return st;
        }
    } else {
        for (const int s : active) {
            MultitalkerPass pass;
            pass.use_kernel = use_kernel;
            pass.spk.resize(static_cast<size_t>(T_diar));
            pass.bg.assign(static_cast<size_t>(T_diar), 0.0f);
            for (int t = 0; t < T_diar; ++t) {
                pass.spk[static_cast<size_t>(t)] = probs[static_cast<size_t>(t) * n_spk + s];
            }
            for (const int o : active) {
                if (o == s) {
                    continue;
                }
                for (int t = 0; t < T_diar; ++t) {
                    if (probs[static_cast<size_t>(t) * n_spk + o] > 0.5f) {
                        pass.bg[static_cast<size_t>(t)] = 1.0f;
                    }
                }
            }

            // Kernel-mode chunk gating (reference cache_gating=true,
            // cache_gating_buffer_size=2): a speaker's conformer/decoder only
            // ever sees chunks where _find_active_speakers selected it — max
            // of the accumulated preds over the last 2 chunks (incl. current)
            // > 0.5. Without this, a mostly-inactive speaker's pass
            // transcribes everyone else's speech (the trained kernels only
            // ever saw gated chunks). Masked mode hard-silences non-target
            // features instead, so it stays ungated here.
            if (use_kernel) {
                const int chunk    = P.chunk_len;  // 14 = nframes_per_chunk
                const int n_chunks = (T_diar + chunk - 1) / chunk;
                // +1 covers the encoder length trailing the diar frame count
                // at the tail; run_one_shot_inner clamps to the true length.
                pass.keep_enc_frames.reserve(static_cast<size_t>(T_diar) + 1);
                for (int k = 0; k < n_chunks; ++k) {
                    if (chunk_active(probs.data(), T_diar, n_spk, chunk, s, k)) {
                        const int end     = std::min((k + 1) * chunk, T_diar);
                        const int enc_end = (k == n_chunks - 1) ? end + 1 : end;
                        for (int t = k * chunk; t < enc_end; ++t) {
                            pass.keep_enc_frames.push_back(t);
                        }
                    }
                }
                if (pass.keep_enc_frames.empty()) {
                    continue;  // globally active but never chunk-selected
                }
            }

            // Decode at full timestamp resolution; the merged result is elided
            // to the caller's request at the end.
            transcribe_run_params sp = *params;
            sp.timestamps            = TRANSCRIBE_TIMESTAMPS_AUTO;

            pc->clear_result();
            char prefix[32];
            std::snprintf(prefix, sizeof(prefix), "mt.spk%d", s);
            transcribe::debug::push_name_prefix(prefix);
            const transcribe_status st = run_one_shot_inner(pc, pm, pcm, n_samples, &sp, &pass);
            transcribe::debug::pop_name_prefix();
            if (st != TRANSCRIBE_OK) {
                return st;
            }
            mel_us += pc->t_mel_us;
            enc_us += pc->t_encode_us;
            dec_us += pc->t_decode_us;

            SpeakerDecode r;
            r.speaker   = s;
            r.tokens    = std::move(pc->tokens);
            r.words     = std::move(pc->words);
            r.segments  = std::move(pc->segments);
            r.full_text = std::move(pc->full_text);
            r.raw_text  = std::move(pc->raw_text);

            // Gated pass timestamps are in the gathered timeline; map each
            // frame back through the keep list to absolute clip time.
            if (!pass.keep_enc_frames.empty()) {
                const auto & keep     = pass.keep_enc_frames;
                const double f_ms     = diar_ms_per_frame;  // 80 ms encoder cadence
                auto         remap_t0 = [&](int64_t t_ms) {
                    size_t f = static_cast<size_t>(std::max<int64_t>(0, t_ms) / static_cast<int64_t>(f_ms));
                    f        = std::min(f, keep.size() - 1);
                    return static_cast<int64_t>(keep[f]) * static_cast<int64_t>(f_ms);
                };
                auto remap_t1 = [&](int64_t t_ms) {
                    size_t f = static_cast<size_t>(std::max<int64_t>(0, t_ms - 1) / static_cast<int64_t>(f_ms));
                    f        = std::min(f, keep.size() - 1);
                    return static_cast<int64_t>(keep[f] + 1) * static_cast<int64_t>(f_ms);
                };
                for (auto & tk : r.tokens) {
                    tk.t0_ms = remap_t0(tk.t0_ms);
                    tk.t1_ms = remap_t1(tk.t1_ms);
                }
                for (auto & wd : r.words) {
                    wd.t0_ms = remap_t0(wd.t0_ms);
                    wd.t1_ms = remap_t1(wd.t1_ms);
                }
                for (auto & sg : r.segments) {
                    sg.t0_ms = remap_t0(sg.t0_ms);
                    sg.t1_ms = remap_t1(sg.t1_ms);
                }
            }
            per_spk.push_back(std::move(r));
        }
    }

    // ---- 4. Merge into one speaker-tagged result ----
    pc->clear_result();
    std::vector<const SpeakerDecode *> refs;
    refs.reserve(per_spk.size());
    for (const SpeakerDecode & r : per_spk) {
        refs.push_back(&r);
    }
    merge_speakers(pc, pm, params, refs, probs.data(), T_diar, n_spk, diar_ms_per_frame);

    pc->t_mel_us    = mel_us;
    pc->t_encode_us = enc_us + t_diar_us;
    pc->t_decode_us = dec_us;
    return TRANSCRIBE_OK;
}

// ---- Streaming API ------------------------------------------------------ //

struct MultitalkerStream {
    ChunkGrid                  grid;
    sf::SortformerStreamParams P;
    bool                       use_kernel        = true;
    int                        n_spk             = 0;
    int                        diar_sub          = 8;
    double                     diar_ms_per_frame = 80.0;

    // Embedded diarizer: its own scratch and scheduler, so its per-chunk
    // graphs never invalidate a speaker instance's retained graph.
    sf::DiarStreamScratch diar;
    ggml_backend_sched_t  diar_sched  = nullptr;
    int64_t               diar_chunks = 0;        // diarizer chunks run
    int64_t               diar_end    = 0;        // absolute diar mel column consumed up to

    int64_t                      asr_chunks = 0;  // grid index of the next ASR chunk
    std::vector<int>             active;          // slots seen above 0.5, ascending
    std::vector<SpeakerInstance> inst;            // one per diarizer slot

    // Sliding PCM window: pcm[0] is absolute sample pcm_start (a hop
    // multiple), trimmed after every feed to what the next chunk's windows
    // still read.
    std::vector<float> pcm;
    int64_t            pcm_start = 0;
    int64_t            n_input   = 0;

    std::vector<float> asr_mel, diar_mel, chunk_buf, win_buf;

    ~MultitalkerStream() {
        for (auto & I : inst) {
            free_instance(I);
        }
        transcribe::safe_sched_free(diar_sched);
        diar_sched = nullptr;
    }
};

void MultitalkerStreamDeleter::operator()(MultitalkerStream * s) const noexcept {
    delete s;
}

namespace {

int64_t samples_to_us(const ParakeetModel * pm, int64_t n_samples) {
    return n_samples * 1000000 / pm->hparams.fe_sample_rate;
}

// Chunk k is complete once its last mel column is final: that column's
// STFT window ends n_fft/2 samples past its centre, and until those samples
// arrive the zero pad at the buffer end stands in for them.
bool chunk_ready(const MultitalkerStream & S, const ParakeetModel * pm, int64_t k) {
    const int64_t last = S.grid.new_col(k) + S.grid.advance(k) - 1;
    return S.n_input >= last * pm->hparams.fe_hop_length + pm->hparams.fe_n_fft / 2;
}

// Diarizer chunk S.diar_chunks over absolute diar mel columns
// [win_lo, end), read from S.diar_mel (column 0 = absolute column col0).
transcribe_status diar_chunk(ParakeetSession *   pc,
                             ParakeetModel *     pm,
                             MultitalkerStream & S,
                             int                 n_mels,
                             int64_t             n_cols,
                             int64_t             col0,
                             int64_t             end) {
    const int64_t k      = S.diar_chunks;
    const int64_t stt    = S.diar_end;
    const int64_t win_lo = (k == 0) ? 0 : std::max<int64_t>(0, stt - S.P.feat_cache);
    const int     drop   = (k == 0) ? 0 : S.P.drop_pre_encode;
    const int     M      = static_cast<int>(end - win_lo);
    if (win_lo < col0 || end - col0 > n_cols || M <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet multitalker stream: diarizer window [%lld, %lld) outside buffer",
                static_cast<long long>(win_lo), static_cast<long long>(end));
        return TRANSCRIBE_ERR_BACKEND;
    }
    S.win_buf.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(M));
    for (int m = 0; m < n_mels; ++m) {
        const float * src = S.diar_mel.data() + static_cast<size_t>(m) * static_cast<size_t>(n_cols) +
                            static_cast<size_t>(win_lo - col0);
        std::copy(src, src + M, S.win_buf.data() + static_cast<size_t>(m) * M);
    }
    const sf::SortformerEmbedded & d = *pm->diar;
    const transcribe_status        st =
        sf::run_diar_stream_chunk(S.diar, d.hp, d.conformer_hp, d.conformer, d.weights, pm->backend.c_str(),
                                  S.diar_sched, pc->n_threads, S.P, S.win_buf.data(), n_mels, M, drop, 0, 0);
    if (st != TRANSCRIBE_OK) {
        return st;
    }
    S.diar_end = end;
    S.diar_chunks += 1;
    return TRANSCRIBE_OK;
}

// Rebuild the session's result snapshot from the instances (re-decoding
// only speakers whose tokens grew) and the first T_diar diarizer frames.
transcribe_status rebuild_snapshot(ParakeetSession * pc, ParakeetModel * pm, MultitalkerStream & S, int T_diar) {
    std::vector<const SpeakerDecode *> per_spk;
    for (const int s : S.active) {
        SpeakerInstance & I = S.inst[static_cast<size_t>(s)];
        if (I.raw.empty()) {
            continue;
        }
        if (!I.has_built || I.built_from != I.raw.size()) {
            if (const transcribe_status st = build_speaker(pc, pm, &pc->stream_run_params, s, I.raw, I.fed, I.built);
                st != TRANSCRIBE_OK) {
                return st;
            }
            I.built_from = I.raw.size();
            I.has_built  = true;
        }
        per_spk.push_back(&I.built);
    }
    merge_speakers(pc, pm, &pc->stream_run_params, per_spk, S.diar.stream.total_preds.data(), T_diar, S.n_spk,
                   S.diar_ms_per_frame);
    return TRANSCRIBE_OK;
}

void fill_update(const ParakeetSession * pc, transcribe_stream_update * update, bool changed, bool is_final) {
    if (update == nullptr) {
        return;
    }
    update->result_changed     = changed;
    update->is_final           = is_final;  // the dispatcher forces it on finalize as well
    update->revision           = pc->stream_revision;
    update->input_received_ms  = pc->stream_audio_input_us / 1000;
    update->audio_committed_ms = pc->stream_audio_committed_us / 1000;
    update->buffered_ms        = (pc->stream_audio_input_us - pc->stream_audio_committed_us) / 1000;
}

}  // namespace

transcribe_status multitalker_stream_begin(ParakeetSession *             pc,
                                           ParakeetModel *               pm,
                                           const transcribe_run_params * run_params,
                                           int                           att_context_left,
                                           int                           att_context_right) {
    (void) run_params;  // captured by the caller as pc->stream_run_params
    pc->mt_stream.reset();
    if (pm->diar == nullptr || !pm->diar_mel.has_value() || !pm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const auto & hp = pm->hparams;
    if (hp.enc_subsampling_factor <= 0 || hp.enc_stream_sampling_frames_first <= 0 ||
        hp.enc_stream_pre_encode_cache_size <= 0 || hp.enc_stream_drop_extra_pre_encoded < 0 ||
        pm->diar->hp.enc_subsampling_factor != hp.enc_subsampling_factor ||
        pm->diar_mel->num_mels() != pm->mel->num_mels()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet multitalker stream: bundle streaming geometry is incomplete or the diarizer's frame "
                "rate differs from the ASR encoder's");
        return TRANSCRIBE_ERR_GGUF;
    }

    std::unique_ptr<MultitalkerStream, MultitalkerStreamDeleter> S(new MultitalkerStream);
    S->grid       = make_grid(hp, att_context_left, att_context_right);
    S->P          = diar_stream_params(pm, S->grid);
    S->use_kernel = multitalker_use_kernel();
    S->n_spk      = pm->diar->hp.max_speakers;
    S->diar_sub   = pm->diar->hp.enc_subsampling_factor;
    S->diar_ms_per_frame =
        1000.0 * static_cast<double>(pm->diar->hp.frame_hop) / static_cast<double>(pm->diar->hp.fe_sample_rate);
    S->inst.resize(static_cast<size_t>(S->n_spk));
    S->diar.stream.reset(pm->diar->conformer_hp.enc_d_model);
    S->diar_sched = ggml_backend_sched_new(pm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(pm->plan.scheduler_list.size()),
                                           /*graph_size=*/8192, /*parallel=*/false, /*op_offload=*/true);
    if (S->diar_sched == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet multitalker stream: ggml_backend_sched_new failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    pc->mt_stream = std::move(S);
    pc->raw_tokens.clear();
    pc->stream_audio_input_samples = 0;
    return TRANSCRIBE_OK;
}

transcribe_status multitalker_stream_feed(ParakeetSession *          pc,
                                          ParakeetModel *            pm,
                                          const float *              pcm,
                                          int                        n_samples,
                                          transcribe_stream_update * update) {
    MultitalkerStream & S = *pc->mt_stream;
    if (pc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }
    S.pcm.insert(S.pcm.end(), pcm, pcm + n_samples);
    S.n_input += n_samples;
    pc->stream_audio_input_samples += n_samples;
    pc->stream_audio_input_us = samples_to_us(pm, pc->stream_audio_input_samples);

    const int hop     = pm->hparams.fe_hop_length;
    bool      changed = false;
    if (chunk_ready(S, pm, S.asr_chunks)) {
        // One mel pass per feed covers every chunk that became complete.
        const int64_t t_mel_start = ggml_time_us();
        int           n_mels = 0, n_cols = 0, d_mels = 0, d_cols = 0;
        if (const transcribe_status st =
                pm->mel->compute(S.pcm.data(), S.pcm.size(), S.asr_mel, n_mels, n_cols, pc->n_threads);
            st != TRANSCRIBE_OK) {
            return st;
        }
        if (const transcribe_status st =
                pm->diar_mel->compute(S.pcm.data(), S.pcm.size(), S.diar_mel, d_mels, d_cols, pc->n_threads);
            st != TRANSCRIBE_OK) {
            return st;
        }
        pc->t_mel_us += ggml_time_us() - t_mel_start;
        const int64_t col0 = S.pcm_start / hop;

        const int64_t t_enc_start = ggml_time_us();
        while (chunk_ready(S, pm, S.asr_chunks)) {
            if (pc->poll_abort()) {
                return TRANSCRIBE_ERR_ABORTED;
            }
            const int64_t   k = S.asr_chunks;
            const ChunkGrid g = S.grid;

            // Diarizer step k first: ASR chunk k gates on its preds.
            const int prev_n = S.diar.stream.total_n;
            if (const transcribe_status st = diar_chunk(pc, pm, S, d_mels, d_cols, col0, g.new_col(k) + g.advance(k));
                st != TRANSCRIBE_OK) {
                return st;
            }
            const int T_diar = S.diar.stream.total_n;
            note_active_speakers(S.diar.stream.total_preds.data(), S.n_spk, prev_n, T_diar, S.active);

            AsrChunk ch;
            ch.k      = k;
            ch.mel    = S.asr_mel.data();
            ch.n_mels = n_mels;
            ch.n_cols = n_cols;
            ch.col0   = col0;
            ch.c0_new = g.new_col(k);
            ch.c0     = ch.c0_new - g.history(k);
            ch.c1     = ch.c0_new + g.advance(k);
            if (const transcribe_status st =
                    run_asr_chunk(pc, pm, g, S.inst, S.active, ch, S.diar.stream.total_preds.data(), T_diar, S.n_spk,
                                  S.use_kernel, S.chunk_buf);
                st != TRANSCRIBE_OK) {
                return st;
            }
            S.asr_chunks += 1;
            changed = true;
        }
        pc->t_encode_us += ggml_time_us() - t_enc_start;

        // Keep what the next chunk's windows read, plus one hop: the
        // pre-emphasis filter reads the sample before every window.
        const int64_t need_col  = S.grid.new_col(S.asr_chunks) - S.grid.history(S.asr_chunks);
        int64_t       keep_from = need_col * hop - pm->hparams.fe_n_fft / 2 - hop;
        keep_from               = (std::max<int64_t>(0, keep_from) / hop) * hop;
        if (keep_from > S.pcm_start) {
            const int64_t drop = std::min<int64_t>(keep_from - S.pcm_start, static_cast<int64_t>(S.pcm.size()));
            S.pcm.erase(S.pcm.begin(), S.pcm.begin() + static_cast<ptrdiff_t>(drop));
            S.pcm_start += drop;
        }
    }

    if (changed) {
        const int64_t t_build_start = ggml_time_us();
        if (const transcribe_status st = rebuild_snapshot(pc, pm, S, S.diar.stream.total_n); st != TRANSCRIBE_OK) {
            return st;
        }
        pc->t_decode_us += ggml_time_us() - t_build_start;
        // Rows of a multitalker snapshot are rewritten as speakers grow, so
        // nothing is committed before finalize: committed_text stays empty
        // and the whole merged text is tentative.
        pc->n_committed_tokens   = 0;
        pc->n_committed_words    = 0;
        pc->n_committed_segments = 0;
        pc->stream_revision += 1;
    }
    pc->stream_audio_committed_us = samples_to_us(pm, S.grid.new_col(S.asr_chunks) * hop);
    fill_update(pc, update, changed, false);
    return TRANSCRIBE_OK;
}

transcribe_status multitalker_stream_finalize(ParakeetSession *          pc,
                                              ParakeetModel *            pm,
                                              transcribe_stream_update * update) {
    MultitalkerStream & S   = *pc->mt_stream;
    const int           hop = pm->hparams.fe_hop_length;

    // Final mels: the zero pad past the last sample is now the real end of
    // the stream, exactly as transcribe_run pads the whole clip.
    int n_mels = 0, n_cols = 0, d_mels = 0, d_cols = 0;
    if (!S.pcm.empty()) {
        const int64_t           t_mel_start = ggml_time_us();
        const transcribe_status ast =
            pm->mel->compute(S.pcm.data(), S.pcm.size(), S.asr_mel, n_mels, n_cols, pc->n_threads);
        const transcribe_status dst =
            pm->diar_mel->compute(S.pcm.data(), S.pcm.size(), S.diar_mel, d_mels, d_cols, pc->n_threads);
        // INVALID_ARG = too little audio for a frame: nothing left to run.
        if (ast != TRANSCRIBE_OK && ast != TRANSCRIBE_ERR_INVALID_ARG) {
            return ast;
        }
        if (dst != TRANSCRIBE_OK && dst != TRANSCRIBE_ERR_INVALID_ARG) {
            return dst;
        }
        if (ast != TRANSCRIBE_OK || dst != TRANSCRIBE_OK) {
            n_cols = 0;
            d_cols = 0;
        }
        pc->t_mel_us += ggml_time_us() - t_mel_start;
    }
    const int64_t col0     = S.pcm_start / hop;
    const int64_t n_frames = n_cols > 0 ? col0 + n_cols : 0;  // absolute ASR mel columns
    const int64_t feat_len = d_cols > 0 ? col0 + d_cols : 0;  // absolute diar mel columns

    const int64_t t_enc_start = ggml_time_us();

    // Remaining diarizer chunks, with transcribe_run's tail rule.
    while (S.diar_end < feat_len) {
        if (pc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const int64_t k   = S.diar_chunks;
        const int64_t end = (k == 0) ? std::min<int64_t>(S.P.feat_first_chunk, feat_len) :
                                       std::min<int64_t>(S.diar_end + S.P.chunk_len * S.diar_sub, feat_len);
        if (const transcribe_status st = diar_chunk(pc, pm, S, d_mels, d_cols, col0, end); st != TRANSCRIBE_OK) {
            return st;
        }
    }
    const int T_diar = std::min<int>(S.diar.stream.total_n, static_cast<int>((feat_len + S.diar_sub - 1) / S.diar_sub));
    note_active_speakers(S.diar.stream.total_preds.data(), S.n_spk, 0, T_diar, S.active);

    // Remaining ASR chunks, with transcribe_run's tail rule.
    for (int64_t k = S.asr_chunks;; ++k) {
        const ChunkGrid & g      = S.grid;
        const int64_t     c0_new = g.new_col(k);
        if (c0_new >= n_frames) {
            break;
        }
        const int64_t c0 = c0_new - g.history(k);
        const int64_t c1 = std::min<int64_t>(c0_new + g.advance(k), n_frames);
        if (c1 - c0 <= 0 || c1 - c0_new <= 0) {
            break;
        }
        if (pc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        AsrChunk ch;
        ch.k      = k;
        ch.mel    = S.asr_mel.data();
        ch.n_mels = n_mels;
        ch.n_cols = n_cols;
        ch.col0   = col0;
        ch.c0     = c0;
        ch.c1     = c1;
        ch.c0_new = c0_new;
        if (const transcribe_status st =
                run_asr_chunk(pc, pm, g, S.inst, S.active, ch, S.diar.stream.total_preds.data(), T_diar, S.n_spk,
                              S.use_kernel, S.chunk_buf);
            st != TRANSCRIBE_OK) {
            return st;
        }
        S.asr_chunks = k + 1;
    }
    pc->t_encode_us += ggml_time_us() - t_enc_start;

    const int64_t t_build_start = ggml_time_us();
    if (S.active.empty()) {
        // No speech anywhere: the empty-but-valid result transcribe_run gives.
        clear_result_rows(pc);
        pc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        pc->has_result  = true;
    } else if (const transcribe_status st = rebuild_snapshot(pc, pm, S, T_diar); st != TRANSCRIBE_OK) {
        return st;
    }
    pc->t_decode_us += ggml_time_us() - t_build_start;

    // The stream is over: release the instances and the diarizer now
    // rather than at the next begin. The snapshot lives on the session.
    pc->mt_stream.reset();

    pc->stream_audio_committed_us = pc->stream_audio_input_us;
    pc->n_committed_tokens        = static_cast<int>(pc->tokens.size());
    pc->n_committed_words         = static_cast<int>(pc->words.size());
    pc->n_committed_segments      = static_cast<int>(pc->segments.size());
    pc->stream_revision += 1;
    fill_update(pc, update, pc->has_result, true);
    return TRANSCRIBE_OK;
}

}  // namespace transcribe::parakeet
