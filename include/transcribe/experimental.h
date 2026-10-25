/*
 * include/transcribe/experimental.h - unstable additions outside the
 * ABI-pinned public surface (extensions.h does not include this header).
 */
#pragma once

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scores alternative spellings of a stretch of one speaker's words in the
 * session's stream: the model's log-probability (nats) of each candidate
 * text over the audio of `n_words` of speaker `speaker`'s words from
 * `first_word` (speaker ids as in the snapshot's segments, 1 for a stream
 * without speakers; words numbered per speaker in snapshot order), summed
 * over alignments from the decoder state before them. `transcribed` gets
 * the same for the words as transcribed. Works on the active stream and on
 * a finished one until the next transcribe_stream_begin. A score the
 * stream cannot give (audio no longer kept, a candidate the vocabulary
 * cannot spell, a span too long) is -INFINITY.
 *
 * Returns TRANSCRIBE_ERR_NOT_IMPLEMENTED for models without it and
 * TRANSCRIBE_ERR_INVALID_ARG for a speaker or word range the stream does
 * not have.
 */
TRANSCRIBE_API transcribe_status transcribe_stream_score_alternatives(struct transcribe_session * session,
                                                                      int                         speaker,
                                                                      int                         first_word,
                                                                      int                         n_words,
                                                                      const char * const *        candidates,
                                                                      int                         n_candidates,
                                                                      float *                     scores,
                                                                      float *                     transcribed);

#ifdef __cplusplus
}
#endif
