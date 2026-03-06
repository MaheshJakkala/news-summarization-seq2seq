"""
inference.py
------------
Inference utilities for the trained Seq2Seq LSTM summarization model.

Provides:
  - decode_sequence(): greedy token-by-token decoding
  - seq2text(): convert integer sequence back to source text
  - seq2summary(): convert integer sequence back to target summary
  - summarize(): end-to-end pipeline from raw text → predicted headline
"""

import numpy as np
from src.preprocess import MAX_SUMMARY_LEN, START_TOKEN, END_TOKEN


def decode_sequence(
    input_seq: np.ndarray,
    encoder_model,
    decoder_model,
    target_word_index: dict,
    reverse_target_word_index: dict,
    max_summary_len: int = MAX_SUMMARY_LEN,
) -> str:
    """
    Greedy decoding: generate summary one token at a time.

    Steps:
    1. Run input_seq through encoder → get encoder outputs + (h, c) states.
    2. Seed decoder with START token.
    3. At each step, pick the argmax token from softmax output.
    4. Stop when END token is generated or max_summary_len is reached.

    Args:
        input_seq: shape (1, max_text_len) padded integer sequence.
        encoder_model: inference encoder.
        decoder_model: inference decoder.
        target_word_index: word → index mapping for target vocab.
        reverse_target_word_index: index → word mapping for target vocab.
        max_summary_len: maximum tokens to generate.

    Returns:
        str: decoded summary string (without start/end tokens).
    """
    e_out, e_h, e_c = encoder_model.predict(input_seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index[START_TOKEN]

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq, e_out, e_h, e_c], verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index.get(sampled_token_index, "")

        if sampled_token != END_TOKEN:
            decoded_sentence += " " + sampled_token

        if sampled_token == END_TOKEN or len(decoded_sentence.split()) >= (max_summary_len - 1):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        e_h, e_c = h, c

    return decoded_sentence.strip()


def seq2summary(input_seq: np.ndarray, target_word_index: dict, reverse_target_word_index: dict) -> str:
    """
    Convert an integer-encoded summary sequence back to a readable string.
    Filters out padding (0), START token, and END token.

    Args:
        input_seq: 1D integer array of target token indices.
        target_word_index: word → index for target vocab.
        reverse_target_word_index: index → word for target vocab.

    Returns:
        str: human-readable summary.
    """
    tokens = []
    for idx in input_seq:
        if idx != 0 and idx != target_word_index.get(START_TOKEN) and idx != target_word_index.get(END_TOKEN):
            tokens.append(reverse_target_word_index.get(idx, ""))
    return " ".join(tokens)


def seq2text(input_seq: np.ndarray, reverse_source_word_index: dict) -> str:
    """
    Convert an integer-encoded source text sequence back to a readable string.
    Filters out padding (0).

    Args:
        input_seq: 1D integer array of source token indices.
        reverse_source_word_index: index → word for source vocab.

    Returns:
        str: human-readable source text.
    """
    return " ".join(
        reverse_source_word_index.get(idx, "")
        for idx in input_seq if idx != 0
    )


def summarize(
    raw_text: str,
    x_tokenizer,
    encoder_model,
    decoder_model,
    target_word_index: dict,
    reverse_target_word_index: dict,
    max_text_len: int = 100,
    max_summary_len: int = MAX_SUMMARY_LEN,
) -> str:
    """
    End-to-end: raw news text → predicted headline.

    Args:
        raw_text: raw input string (not yet cleaned/tokenized).
        x_tokenizer: fitted source Tokenizer.
        encoder_model, decoder_model: inference models.
        target_word_index, reverse_target_word_index: target vocab dicts.
        max_text_len: encoder sequence length.
        max_summary_len: max tokens to generate.

    Returns:
        str: predicted headline.
    """
    from src.preprocess import text_strip
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = list(text_strip([raw_text]))[0]
    seq = x_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_text_len, padding="post")

    return decode_sequence(
        padded, encoder_model, decoder_model,
        target_word_index, reverse_target_word_index, max_summary_len
    )
