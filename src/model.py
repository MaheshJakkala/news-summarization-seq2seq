"""
model.py
--------
Stacked Seq2Seq LSTM encoder-decoder architecture for abstractive news summarization.

Architecture:
  Encoder:
    Input(max_text_len=100)
    → Embedding(x_voc=33412, dim=200)
    → LSTM(300, return_sequences=True, dropout=0.4)   [encoder_lstm1]
    → LSTM(300, return_sequences=True, dropout=0.4)   [encoder_lstm2]
    → LSTM(300, return_sequences=True, dropout=0.4)   [encoder_lstm3]
    → (encoder_outputs, state_h, state_c)

  Decoder:
    Input(None,)
    → Embedding(y_voc=11581, dim=200)
    → LSTM(300, return_sequences=True, dropout=0.4, initial_state=[state_h, state_c])
    → TimeDistributed(Dense(y_voc, activation='softmax'))

Total trainable parameters: 15,129,281
Compiled with: optimizer='rmsprop', loss='sparse_categorical_crossentropy'

Note: No attention mechanism — this is a vanilla Seq2Seq baseline.
      See Future Work in README for attention-based improvements.
"""

from tensorflow.keras.layers import (
    Input, LSTM, Embedding, Dense, TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
LATENT_DIM = 300
EMBEDDING_DIM = 200


# ─────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────
def build_seq2seq_model(x_voc: int, y_voc: int, max_text_len: int, max_summary_len: int):
    """
    Build and compile the stacked Seq2Seq LSTM model.

    Args:
        x_voc (int): Source vocabulary size (encoder side).
        y_voc (int): Target vocabulary size (decoder side).
        max_text_len (int): Maximum encoder input length.
        max_summary_len (int): Maximum decoder output length (unused in build, kept for docs).

    Returns:
        model: Compiled Keras Model.
        encoder_inputs: Encoder input tensor (needed to build inference encoder).
        decoder_inputs: Decoder input tensor.
        dec_emb_layer: Decoder embedding layer (shared for inference decoder).
        decoder_lstm: Decoder LSTM layer (shared for inference decoder).
        decoder_dense: Decoder dense layer (shared for inference decoder).
        encoder_outputs: Full encoder output sequence.
        state_h, state_c: Final encoder hidden/cell states passed to decoder.
    """
    K.clear_session()

    # ── Encoder ──────────────────────────────────────────────────────────
    encoder_inputs = Input(shape=(max_text_len,), name="encoder_input")
    enc_emb = Embedding(x_voc, EMBEDDING_DIM, trainable=True, name="encoder_embedding")(encoder_inputs)

    encoder_lstm1 = LSTM(
        LATENT_DIM, return_sequences=True, return_state=True,
        dropout=0.4, recurrent_dropout=0.4, name="encoder_lstm1"
    )
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

    encoder_lstm2 = LSTM(
        LATENT_DIM, return_sequences=True, return_state=True,
        dropout=0.4, recurrent_dropout=0.4, name="encoder_lstm2"
    )
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    encoder_lstm3 = LSTM(
        LATENT_DIM, return_sequences=True, return_state=True,
        dropout=0.4, recurrent_dropout=0.4, name="encoder_lstm3"
    )
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    # ── Decoder ──────────────────────────────────────────────────────────
    decoder_inputs = Input(shape=(None,), name="decoder_input")

    dec_emb_layer = Embedding(y_voc, EMBEDDING_DIM, trainable=True, name="decoder_embedding")
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(
        LATENT_DIM, return_sequences=True, return_state=True,
        dropout=0.4, recurrent_dropout=0.2, name="decoder_lstm"
    )
    decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
        dec_emb, initial_state=[state_h, state_c]
    )

    decoder_dense = TimeDistributed(
        Dense(y_voc, activation="softmax"), name="decoder_dense"
    )
    decoder_outputs = decoder_dense(decoder_outputs)

    # ── Full Model ────────────────────────────────────────────────────────
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq")
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    return (
        model,
        encoder_inputs, decoder_inputs,
        dec_emb_layer, decoder_lstm, decoder_dense,
        encoder_outputs, state_h, state_c,
    )


# ─────────────────────────────────────────────
# Inference Models
# ─────────────────────────────────────────────
def build_inference_models(
    encoder_inputs, decoder_inputs,
    dec_emb_layer, decoder_lstm, decoder_dense,
    encoder_outputs, state_h, state_c,
    max_text_len: int,
):
    """
    Build separate encoder and decoder inference models from trained layers.

    These share weights with the training model — no re-training needed.

    Args:
        (All shared Keras tensors/layers from build_seq2seq_model outputs)
        max_text_len: encoder sequence length

    Returns:
        encoder_model: Model(encoder_inputs → [encoder_outputs, state_h, state_c])
        decoder_model: Model([decoder_inputs, hidden_state, h, c] → [outputs, h2, c2])
    """
    # Encoder inference
    encoder_model = Model(
        inputs=encoder_inputs,
        outputs=[encoder_outputs, state_h, state_c],
        name="encoder_inference"
    )

    # Decoder inference
    decoder_state_input_h = Input(shape=(LATENT_DIM,), name="decoder_state_h")
    decoder_state_input_c = Input(shape=(LATENT_DIM,), name="decoder_state_c")
    decoder_hidden_state_input = Input(shape=(max_text_len, LATENT_DIM), name="decoder_hidden")

    dec_emb2 = dec_emb_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )
    decoder_outputs2 = decoder_dense(decoder_outputs2)

    decoder_model = Model(
        inputs=[decoder_inputs, decoder_hidden_state_input,
                decoder_state_input_h, decoder_state_input_c],
        outputs=[decoder_outputs2, state_h2, state_c2],
        name="decoder_inference"
    )

    return encoder_model, decoder_model
