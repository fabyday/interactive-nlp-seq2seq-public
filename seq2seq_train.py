from gc import callbacks
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
from dataset import *



def create_model(src_vocab_size, tar_vocab_size, test=False):
    encoder_inputs = Input(shape=(None, src_vocab_size), name="enc_input")
    encoder_lstm = LSTM(units=256, return_state=True, name="enc_lstm")

    # encoder_outputs은 여기서는 불필요
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
    encoder_states = [state_h, state_c]



    decoder_inputs = Input(shape=(None, tar_vocab_size), name="dec_input")
    decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True, name="dec_lstm")

    # 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
    decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax', name="dec_softmax")
    decoder_outputs = decoder_softmax_layer(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    if(test):
        return model, encoder_inputs, encoder_states, decoder_inputs
    return model

def load_model(path, src_vocab_size, tar_vocab_size):
    model, encoder_inputs, encoder_states, decoder_inputs = create_model(src_vocab_size, tar_vocab_size, True)
    model.load_weights(path)
    return model, encoder_inputs, encoder_states, decoder_inputs


if __name__ == "__main__":

    Config.verbose = False

    encoder_input, decoder_input, decoder_target, \
    src_vocab_size, tar_vocab_size,\
    src_to_index, tar_to_index,\
    max_src_len,max_tar_len, lines  = load()
    model = create_model(src_vocab_size, tar_vocab_size)
    import os 
    ckpt_path = "./train"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_path, "cp.ckpt"),
                                                    save_weights_only=True,
                                                    verbose=1)



    model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=64, epochs=40, validation_split=0.2, callbacks=[cp_callback])



