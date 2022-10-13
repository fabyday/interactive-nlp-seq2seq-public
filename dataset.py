import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# dataset preprocess 

class Config:
    verbose = False


def vprint(*args):
    if Config.verbose:
        print(*args)

def load():
    lines = pd.read_csv('spa-eng/spa.txt', names=['src', 'tar', 'lic'], sep='\t')
    del lines['lic']
    vprint('전체 샘플의 개수 :',len(lines))




    lines = lines.loc[:, 'src':'tar']
    lines = lines[0:60000] # 6만개만 저장
    lines.sample(10)


    lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')
    lines.sample(10)



    # 문자 집합 구축
    src_vocab = set()
    for line in lines.src: # 1줄씩 읽음
        for char in line: # 1개의 문자씩 읽음
            src_vocab.add(char)

    tar_vocab = set()
    for line in lines.tar:
        for char in line:
            tar_vocab.add(char)


    src_vocab_size = len(src_vocab)+1
    tar_vocab_size = len(tar_vocab)+1
    vprint('source 문장의 char 집합 :',src_vocab_size)
    vprint('target 문장의 char 집합 :',tar_vocab_size)



    src_vocab = sorted(list(src_vocab))
    tar_vocab = sorted(list(tar_vocab))
    vprint(src_vocab[45:75])
    vprint(tar_vocab[45:75])


    src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
    tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
    vprint(src_to_index)
    vprint(tar_to_index)
    
    encoder_input = []

    # 1개의 문장
    for line in lines.src:
        encoded_line = []
        # 각 줄에서 1개의 char
        for char in line:
            # 각 char을 정수로 변환
            encoded_line.append(src_to_index[char])
        encoder_input.append(encoded_line)
    vprint('source 문장의 정수 인코딩 :',encoder_input[:5])


    decoder_input = []
    for line in lines.tar:
        encoded_line = []
        for char in line:
            encoded_line.append(tar_to_index[char])
        decoder_input.append(encoded_line)
    vprint('target 문장의 정수 인코딩 :',decoder_input[:5])


    decoder_target = []
    for line in lines.tar:
        timestep = 0
        encoded_line = []
        for char in line:
            if timestep > 0:
                encoded_line.append(tar_to_index[char])
            timestep = timestep + 1
        decoder_target.append(encoded_line)
    vprint('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])

    max_src_len = max([len(line) for line in lines.src])
    max_tar_len = max([len(line) for line in lines.tar])
    vprint('source 문장의 최대 길이 :',max_src_len)
    vprint('target 문장의 최대 길이 :',max_tar_len)


    encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
    decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
    decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')


    encoder_input = to_categorical(encoder_input)
    decoder_input = to_categorical(decoder_input)
    decoder_target = to_categorical(decoder_target)


    return encoder_input, decoder_input, decoder_target, src_vocab_size, tar_vocab_size,\
        src_to_index, tar_to_index, max_src_len,max_tar_len, lines 

if __name__ == "__main__":
    Config.verbose = True
    load()