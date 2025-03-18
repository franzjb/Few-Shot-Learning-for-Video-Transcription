'''
train_spm_ggl.py

--input=/home/frb6002/Documents/lipread_files_wav/merged_txt_files.txt
--model_prefix=spm_model_ggl.model
--vocab_size=1023
--character_coverage=1.0
--model_type=unigram
'''

import sentencepiece as spm

train_command = (
    '--input= / home / frb6002 / Documents / lipread_files_wav / pretrain'
    '--model_prefix=spm_model.model'
    '--vocab_size=1023'
    '--character_coverage=1.0'
    '--model_type=unigram'
)

spm.SentencePieceTrainer.train(train_command)