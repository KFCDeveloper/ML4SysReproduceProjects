# coding=utf-8

class DOTE_Cons:
    HIST_LEN = 12 # correspond to DOTE hist_len # this assumes that we sample 12 TMs per hour

class RNN_Cons:
    NUM_FEATURES_PER_SHOT = 13  # 11 (obs) + 1 (action) + 1 (reward)
    RNN_HIDDEN_SIZE = 256  # equal to the max steps of each episode
    RNN_NUM_LAYERS = 2
    EMBEDDING_DIM = 32