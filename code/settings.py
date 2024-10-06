"""
(train & tune)
    WARM_UP_RATE = 0.2
    BATCH_SIZE = 4 * nGPU(4) = 16

(train) BERT-base + Smooth K2 Loss(k = 2, threshold = 0.25) => 76.03
    BERT_LEARNING_RATE = 5e-5

(train) RoBERTa-base + Smooth K2 Loss(k = 3, threshold = 0.25) => 76.04
    BERT_LEARNING_RATE = 2e-5
    GRADIENT_CLIPPING = 2.0
    
(train) BERT-base + Translated ReLU Loss(k = 2.5, threshold = 0.25) => 74.28
    BERT_LEARNING_RATE = 2e-5

(train) RoBERTa-base + Translated ReLU Loss(k = 1, threshold = 0.25) => 74.28
    BERT_LEARNING_RATE = 2e-5
"""

EPOCH = 1
BATCH_SIZE = 4
MAX_SEQUENCE_LENGTH = 256

RATE_DECAY_FACTOR = 0.95 
BERT_LEARNING_RATE = 5e-5

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-8
ADAM_WEIGHT_DECAY = 1e-2

WARM_UP_RATE = 0.2
GRADIENT_CLIPPING = 10.0
EVALUATION_PER_STEP = 250
