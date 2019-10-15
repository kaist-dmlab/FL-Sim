MODEL_NAME = 'svm' # svm, sr / cnn / 2nn
DATA_NAME = 'cifar10' # mnist-o / mnist-f / cifar10
NUM_NODES = 100
NUM_EDGES = 10

FED_AVG_TAU_1 = 1
HIER_FAVG_TAU_1 = 1
HIER_FAVG_TAU_2 = 1

################################################# Model parameter

CH_FEDAVG_NUM_SAMPLE_NODE = 100
CH_FEDAVG_MAX_STEADY_STEPS = 5
CH_FEDAVG_IID_GROUPING_INTERVAL = 10000

# Terminating Condition
MAX_EPOCH = 1000

# Learning Rates
LR_DECAY_RATE = 0.99

# SGD
BATCH_SIZE = 128

# Evaluation
NUM_TEST_ITERS = int(10000 / BATCH_SIZE)

if MODEL_NAME == 'sr':
    SGD_ENABLED = False
    MAX_TIME = 50
    if DATA_NAME == 'mnist-o' or DATA_NAME == 'mnist-f':
        MODEL_SIZE = 7850
        LR_INITIAL = 0.1
#     if DATA_NAME == 'cifar10': # 'sr', 'cifar10' 의 경우는 데이터에 비해 모델이 너무 단순해서 실험에서 제외
#         MODEL_SIZE = 30730
#         LR_INITIAL = 0.01
    else:
        raise Exception(MODEL_NAME, DATA_NAME)
elif MODEL_NAME == '2nn':
    SGD_ENABLED = True
    MAX_TIME = 500
    LR_INITIAL = 0.1
    if DATA_NAME == 'mnist-o' or DATA_NAME == 'mnist-f':
        MODEL_SIZE = 199210
    else:
        raise Exception(MODEL_NAME, DATA_NAME)
elif MODEL_NAME == 'cnn':
    SGD_ENABLED = True
    if DATA_NAME == 'mnist-o' or DATA_NAME == 'mnist-f':
        MAX_TIME = 1000
        MODEL_SIZE = 369098
        LR_INITIAL = 0.1
    elif DATA_NAME == 'cifar10':
        MAX_TIME = 10000
        MODEL_SIZE = 519754
        LR_INITIAL = 0.01
    else:
        raise Exception(MODEL_NAME, DATA_NAME)
elif MODEL_NAME == 'svm':
    SGD_ENABLED = False
    MAX_TIME = 25
    LR_INITIAL = 0.01
    if DATA_NAME == 'cifar10':
        MODEL_SIZE = 3073
    else:
        raise Exception(MODEL_NAME, DATA_NAME)
else:
    raise Exception(MODEL_NAME)