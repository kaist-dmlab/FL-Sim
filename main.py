import importlib
import traceback
import os

import tensorflow as tf

import random
import numpy as np

import fl_const
import fl_util

alg = None

def main():
    args = fl_util.parseArgs()
    
    # Suppress TensorFlow Logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
#     tf.logging.set_verbosity(tf.logging.ERROR)

    # Initialize C log file
    open(os.path.join(fl_const.LOG_DIR_NAME, fl_const.C_LOG_FILE_NAME), 'w')
    
    # tf.set_random_seed 는 model.abc 에서 Graph 생성 후 수행
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    
    algPackagePath = 'algorithm.' + args.algName
    algModule = importlib.import_module(algPackagePath)
    Algorithm = getattr(algModule, 'Algorithm')
    global alg
    alg = Algorithm(args)
    alg.run()
    
if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        alg.finalize()