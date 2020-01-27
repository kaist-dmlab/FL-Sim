import importlib
import traceback
import os
import tensorflow as tf

import fl_util

alg = None

def main():
    args = fl_util.parseArgs()
    
    # Suppress TensorFlow Logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
#     tf.logging.set_verbosity(tf.logging.ERROR)
    
    fl_util.initialize(args.seed)
    
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