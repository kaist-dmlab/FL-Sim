import importlib

import fl_util

def main():
    args = fl_util.parse_args()
    
    algPath = 'algorithm.%s' % (args.algName)
    alg = importlib.import_module(algPath)
    Algorithm = getattr(alg, 'Algorithm')
    Algorithm(args).run()
    
if __name__ == '__main__':
    main()