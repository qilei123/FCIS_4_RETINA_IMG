import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'fcis'))
from config.config import config
import test

if __name__ == "__main__":
    test_epoch_list = [43]
    for test_epoch in test_epoch_list:
        print 'test_epoch:'+str(test_epoch)
        config.TEST.test_epoch = test_epoch
        test.main()
