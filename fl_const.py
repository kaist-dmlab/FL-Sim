import os

HOME_DIR_PATH = os.path.join(os.getenv('HOME'), 'FL-Sim')
DATA_DIR_PATH = os.path.join(HOME_DIR_PATH, 'data')
LOG_DIR_PATH = os.path.join(HOME_DIR_PATH, 'log')
PCAP_DIR_PATH = os.path.join(HOME_DIR_PATH, 'pcap')
EPOCH_CSV_POSTFIX = 'epoch.csv'
TIME_CSV_POSTFIX = 'time.csv'
C_LOG_FILE_NAME = 'c_log.txt'

DEFAULT_LINK_SPEED = 10
DEFAULT_PACKET_SIZE = 10240