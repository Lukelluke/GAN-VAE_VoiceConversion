class hyperparams:
    def __init__(self):
        #################################################################################
        #                                                                               #
        #                            Preprocess Hyperparams                             #
        #                                                                               #
        #################################################################################

        # ------------------------ Paths And Directory ------------------------ #
        self.CSV_PATH = './data/dataset.csv'
        # Format: spk|fpath
        # Example: xyb|0001.wav
        self.TRAIN_DATASET_PATH = './data/train_data'
        self.TEST_DATASET_PATH = './data/test_data'
        self.WAVS_PATH = './data/wavs'
        self.TRAIN_RATE = 1
        # ------------------------ Setting And Hyperparams -------------------- #
        self.MULTI_PROCESS = True
        self.CPU_RATE = 1
        self.SR = 24000
        self.N_FFT = 1024
        self.CODED_DIM = 60
        self.SPK_NUM = 14

        #################################################################################
        #                                                                               #
        #                               Train Hyperparams                               #
        #                                                                               #
        #################################################################################

        # ------------------------ Paths And Directory ------------------------ #
        self.LOG_DIR = './logs'
        self.MODEL_DIR = './models'
        # ------------------------ Setting And Hyperparams -------------------- #
        self.GPU_RATE = 0.9
        self.GPU_OPTION = 1
        self.NUM_EPOCHS = 150
        self.BATCH_SIZE = 8
        self.EMBED_SIZE = 256
        self.G_LR = 0.001
        self.D_LR = 0.001
        # VAE LOSS WEIGHT
        self.X0 = 2500
        self.K = 0.0025
        self.DROPOUT_RATE = 0.5
        self.VAE_GAUSSION_UNITS = 16
        self.PER_STEPS = 1000
        # ENCODER SETTING
        self.ENCODER_FILTER_NUMS = 512
        self.ENCODER_FILTER_SIZE = 30
        self.ENCODER_GRU_UNITS = 256
        # DECODER SETTING
        self.DECODER_FILTER_NUMS = 256
        self.DECODER_FILTER_SIZE = 30
        self.DECODER_GRU_UNITS = 256
        # DISCRIMINATOR
        self.LSTM_UNITS = 256

