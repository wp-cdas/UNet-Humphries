from model import *
from data import *
from PIL import Image
import cv2

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
trainGene = trainGenerator(4,
                           '/data/spacenet/road/AllTrain',
                           'PAN-8BIT',
                           'PAN-GT',
                           data_gen_args,
                           save_to_dir = None)
validGene = trainGenerator(4,
                           '/data/spacenet/road/AllValid',
                           'PAN-8BIT',
                           'PAN-GT',
                           data_gen_args,
                           save_to_dir = None)
model = unet()
model.load_weights('/lfs/jonas/oldunet/roadweights.hdf5')
model_checkpoint = ModelCheckpoint('/lfs/jonas/oldunet/roadweights.hdf5', 
                                   monitor='val_loss',
                                   verbose=1, 
                                   save_best_only=True)
model.fit(trainGene,
          steps_per_epoch=2000,
          epochs=100,
          callbacks=[model_checkpoint], 
          validation_data=validGene, 
          validation_steps=2,
          verbose=1,
          max_queue_size = 16,
          workers = 8,
          use_multiprocessing = True)