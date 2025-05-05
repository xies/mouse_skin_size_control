import numpy as np
from skimage import io
from careamics import CAREamist
from careamics.config import create_n2v_configuration

from glob import glob
import pickle as pkl
from os import path
from tqdm import tqdm
from natsort import natsorted

#%%
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
filenames = natsorted(glob(path.join(dirname,'im_seq/t*.tif')))

# split train/val
train_img = io.imread(filenames[10])
test_img = io.imread(filenames[1])

config = create_n2v_configuration(
    experiment_name="mesa_mix_chanel_3d",
    data_type="array",
    axes="ZYXC",
    n_channels=3,
    patch_size=(16, 64, 64),
    batch_size=8,
    num_epochs=50,
    logger='tensorboard',
)

# Train
careamist = CAREamist(source=config,
                      work_dir=path.join(dirname,'im_seq_denoise'))
careamist.train(
    train_source=train_img,
    val_source=test_img,
)

with open(path.join(dirname,'im_seq_denoise/N2V.pkl'),'rb') as f:
    careamist = pkl.load(f)

for t,f in tqdm(enumerate(sorted(filenames))):
    im = io.imread(f)
    prediction = careamist.predict(
        source=im,
        tile_size=(32, 128, 128),
        tile_overlap=(8, 48, 48),
        batch_size=1,
        tta=False,
    )
    io.imsave(path.join(dirname,f'im_seq_denoise/t{t}_denoise.tif'),
        np.squeeze(prediction[0]))

# # Export the model
# careamist.export_to_bmz(
#     path_to_archive='mesa_3d_mix_chan_model.zip',
#     friendly_model_name="Mesa_3D_mix_channel",
#     input_array=train_img,
#     authors=[{"name": "xies", "affiliation": "Stanford"}],
#     general_description='Mesa et al',
#     data_description='Denoising on t=10',
)
