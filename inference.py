import os
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread

def custom_predict(test_folder, checkpoint_filepath=None):
    # predicted_images
    dest_folder = r'./preds/'
    os.makedirs(dest_folder,exist_ok=True)
    for image_file in tqdm(os.listdir(test_folder)):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image = tf.keras.preprocessing.image.load_img(test_folder + '/' + image_file, target_size=(256,448))
                image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255
                #print(image.shape)
                x_inp=image.reshape(1,256,448,3)
                autoencoder_best = load_model(checkpoint_filepath)
                #autoencoder_best = autoencoder
                pred = autoencoder_best.predict(x_inp)
                result = pred.reshape(256,448,3)
                img = Image.fromarray((result*255).astype(np.uint8))
                #img_name = "img_{}_{}.png".format(i,j)
                img.save(dest_folder + '/' + image_file)
    #clean_frames = np.array(clean_frames)
    #return clean_frames, labels

    # Example usage:
test_folder = r'../custom_test/blur/'
ckpt_path = r'./sanika/autoencoder_ckpt.keras'
custom_predict(test_folder, ckpt_path)

def psnr_between_folders(folder1, folder2):
    psnr_values = []
    
    # Get list of filenames in folder1
    filenames = os.listdir(folder1)
    
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read corresponding images from both folders
            img_path1 = os.path.join(folder1, filename)
            img_path2 = os.path.join(folder2, filename)
            img1 = imread(img_path1)
            img2 = imread(img_path2)
            
            # Compute PSNR between corresponding images
            psnr = peak_signal_noise_ratio(img1, img2)
            psnr_values.append(psnr)
    
    # Compute average PSNR across all images
    avg_psnr = sum(psnr_values) / len(psnr_values)

    print (len(psnr_values))
    
    return avg_psnr

# Example usage:
folder1 = r'./custom_test/sharp/'
folder2 = r'./sanika/preds/'

avg_psnr = psnr_between_folders(folder1, folder2)
print(f"Average PSNR between corresponding images: {avg_psnr} dB")