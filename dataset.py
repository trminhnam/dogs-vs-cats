import os
import cv2
from PIL import Image
import random
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root, datalist, INPUT_SIZE, BATCH_SIZE, shuffle=True, seed=42):
        super(DataGenerator, self).__init__()
        self.root = root
        self.INPUT_SIZE  = INPUT_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.shuffle = shuffle
        self.datalist = datalist
        random.seed(seed)
        self.image_list = list()
        self.mask_list = list()
        for i in self.datalist:
            image_path = os.path.join(self.root, f'data{i}/data{i}/CameraRGB/')
            mask_path = os.path.join(self.root, f'data{i}/data{i}/CameraSeg/')
            
            images = os.listdir(image_path)
            masks = os.listdir(mask_path)
            
            self.image_list += [image_path + i for i in images]
            self.mask_list += [mask_path + i for i in masks]
        
        #Sort the list 
        self.image_list = sorted(self.image_list)
        self.mask_list = sorted(self.mask_list)

        self.data_list = list(zip(self.image_list, self.mask_list))
        if(self.shuffle):
            random.shuffle(self.data_list)
        self.data_list = list(self.__chunks(self.data_list, self.BATCH_SIZE))

    def __len__(self):
        return len(self.data_list)

    def __chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
            
    def __getitem__(self, idx):
        images_batch = list()
        masks_batch  = list()
        for image_path, mask_path in self.data_list[idx]:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY if self.INPUT_SIZE[2] == 1 else cv2.COLOR_BGR2RGB)
            mask  = Image.open(mask_path)
            mask = np.array(mask)
            # mask  = cv2.imread(mask_path)
            image = cv2.resize(image, (self.INPUT_SIZE[0], self.INPUT_SIZE[1]), interpolation=cv2.INTER_NEAREST)
            mask  = cv2.resize(mask , (self.INPUT_SIZE[0], self.INPUT_SIZE[1]), interpolation=cv2.INTER_NEAREST)
            image = image.astype('float32')
            image /= 255.0          # normalization
            images_batch.append(image)  
            masks_batch.append(mask)

        images_batch = np.array(images_batch)
        masks_batch  = np.array(masks_batch)
        images_batch = images_batch.reshape(-1, self.INPUT_SIZE[0], self.INPUT_SIZE[1], 1 if self.INPUT_SIZE[2] == 1 else 3)
        masks_batch  = masks_batch.reshape(-1, self.INPUT_SIZE[0], self.INPUT_SIZE[1], 1)
        return images_batch, masks_batch

def Dataset(root, datalist, INPUT_SIZE, BATCH_SIZE):

    def process_path(image_path, mask_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        #this do the same as dividing by 255 to set the values between 0 and 1 (normalization)
        img = tf.image.convert_image_dtype(img, tf.float32) 
        
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=3)
        mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
        return img , mask

    def preprocess(image,mask) : 
        input_image = tf.image.resize(image,(INPUT_SIZE[0],INPUT_SIZE[1]),method='nearest')
        input_mask = tf.image.resize(mask,(INPUT_SIZE[0],INPUT_SIZE[1]),method='nearest')
        
        return input_image , input_mask
        
    image_list = list()
    mask_list = list()
    for i in datalist:
        image_path = os.path.join(root, f'data{i}/data{i}/CameraRGB/')
        mask_path = os.path.join(root, f'data{i}/data{i}/CameraSeg/')
        
        images = os.listdir(image_path)
        masks = os.listdir(mask_path)
        
        image_list += [image_path+i for i in images]
        mask_list += [mask_path+i for i in masks]

    #Sort the list 
    image_list = sorted(image_list)
    mask_list = sorted(mask_list)

    # image_list_dataset = tf.data.Dataset.list_files(image_list ,shuffle=False)
    # mask_list_dataset = tf.data.Dataset.list_files(mask_list , shuffle=False)

    images_filenames = tf.constant(image_list)
    masks_filenames = tf.constant(mask_list)

    dataset = tf.data.Dataset.from_tensor_slices((images_filenames ,masks_filenames))
    image_ds = dataset.map(process_path) # apply the preprocces_path function to our dataset
    processed_image_ds = image_ds.map(preprocess) # apply the preprocess function to our dataset

    processed_image_ds.batch(BATCH_SIZE)
    train_dataset = processed_image_ds.cache().shuffle(500).batch(BATCH_SIZE)
    return train_dataset

if __name__ == '__main__':
    DATA_ROOT = "F:/Semantic Segmentation for Self Driving Cars"
    INPUT_SIZE = (96, 96, 3)
    BATCH_SIZE = 1

    train_datagen = Dataset(DATA_ROOT, ['A'], INPUT_SIZE, BATCH_SIZE)
    for i in range(len(train_datagen)):
        data = train_datagen[i]