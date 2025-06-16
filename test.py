from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen = ImageDataGenerator(rescale=1./255)
data = gen.flow_from_directory("dataset/train", target_size=(224, 224))

# OR
# data = gen.flow_from_directory("dataset/train", target_size=(224,224))
