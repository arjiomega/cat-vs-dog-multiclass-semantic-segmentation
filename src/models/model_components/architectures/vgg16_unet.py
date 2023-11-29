import tensorflow as tf
import keras.layers as tfl

width, height = 224, 224

base_model = tf.keras.applications.vgg16.VGG16(
    include_top=False, input_shape=(width, height, 3))

layer_names = [
    'block1_conv2', # 224,224,64
    'block2_conv2', # 112,112,128
    'block3_conv3', # 56,56,256
    'block4_conv3', # 28,28,512
    'block5_conv3', # 14,14,512
    'block5_pool',  # 7,7,512
]

base_model_outputs = [base_model.get_layer(
    name).output for name in layer_names]
base_model.trainable = False # False

# change to mod_VGG_16
VGG_16 = tf.keras.models.Model(base_model.input,
                               base_model_outputs)

def VGG16_Unet_decoder(expansive_input,
                     contractive_input,
                     n_filters=32):

  # deconvolution
  # new shape = [(shape + 2(padding) - kernel_size)/stride]  + 1 (reverse this formula for deconvolution new shape)
  # doubles shape size
  up = tfl.Conv2DTranspose(filters = n_filters,
                         kernel_size = 3,
                         strides = 2,
                         padding = 'same')(expansive_input)

  # combine expansive and contractive conv
  merge = tfl.concatenate([up,contractive_input], axis = 3)

  # kernel_initializer="he_normal" helps to mitigate vanishing and exploding gradients problem
  conv = tfl.Conv2D(filters = n_filters,
                  kernel_size = 3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)

  conv = tfl.Conv2D(filters = n_filters,
                  kernel_size = 3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

  return conv

def VGG16_Unet(n_classes):
    """
        VGG16 = contract1,contract2,contract3, contract4, contract5
        contract5 -> conv2d -> conv2d

        'block1_conv2', # 224,224,64
        'block2_conv2', # 112,112,128
        'block3_conv3', # 56,56,256
        'block4_conv3', # 28,28,512
        'block5_conv3', # 14,14,512
        'block5_pool',  # 7,7,512
    """
    inputs = tf.keras.layers.Input(shape=(224,224,3))

    contract_blocks = VGG_16(inputs)

    # last maxpool to connect to conv2d
    cb_last = contract_blocks[-1]

    # cb_last contains 512 filters, (7x7)
    # applying padding="same" results in p = 1
    # new shape = [(shape + 2(padding) - kernel_size)/stride]  + 1
    # new shape = 7
    conv_last1 = tfl.Conv2D(filters = 1024,
                    kernel_size = 3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(cb_last)

    conv_last2 = tfl.Conv2D(filters = 1024,
                    kernel_size = 3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv_last1)


    # expansive path (concatenate )

    ## fifth block
    conv_5 = VGG16_Unet_decoder(expansive_input = conv_last2,
                        contractive_input = contract_blocks[4],
                        n_filters= 512)

    ## fourth block
    conv_4 = VGG16_Unet_decoder(expansive_input = conv_5,
                        contractive_input = contract_blocks[3],
                        n_filters= 512)

    ## third block

    conv_3 = VGG16_Unet_decoder(expansive_input = conv_4,
                        contractive_input = contract_blocks[2],
                        n_filters= 256)

    ## second block
    conv_2 = VGG16_Unet_decoder(expansive_input = conv_3,
                        contractive_input = contract_blocks[1],
                        n_filters= 128)

    ## first block

    conv_1 = VGG16_Unet_decoder(expansive_input = conv_2,
                        contractive_input = contract_blocks[0],
                        n_filters= 64)


    conv_out = tfl.Conv2D(filters = 64,
                    kernel_size = 3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(conv_1)



    conv_out = tfl.Conv2D(filters = n_classes, kernel_size = 1, padding='same',activation='softmax')(conv_out)

    model = tf.keras.Model(inputs=inputs,outputs=conv_out)

    return model