import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, name="DiceLoss", smooth=1e-7, gamma=2, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
        self.name = name
        self.smooth = smooth
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # shape (batch_size,width,height,n_classes)
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        
        # shape (batch_size,n_classes)
        numerator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true),axis=(1,2)) + self.smooth
        denominator = tf.reduce_sum(y_pred ** self.gamma,axis=(1,2)) + tf.reduce_sum(y_true ** self.gamma,axis=(1,2)) + self.smooth
        
        # get the average dice loss of all the classes
        # shape (batch_size,)
        dice_loss_per_batch = 1 - tf.reduce_mean(tf.divide(numerator, denominator),axis=-1)

        return dice_loss_per_batch
    
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.name = 'FocalLoss'
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha
        self.BCE = tf.keras.losses.BinaryCrossentropy()

    def call(self,y_true,y_pred):

        # shape (batch_size,width,height,n_classes)
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)

        # get BCE
        loss_bce = self.BCE(y_true, y_pred)

        # pt
        pt = (y_true*y_pred) + ((1-y_true)*(1-y_pred))

        # alpha factor
        alpha_t = (y_true*self.alpha) + ((1-y_true)*(1-self.alpha))

        # modulating factor
        mod_f = (1-pt)**self.gamma
        #tf.print("mod_f shape: ",mod_f.shape)
        loss_fl = alpha_t * mod_f * loss_bce

        class_FL = tf.reduce_mean(loss_fl,axis=(1,2))
        class_FL = tf.reduce_mean(class_FL,axis=-1)
        batch_FL = tf.reduce_mean(class_FL)

        return batch_FL
    
class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.name = 'TverskyLoss'
        self.smooth = smooth
        self.beta = beta

    def call(self,y_true,y_pred):

        # shape (batch_size,width,height,n_classes)
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)

        # shape (batch_size,1,1,n_classes)
        numerator = tf.reduce_sum(tf.multiply(y_pred, y_true),axis=(1,2),keepdims=True) + self.smooth

        denominator = tf.reduce_sum(tf.multiply(y_pred, y_true),axis=(1,2),keepdims=True) + \
                      tf.reduce_sum(self.beta * tf.multiply((1-y_true),y_pred),axis=(1,2),keepdims=True) + \
                      tf.reduce_sum((1-self.beta)  * tf.multiply(y_true,(1-y_pred)),axis=(1,2),keepdims=True) + \
                      self.smooth
        
        # shape (batch_size,1,1,1)
        TL_per_class = 1 - tf.reduce_mean(tf.divide(numerator, denominator),axis=-1)

        # shape (1,)
        batch_TL = tf.reduce_mean(TL_per_class)

        return batch_TL
    
class FocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, beta=0.7, gamma=2):
        super(FocalTverskyLoss, self).__init__()
        self.name = 'FocalTverskyLoss'
        self.smooth = smooth
        self.beta = beta
        self.gamma = gamma

    def call(self,y_true,y_pred):

        # shape (width,height,n_classes)
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)

        # shape (1,1,n_classes)
        numerator = tf.reduce_sum(tf.multiply(y_pred, y_true),axis=(0,1),keepdims=True) + self.smooth

        denominator = tf.reduce_sum(tf.multiply(y_pred, y_true),axis=(0,1),keepdims=True) + \
                      tf.reduce_sum(self.beta * tf.multiply((1-y_true),y_pred),axis=(0,1),keepdims=True) + \
                      tf.reduce_sum((1-self.beta)  * tf.multiply(y_true,(1-y_pred)),axis=(0,1),keepdims=True) + \
                      self.smooth
        
        FTL = tf.reduce_mean(1 - tf.divide(numerator, denominator))**self.gamma

        return FTL