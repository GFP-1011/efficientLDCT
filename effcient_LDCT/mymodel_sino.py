import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime


class CTReconModel(tf.keras.Model):
    def __init__(self,scale):
        super(CTReconModel, self).__init__()
        # initialize  the sinLayers
        self.sinModule = []
        self.sinModule.append(
            tf.keras.layers.Conv2D(32, 5, padding='valid', name='sine_conv1', activation='tanh'))
        self.sinModule.append(
            tf.keras.layers.Conv2D(32, 5, padding='valid', name='sine_conv2', activation='tanh'))
        self.sinModule.append(
            tf.keras.layers.Conv2D(32, 5, padding='valid', name='sine_conv3', activation='tanh'))
        self.sinModule.append(
            tf.keras.layers.Conv2D(scale-1, kernel_size=1, padding='same', name='sine_conv4'))
        self.cut_sz = 6  # (5-1)/2 + (5-1)/2 + (5-1)/2 + (1-1)/2

        # initialize FbpLayer
        self.fbpModule = FbpLayer(name='fbp_layer')

        # initialize the CTLayers
        self.ctModule = []

        # initialize training mode and trainable variables
        self.trainMode = None
        self.trainable_var = []
    def inputModule(self, sin_in):
        # extend input for sinLayer
        channel = 360 // sin_in.shape[1]
        axis1_ext_end = tf.gather(sin_in, range(0, self.cut_sz), axis=1)
        axis1_ext_start = tf.gather(sin_in, range(sin_in.shape[1] - self.cut_sz, sin_in.shape[1]), axis=1)
        sin_in_ex = tf.concat([axis1_ext_start, sin_in, axis1_ext_end], 1)
        axis2_ext = tf.zeros([sin_in_ex.shape[0], sin_in_ex.shape[1], self.cut_sz, 1], tf.float32)
        # linear interpolation
        sin_map = ((channel-1) * sin_in / channel  + tf.gather(sin_in_ex, range(self.cut_sz + 1, self.cut_sz
                                                                    + sin_in.shape[1] + 1), axis=1) / channel )
        for i in range(channel-2):
            sin_interp = ((channel - 2 - i) * sin_in / channel + (i + 2) * tf.gather(sin_in_ex,range(self.cut_sz + 1, self.cut_sz
                                                                                           + sin_in.shape[1] + 1),                                                                         axis=1) / channel)
            sin_map = tf.concat([sin_map, sin_interp], 3)
        return tf.concat([axis2_ext, sin_in_ex, axis2_ext], 2), sin_map

    def call(self, train_batch, training=False):  # 定义正向传播过程
        # extend input for sinLayer
        if training:
            sin_in = train_batch[0]
        else:
            sin_in = train_batch
        sin_in_ex, sin_interp = self.inputModule(sin_in)
        # print('extended input: ', tf.shape(sin_in_ex))
        # 正弦域网络
        sin_out = self.sinModule[0](sin_in_ex)
        sin_out = self.sinModule[1](sin_out)
        sin_out = self.sinModule[2](sin_out)
        sin_out = self.sinModule[3](sin_out)
        sin_out = sin_interp + sin_out
        # combine the input and output of the sinLayer
        sin_map = self.concatSino(sin_out,sin_in)
        # fbp layer
        fbp_out = self.fbpModule(sin_map)  # 调用decode完成fbp
        # CT域网络
        ct_out = fbp_out

        model_out = [sin_out, fbp_out, ct_out, sin_interp]
        if training:
            return model_out, self.loss(train_batch[1], train_batch[2], model_out)
        else:
            return model_out

    def setTrainMode(self, sinLayers=True, fbpLayer=False, ctLayers=False):
        self.trainMode = (sinLayers, fbpLayer, ctLayers)
        # update my trainable variables
        self.trainable_var = []
        if sinLayers:
            self.trainable_var.extend(self.trainable_variables[0:8])
        if fbpLayer:
            self.trainable_var.extend(self.trainable_variables[8:11])
        if ctLayers:
            self.trainable_var.extend(self.trainable_variables[11:30])
        print('Training mode:', self.trainMode)

    def loss(self, sin_label, ct_label, model_out, weights=(1.0, 1.0, 1.0)):
        # model_out = [sin_out, fbp_out, ct_out, sin_interp]
        loss = 0
        if self.trainMode[0]:  # sine loss
            loss = weights[0] * tf.reduce_mean(tf.math.square(model_out[0] - sin_label))
        if self.trainMode[1]:  # fbp loss
            loss += weights[1] * tf.reduce_mean(tf.math.square(model_out[1] - ct_label))
        if self.trainMode[2]:  # ct loss
            loss += weights[2] * tf.reduce_mean(tf.math.square(model_out[2] - ct_label))
        return loss

    def concatSino(self, sin_interp, sin_in):
        channel = sin_interp.shape[3]
        sin_map = sin_in
        for i in range(channel):
            sin_map = tf.concat([sin_map, tf.expand_dims(sin_interp[:, :, :, i], 3)], 2)
        sin_map = tf.reshape(sin_map, [sin_in.shape[0], -1, sin_in.shape[2], 1])
        return sin_map


class FbpLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FbpLayer, self).__init__(**kwargs)
        # load AT, fbp_filter
        _rawAT = np.load('Data/My_AT_256.npz')
        _AT = tf.sparse.SparseTensor(_rawAT['arr_0'].astype('int32'), _rawAT['arr_1'].astype('float32'),
                                     _rawAT['arr_2'])  # 使用index,val,shape构建稀疏反投影矩阵 #!!!!!!!!!!!!!!!!!!!!!!1
        # self.A_Matrix = tf.sparse.transpose(_AT)
        self.A_Matrix = _AT
        _out_sz = round(np.sqrt(float(self.A_Matrix.shape[1])))
        self.out_shape = (_out_sz, _out_sz)
        # FBP时使用的滤波器
        self.fbp_filter = tf.Variable(_rawAT['arr_3'].astype('float32').reshape(-1, 1, 1),
                                      name=self.name + '/fbp_filter')
        self.scale = tf.Variable([10.0], name=self.name + '/scale')  # scale for CT image
        self.bias = tf.Variable([0.0], name=self.name + '/bias')

    def call(self, sin_fan):
        sin_sz = sin_fan.shape[1] * sin_fan.shape[2] * sin_fan.shape[3]
        sin_fan_flt = tf.nn.conv1d(sin_fan, self.fbp_filter, stride=1, padding='SAME')
        # print(tf.shape(sin_fan_flt))
        fbpOut = tf.sparse.sparse_dense_matmul(tf.reshape(sin_fan_flt, [-1, sin_sz]), self.A_Matrix)
        fbpOut = tf.reshape(fbpOut, [-1, self.out_shape[0], self.out_shape[1], 1])

        output = fbpOut * self.scale + self.bias
        return output


def train_step(data_batch, model, optimizer):
    # 开启上下文管理，参数watch_accessed_variables=False表示手动设置可训练参数
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_var)
        model_out, model_loss = model(data_batch, training=1)
    grads = tape.gradient(model_loss, model.trainable_var)  # 使用loss对后者进行求导
    optimizer.apply_gradients(zip(grads, model.trainable_var))
    return model_out, model_loss

def concatSino(sin_interp, sin_in):
    channel = sin_interp.shape[3]
    sin_map = sin_in
    for i in range(channel):
        sin_map = tf.concat([sin_map, tf.expand_dims(sin_interp[:, :, :, i], 3)], 2)
    sin_map = tf.reshape(sin_map, [sin_in.shape[0], -1, sin_in.shape[2], 1])
    return sin_map

def compute_psnr(data_batch, model_out):
    def psnr(ref, out):
        return tf.reduce_mean(tf.image.psnr(ref, out, max_val=tf.reduce_max(ref)-tf.reduce_min(ref))).numpy()

    sin_out, fbp_out, ct_out, sin_interp = model_out
    sin_in, sin_label, ct_label = data_batch
    sin_interps_large = concatSino(sin_interp,sin_in)
    sin_label_large = concatSino(sin_label,sin_in)
    sin_out_large = concatSino(sin_out,sin_in)
    # psnr for sin_in vs sin_label, sin_out vs sin_label, fbp_out vs ct_label, ct_out vs ct_label
    return [psnr(sin_label_large , sin_interps_large), psnr(sin_label_large, sin_out_large),psnr(ct_label, fbp_out), psnr(ct_label, ct_out)]


def show_image(train_data, model_out):
    sin_out, fbp_out, ct_out, sin_interp = model_out
    sin_in, sin_label, ct_label = train_data
    sin_label_large = concatSino(sin_label,sin_in)
    sin_out_large = concatSino(sin_out,sin_in)
    # sin_map = tf.reshape(tf.concat([sin_in, sin_out], 2), [sin_in.shape[0], -1, sin_in.shape[2], 1])
    for i in range(sin_out.shape[0]):
        plt.cla()
        plt.subplot(3, 2, 1)
        plt.title("sine label %d" % i, loc='center')
        plt.imshow(sin_label_large[i, :, :, :])
        plt.subplot(3, 2, 2)
        plt.title("sine output %d" % i, loc='center')
        plt.imshow(sin_out_large[i, :, :, :])

        plt.subplot(3, 2, 3)
        plt.title("CT label %d" % i, loc='center')
        plt.imshow(ct_label[i, :, :, :])
        plt.subplot(3, 2, 4)
        plt.title("CT output %d" % i, loc='center')
        plt.imshow(ct_out[i, :, :, :])

        plt.subplot(3, 2, 5)
        plt.title("CT label %d" % i, loc='center')
        plt.imshow(ct_label[i, :, :, :])
        plt.subplot(3, 2, 6)
        plt.title("FBP output %d" % i, loc='center')
        plt.imshow(fbp_out[i, :, :, :])
        plt.pause(0.5)


def load_traindata(trainDataDir="./Data/mymodel/My_data_256_180.npz", val_sz=2, c=2):
    # data = np.load(trainDataDir)
    train_data = np.load(trainDataDir)
    f_img = train_data['f_img'].astype('float32')  # 正弦域input
    ct_label = train_data['ct_label'].astype('float32')  # 正弦域label

    sin_input = np.expand_dims(f_img[:, 0::c, :], 3)
    sin_label = np.zeros([f_img.shape[0], int(f_img.shape[1] / c), f_img.shape[2], c - 1])
    for i in range(c - 1):
        sin_label[:, :, :, i] = f_img[:, i + 1::c, :]

    sin_label = sin_label.astype('float32')

    print('shapes of ct_label, sin_label, :', ct_label.shape)
    print('shape of sin_label:', sin_label.shape)
    print('shape of sin_input:', sin_input.shape)

    # 处理数据集，随机选取前dataset_sz-val_sz个作为数据并shuffle，剩下的val_sz个作为验证集
    dataset_sz = sin_input.shape[0]
    train_sz = dataset_sz - val_sz
    ids = np.random.permutation(dataset_sz)
    train_ids = ids[0:train_sz]
    val_ids = ids[train_sz:dataset_sz]
    val_data = [sin_input[val_ids], sin_label[val_ids], ct_label[val_ids]]

    train_data = tf.data.Dataset.from_tensor_slices((sin_input[train_ids], sin_label[train_ids], ct_label[train_ids])). \
        shuffle(dataset_sz - val_sz)
    print('shape of val_data:', val_data[0].shape, val_data[1].shape, val_data[2].shape)
    return train_data, val_data


def train(epoch=50, batch_sz=2,c =2):
    # load data
    train_data, val_data = load_traindata("./Data/mymodel/My_data_256_180.npz",c=c)

    # create and build model
    ct_model = CTReconModel(c)
    ct_model.build((1, int(360/c), 357, 1))  # Manually build the model and initialize the weights
    # ct_model.build((1, int(360/c), 605, 1))  # Manually build the model and initialize the weights
    ct_model.summary()
    # create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    current_time = datetime.datetime.now()
    plt.ion()
    for i in range(epoch):
        # set training mode
        if i <= 200:
            ct_model.setTrainMode(True, False, False)
        # if 5 < i <= 30:
        #     # optimizer.learning_rate = 0.01
        #     ct_model.setTrainMode(False, False, True)
        # if 30 < i <= 50:
        #     optimizer.learning_rate = 0.0001
        #     ct_model.setTrainMode(True, True, True)

        for iterNo, data_batch in enumerate(train_data.batch(batch_sz)):
            model_out, model_loss = train_step(data_batch, ct_model, optimizer)
            if iterNo%10==0 & i%20==0:
                psnr_train = compute_psnr(data_batch, model_out)  # psnr for training dataset
                print(iterNo, "/", i, ":", model_loss.numpy(), "; psnr_train:", psnr_train)

        # psnr for valid dataset
        val_out = ct_model(val_data[0], training=0)
        psnr_valid = compute_psnr(val_data, val_out)
        # show_image(val_data, val_out)
        print("epoch: ", i, ":", "psnr_valid", psnr_valid)
        ckpt = './256x256/' + str(c) + 'x_weights_' + str(i) + '_epoch/ckpt'
        ct_model.save_weights(ckpt)
    plt.ioff()
    plt.show()

    print('the time of training' + str(datetime.datetime.now() - current_time))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train(epoch=200, batch_sz=2,c = 2)
