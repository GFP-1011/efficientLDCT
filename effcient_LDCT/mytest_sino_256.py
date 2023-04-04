import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as s_ssim
import cv2

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
        self.ctModule.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='ct_conv0', activation=tf.nn.relu))
        self.M2 = 5
        for l_i in range(1, self.M2 + 1):
            self.ctModule.append(tf.keras.layers.Conv2D(64, 5, padding='same', name='ct_conv%d' % l_i, use_bias=False))
            self.ctModule.append(tf.keras.layers.BatchNormalization())
            self.ctModule.append(tf.keras.layers.ReLU())
        l_i += 1
        self.ctModule.append(tf.keras.layers.Conv2D(1, 5, name='ct_conv%d' % l_i, padding='same'))

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
        ct_out = self.ctModule[0](fbp_out)
        sum_res = ct_out
        for k in range(1, self.M2 + 1):
            for j in range(0, 3):
                ct_out = self.ctModule[3 * k + j - 2](ct_out)
            sum_res = sum_res + ct_out
        ct_out = self.ctModule[3 * self.M2 + 1](sum_res / self.M2) + fbp_out

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


def compute_psnr(data_batch, model_out):
    def psnr(ref, out):
        return tf.reduce_mean(tf.image.psnr(ref, out, max_val=tf.reduce_max(ref)-tf.reduce_min(ref))).numpy()

    sin_out, fbp_out, ct_out, sin_interp = model_out
    sin_in, sin_label, ct_label = data_batch
    sin_interps_large = tf.reshape(tf.concat([sin_in, sin_interp], 3), [sin_in.shape[0], -1, sin_in.shape[2], 1])
    sin_label_large = tf.reshape(tf.concat([sin_in, sin_label], 3), [sin_in.shape[0], -1, sin_in.shape[2], 1])
    sin_out_large = tf.reshape(tf.concat([sin_in, sin_out], 3), [sin_in.shape[0], -1, sin_in.shape[2], 1])
    # psnr for sin_in vs sin_label, sin_out vs sin_label, fbp_out vs ct_label, ct_out vs ct_label
    return [psnr(sin_label_large , sin_interps_large), psnr(sin_label_large, sin_out_large),psnr(ct_label, fbp_out), psnr(ct_label, ct_out)]


def compute_ssim(data_batch, model_out):
    def ssim(ref, out):
        ref = tf.squeeze(ref).numpy()
        out = tf.squeeze(out).numpy()
        temp = []
        for i in range(np.shape(ref)[0]):
            temp.append(s_ssim(ref[i, :, :], out[i, :, :]))
        return np.mean(temp)

    sin_out, fbp_out, ct_out, sin_interp = model_out
    sin_in, sin_label, ct_label = data_batch

    return [ssim(ct_label, fbp_out), ssim(ct_label, ct_out)]

def show_image(train_data, model_out):
    sin_out, fbp_out, ct_out, sin_interp = model_out
    sin_in, sin_label, ct_label = train_data
    # sin_map = tf.reshape(tf.concat([sin_in, sin_out], 2), [sin_in.shape[0], -1, sin_in.shape[2], 1])
    for i in range(sin_out.shape[0]):
        plt.cla()
        plt.subplot(3, 2, 1)
        plt.title("sine label %d" % i, loc='center')
        plt.imshow(sin_label[i, :, :, :])

        plt.subplot(3, 2, 2)

        plt.title("sine output %d" % i, loc='center')
        plt.imshow(sin_out[i, :, :, :])

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


def load_testdata(trainDataDir="./Data/mytest/My_data_512_200.npz", c=2):
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
    train_data = tf.data.Dataset.from_tensor_slices((sin_input, sin_label, ct_label))
    return train_data


def save_data(data_batch, test_out,i,c):
    sin_out, fbp_out, ct_out, sin_intp = test_out
    sin_in = data_batch[0]
    # sin_map = tf.reshape(tf.concat([sin_in, sin_out], 2), [sin_in.shape[0], -1, sin_in.shape[2], 1]).numpy()[0,:,:,0]
    fbp_out = fbp_out.numpy()[0,:,:,0]*255
    # sin_map = (sin_map - np.min(sin_map)) / (np.max(sin_map) - np.min(sin_map)) * 255
    cv2.imwrite('./Result/public/' + str(c) + 'x/mymodel_sine/' + str(i) + '_CNNmodel_fbp_out.bmp', fbp_out)


def test_mymodel(ckpt='./256x256/weights/new_model_lambda=0.5',testDataDir = '',test_sz =2 ,c=2):
    ct_model = CTReconModel(c)
    ct_model.build((1, 360//c, 357, 1))  # Manually build the model and initialize the weights
    ct_model.load_weights(ckpt)
    test_data  = load_testdata(testDataDir,c)
    total_psnr = [0, 0, 0, 0, 0, 0]
    total_ssim = [0, 0, 0, 0, 0, 0]
    result = []
    for iterNo, data_batch in enumerate(test_data.batch(test_sz)):
        test_out = ct_model(data_batch[0], training=0)
        psnr_test = compute_psnr(data_batch, test_out)  # psnr for training dataset
        ssim_test = compute_ssim(data_batch, test_out)
        total_psnr=list(map(lambda x: x[0] + x[1], zip(total_psnr, psnr_test)))
        total_ssim=list(map(lambda x: x[0] + x[1], zip(total_ssim, ssim_test)))
        save_data(data_batch,test_out,iterNo,c)

        if iterNo% 20==0:
            print("iterNo: ", iterNo, ":", "psnr_valid", psnr_test)
            print("iterNo: ", iterNo, ":", "ssim_valid", ssim_test)
            # show_image(data_batch, test_out)

    print("the average psnr :" ,total_psnr/(test_data.cardinality().numpy())*test_sz)
    print("the average ssim :", total_ssim / (test_data.cardinality().numpy()) * test_sz)

if __name__ == '__main__':
    testDataDir = './Data/mytest/My_data_256_20.npz'
    c = 2
    i = 100
    # iterNo = 800
    ckpt = './256x256/' + str(c) + 'x_weights_' + str(i) + '_epoch/ckpt'
    test_sz = 2
    test_mymodel(ckpt,testDataDir,test_sz,c)