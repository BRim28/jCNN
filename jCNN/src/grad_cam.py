from all_imports import *
class GradCAM:
    def __init__(self, layerName=None):
        self.model = model
        self.layerName = layerName

    def compute_heatmap(self, data, eps=1e-8):
        grads = []
        guided_grads = []
        loss = []
        global pred_idx
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,	self.model.output])
        with tf.GradientTape() as tape:
            (convOutputs, predictions) = gradModel(data) 
            predictions = predictions[1]
            pred_idx = [np.argmax(x) for x in predictions]
            loss = [predictions[0,pred_idx[0]]]
        grads = tape.gradient(loss, convOutputs) #derivative of class wrt feature maps
        if grads==None:
            return 0
        castConvOutputs = tf.cast(convOutputs > 0, "float32") #convert to float32
        castGrads = tf.cast(grads > 0, "float32") #Select positive grads
        guidedGrads = castGrads * castGrads * grads
        heatmaps = []

        convOutputs_i = convOutputs[0]
        guidedGrads_i = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads_i, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs_i), axis=-1) 
        (w, h) = (input_size, input_size)
        heatmap = cv2.resize(cam.numpy(), (w, h), interpolation=cv2.INTER_CUBIC)

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmaps.append(heatmap)

        return heatmaps[0]

    #***************************************************************************************************************************
    # GradCAM and GradCAM++

def grad_cam(input_model, image, layer_name,H=input_size,W=input_size):
    grads = []
    y_c = []
    global cls
    gradModel = Model(
        inputs=[input_model.inputs],
        outputs=[input_model.get_layer(layer_name).output,input_model.output])
    with tf.GradientTape() as tape:
        (conv_output, predictions) = gradModel(image) 
        predictions = predictions[1]
        cls = [np.argmax(x) for x in predictions]
        y_c = predictions[0,cls[0]]
    grads = tape.gradient(y_c, conv_output) #derivative of class wrt feature maps
    if grads==None:
        return 0

    output, grads_val = conv_output[0, :], grads[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (H, W), interpolation=cv2.INTER_CUBIC)
    #cam = zoom(cam,H/cam.shape[0])
    cam = cam / cam.max()
    return cam

def grad_cam_plus(input_model, img, layer_name,H=input_size,W=input_size):
    grads = []
    y_c = []
    global cls
    gradModel = Model(
        inputs=[input_model.inputs],
        outputs=[input_model.get_layer(layer_name).output,input_model.output])
    with tf.GradientTape() as tape:
        (conv_output, predictions) = gradModel(img) 
        predictions = predictions[1]
        cls = [np.argmax(x) for x in predictions]
        y_c = predictions[0,cls[0]]
    grads = tape.gradient(y_c, conv_output) #derivative of class wrt feature maps
    if grads==None:
        return 0
    first = K.exp(y_c)*grads
    second = K.exp(y_c)*grads*grads
    third = K.exp(y_c)*grads*grads*grads
    CO = conv_output.numpy()

    global_sum = np.sum(CO[0].reshape((-1,first[0].shape[2])), axis=0)
    alpha_num = second[0]
    alpha_denom = second[0]*2.0 + third[0]*global_sum.reshape((1,1,first[0].shape[2]))
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num/alpha_denom

    weights = np.maximum(first[0], 0.0)

    alpha_normalization_constant = np.sum(np.sum(alphas, axis=0),axis=0)

    alphas /= alpha_normalization_constant.reshape((1,1,first[0].shape[2]))
    IM =(weights*alphas).numpy()
    deep_linearization_weights = np.sum(IM.reshape((-1,first[0].shape[2])),axis=0)
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    # Passing through ReLU
    cam = np.maximum(grad_CAM_map, 0)
    #cam = zoom(cam,H/cam.shape[0])
    cam = cv2.resize(cam, (H, W), interpolation=cv2.INTER_CUBIC)
    cam = cam / np.max(cam) # scale 0 to 1.0    
    return cam
