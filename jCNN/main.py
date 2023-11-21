from logging import config
from src.all_imports import *
from src.plot_functions import *

class jCNN:

    def __init__(self,path):
        self.config = Config(path)
        self.out_path = self.config.OUT_PATH+self.config.EXPERIMENT_NAME
        self.setup_experiment()
        self.log = open(self.out_path+self.config.LOG_BOOK, 'a+')  
        self.label_binarizer = preprocessing.LabelEncoder()
        self.log.write("*"*50+'\n'+str(datetime.datetime.now())+"\t Configuration:\n\n\t Details: "+str(self.config.DESCRIPTION)+"\n\n"+"*"*50+'\n')
        self.config.CHANNELS = 3 if self.config.RGB else 1 # Number of channels in input image
        
    def setup_experiment(self):
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        if not os.path.exists(self.out_path+'/Plots'):
            os.makedirs(self.out_path+'/Plots')
        if not os.path.exists(self.out_path+'/Models'):
            os.makedirs(self.out_path+'/Models')


    def fetchTrainData(self):
        self.train_data = []
        self.train_labels = []
        self.train_masks = []
        for sub_ds in range(self.config.NUM_SUB_DATASETS):
            if sub_ds==0:
                self.train_data = np.load(self.config.DATA_PATH+'train_data_'+str(sub_ds+1)+'.npy')
                self.train_labels = np.load(self.config.DATA_PATH+'train_labels_'+str(sub_ds+1)+'.npy')
                self.train_masks = np.load(self.config.DATA_PATH+'train_masks_'+str(sub_ds+1)+'.npy')
            else:
                self.train_data = np.append(self.train_data,np.load(self.config.DATA_PATH+'train_data_'+str(sub_ds+1)+'.npy'),axis=0)
                self.train_labels = np.append(self.train_labels,np.load(self.config.DATA_PATH+'train_labels_'+str(sub_ds+1)+'.npy'),axis=0)
                self.train_masks = np.append(self.train_masks,np.load(self.config.DATA_PATH+'train_masks_'+str(sub_ds+1)+'.npy'),axis=0)
        # Select Training Masks
        self.select_masks()
        
        self.val_data = np.load(self.config.DATA_PATH+'val_data.npy')
        self.val_labels = np.load(self.config.DATA_PATH+'val_labels.npy')
        self.val_masks = np.load(self.config.DATA_PATH+'val_masks.npy')
        
        self.train_data, self.train_labels = self.preProcess(self.train_data, self.train_labels)
        self.val_data, self.val_labels = self.preProcess(self.val_data, self.val_labels)
        
        self.train_masks = np.expand_dims(self.train_masks,-1)
        self.val_masks = np.expand_dims(self.val_masks,-1)
        if not self.config.USE_FNN:
            self.val_masks = np.array([self.val_masks[x]*self.val_labels[x] for x in range(self.val_masks.shape[0])])
            self.train_masks = np.array([self.train_masks[x]*self.train_labels[x] for x in range(self.train_masks.shape[0])])
        
        return 0
    
    def fetchTestData(self):
        self.test_data = np.load(self.config.DATA_PATH+'test_data.npy')
        self.test_labels = np.load(self.config.DATA_PATH+'test_labels.npy')
        self.test_masks = np.load(self.config.DATA_PATH+'test_masks.npy')
        
        self.test_data, self.test_labels = self.preProcess(self.test_data, self.test_labels)
        
        self.test_masks = np.expand_dims(self.test_masks,-1)
        if not self.config.USE_FNN:
            self.test_masks = np.array([self.test_masks[x]*self.test_labels[x] for x in range(self.test_masks.shape[0])])
            
        return 0
    
    def select_masks(self):
        random.seed(8)
        selected_masks = random.sample(range(self.config.NUM_SUB_DATASETS*1000), self.config.HM_COUNT)
        all_masks = range(self.train_masks.shape[0])
        unselected_masks = [x for x in all_masks if x not in selected_masks]
        self.train_masks[unselected_masks] = np.ones_like(self.train_masks[0])
        #print(str(len(unselected_masks))+" masks not selected")
            

    def preProcess(self,data,labels):
        data = np.array(data[...,::-1])/255 # transforming from BGR to RGB and normalizing to range 0-1
        labels = np.array(labels)
        labels = preprocessing.LabelEncoder().fit_transform(labels)
        labels = to_categorical(labels)
        return data,labels

    def model(self,spec):
        inp = Input(shape=(self.config.INPUT_SIZE,self.config.INPUT_SIZE,self.config.CHANNELS))
        for i in range(len(spec)):
            if i==0:
                convBase = Conv2D(filters=spec[i],activation = 'relu', kernel_size=(5,5),strides=(1,1),padding=self.config.PADDING, name="Conv_"+str(i+1)+"_"+str(spec[i]))(inp)
            else:
                if i%3==2:
                    convBase = Conv2D(filters=spec[i],activation = 'relu', kernel_size=(5,5),strides=(1,1),padding=self.config.PADDING, name="Conv_"+str(i+1)+"_"+str(spec[i]))(convBase)
                else:
                    convBase = Conv2D(filters=spec[i],activation = 'relu', kernel_size=(5,5),strides=(1,1),padding=self.config.PADDING, name="Conv_"+str(i+1)+"_"+str(spec[i]))(convBase)
            convBase = BatchNormalization(axis=-1,name="BN_"+str(i+1)+"_"+str(spec[i]))(convBase)
            convBase = Dropout(0.3)(convBase)
        if self.config.FNN:
            convBase1x1=convBase = Conv2D(filters=16,activation ='relu', kernel_size=(1,1),strides=(1,1),padding=self.config.PADDING, name="JUSTIFICATION")(convBase)
            convBase = Flatten(name='Flatten')(convBase)
            convBase = Dense(units=self.config.NUM_CLASSES,name="Output")(convBase)
        else:
            convBase1x1 = convBase = Conv2D(filters=self.config.NUM_CLASSES,activation ='relu', kernel_size=(1,1),strides=(1,1),padding=self.config.PADDING, name="JUSTIFICATION")(convBase)
            convBase = GlobalAveragePooling2D(name='GAP')(convBase)
        convBase = Softmax(name="CATEGORICAL")(convBase)
        model = Model(inputs=inp, outputs=[convBase1x1,convBase])
        tf.keras.utils.plot_model(model,self.out_path+'/model.png')
        #trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
        #nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
        #totalParams = trainableParams + nonTrainableParams
        #print(model.summary())
        #print("Total number of parameters: ",totalParams)
        return model

    def justificationLoss(self,Y,Y_hat):  
        HM_true = tf.cast(Y,dtype=tf.float32)
        HM_pred = tf.cast(tf.image.resize(Y_hat,[self.config.INPUT_SIZE,self.config.INPUT_SIZE]),dtype=tf.float32)
        return tf.math.subtract(1.,tf.math.reduce_mean((tf.math.reduce_sum(tf.math.multiply(HM_true,HM_pred),axis=(1,2,3)))/(tf.math.reduce_sum(HM_pred,axis=(1,2,3)) + 1e-10)))


    def trainModel(self,model):    
        print(self.train_data.shape, self.train_labels.shape)   
        callback = EarlyStopping(monitor='val_CATEGORICAL_loss', patience=self.config.PATIENCE, restore_best_weights=True)
        print("[INFO] compiling model...")
        
        opt = Adam(lr=self.config.LEARNING_RATE, decay=self.config.DECAY)
        #opt = SGD(learning_rate=lr, decay=1e-6, momentum=0.9)
        
        losses = {
        "CATEGORICAL": tf.keras.losses.CategoricalCrossentropy(),
        "JUSTIFICATION": self.justificationLoss,
        }
        
        #monitor_quantity = 'val_categorical_crossentropy' if VAL else 'categorical_crossentropy'
        model.compile(loss=losses, optimizer=opt, metrics=['accuracy'],loss_weights=[self.config.ALPHA,1])
        if self.config.AUGMENT:
            data_gen_args = dict(rotation_range=45,
                                zoom_range=0.3,
                                horizontal_flip=True, 
                                fill_mode = 'nearest'
                                )
        else:
            data_gen_args = dict(rotation_range=0,
                                zoom_range=0.0,
                                horizontal_flip=False
                                )
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        image_generator = image_datagen.flow(self.train_data, #save_to_dir='../Images1',
            seed=seed)
        mask_generator = mask_datagen.flow((self.train_masks,self.train_labels), #save_to_dir='../Images2',
            seed=seed)
        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)

        
        H = model.fit(train_generator,
                steps_per_epoch=self.train_data.shape[0]//self.config.BATCH_SIZE,
                epochs=self.config.EPOCHS,
                shuffle=True,
                callbacks = [callback],
                #class_weight={'JUSTIFICATION':None,'CATEGORICAL':class_weight},
                validation_data = (self.val_data,[self.val_masks,self.val_labels])
                )    
        
            
        # tic = time.time()
        # predIdxs = model.predict(self.train_data, batch_size=self.config.BATCH_SIZE)
        # toc = time.time()
        # train_time = (toc-tic)/self.train_labels.shape[0]
        # print('Train Pred time (s/sample): ',train_time)
        
        tic = time.time()
        predIdxs = model.predict(self.val_data, batch_size=self.config.BATCH_SIZE)
        toc = time.time()
        val_time = (toc-tic)/self.val_labels.shape[0]
        print('Val Pred time (s/sample): ',val_time)
        
        predY = predIdxs[1].copy()
        predHM = predIdxs[0].copy()
        predIdxs = np.argmax(predIdxs[1], axis=-1)
        
        # Calculate CCE:
        tr_cce = tf.keras.losses.CategoricalCrossentropy()(self.val_labels,predY).numpy()
        tr_hm = self.justificationLoss(self.val_masks,predHM).numpy()
        tr_f1_macro = f1_score(np.argmax(self.val_labels,-1), predIdxs, average='macro') 
        tr_roc = roc_auc_score(self.val_labels, predY)
        
        self.label_binarizer.classes_ = self.config.CLASS_NAMES
        
        print("\n*****INFO: Displaying Test*****\n")
        cm = confusion_matrix(self.val_labels.argmax(axis=-1), predIdxs)
        #print(cm)
        report = (classification_report(self.val_labels.argmax(axis=-1), predIdxs,target_names=self.label_binarizer.classes_))
        print(report)
        accuracy = (accuracy_score(self.val_labels.argmax(axis=-1), predIdxs))
        print(accuracy)
        model.save(self.out_path+'/Models/'+self.config.MODEL_NAME+'.model', save_format='h5')
        self.plotTrainHistory(H)
        metrics = {'Histort':H,
                   #'Train_Time':train_time,
                   'Val_Time':val_time,
                   'Val_Accuracy':accuracy,
                   'Val_CCE':tr_cce,
                   'Val_Justification_Loss':tr_hm,
                   'Val_F1_Macro':tr_f1_macro,
                   'Val_ROC':tr_roc,
                   }
        return metrics

    #Model Evaluation

        
    def plotTrainHistory(self,H):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, len(H.history["CATEGORICAL_accuracy"])), H.history["CATEGORICAL_accuracy"], label="tr_acc",scaley=False,color='g')
        plt.plot(np.arange(0, len(H.history["CATEGORICAL_loss"])), H.history["CATEGORICAL_loss"], label="tr_cce",scaley=False,color='r')
        plt.plot(np.arange(0, len(H.history["JUSTIFICATION_loss"])), H.history["JUSTIFICATION_loss"], label="tr_act",scaley=False,color='b')
        plt.plot(np.arange(0, len(H.history["val_CATEGORICAL_accuracy"])), H.history["val_CATEGORICAL_accuracy"], label="val_acc",scaley=False, linestyle= '--',color='g')
        plt.plot(np.arange(0, len(H.history["val_CATEGORICAL_loss"])), H.history["val_CATEGORICAL_loss"], label="val_cce",scaley=False, linestyle= '--',color='r')
        plt.plot(np.arange(0, len(H.history["val_JUSTIFICATION_loss"])), H.history["val_JUSTIFICATION_loss"], label="val_act",scaley=False, linestyle= '--',color='b')
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper left")
        plt.savefig(self.out_path+'/Plots/'+self.config.MODEL_NAME+"_training_loss.png")
        np.save(self.out_path+'/Models/'+'/train_history.npy',H.history)
        plt.clf()
            
        
    def testModel(self,model):
        pr_auc=[]
        roc_auc=[]
        accuracy=[]
        f1_scores=[]
        
        testX = np.array(self.test_data)
        testY = np.array(self.test_labels)  
        
        tic = time.time()
        predIdxs = model.predict(testX, batch_size=self.config.BATCH_SIZE)
        toc = time.time()
        test_time = (toc-tic)/testY.shape[0]
        print('Test Pred time (s/sample): ',test_time)
        
        predHM = predIdxs[0].copy()
        predY = predIdxs[1].copy()
        logits = predIdxs[1].copy()
        predIdxs = np.argmax(predIdxs[1], axis=-1)
        # show a nicely formatted classification report
        self.label_binarizer.classes_ = self.config.CLASS_NAMES
        # Display results
        print("\n*****INFO: Displaying Test Results for fold {}*****\n")
        report = (classification_report(testY.argmax(axis=1), predIdxs,target_names=self.label_binarizer.classes_))
        print(report)
        conf_mat = confusion_matrix(testY.argmax(axis=1), predIdxs)
        print(conf_mat)
        accuracy = (accuracy_score(testY.argmax(axis=1), predIdxs))
        print(accuracy)

        test_cce = tf.keras.losses.CategoricalCrossentropy()(testY,predY).numpy()
        test_hm_loss = self.justificationLoss(self.test_masks,predHM).numpy()
        te_f1_macro = f1_score(testY.argmax(axis=-1), predIdxs, average='macro') 
        te_roc = roc_auc_score(testY, logits)
        
        # Plot AUROC curve
        roc_auc = (plot_AUROC(testY, predY,num_classes=self.config.NUM_CLASSES,out_path=self.out_path, modelName=self.config.MODEL_NAME,lw=1.2))
        # Plot PR_AUC curve
        pr_auc = (plot_PRAUC(testY, predY,num_classes=self.config.NUM_CLASSES,out_path=self.out_path, modelName=self.config.MODEL_NAME, lw=1.2))
            
        print(f"Prediction Time:{toc - tic:0.4f} seconds")
        con_mat_norm = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals=2)

        f1=f1_score(testY.argmax(axis=-1), predIdxs,average='macro')
        #print('f1 score is ', f1)
        f1_scores.append(f1)

        con_mat_df = pd.DataFrame(con_mat_norm, index = self.label_binarizer.classes_, columns = self.label_binarizer.classes_)
        figure = plt.figure(figsize=(4, 4))
        #model.summary(print_fn=lambda x: f.write(x + '\n'))
        sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(self.out_path+'/Plots/'+self.config.MODEL_NAME+'_confusion_matrix.png')
        plt.clf()
        #model.summary(print_fn=lambda x: f.write(x + '\n'))
        metrics = {'Test_Time':test_time,
                   'Test_Accuracy':accuracy,
                   'Test_CCE':test_cce,
                   'Test_Justification_Loss':test_hm_loss,
                   'Test_F1_Macro':te_f1_macro,
                   'Test_ROC':te_roc,
                   'Test_Report':report,
                   'Test_Conf_Matrix':conf_mat,
                   }
        
        return metrics

    

if __name__=='__main__':
    path = "configs/oxford_1K.yaml"
    ########TRAINING########
    jcnn_trainer = jCNN(path)
    # fetch and preprocess data
    jcnn_trainer.fetchTrainData()
    #print(jcnn_trainer.train_data.shape)
    jcnn_trainer.log.write(str(datetime.datetime.now())+"\tRunning for specification: "+str(jcnn_trainer.config.MODEL_SPECS)+"\n")
    jcnn_trainer.log.write(str(datetime.datetime.now())+"\tCreated the model: "+jcnn_trainer.config.MODEL_NAME+" \n") 
    jcnn_trainer.log.write(str(datetime.datetime.now())+"\tTraining the model.\n") 
    model = jcnn_trainer.model(jcnn_trainer.config.MODEL_SPECS)
    print("*****INFO: Training the Model*****")
    validation_metrics = jcnn_trainer.trainModel(model)
    # with open(jcnn_trainer.out_path+'/validation_metrics.pkl', 'wb') as filehandle:
    #     pickle.dump(validation_metrics, filehandle)
    jcnn_trainer.log.write(str(datetime.datetime.now())+"\tCompleted training!\n") 
    del jcnn_trainer

    ########TESTING########
    jcnn_tester = jCNN(path)  
    jcnn_tester.fetchTestData()
    # print('Test images: ',jcnn_tester.test_data.shape[0])
    model = tf.keras.models.load_model(jcnn_tester.out_path+'/Models/'+jcnn_tester.config.MODEL_NAME+'.model',custom_objects = {"justificationLoss": jcnn_tester.justificationLoss})
    test_metrics = jcnn_tester.testModel(model)
    # with open(jcnn_tester.out_path+'/test_metrics.pkl', 'wb') as filehandle:
    #     pickle.dump(test_metrics, filehandle)
