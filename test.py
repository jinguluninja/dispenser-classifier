import os
import keras
import numpy as np
np.random.seed(135109)
import sys

sensor = sys.argv[1]
print(sensor)

def dataset():
    X = []
    y = []
    data_dir_0 = sorted(os.listdir(os.path.join('pac_data', sensor, '0')))   
    i = 0
    for f in data_dir_0:
        if i % 100 == 0: print(i)
        i += 1
        X.append(np.load(os.path.join('pac_data', sensor, '0', f))['x'])
        y.append(0)
    X = np.asarray(X, dtype=np.float16)
    y = np.asarray(y, dtype=int)
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    X_train = X[indices[:int(0.8*len(indices))]]
    y_train = y[indices[:int(0.8*len(indices))]]
    X_val = X[indices[int(0.8*len(indices)):]]
    y_val = y[indices[int(0.8*len(indices)):]]
    labels_present = os.listdir(os.path.join('pac_data', sensor))
    if '1' in labels_present:
        X = []
        y = []
        data_dir_1 = sorted(os.listdir(os.path.join('pac_data', sensor, '1')))
        i = 0
        for f in data_dir_1:
            if i % 100 == 0: print(i)
            i += 1
            X.append(np.load(os.path.join('pac_data', sensor, '1', f))['x'])
            y.append(1)        
        X = np.asarray(X, dtype=np.float16)
        y = np.asarray(y, dtype=int)
        indices = np.arange(y.shape[0])
        np.random.shuffle(indices)
        X_train = np.concatenate([X_train, X[indices[:int(0.8*len(indices))]]])
        y_train = np.concatenate([y_train, y[indices[:int(0.8*len(indices))]]])
        X_val = np.concatenate([X_val, X[indices[int(0.8*len(indices)):]]])
        y_val = np.concatenate([y_val, y[indices[int(0.8*len(indices)):]]])
    X_train = np.expand_dims(np.asarray(X_train, dtype=np.float16), axis=3)
    y_train = np.asarray(y_train, dtype=int)
    X_val = np.expand_dims(np.asarray(X_val, dtype=np.float16), axis=3)
    y_val = np.asarray(y_val, dtype=int)
    X_train /= 5.
    X_val /= 5.
    mean_arr = np.mean(X_train, axis = 0)
    np.save(os.path.join('pac_data', sensor, 'mean.npy'), mean_arr)
    X_train -= mean_arr 
    X_val -= mean_arr  
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = dataset()


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

from keras.models import load_model
model_name = 'model' + sensor + '.h5'
trained_model = load_model(model_name)
print(trained_model.metrics_names)
pred_train = np.argmax(trained_model.predict(x=X_train, verbose=0), axis=1)
pred_val = np.argmax(trained_model.predict(x=X_val, verbose=0), axis=1)

TN_train = 0
FP_train = 0
FN_train = 0
TP_train = 0
for i in range(len(y_train)):
    if y_train[i] == 0:
        if pred_train[i] == 0:
            TN_train += 1
        else:
            FP_train += 1
    else:
        if pred_train[i] == 0:
            FN_train += 1
        else:
            TP_train += 1
acc_train = float(TN_train + TP_train) / len(y_train)
#prec_train = float(TP_train) / (TP_train + FP_train)
#sens_train = float(TP_train) / (TP_train + FN_train)
spec_train = float(TN_train) / (TN_train + FP_train)

TN_val = 0
FP_val = 0
FN_val = 0
TP_val = 0
for i in range(len(y_val)):
    if y_val[i] == 0:
        if pred_val[i] == 0:
            TN_val += 1
        else:
            FP_val += 1
    else:
        if pred_val[i] == 0:
            FN_val += 1
        else:
            TP_val += 1
acc_val = float(TN_val + TP_val) / len(y_val)
#prec_val = float(TP_val) / (TP_val + FP_val)
#sens_val = float(TP_val) / (TP_val + FN_val)
spec_val = float(TN_val) / (TN_val + FP_val)

with open('test.out', 'a') as f:
    f.write(sensor + '\n')
    f.write('TRAIN' + '\n')
    f.write('TN: %s, FP: %s, FN: %s, TP: %s\n' % (TN_train, FP_train, FN_train, TP_train))
    f.write('acc: %s\n' % acc_train)
    #f.write('prec: %s\n' % prec_train)
    #f.write('sens: %s\n' % sens_train)
    f.write('spec: %s\n' % spec_train)
    f.write('VAL' + '\n')
    f.write('TN: %s, FP: %s, FN: %s, TP: %s\n' % (TN_val, FP_val, FN_val, TP_val))
    f.write('acc: %s\n' % acc_val)
    #f.write('prec: %s\n' % prec_val)
    #f.write('sens: %s\n' % sens_val)
    f.write('spec: %s\n\n' % spec_val)
