"""
Custom callbacks.
"""
import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score, classification_report


class F1score(Callback):

    def __init__(self, seq, preprocessor=None):
        super(F1score, self).__init__()
        self.seq = seq
        self.p = preprocessor

    def get_lengths(self, y_true):
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)

        return lengths

    def on_epoch_end(self, epoch, logs={}):
        label_true = []
        label_pred = []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            lengths = self.get_lengths(y_true)
            y_pred = self.model.predict_on_batch(x_true)

            y_true = self.p.inverse_transform(y_true, lengths)
            y_pred = self.p.inverse_transform(y_pred, lengths)

            label_true.extend(y_true)
            label_pred.extend(y_pred)

        score = f1_score(label_true, label_pred)
        print(' - f1: {:04.2f}'.format(score * 100))
        print(classification_report(label_true, label_pred))
        logs['f1'] = score

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='f1', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, preprocessor=None):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.p = preprocessor

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure') or self.monitor.startswith('f1'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            weights_file = str(os.path.join(filepath, 'weights.h5'))
            params_file = str(os.path.join(filepath, 'params.json'))
            preprocessor_file = str(os.path.join(filepath, 'preprocessor.json'))
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current

                        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
                        if self.save_weights_only:
                            self.model.save_weights(weights_file, overwrite=True)
                        else:
                            self.save_anago_model(self.model, weights_file, params_file, self.p, preprocessor_file)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(weights_file, overwrite=True)
                else:
                    self.save_anago_model(self.model, weights_file, params_file, self.p, preprocessor_file)

    def save_anago_model(self, model, weights_file, params_file, preprocessor=None, preprocessor_file=None):
        with open(params_file, 'w') as f:
            params = model.to_json()
            json.dump(json.loads(params), f, sort_keys=True, indent=4)
            model.save_weights(weights_file)
        if preprocessor and preprocessor_file:
            preprocessor.save(preprocessor_file)
