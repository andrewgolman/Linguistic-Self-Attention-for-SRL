import tensorflow as tf
from tqdm import tqdm
import os


def print_model_metrics(metrics, file=None):
    print("Validation losses", file=file)
    print(metrics[(None, 'ValLosses')], file=file)
    print("Validation metrics:", file=file)
    for k, v in metrics.items():
        if k[0]:
            print(k, ":", v, file=file)


class EvalMetricsCallBack(tf.keras.callbacks.Callback):
    """
    On every epoch end: sets evaluation mode, passes validation data, prints metrics
    TODO: do it in within graph, currently eval_fns_np fails due to tf2.x data format
    """
    def __init__(self, dataset, log_file, teacher_forcing_on_train=False, eval_every=1):
        super(EvalMetricsCallBack, self).__init__()
        self.ds = dataset
        self.enable_teacher_forcing = not teacher_forcing_on_train
        self.log_file = log_file
        self.eval_every = eval_every

        with open(self.log_file, "w") as fout:
            pass

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.eval_every == 0:
            self.model.start_custom_eval()
            for batch in tqdm(self.ds.as_numpy_iterator()):
                self.model(batch)
                # self.model.predict(batch)

            metrics = self.model.get_metrics()
            print("=" * 20)
            print("EPOCH:", epoch + 1)
            print_model_metrics(metrics)
            with open(self.log_file, "a") as fout:
                print_model_metrics(metrics, fout)
            self.model.end_custom_eval(enable_teacher_forcing=self.enable_teacher_forcing)


class SaveCallBack(tf.keras.callbacks.Callback):
    def __init__(self, path, save_every=1):
        os.mkdir("{}/checkpoints".format(path))
        self.path = "{}/checkpoints/".format(path)
        self.save_every = save_every

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.save_every == 0:
            save_path = "{}/epoch_{}".format(self.path, epoch + 1)
            print("Epoch {}, saving model into {}".format(epoch + 1, save_path))
            self.model.save_weights(save_path, save_format='tf')
