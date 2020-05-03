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
    def __init__(self, dataset, log_file, enable_teacher_forcing=True, eval_every=1):
        super(EvalMetricsCallBack, self).__init__()
        self.ds = dataset
        self.enable_teacher_forcing = enable_teacher_forcing
        self.log_file = log_file
        self.eval_every = eval_every

        # with open(self.log_file, "w") as fout:
        #     pass

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.eval_every == 0:
            self.model.start_custom_eval()
            for batch in tqdm(self.ds.as_numpy_iterator()):
                self.model(batch)
                # self.model.predict(batch)

            metrics = self.model.get_metrics()
            print_model_metrics(metrics)
            with open(self.log_file, "a") as fout:
                print("=" * 20, file=fout)
                print("EPOCH:", epoch + 1, file=fout)
                print_model_metrics(metrics, fout)
            self.model.end_custom_eval(enable_teacher_forcing=self.enable_teacher_forcing)


class SaveCallBack(tf.keras.callbacks.Callback):
    def __init__(self, path, save_every=1, start_epoch=0):
        self.path = "{}/checkpoints/".format(path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_every = save_every
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.save_every == 0:
            save_path = "{}/epoch_{}".format(self.path, epoch + 1 + self.start_epoch)
            print("Epoch {}, saving model into {}".format(epoch + 1 + self.start_epoch, save_path))
            self.model.save_weights(save_path, save_format='tf')
