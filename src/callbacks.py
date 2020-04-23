import tensorflow as tf


def print_model_metrics(model):
    metrics = model.get_metrics()
    print("Validation losses")
    print(metrics[(None, 'ValLosses')])
    print("Validation metrics:")
    for k, v in metrics.items():
        if k[0]:
            print(k, ":", v)


from tqdm import tqdm

class EvalMetricsCallBack(tf.keras.callbacks.Callback):
    """
    On every epoch end: sets evaluation mode, passes validation data, prints metrics
    TODO: do it in within graph, currently eval_fns_np fails due to tf2.x data format
    """
    def __init__(self, dataset):
        super(EvalMetricsCallBack, self).__init__()
        self.ds = dataset

    # def on_epoch_end(self, epoch, logs={}):
    #     self.model.start_custom_eval()
    #     for batch in self.ds.as_numpy_iterator():
    #         self.model(batch)
    #         # self.model.predict(batch)

    def on_epoch_end(self, epoch, logs={}):
        self.model.start_custom_eval()
        for batch in tqdm(self.ds):
            self.model(batch)
            # self.model.predict(batch)

        print("=" * 20)
        print("EPOCH:", epoch + 1)
        print_model_metrics(self.model)
        self.model.end_custom_eval()


class SaveCallBack(tf.keras.callbacks.Callback):
    def __init__(self, path, save_every=1):
        self.path = path
        self.save_every = save_every

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.save_every == 0:
            print("Epoch {}, saving model into {}".format(epoch + 1, self.path))
            self.model.save_weights(self.path, save_format='tf')
