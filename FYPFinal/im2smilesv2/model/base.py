import os
import sys
import time
import tensorflow as tf


from .utils.general import init_dir, get_logger


class BaseModel(object):

    def __init__(self, config, dir_output):

        self._config = config
        self._dir_output = dir_output
        init_dir(self._dir_output)
        self.logger = get_logger(self._dir_output + "model.log")
        tf.compat.v1.reset_default_graph() # saveguard if previous model was defined


    def build_train(self, config=None):

        raise NotImplementedError


    def build_pred(self, config=None):
        """Similar to build_train but no need to define train_op"""
        raise NotImplementedError


    def _add_train_op(self, lr_method, lr, loss, clip=-1):

        _lr_m = lr_method.lower() # lower to make sure

        with tf.compat.v1.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.compat.v1.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            # for batch norm beta gamma
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if clip > 0: # gradient clipping if clip is positive
                    grads, vs     = zip(*optimizer.compute_gradients(loss))
                    grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                    self.train_op = optimizer.apply_gradients(zip(grads, vs))
                else:
                    self.train_op = optimizer.minimize(loss)


    def init_session(self):

        # self.sess = tf.Session()
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()


    def restore_session(self, dir_model):

        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)


    def save_session(self):

        print("==SAVING WEIGHTS===")
        # check dir one last time
        dir_model = self._dir_output + "model.weights/"
        init_dir(dir_model)

        # logging
        sys.stdout.write("\r- Saving model...")
        sys.stdout.flush()

        # saving
        self.saver.save(self.sess, dir_model)

        # logging
        sys.stdout.write("\r")
        sys.stdout.flush()
        self.logger.info("- Saved model in {}".format(dir_model))


    def close_session(self):
        self.sess.close()


    def _add_summary(self):

        self.merged      = tf.compat.v1.summary.merge_all()
        self.file_writer = tf.compat.v1.summary.FileWriter(self._dir_output,
                self.sess.graph)


    def train(self, config, train_set, val_set, lr_schedule):

        best_score = None

        for epoch in range(config.n_epochs):
            # logging
            tic = time.time()
            self.logger.info("Epoch {:}/{:}".format(epoch+1, config.n_epochs))

            # epoch
            score = self._run_epoch(config, train_set, val_set, epoch,
                    lr_schedule)

            # save weights if we have new best score on eval
            if best_score is None or score >= best_score:
                best_score = score
                self.logger.info("- New best score ({:04.2f})!".format(
                        best_score))
                self.save_session()
            if lr_schedule.stop_training:
                self.logger.info("- Early Stopping.")
                break

            # logging
            toc = time.time()
            self.logger.info("- Elapsed time: {:04.2f}, lr: {:04.5f}".format(
                            toc-tic, lr_schedule.lr))

        return best_score


    def _run_epoch(config, train_set, val_set, epoch, lr_schedule):
        raise NotImplementedError


    def evaluate(self, config, test_set):

        # logging
        sys.stdout.write("\r- Evaluating...")
        sys.stdout.flush()

        # evaluate
        scores = self._run_evaluate(config, test_set)

        # logging
        sys.stdout.write("\r")
        sys.stdout.flush()
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in scores.items()])
        self.logger.info("- Eval: {}".format(msg))

        return scores


    def _run_evaluate(config, test_set):
        raise NotImplementedError