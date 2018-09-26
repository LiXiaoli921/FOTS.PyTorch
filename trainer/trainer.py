import numpy as np
import torch
from base import BaseTrainer
from utils.bbox import Toolbox
from torch.autograd import Variable

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, toolbox: Toolbox, valid_data_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        self.log_step = int(np.sqrt(self.batch_size))
        self.toolbox = toolbox

    def _to_tensor(self, *tensors):
        t = []
        for __tensors in tensors:
            t.append(__tensors.to(self.device))
        return t

    def _eval_metrics(self, output, target, mask):
        acc_metrics = np.zeros(len(self.metrics))

        print(output[0].shape)

        print(torch.numel(output[1]),"hahhaha"*9)
        print(torch.numel(output[0]),"hahhaha"*9)



        output = np.asarray(output)
        target = np.asarray(target)
        print(output.shape,"meibianhua----"*9)

        # # print("*****"*9)
        print(output)
        print(target)


        output_0 = output[0]
        target_0 = target[0]
        output_1 = output[1]
        target_1 = target[1]


        print("****biamhua*"*9)
        print(output_1)
        print(target_1)
        print(output.shape) # (3,)
        print(output_0.shape) # torch.Size([5, 128, 128])
        print(output_1.shape) # torch.Size([128, 128])
        print(target_0.shape)
        print(target_1.shape)





        output_0 = output_0.data.cpu().numpy()
        output_1 = output_1.data.cpu().numpy()
        target_0 = target_0.data.cpu().numpy()
        target_1 = target_1.data.cpu().numpy()

        # output = torch.cat((output_1, output_0))
        # target = torch.cat((target_1, target_0))
        #
        output = np.argmax(output_0, axis=1)
        print(output,'outy*8888'*9)
        print(output_0.shape)
        print(target.shape)
        print(target_0.shape)

        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output_0, target_0)
            acc_metrics[i] += metric(output_1, target_1)
            print('acc_metrics')
            print(acc_metrics)
        return acc_metrics
            # print(output.shape, "meibianhua----" * 9)
            # output = output[i].squeeze()
            # target = target[i].squeeze()
            # print(output.shape)
            # # print("*****"*9)
            # # print(target)
            # # print(output.shape)
            # output = output.data.cpu().numpy()
            # target = target.data.cpu().numpy()

            # output = np.argmax(output, axis=1)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, gt in enumerate(self.data_loader):
            print(batch_idx,"batchindex"*9)
            img, score_map, geo_map, training_mask = gt #ICDAR 没有transcript，所以ｇｔ信息只有４个
            img, score_map, geo_map, training_mask = self._to_tensor(img, score_map, geo_map, training_mask)
            recog_map = None

            self.optimizer.zero_grad()
            pred_score_map, pred_geo_map, pred_recog_map = self.model(img)

            loss = self.loss(score_map, pred_score_map, geo_map, pred_geo_map, pred_recog_map, recog_map, training_mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            #total_metrics += self._eval_metrics(output, target)

            total_metrics += 0

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    len(self.data_loader) * self.data_loader.batch_size,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            # for batch_idx, (img, score_map, geo_map, training_mask,transcript) in enumerate(self.data_loader):#验证的时候没有transcript
            for batch_idx, gt in enumerate(self.data_loader):
                img, score_map, geo_map, training_mask = gt
                print('****start_geomap**'*7)
                print(geo_map)
                print('img****'*7)
                print(img)
                print('score****'*7)
                print(score_map)

                img, score_map, geo_map, training_mask = self._to_tensor(img, score_map, geo_map, training_mask)
                recog_map = None

                pred_score_map, pred_geo_map, pred_recog_map = self.model(img)

                loss = self.loss(score_map, pred_score_map, geo_map, pred_geo_map, pred_recog_map, recog_map,
                                 training_mask)


                total_val_loss += loss.item()

                output = (pred_score_map, pred_geo_map, pred_recog_map)
                target = (score_map, geo_map, recog_map)
                total_val_metrics += self._eval_metrics(output, target, training_mask)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
