import logging


class Client:
    """
    get_sample_number: return the number of data for current client
    train: (local)
    local_test: (local)
    """

    def __init__(self, client_idx,local_label_training_data, local_training_data, local_test_data, local_train_sample_number, args, device,
                 model_trainer, model):
        self.client_idx = client_idx
        self.local_label_training_data = local_label_training_data
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_train_sample_number = local_train_sample_number
        logging.info("self.local_train_sample_number = " + str(self.local_train_sample_number))

        self.args = args
        self.device = device
        
        self.model = model
        self.model_trainer = model_trainer

    def get_sample_number(self):
        return self.local_train_sample_number

    def train(self, epochs=None):
        loss, loss_labeled, loss_unlabeled = self.model_trainer.train(self.model,self.local_label_training_data, self.local_training_data, self.device, self.args, epochs=epochs, client_idx=self.client_idx)
        weights = self.model_trainer.get_model_params(self.model)
        return weights, loss , loss_labeled, loss_unlabeled
        

    def local_test(self):
        test_data = self.local_test_data
        metrics = self.model_trainer.test(self.model, test_data, self.device, self.args)
        return metrics


