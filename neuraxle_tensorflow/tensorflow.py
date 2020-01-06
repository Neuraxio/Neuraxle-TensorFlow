from neuraxle.base import BaseStep


class BaseTensorflowModelStep(BaseStep):
    def __init__(self, create_model, create_loss, create_optimizer, step_saver):
        self.create_model = create_model
        self.create_loss = create_loss
        self.create_optimizer = create_optimizer

        self.set_hyperparams(self.__class__.HYPERPARAMS)
        self.set_hyperparams_space(self.__class__.HYPERPARAMS_SPACE)

        BaseStep.__init__(
            self,
            savers=[step_saver],
            hyperparams=self.HYPERPARAMS
        )
