from neuraxle.base import BaseStep


class BaseTensorflowModelStep(BaseStep):
    def __init__(
        self,
        create_model,
        create_loss,
        create_optimizer,
        step_saver,
        create_inputs=None,
        data_inputs_dtype=None,
        expected_outputs_dtype=None,
        print_loss=False,
        print_func=None
    ):
        BaseStep.__init__(
            self,
            savers=[step_saver],
            hyperparams=self.__class__.HYPERPARAMS,
            hyperparams_space=self.__class__.HYPERPARAMS_SPACE
        )

        self.create_inputs = create_inputs
        self.create_model = create_model
        self.create_loss = create_loss
        self.create_optimizer = create_optimizer

        self.expected_outputs_dtype = expected_outputs_dtype
        self.data_inputs_dtype = data_inputs_dtype

        self.train_losses = []
        self.test_losses = []
        self.print_loss = print_loss
        if print_func is None:
            print_func = print
        self.print_func = print_func

    def add_new_loss(self, loss, test_only=False):
        if test_only:
            if not self.is_train:
                self.test_losses.append(loss)
            else:
                return

        if self.is_train:
            self.train_losses.append(loss)

        if self.print_loss:
            self.print_func('{} Loss: {}'.format('Train' if self.is_train else 'Test', loss))
