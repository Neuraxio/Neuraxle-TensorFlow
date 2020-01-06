class BaseTensorflowModelStep(BaseStep):
    def __init__(self, create_graph, create_loss, create_optimizer):
        self.create_graph = create_graph
        self.create_loss = create_loss
        self.create_optimizer = create_optimizer

        self.set_hyperparams(self.__class__.HYPERPARAMS)
        self.set_hyperparams_space(self.__class__.HYPERPARAMS_SPACE)