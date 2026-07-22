import torch


class EasyProperty:
    def __init__(self, input, input_lb, input_ub, out_constr):
        if input is not None:
            self.input = input.flatten()
        else:
            self.input = None
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.out_constr = out_constr
        self.input_props = [EasyInputProp(input_lb, input_ub)]


class EasyInputProp:
    def __init__(self, input_lb, input_ub):
        self.input_lb = input_lb
        self.input_ub = input_ub


class EasyOutConstr:
    def __init__(self, label):
        self.label = torch.tensor(label)
