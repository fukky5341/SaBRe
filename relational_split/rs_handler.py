import copy
import torch
import random
from common import Status


class RS_handler:
    def __init__(self):
        self.start_layer_idx = 0

    def duplicate_RS(self):
        rs = copy.deepcopy(self)
        rs.IARb = self.duplicate_IARb(self.IARb)
        return rs

    def duplicate_IARb(self, IARb):
        new_IARb = copy.deepcopy(IARb)
        new_IARb.inp1_relu_input_info = copy.deepcopy(IARb.inp1_relu_input_info)
        new_IARb.inp2_relu_input_info = copy.deepcopy(IARb.inp2_relu_input_info)
        return new_IARb

    def copy_grb_model(self, rs):
        if self.IARb.global_robustness_lp.grb_model is not None:
            rs.IARb.global_robustness_lp.grb_model = self.IARb.global_robustness_lp.grb_model.copy()

    def get_rs_random_order(self, ds_mode, layer_idx, inp1_lbs, inp1_ubs, inp2_lbs, inp2_ubs, d_lbs, d_ubs):
        """
        layer_idx indicates the index of the activation layer (ReLU layer) in the net
        layer_idx also indicates the index of the preactivation lubs in the lubs list
        net: [linear, relu, linear, relu, ...]
        lubs: [inp, linear, relu, linear, relu, ...]
        """
        ds_conditions = self.sort_into_single(inp1_lbs[layer_idx], inp1_ubs[layer_idx], inp2_lbs[layer_idx],
                                           inp2_ubs[layer_idx], d_lbs[layer_idx], d_ubs[layer_idx])
        if not ds_conditions['common'].any():
            return []

        temp_order = []
        if ds_mode == 'RS_random_Z':
            active_idx = torch.where(ds_conditions['RSZ'])[0]  # e.g., tensor([1, 2, 4, 5, ...])
            active_idx_list = active_idx.tolist()
            for idx in active_idx_list:
                temp_order.append(['RSZ', layer_idx, idx])

        if len(temp_order) == 0:
            return []
        random.shuffle(temp_order)
        rs_order = []
        seen_indices = set()
        for item in temp_order:  # item is [rs_type, layer_idx, pos]
            if item[2] not in seen_indices:
                candidate = [item[0], layer_idx, item[2]]
                rs_order.append(candidate)
                seen_indices.add(item[2])
        return rs_order
    
    def sort_into_single(self, inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub):
        ds_conditions = {}
        # conditions for sorting
        ds_conditions['RSZ'] = ((d_lb < 0) & (0 < d_ub) & ((inp1_lb < -1e-6) & (1e-6 < inp1_ub))) | \
                ((d_lb < 0) & (0 < d_ub) & ((inp2_lb < -1e-6) & (1e-6 < inp2_ub)))
        ds_conditions['common'] = ds_conditions['RSZ']

        return ds_conditions

    def collect_final_status(self, RS_list):
        status = Status.UNKNOWN
        inp1_label = None
        inp2_label = None
        for rs in RS_list:
            if rs.status == Status.VERIFIED:
                status = Status.VERIFIED
                if inp1_label is None:
                    inp1_label = rs.inp1_label
                else:
                    if inp1_label != rs.inp1_label:
                        raise ValueError(f"Multiple inp1 labels found: {inp1_label} and {rs.inp1_label}.")
                if inp2_label is None:
                    inp2_label = rs.inp2_label
                else:
                    if inp2_label != rs.inp2_label:
                        raise ValueError(f"Multiple inp2 labels found: {inp2_label} and {rs.inp2_label}.")
            elif rs.status == Status.ADV_EXAMPLE:
                return Status.ADV_EXAMPLE, rs.inp1_label, rs.inp2_label
            elif rs.status == Status.UNREACHABLE:
                continue
            else:
                return Status.UNKNOWN, None, None

        return status, inp1_label, inp2_label
