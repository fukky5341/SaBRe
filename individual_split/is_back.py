import copy
import time
import torch
import gc
from common.network import LayerType
from individual_split.is_handler import IS_handler
from common import Status
from relational_property.relational_analysis import relational_analysis_back, RelationalProperty
from dual.dual_network import get_relational_order_is
from dual.dual_network_ind import get_relational_order_is_ind


class IS_info:
    def __init__(self, is_inp, layer_idx, pos, split_value):

        self.is_type = None  # None for now, will be set as ("A1", "A2", "B1", "B2")
        self.is_inp = is_inp  # str: A or B, which indicates which individual network to split
        self.layer_idx = layer_idx  # int
        self.pos = pos  # int
        self.split_value = split_value


class IS(IS_handler):
    def __init__(self, log_file='log', IS_mode='IS_random', split_limit=10, relational_prop=RelationalProperty.GLOBAL_ROBUSTNESS):

        # self.IARb = IndividualAndRelationalBounds
        self.IS_mode = IS_mode
        self.IS_history = []
        self.IS_failed_list = []
        self.next_candidate_input = 'A'  # 'A' or 'B'
        self.IS_candidates = {}
        self.IS_candidates_A = {}
        self.IS_candidates_B = {}
        self.split_limit = split_limit
        self.log_file = log_file
        self.name = "IS"
        self.split_count = 0
        self.status = Status.UNKNOWN
        self.inp1_label = None
        self.inp2_label = None
        self.relational_output_dist = None  # {dim: [lb, ub], ...}
        self.relational_prop = relational_prop
        self.selection_start_layer = 0  # to speed up candidate selection
        if IS_mode == 'IS_random':
            self.get_is_candidates = self.get_is_candidates_random
        elif IS_mode == 'IS_dual':
            self.get_is_candidates = self.get_is_candidates_dual
        elif IS_mode == 'IS_dual_ind':
            self.get_is_candidates = self.get_is_candidates_dual_ind
        else:
            raise ValueError(f"Unknown IS_mode: {IS_mode}")

    def get_IS_info(self, is_position):
        if len(is_position) != 3:
            raise ValueError(f"Expected is_position to have length 3, but got {len(is_position)}. "
                             f"This indicates that the IS position is not valid.")

        is_inp, lubs_layer_idx, pos = is_position[0], is_position[1], is_position[2]
        split_value = 0

        is_info = IS_info(is_inp=is_inp, layer_idx=lubs_layer_idx, pos=pos, split_value=split_value)
        return is_info

    def candidate_to_IS_info(self, candidate):
        # candidate = [is_inp,lubs_layer_idx, pos]
        is_info = self.get_IS_info(is_position=candidate)
        return is_info

    # todo
    def get_is_candidates_dual(self, IARb, layer_idx, DN=None, RelAna=None):
        if RelAna is not None:
            target_dim = RelAna.inp1_correct_label
            C = torch.zeros_like(IARb.inp1_lbs[-1])
            C[target_dim] = 1
        else:
            C = torch.ones_like(IARb.inp1_lbs[-1])
        is_order = get_relational_order_is(IARb.net, C, self.IS_mode, layer_idx, IARb.shapes, IARb.inp1_lbs, IARb.inp1_ubs,
                                           IARb.inp2_lbs, IARb.inp2_ubs, IARb.d_lbs, IARb.d_ubs)
        # is_order = [[is_inp,lubs_layer_idx, pos], ...]
        return None, is_order

    def get_is_candidates_dual_ind(self, IARb, layer_idx, DNI=None, RelAna=None, input_ab='A'):
        if RelAna is not None:
            target_dim = RelAna.inp1_correct_label
            C = torch.zeros_like(IARb.inp1_lbs[-1])
            C[target_dim] = 1
        else:
            C = torch.ones_like(IARb.inp1_lbs[-1])
        if input_ab == 'A':
            lbs = IARb.inp1_lbs
            ubs = IARb.inp1_ubs
        elif input_ab == 'B':
            lbs = IARb.inp2_lbs
            ubs = IARb.inp2_ubs
        else:
            raise ValueError(f"Unknown input_ab: {input_ab}")
        is_order = get_relational_order_is_ind(IARb.net, C, self.IS_mode, layer_idx, IARb.shapes, lbs, ubs)
        # is_order = [[is_inp, lubs_layer_idx, pos], ...]

        if input_ab == 'B':
            # change is_inp from 'A' to 'B'
            for item in is_order:
                item[0] = 'B'

        return None, is_order

    def run_iterative_IS_backend(self, IARb, RelAna, time_budget=1000, lp_analysis=False):
        self.IARb = IARb
        bfs_rsis_list = [self]
        RSIS_list = []
        self.status = Status.UNKNOWN
        if self.split_count >= self.split_limit:
            return Status.UNKNOWN, None, None

        time_start_grd = time.time()  # time tracking start
        split_count = 0
        bfs_res = None
        rsis_order_selection = None
        first_relu_layer_skip = False  # skip the first relu layer, because splitting relu at the first layer by RSM and RSZ is same
        termination_condition = (len(bfs_rsis_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        while not termination_condition:
            left_time_budget = time_budget - (time.time() - time_start_grd)
            bfs_res, next_bfs_list = self.bfs_loop(bfs_rsis_list, left_time_budget, RelAna, first_relu_layer_skip, RSIS_list, split_count, rsis_order_selection)
            if len(next_bfs_list) == 1 and next_bfs_list[0] == self:
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nNo further splits possible on Root problem. Terminating RS process.\n")
                break
            bfs_rsis_list = next_bfs_list
            split_count += 1
            termination_condition = (len(bfs_rsis_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        return bfs_res

    def bfs_loop(self, bfs_list, time_budget, RelAna, first_relu_layer_skip, IS_list, split_count, is_order_selection):
        """
        for each bfs is instance in the bfs list:
        1. backsubstitution based on the is history
        2. candidate selection
        3. lp analysis
        """
        time_start_grd = time.time()
        is_res_level = []  # to keep is sets of this candidate excluding unreachable ones
        next_bfs_list = []
        for bfs_idx in range(len(bfs_list)):
            time_progress = time.time() - time_start_grd
            exe_flag = False
            if time_progress > time_budget:
                is_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                next_bfs_list = []
                break
            time_bfs_start = time.time()

            # current IS instance
            bfs_is = bfs_list[bfs_idx]
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## BFS IS instance: {bfs_is.name}\n")

            # ---- update bounds by backsubstitution ----
            update_start_time = time.time()
            curr_iarb = copy.deepcopy(bfs_is.IARb)
            if len(bfs_is.IS_history) > 0 and curr_iarb.feasible_flag is not False:
                split_layer_list = [is_info.layer_idx for is_info in bfs_is.IS_history]
                split_layer_list = list(set(split_layer_list))  # unique layer indices, e.g., [1,1,3,3,5] -> [1,3,5]
                curr_iarb = curr_iarb.update_bounds_IAR(is_history=bfs_is.IS_history, split_layer_list=split_layer_list)
                if curr_iarb is None:
                    # backsubstitution failed
                    bfs_is.IARb.feasible_flag = False  # mark as infeasible
                    curr_iarb = bfs_is.IARb

                # ---- debug ----
                if curr_iarb.feasible_flag is not False:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Backsubstitution after applying IS history:\n")
                        for dim in range(len(curr_iarb.inp1_lbs[-1])):
                            f.write(f"{dim}: {curr_iarb.inp1_lbs[-1][dim].item():.7f}, {curr_iarb.inp1_ubs[-1][dim].item():.7f}, "
                                    f"{curr_iarb.inp2_lbs[-1][dim].item():.7f}, {curr_iarb.inp2_ubs[-1][dim].item():.7f}, "
                                    f"{curr_iarb.d_lbs[-1][dim].item():.7f}, {curr_iarb.d_ubs[-1][dim].item():.7f}\n")
                # ---- debug ----
            update_time = time.time() - update_start_time
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\nTime for backsubstitution: {update_time:.2f} seconds\n")

            # ---- candidate selection ----
            candidate_selection_start = time.time()
            start_layer = bfs_is.selection_start_layer if self.IS_mode != 'IS_dual_ind' else 0
            for idx in range(start_layer, len(curr_iarb.net)):
                if exe_flag:
                    break  # proceed to next bfs is instance
                layer = curr_iarb.net[idx]
                time_progress = time.time() - time_start_grd
                if time_progress > time_budget:
                    is_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                    next_bfs_list = []
                    break
                if layer.type is LayerType.ReLU and first_relu_layer_skip is False:
                    if self.IS_mode == 'IS_dual':
                        if curr_iarb.feasible_flag is False and bfs_is.IS_candidates != {}:
                            is_candidates = bfs_is.IS_candidates[idx]
                        else:
                            if is_order_selection:
                                is_order_selection, is_candidates = self.get_is_candidates(curr_iarb, idx, is_order_selection)
                                bfs_is.IS_candidates[idx] = is_candidates
                            else:
                                is_order_selection, is_candidates = self.get_is_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna)  # candidates: [[is_type, lubs_layer_idx, pos], ...]
                                bfs_is.IS_candidates[idx] = is_candidates
                    elif self.IS_mode == 'IS_dual_ind':
                        if bfs_is.next_candidate_input == 'A':
                            if curr_iarb.feasible_flag is False and bfs_is.IS_candidates_A != {}:
                                is_candidates = bfs_is.IS_candidates_A[idx]
                            else:
                                if is_order_selection:
                                    is_order_selection, is_candidates = self.get_is_candidates(curr_iarb, idx, is_order_selection, input_ab='A')
                                    bfs_is.IS_candidates_A[idx] = is_candidates
                                else:
                                    is_order_selection, is_candidates = self.get_is_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna,
                                                                                               input_ab='A')  # candidates: [[is_type, lubs_layer_idx, pos], ...]
                                    bfs_is.IS_candidates_A[idx] = is_candidates
                        elif bfs_is.next_candidate_input == 'B':
                            if curr_iarb.feasible_flag is False and bfs_is.IS_candidates_B != {}:
                                is_candidates = bfs_is.IS_candidates_B[idx]
                            else:
                                if is_order_selection:
                                    is_order_selection, is_candidates = self.get_is_candidates(curr_iarb, idx, is_order_selection, input_ab='B')
                                    bfs_is.IS_candidates_B[idx] = is_candidates
                                else:
                                    is_order_selection, is_candidates = self.get_is_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna,
                                                                                               input_ab='B')  # candidates: [[is_type, lubs_layer_idx, pos], ...]
                                    bfs_is.IS_candidates_B[idx] = is_candidates
                        else:
                            raise ValueError(f"Unknown next_candidate_input: {bfs_is.next_candidate_input}")
                elif idx == len(curr_iarb.net) - 1:  # last layer
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo IS candidates found\n")
                    break
                else:
                    bfs_is.selection_start_layer += 1
                    continue

                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\n### IS candidates at layer {idx}\n")
                if len(is_candidates) == 0 or is_candidates is None:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo IS candidates found\n")
                    continue
                else:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        for candidate in is_candidates:
                            f.write(f"type: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")

                candidate_selection_time = time.time() - candidate_selection_start
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nTime for candidate selection: {candidate_selection_time:.2f} seconds\n")

                progress_time = time.time() - time_start_grd
                if progress_time > time_budget:
                    break

                # ---- split LP analysis ----
                lp_analysis_start = time.time()
                exe_flag = False
                for candidate in is_candidates:
                    progress_time = time.time() - time_start_grd
                    if progress_time > time_budget:
                        is_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                        next_bfs_list = []
                        break
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Candidate\ntype: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")
                    is_info = self.candidate_to_IS_info(candidate)
                    # check whether the same split has been done
                    prev_layer_pos = [(h.is_type, h.layer_idx, h.pos) for h in bfs_is.IS_history]
                    if (is_info.is_type, is_info.layer_idx, is_info.pos) in prev_layer_pos:
                        # try next candidate
                        continue
                    # check whether the same split has failed before
                    prev_failed_splits = [(h.is_type, h.layer_idx, h.pos) for h in bfs_is.IS_failed_list]
                    if (is_info.is_type, is_info.layer_idx, is_info.pos) in prev_failed_splits:
                        # try next candidate
                        continue
                    is_info_1 = copy.deepcopy(is_info)
                    is_info_2 = copy.deepcopy(is_info)
                    is_info_1.is_type = f"{is_info.is_inp}1"
                    is_info_2.is_type = f"{is_info.is_inp}2"
                    is1 = copy.deepcopy(bfs_is)
                    is1.name = f"{bfs_is.name}_{is_info_1.is_type}"
                    is1.IS_history.append(is_info_1)
                    is2 = copy.deepcopy(bfs_is)
                    is2.name = f"{bfs_is.name}_{is_info_2.is_type}"
                    is2.IS_history.append(is_info_2)
                    status1, status2 = self.perform_is_1_2(is1, is2, RelAna, bfs_is.IARb)
                    # aggregate results of the two branches
                    is_analyzed = []
                    if status1 == Status.UNREACHABLE or status2 == Status.UNREACHABLE:
                        # free up memory
                        del is1
                        del is2
                        gc.collect()
                        continue
                    else:
                        is_analyzed.append((is1, status1))
                        is_analyzed.append((is2, status2))

                    exe_flag = True
                    for curr_is, curr_status in is_analyzed:
                        curr_is.IARb = curr_iarb
                        if self.IS_mode == 'IS_dual_ind':
                            if bfs_is.next_candidate_input == 'A':
                                curr_is.next_candidate_input = 'B'
                            else:
                                curr_is.next_candidate_input = 'A'
                        if curr_status == Status.UNKNOWN:
                            next_bfs_list.append(curr_is)
                            is_res_level.append(curr_is)
                        elif curr_status == Status.ADV_EXAMPLE:
                            IS_list.append(curr_is)
                            return (curr_status, curr_is.inp1_label, curr_is.inp2_label), []
                        elif curr_status == Status.VERIFIED:
                            IS_list.append(curr_is)
                            is_res_level.append(curr_is)
                        elif curr_status == Status.UNREACHABLE:
                            raise ValueError("This case should have been handled earlier.")
                        else:
                            raise ValueError(f"Unknown status: {curr_status}")
                    if exe_flag:
                        break
                if not exe_flag:
                    bfs_is.selection_start_layer += 1
            if not exe_flag:
                return (Status.UNKNOWN, None, None), []  # we cannot reach satisfying the output specification

        # ---- show results of bfs loop ----
        bfs_time = time.time() - time_bfs_start
        if len(is_res_level) > 0:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## Summary of splitting at layer (split count: {split_count})\n")
                f.write(f"- Time for IS candidates: {bfs_time:.2f} seconds\n")
                for is_ in is_res_level:
                    f.write(f"{is_.name}, status: {is_.status}, split count: {is_.split_count}, time: {bfs_time:.2f}\n")
                    for dim, dist in is_.relational_output_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")

        status, inp1_out_label, inp2_out_label = self.collect_final_status(IS_list + is_res_level)

        return (status, inp1_out_label, inp2_out_label), next_bfs_list

    def perform_is_1_2(self, is1, is2, RelAna, IARb):
        for curr_is in [is1, is2]:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## Relational analysis of {curr_is.name}\n")
            ra_time_start = time.time()
            curr_status, inp1_label, inp2_label, rel_out_dist = relational_analysis_back(IARb=IARb, RelAna=RelAna, IS_history=curr_is.IS_history,
                                                                                         log_file=self.log_file)
            if curr_status == Status.UNREACHABLE:
                return Status.UNREACHABLE, Status.UNREACHABLE
            curr_is.split_count += 1
            curr_is.relational_output_dist = rel_out_dist
            curr_is.status = curr_status
            curr_is.inp1_label = inp1_label
            curr_is.inp2_label = inp2_label
            ra_time_end = time.time()
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n### Relational analysis result of {curr_is.name}\n")
                f.write(f"Status: {curr_status}\n")
                if rel_out_dist is not None:
                    for dim, dist in rel_out_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")
                f.write(f"time: {(ra_time_end - ra_time_start):.2f} seconds\n")
        return is1.status, is2.status
