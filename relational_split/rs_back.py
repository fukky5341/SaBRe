import copy
import time
import torch
import gc
from common.network import LayerType
from relational_split.rs_handler import RS_handler
from common import Status
from relational_property.relational_analysis import relational_analysis_back, RelationalProperty
from dual.dual_network import get_relational_order


class RSInfo:
    def __init__(self, rs_type, layer_idx, pos, split_value):

        self.rs_type = rs_type
        self.layer_idx = layer_idx  # int
        self.pos = pos  # int
        self.split_value = split_value


class RS(RS_handler):
    def __init__(self, log_file='log', RS_mode='RS_random_ABCD', split_limit=10, relational_prop=RelationalProperty.GLOBAL_ROBUSTNESS):

        self.IARb = None
        self.RS_mode = RS_mode
        self.RS_history = []
        self.RS_failed_list = []
        self.split_limit = split_limit
        self.log_file = log_file
        self.name = "RS"
        self.split_count = 0
        self.status = Status.UNKNOWN
        self.inp1_label = None
        self.inp2_label = None
        self.relational_output_dist = None
        self.relational_prop = relational_prop
        self.RS_candidates = {}
        self.selection_start_layer = 0
        if RS_mode in ['RS_random_Z']:
            self.get_rs_candidates = self.get_rs_candidates_random
        elif RS_mode in ['RS_dual_Z']:
            self.get_rs_candidates = self.get_rs_candidates_dual
        else:
            raise ValueError(f"Unknown RS mode: {RS_mode}. Expected 'RS_random_Z' or 'RS_dual_Z'.")

    def get_RSInfo(self, rs_type, rs_position, IARb):
        if len(rs_position) != 2:
            raise ValueError(f"Expected rs_position to have length 2, but got {len(rs_position)}. "
                             f"This indicates that the RS position is not valid.")
        lubs_layer_idx, pos = rs_position[0], rs_position[1]
        if rs_type == 'RSZ':
            split_value = 0
        else:
            raise ValueError(f"Unknown RS type: {rs_type}. Expected 'RSZ'.")

        rs_info = RSInfo(rs_type=rs_type, layer_idx=lubs_layer_idx, pos=pos, split_value=split_value)
        return rs_info

    def candidate_to_RSInfo(self, candidate, IARb):
        # candidate = [rs_type, lubs_layer_idx, pos]
        rs_type = candidate[0]
        rs_position = candidate[1:]
        if rs_type in ('RSZ'):
            rs_info = self.get_RSInfo(rs_type=rs_type, rs_position=rs_position, IARb=IARb)
        else:
            raise ValueError(f"Unknown RS type: {candidate[0]}. Expected 'RSZ'.")
        return rs_info

    def get_rs_candidates_dual(self, IARb, layer_idx, DN=None, RelAna=None):
        if RelAna is not None:
            target_dim = RelAna.inp1_correct_label
            C = torch.zeros_like(IARb.inp1_lbs[-1])
            C[target_dim] = 1
        else:
            C = torch.ones_like(IARb.inp1_lbs[-1])
        rs_order = get_relational_order(IARb.net, C, self.RS_mode, layer_idx, IARb.shapes, IARb.inp1_lbs, IARb.inp1_ubs,
                                        IARb.inp2_lbs, IARb.inp2_ubs, IARb.d_lbs, IARb.d_ubs)
        # rs_order = [[rs_type,lubs_layer_idx, pos], ...]
        return None, rs_order

    def get_rs_candidates_random(self, IARb, layer_idx, RelAna=None):
        """
        RS_random_Z: randomly selects from RSZ candidates
        """

        rs_order = self.get_rs_random_order(self.RS_mode, layer_idx, IARb.inp1_lbs, IARb.inp1_ubs, IARb.inp2_lbs, IARb.inp2_ubs,
                                            IARb.d_lbs, IARb.d_ubs)
        # rs_order = [[rs_type, lubs_layer_idx, pos], ...]
        return None, rs_order

    def run_iterative_RS_backend(self, IARb, RelAna, time_budget=1000, lp_analysis=False):
        """
        bfs: breadth-first search

        bfs_rs_list: list of RS instances to be explored
        bfs_relAna_list: list of RelationalAnalysis instances corresponding to bfs_rs_list

        RS_list: list of RS instances that have been verified or found adversarial examples
        RS_history: list of RSInfo instances representing how splitting has been done
        """

        self.IARb = IARb
        bfs_rs_list = [self]
        RS_list = []
        self.status = Status.UNKNOWN
        if self.split_count >= self.split_limit:
            return Status.UNKNOWN, None, None

        time_start_grd = time.time()  # time tracking start
        split_count = 0
        bfs_res = None
        rs_order_selection = None
        first_relu_layer_skip = False  # skip the first relu layer, because splitting relu at the first layer by RSM and RSZ is same
        termination_condition = (len(bfs_rs_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        while not termination_condition:
            left_time_budget = time_budget - (time.time() - time_start_grd)
            bfs_res, next_bfs_list = self.bfs_loop(bfs_rs_list, left_time_budget, RelAna, first_relu_layer_skip, RS_list, split_count, rs_order_selection)
            if len(next_bfs_list) == 1 and next_bfs_list[0] == self:
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nNo further splits possible on Root problem. Terminating RS process.\n")
                break
            bfs_rs_list = next_bfs_list
            split_count += 1
            termination_condition = (len(bfs_rs_list) == 0) or (split_count >= self.split_limit) or ((time.time() - time_start_grd) >= time_budget)

        return bfs_res

    def bfs_loop(self, bfs_list, time_budget, RelAna, first_relu_layer_skip, RS_list, split_count, rs_order_selection):
        """
        for each bfs rs instance in the bfs list:
        1. backsubstitution based on the rs history
        2. candidate selection
        3. lp analysis
        """
        time_start_grd = time.time()
        rs_res_level = []  # to keep rs sets of this candidate excluding unreachable ones
        next_bfs_list = []
        for bfs_idx in range(len(bfs_list)):
            time_progress = time.time() - time_start_grd
            exe_flag = False
            if time_progress > time_budget:
                rs_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                next_bfs_list = []
                break
            time_bfs_start = time.time()

            # current RS instance
            bfs_rs = bfs_list[bfs_idx]
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## BFS RS instance: {bfs_rs.name}\n")

            # ---- update bounds by backsubstitution ----
            update_start_time = time.time()
            curr_iarb = copy.deepcopy(bfs_rs.IARb)
            if len(bfs_rs.RS_history) > 0 and curr_iarb.feasible_flag is not False:
                split_layer_list = [rs_info.layer_idx for rs_info in bfs_rs.RS_history]
                split_layer_list = list(set(split_layer_list))  # unique layer indices, e.g., [1,1,3,3,5] -> [1,3,5]
                curr_iarb = curr_iarb.update_bounds_IAR(rs_history=bfs_rs.RS_history, split_layer_list=split_layer_list)
                if curr_iarb is None:
                    # backsubstitution failed
                    bfs_rs.IARb.feasible_flag = False  # mark as infeasible
                    curr_iarb = bfs_rs.IARb

                # ---- debug ----
                if curr_iarb.feasible_flag is not False:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Backsubstitution after applying RS history:\n")
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
            for idx in range(bfs_rs.selection_start_layer, len(curr_iarb.net)):
                if exe_flag:
                    break  # proceed to next bfs rs instance
                layer = curr_iarb.net[idx]
                time_progress = time.time() - time_start_grd
                if time_progress > time_budget:
                    rs_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                    next_bfs_list = []
                    break
                if layer.type is LayerType.ReLU and first_relu_layer_skip is False:
                    if curr_iarb.feasible_flag is False and bfs_rs.RS_candidates != {}:
                        rs_candidates = bfs_rs.RS_candidates[idx]
                    else:
                        if rs_order_selection:
                            rs_order_selection, rs_candidates = self.get_rs_candidates(curr_iarb, idx, rs_order_selection)
                            bfs_rs.RS_candidates[idx] = rs_candidates
                        else:
                            rs_order_selection, rs_candidates = self.get_rs_candidates(IARb=curr_iarb, layer_idx=idx, RelAna=RelAna)  # candidates: [[rs_type, lubs_layer_idx, pos], ...]
                            bfs_rs.RS_candidates[idx] = rs_candidates
                    if len(rs_candidates) == 0:
                        bfs_rs.selection_start_layer += 1
                        continue
                elif idx == len(curr_iarb.net) - 1:  # last layer
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo RS candidates found\n")
                    break
                else:
                    bfs_rs.selection_start_layer += 1
                    continue

                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\n### RS candidates at layer {idx}\n")
                if len(rs_candidates) == 0 or rs_candidates is None:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\nNo RS candidates found\n")
                    continue
                else:
                    with open(f"{self.log_file}log.md", 'a') as f:
                        for candidate in rs_candidates:
                            f.write(f"type: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")

                candidate_selection_time = time.time() - candidate_selection_start
                with open(f"{self.log_file}log.md", 'a') as f:
                    f.write(f"\nTime for candidate selection: {candidate_selection_time:.2f} seconds\n")

                progress_time = time.time() - time_start_grd
                if progress_time > time_budget:
                    break

                # ---- split LP analysis ----
                lp_analysis_start = time.time()
                for candidate in rs_candidates:
                    progress_time = time.time() - time_start_grd
                    if progress_time > time_budget:
                        rs_res_level.extend(bfs_list[bfs_idx:])  # list = [a,b,c,d,e], list[3:] = [d,e]
                        next_bfs_list = []
                        break
                    with open(f"{self.log_file}log.md", 'a') as f:
                        f.write(f"\n### Candidate\ntype: {candidate[0]}, layer: {candidate[1]}, pos: {candidate[2]}\n")
                    rs_info = self.candidate_to_RSInfo(candidate, bfs_rs.IARb)
                    # check whether the same split has been done
                    prev_layer_pos = [(h.layer_idx, h.pos) for h in bfs_rs.RS_history]
                    if (rs_info.layer_idx, rs_info.pos) in prev_layer_pos:
                        # try next candidate
                        continue
                    # check whether the same split has failed before
                    prev_failed_splits = [(h.layer_idx, h.pos, h.rs_type) for h in bfs_rs.RS_failed_list]
                    if (rs_info.layer_idx, rs_info.pos, rs_info.rs_type) in prev_failed_splits:
                        # try next candidate
                        continue
                    rs_info_1 = copy.deepcopy(rs_info)
                    rs_info_2 = copy.deepcopy(rs_info)
                    rs_info_1.rs_type = f"{rs_info.rs_type}1"
                    rs_info_2.rs_type = f"{rs_info.rs_type}2"
                    rs1 = copy.deepcopy(bfs_rs)
                    rs1.name = f"{bfs_rs.name}_{rs_info_1.rs_type}"
                    rs1.RS_history.append(rs_info_1)
                    rs2 = copy.deepcopy(bfs_rs)
                    rs2.name = f"{bfs_rs.name}_{rs_info_2.rs_type}"
                    rs2.RS_history.append(rs_info_2)
                    status1, status2 = self.perform_rs_1_2(rs1, rs2, RelAna, bfs_rs.IARb, rs_type=rs_info.rs_type)
                    # aggregate results of the two branches
                    rs_analyzed = []
                    if status1 == Status.UNREACHABLE or status2 == Status.UNREACHABLE:
                        # free up memory
                        del rs1
                        del rs2
                        gc.collect()
                        continue
                    else:
                        rs_analyzed.append((rs1, status1))
                        rs_analyzed.append((rs2, status2))

                    exe_flag = True
                    for curr_rs, curr_status in rs_analyzed:
                        curr_rs.IARb = curr_iarb
                        if curr_status == Status.UNKNOWN:
                            next_bfs_list.append(curr_rs)
                            rs_res_level.append(curr_rs)
                        elif curr_status == Status.ADV_EXAMPLE:
                            RS_list.append(curr_rs)
                            return (curr_status, curr_rs.inp1_label, curr_rs.inp2_label), []
                        elif curr_status == Status.VERIFIED:
                            RS_list.append(curr_rs)
                            rs_res_level.append(curr_rs)
                        elif curr_status == Status.UNREACHABLE:
                            raise ValueError("This case should have been handled earlier.")
                        else:
                            raise ValueError(f"Unknown status: {curr_status}")
                    if exe_flag:
                        break
                if not exe_flag:
                    bfs_rs.selection_start_layer += 1
            if not exe_flag:
                return (Status.UNKNOWN, None, None), []  # we cannot reach satisfying the output specification

        # ---- show results of bfs loop ----
        bfs_time = time.time() - time_bfs_start
        if len(rs_res_level) > 0:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n## Summary of splitting (split count: {split_count})\n")
                f.write(f"- Time for RS candidates: {bfs_time:.2f} seconds\n")
                for rs_ in rs_res_level:
                    f.write(f"{rs_.name}, status: {rs_.status}, split count: {rs_.split_count}, time: {bfs_time:.2f}\n")
                    for dim, dist in rs_.relational_output_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")

        status, inp1_out_label, inp2_out_label = self.collect_final_status(RS_list + rs_res_level)

        return (status, inp1_out_label, inp2_out_label), next_bfs_list

    def perform_rs_1_2(self, rs1, rs2, RelAna, IARb, rs_type):
        for curr_rs in [rs1, rs2]:
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n### Relational analysis {rs_type} of {curr_rs.name}\n")
            ra_time_start = time.time()
            curr_status, inp1_label, inp2_label, rel_out_dist = relational_analysis_back(IARb=IARb, RelAna=RelAna, RS_history=curr_rs.RS_history,
                                                                                         log_file=self.log_file)
            if curr_status == Status.UNREACHABLE:
                return Status.UNREACHABLE, Status.UNREACHABLE
            curr_rs.split_count += 1
            curr_rs.relational_output_dist = rel_out_dist
            curr_rs.status = curr_status
            curr_rs.inp1_label = inp1_label
            curr_rs.inp2_label = inp2_label
            ra_time_end = time.time()
            with open(f"{self.log_file}log.md", 'a') as f:
                f.write(f"\n#### Relational analysis {rs_type} result of {curr_rs.name}\n")
                f.write(f"Status: {curr_status}\n")
                if rel_out_dist is not None:
                    for dim, dist in rel_out_dist.items():
                        f.write(f"Output dim: {dim}, lower bound: {dist[0]:.7f}, upper bound: {dist[1]:.7f}\n")
                f.write(f"time: {(ra_time_end - ra_time_start):.2f} seconds\n")
        return rs1.status, rs2.status
