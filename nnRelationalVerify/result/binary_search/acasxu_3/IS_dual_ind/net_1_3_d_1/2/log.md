## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 12.14935128


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3073158, 6.1510162, -7.3073158, 6.1510162, -13.4583321, 13.4583311)
1: (-5.3463583, 8.2614708, -5.3463583, 8.2614708, -13.6078291, 13.6078291)
2: (-7.7331066, 6.6210580, -7.7331066, 6.6210580, -14.3541632, 14.3541632)
3: (-2.6750574, 11.7156067, -2.6750574, 11.7156067, -14.3906612, 14.3906622)
4: (-10.0698843, 7.5773277, -10.0698843, 7.5773277, -17.6472130, 17.6472130)

## BASE Result
execution time: IAR + LP analysis = 1.33 + 1.14 = 2.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -12.2720720, upper bound: 12.2720720


# Binary Search by BASE starts (time budget: 1197.53 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=13.458332061767578
rel_dist={0: [-12.27133016030539, 12.27133016030539]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=13.458332061767578
rel_dist={0: [-12.270288718386343, 12.270288718386343]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=13.458332061767578
rel_dist={0: [-12.269294519006817, 12.269294519006817]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=13.458332061767578
rel_dist={0: [-12.267428912659181, 12.267428912668205]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=13.458332061767578
rel_dist={0: [-12.266075169862887, 12.266075169858812]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=13.458332061767578
rel_dist={0: [-12.265364319813424, 12.265364319813422]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=13.458332061767578
rel_dist={0: [-12.264960605505483, 12.264960605501642]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=13.458332061767578
rel_dist={0: [-12.264476255980234, 12.264476255980234]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=13.458332061767578
rel_dist={0: [-12.264223046454173, 12.264223046453044]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=13.458332061767578
rel_dist={0: [-12.264089660028645, 12.264089660028077]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=13.458332061767578
rel_dist={0: [-12.264022967421505, 12.264022967420932]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=13.458332061767578
rel_dist={0: [-12.263989620702842, 12.263989620702837]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=13.458332061767578
rel_dist={0: [-12.263972948045687, 12.263972948045613]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=13.458332061767578
rel_dist={0: [-12.263964611984576, 12.263964611984576]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=13.458332061767578
rel_dist={0: [-12.263960445676606, 12.263960445676588]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=13.458332061767578
rel_dist={0: [-12.26395932349288, 12.263958364872]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=13.458332061767578
rel_dist={0: [-12.263957452500073, 12.26395781929547]}

## Binary Search Result
Binary search time: 44.59 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1152.94 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.87
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9298306, 6.0086184, -12.9105215, 13.7462702
1: -5.1180196, 8.9492035, -5.0767632, 8.1064682, -13.2244873, 14.0259666
2: -7.5139894, 7.3766956, -7.3569617, 6.4872084, -14.0011978, 14.7336540
3: -2.9837260, 12.5334835, -2.6160150, 11.3937569, -14.3774834, 15.1494980
4: -9.9552670, 8.3942471, -9.6144686, 7.4212675, -17.3765335, 18.0087147

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.40 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.08 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.08
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2139942, upper bound: 12.0628160
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2139942, upper bound: 12.1359441
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1881671, upper bound: 12.0632178
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0545743
time: 0.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1881671, upper bound: 12.1205452
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.43 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2139942, upper bound: 12.0628160
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.2139942, upper bound: 12.1359441
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.1881671, upper bound: 12.0632178
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0545743
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.1881671, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2453409, upper bound: 12.1293270
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0545743
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0545743
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.46 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0545743
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0545743
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664617
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.41 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.42 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664617
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205447
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205448
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.46 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205447
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.46
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205448
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.02
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.39 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1248949
time: 0.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.54 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.21 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.21
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.44 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.55 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.55
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205447
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205447
time: 0.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205447
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.65
Output dim: 0, lower bound: -12.1146670, upper bound: 12.1205447
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.10 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.10
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.41 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.58
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.78 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.78
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.14 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.38 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.66
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.57 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.41 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.64 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
time: 0.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.66 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
Binary search (step 6): status=Status.VERIFIED, low=0.1984375, high=0.2000000, mid=0.1984375, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 7) starts
Candidate diff: 0.1992187


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.59
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1341262
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205447
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.53 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205447
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
Binary search (step 7): status=Status.VERIFIED, low=0.1992187, high=0.2000000, mid=0.1992187, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 8) starts
Candidate diff: 0.1996094


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.09 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.93 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.93
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 8): status=Status.VERIFIED, low=0.1996094, high=0.2000000, mid=0.1996094, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 9) starts
Candidate diff: 0.1998047


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
time: 0.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1205448
Binary search (step 9): status=Status.VERIFIED, low=0.1998047, high=0.2000000, mid=0.1998047, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 10) starts
Candidate diff: 0.1999023


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
Binary search (step 10): status=Status.VERIFIED, low=0.1999023, high=0.2000000, mid=0.1999023, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 11) starts
Candidate diff: 0.1999512


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.49
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.41 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.71 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.71
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 11): status=Status.VERIFIED, low=0.1999512, high=0.2000000, mid=0.1999512, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 12) starts
Candidate diff: 0.1999756


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.37 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.48 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.48
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560732
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560732
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.60 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560732
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560732
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
Binary search (step 12): status=Status.VERIFIED, low=0.1999756, high=0.2000000, mid=0.1999756, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 13) starts
Candidate diff: 0.1999878


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146673, upper bound: 12.0560733
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1146673, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 13): status=Status.VERIFIED, low=0.1999878, high=0.2000000, mid=0.1999878, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 14) starts
Candidate diff: 0.1999939


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.39
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560732
time: 0.56 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.53 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560732
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 14): status=Status.VERIFIED, low=0.1999939, high=0.2000000, mid=0.1999939, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 15) starts
Candidate diff: 0.1999969


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.38 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.40 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1248949
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.44 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 15): status=Status.VERIFIED, low=0.1999969, high=0.2000000, mid=0.1999969, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 16) starts
Candidate diff: 0.1999985


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.05 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.05
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
time: 0.38 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.42 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.42
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.53 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1146671, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.53
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 16): status=Status.VERIFIED, low=0.1999985, high=0.2000000, mid=0.1999985, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 17) starts
Candidate diff: 0.1999992


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2717727, upper bound: 12.2240781
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -7.3073158, 6.1510162, -11.5154934, 12.5373774
1: -3.9401655, 7.1854095, -5.3463583, 8.2614708, -12.2016363, 12.5317650
2: -5.7179251, 5.7401724, -7.7331066, 6.6210580, -12.3389826, 13.4732790
3: -2.2893574, 9.6674576, -2.6750574, 11.7156067, -14.0049534, 12.3425150
4: -7.5769539, 6.5658350, -10.0698843, 7.5773277, -15.1542788, 16.6357193

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -7.3073158, 6.1510162, -13.0529194, 14.1237555
1: -5.1180196, 8.9492035, -5.3463583, 8.2614708, -13.3794899, 14.2955618
2: -7.5139894, 7.3766956, -7.7331066, 6.6210580, -14.1350451, 15.1097975
3: -2.9837260, 12.5334835, -2.6750574, 11.7156067, -14.6993332, 15.2085409
4: -9.9552670, 8.3942471, -10.0698843, 7.5773277, -17.5325928, 18.4641304

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.03
Output dim: 0, lower bound: -12.2240781, upper bound: 12.2240781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -5.3644776, 5.2300639, -10.5945387, 10.5945415
1: -3.9401655, 7.1854095, -3.9401655, 7.1854095, -11.1255751, 11.1255751
2: -5.7179251, 5.7401724, -5.7179251, 5.7401724, -11.4580965, 11.4580965
3: -2.2893574, 9.6674576, -2.2893574, 9.6674576, -11.9568129, 11.9568148
4: -7.5769539, 6.5658350, -7.5769539, 6.5658350, -14.1427889, 14.1427889

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -5.3644776, 5.2300639, -6.9019041, 6.8164396, -12.1809177, 12.1319656
1: -3.9401655, 7.1854095, -5.1180196, 8.9492035, -12.8893681, 12.3034286
2: -5.7179251, 5.7401724, -7.5139894, 7.3766956, -13.0946198, 13.2541590
3: -2.2893574, 9.6674576, -2.9837260, 12.5334835, -14.8228359, 12.6511841
4: -7.5769539, 6.5658350, -9.9552670, 8.3942471, -15.9711990, 16.5210991

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
time: 0.36 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -5.3644776, 5.2300639, -12.1319666, 12.1809177
1: -5.1180196, 8.9492035, -3.9401655, 7.1854095, -12.3034286, 12.8893633
2: -7.5139894, 7.3766956, -5.7179251, 5.7401724, -13.2541590, 13.0946188
3: -2.9837260, 12.5334835, -2.2893574, 9.6674576, -12.6511841, 14.8228359
4: -9.9552670, 8.3942471, -7.5769539, 6.5658350, -16.5211029, 15.9711981

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.9019041, 6.8164396, -6.9019041, 6.8164396, -13.7183437, 13.7183437
1: -5.1180196, 8.9492035, -5.1180196, 8.9492035, -14.0672226, 14.0672226
2: -7.5139894, 7.3766956, -7.5139894, 7.3766956, -14.8906822, 14.8906832
3: -2.9837260, 12.5334835, -2.9837260, 12.5334835, -15.5172100, 15.5172100
4: -9.9552670, 8.3942471, -9.9552670, 8.3942471, -18.3495083, 18.3495083

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949
time: 0.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.2389104, upper bound: 12.0664622
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.2389104, upper bound: 12.1359810
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.0451486, upper bound: 12.1146674
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.1883128, upper bound: 12.0662481
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.1883128, upper bound: 12.1205452
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.38
Output dim: 0, lower bound: -12.1146674, upper bound: 12.1248949

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -5.3644776, 5.2300639, -10.2255659, 10.2914228
1: -3.6672919, 6.8235912, -3.9401655, 7.1854095, -10.8527012, 10.7637568
2: -5.3162446, 5.4366350, -5.7179251, 5.7401724, -11.0564175, 11.1545601
3: -2.1726148, 9.0977764, -2.2893574, 9.6674576, -11.8400726, 11.3871336
4: -7.0500698, 6.2366433, -7.5769539, 6.5658350, -13.6159048, 13.8135939

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9955039, 4.9269452, -6.9019041, 6.8164396, -11.8119431, 11.8288498
1: -3.6672919, 6.8235912, -5.1180196, 8.9492035, -12.6164951, 11.9416103
2: -5.3162446, 5.4366350, -7.5139894, 7.3766956, -12.6929398, 12.9506245
3: -2.1726148, 9.0977764, -2.9837260, 12.5334835, -14.7060947, 12.0815029
4: -7.0500698, 6.2366433, -9.9552670, 8.3942471, -15.4443130, 16.1919060

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2498351, upper bound: 12.1341262
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -5.3644776, 5.2300639, -11.8511896, 11.9933643
1: -4.9106646, 8.7676888, -3.9401655, 7.1854095, -12.0960732, 12.7078476
2: -7.2061682, 7.2035112, -5.7179251, 5.7401724, -12.9463406, 12.9214363
3: -2.9020705, 12.1498394, -2.2893574, 9.6674576, -12.5695286, 14.4391918
4: -9.5700588, 8.1782074, -7.5769539, 6.5658350, -16.1358948, 15.7551613

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.6211271, 6.6288872, -6.9019041, 6.8164396, -13.4375668, 13.5307884
1: -4.9106646, 8.7676888, -5.1180196, 8.9492035, -13.8598680, 13.8857069
2: -7.2061682, 7.2035112, -7.5139894, 7.3766956, -14.5828629, 14.7174997
3: -2.9020705, 12.1498394, -2.9837260, 12.5334835, -15.4355526, 15.1335659
4: -9.5700588, 8.1782074, -9.9552670, 8.3942471, -17.9643059, 18.1334743

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.44 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0451486, upper bound: 12.0451486
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146671
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.0560733, upper bound: 12.1146670
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1146670, upper bound: 12.0560733
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205448
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -12.1255917, upper bound: 12.1205447
Binary search (step 17): status=Status.VERIFIED, low=0.1999992, high=0.2000000, mid=0.1999992, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1999992251396634
execution time: 481.29 seconds
