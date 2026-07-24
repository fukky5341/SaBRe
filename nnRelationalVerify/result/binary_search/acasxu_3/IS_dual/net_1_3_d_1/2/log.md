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
execution time: IAR + LP analysis = 1.39 + 1.17 = 2.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -12.2720720, upper bound: 12.2720720


# Binary Search by BASE starts (time budget: 1197.44 seconds, max iter: 100)

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
Binary search time: 46.67 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1150.77 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2030398, upper bound: 12.2654487
time: 0.47 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2674809, upper bound: 12.2674811
time: 0.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.97 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.97
Output dim: 0, lower bound: -12.2030398, upper bound: 12.2654487
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.97
Output dim: 0, lower bound: -12.2674809, upper bound: 12.2674811

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -7.3041487, 6.1491537, -9.5487633, 11.0977812
1: -2.4989076, 6.0682201, -5.3440590, 8.2597237, -10.7586288, 11.4122791
2: -3.6286805, 4.3000150, -7.7298098, 6.6192513, -10.2479315, 12.0298243
3: -1.6621141, 7.2208080, -2.6742923, 11.7119293, -13.3740435, 9.8950987
4: -4.9867439, 4.9192414, -10.0658026, 7.5752449, -12.5619888, 14.9850445

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2010074
time: 0.52 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2654487
time: 0.40 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2199450, 5.5384865, -7.3073158, 6.1510162, -12.3709612, 12.8458023
1: -4.5570164, 7.5879345, -5.3463583, 8.2614708, -12.8184872, 12.9342928
2: -6.5976086, 6.0316262, -7.7331066, 6.6210580, -13.2186661, 13.7647324
3: -2.4391820, 10.4356041, -2.6750574, 11.7156067, -14.1547890, 13.1106606
4: -8.6541815, 6.9089074, -10.0698843, 7.5773277, -16.2315063, 16.9787922

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2654487, upper bound: 12.2030398
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2654487, upper bound: 12.2674811
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.14 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2010074
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2654487
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2654487, upper bound: 12.2030398
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -12.2654487, upper bound: 12.2674811

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -3.3996100, 3.7936325, -7.1932425, 7.1932421
1: -2.4989076, 6.0682201, -2.4989076, 6.0682201, -8.5671263, 8.5671253
2: -3.6286805, 4.3000150, -3.6286805, 4.3000150, -7.9286952, 7.9286957
3: -1.6621141, 7.2208080, -1.6621141, 7.2208080, -8.8829193, 8.8829212
4: -4.9867439, 4.9192414, -4.9867439, 4.9192414, -9.9059830, 9.9059849

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642959, upper bound: 12.1854254
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875843, upper bound: 12.1875846
time: 0.34 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -6.2199450, 5.5384865, -8.9380951, 10.0135775
1: -2.4989076, 6.0682201, -4.5570164, 7.5879345, -10.0868416, 10.6252365
2: -3.6286805, 4.3000150, -6.5976086, 6.0316262, -9.6603069, 10.8976231
3: -1.6621141, 7.2208080, -2.4391820, 10.4356041, -12.0977182, 9.6599903
4: -4.9867439, 4.9192414, -8.6541815, 6.9089074, -11.8956509, 13.5734205

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1854254, upper bound: 12.2654487
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875846, upper bound: 12.2172286
time: 0.41 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.2199450, 5.5384865, -3.3996100, 3.7936325, -10.0135775, 8.9380970
1: -4.5570164, 7.5879345, -2.4989076, 6.0682201, -10.6252356, 10.0868416
2: -6.5976086, 6.0316262, -3.6286805, 4.3000150, -10.8976221, 9.6603041
3: -2.4391820, 10.4356041, -1.6621141, 7.2208080, -9.6599903, 12.0977182
4: -8.6541815, 6.9089074, -4.9867439, 4.9192414, -13.5734215, 11.8956509

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2654485, upper bound: 12.1896169
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2172286, upper bound: 12.1896169
time: 0.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.2199450, 5.5384865, -6.2199450, 5.5384865, -11.7584314, 11.7584314
1: -4.5570164, 7.5879345, -4.5570164, 7.5879345, -12.1449509, 12.1449509
2: -6.5976086, 6.0316262, -6.5976086, 6.0316262, -12.6292334, 12.6292343
3: -2.4391820, 10.4356041, -2.4391820, 10.4356041, -12.8747864, 12.8747864
4: -8.6541815, 6.9089074, -8.6541815, 6.9089074, -15.5630894, 15.5630894

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2147627
time: 0.42 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2159045
time: 0.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1642959, upper bound: 12.1854254
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1875843, upper bound: 12.1875846
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1854254, upper bound: 12.2654487
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.1875846, upper bound: 12.2172286
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2654485, upper bound: 12.1896169
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2172286, upper bound: 12.1896169
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2147627
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.41
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2159045

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -3.3996100, 3.7936325, -5.5657253, 6.5275230
1: -1.3520081, 5.2492390, -2.4989076, 6.0682201, -7.4202261, 7.7481465
2: -2.0487065, 3.6625366, -3.6286805, 4.3000150, -6.3487191, 7.2912169
3: -1.4298269, 5.8484859, -1.6621141, 7.2208080, -8.6506348, 7.5105991
4: -3.0981421, 4.2000217, -4.9867439, 4.9192414, -8.0173817, 9.1867647

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1621367
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1854254
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -3.0542848, 3.6593857, -4.9641008, 6.6220312
1: -1.0992223, 5.5531120, -2.2543423, 5.9155998, -7.0148196, 7.8074536
2: -1.7584276, 4.1799893, -3.2838345, 4.1678972, -5.9263248, 7.4638214
3: -1.5837383, 6.1448326, -1.6035967, 6.9553394, -8.5390778, 7.7484288
4: -2.9098625, 4.7395120, -4.5699902, 4.7674265, -7.6772876, 9.3095016

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1642959
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1875846
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -4.5359459, 4.7002478, -8.0998573, 8.3295765
1: -2.4989076, 6.0682201, -3.3364797, 6.6092453, -9.1081486, 9.4046974
2: -3.6286805, 4.3000150, -4.8469982, 5.2329617, -8.8616419, 9.1470118
3: -1.6621141, 7.2208080, -2.0921249, 8.6418915, -10.3040037, 9.3129330
4: -4.9867439, 4.9192414, -6.4802928, 6.0003052, -10.9870491, 11.3995342

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2150694
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2172286
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -3.0542848, 3.6593857, -5.5292892, 5.9653769, -9.0196619, 9.1886749
1: -2.2543423, 5.9155998, -4.1178484, 8.0096436, -10.2639847, 10.0334473
2: -3.2838345, 4.1678972, -6.0670018, 6.5351744, -9.8190088, 10.2348995
3: -1.6035967, 6.9553394, -2.6416011, 10.8504009, -12.4539928, 9.5969400
4: -4.5699902, 4.7674265, -8.1326199, 7.4373612, -12.0073509, 12.9000463

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642959, upper bound: 12.2150695
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642959, upper bound: 12.2172286
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.3996100, 3.7936325, -8.3295774, 8.0998573
1: -3.3364797, 6.6092453, -2.4989076, 6.0682201, -9.4046984, 9.1081514
2: -4.8469982, 5.2329617, -3.6286805, 4.3000150, -9.1470089, 8.8616419
3: -2.0921249, 8.6418915, -1.6621141, 7.2208080, -9.3129330, 10.3040056
4: -6.4802928, 6.0003052, -4.9867439, 4.9192414, -11.3995342, 10.9870491

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1896169
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -3.0542848, 3.6593857, -9.1886749, 9.0196619
1: -4.1178484, 8.0096436, -2.2543423, 5.9155998, -10.0334454, 10.2639856
2: -6.0670018, 6.5351744, -3.2838345, 4.1678972, -10.2348957, 9.8190088
3: -2.6416011, 10.8504009, -1.6035967, 6.9553394, -9.5969410, 12.4539948
4: -8.1326199, 7.4373612, -4.5699902, 4.7674265, -12.9000463, 12.0073509

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1896169
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -6.2199450, 5.5384865, -4.5359459, 4.7002478, -10.9201927, 10.0744305
1: -4.5570164, 7.5879345, -3.3364797, 6.6092453, -11.1662617, 10.9244137
2: -6.5976086, 6.0316262, -4.8469982, 5.2329617, -11.8305693, 10.8786182
3: -2.4391820, 10.4356041, -2.0921249, 8.6418915, -11.0810738, 12.5277271
4: -8.6541815, 6.9089074, -6.4802928, 6.0003052, -14.6544867, 13.3892002

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.42 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -5.8518510, 5.4007053, -5.5292892, 5.9653769, -11.8172283, 10.9299927
1: -4.2941132, 7.4372272, -4.1178484, 8.0096436, -12.3037567, 11.5550756
2: -6.2283597, 5.9039474, -6.0670018, 6.5351744, -12.7635345, 11.9709492
3: -2.3826027, 10.1186829, -2.6416011, 10.8504009, -13.2330036, 12.7602844
4: -8.2087297, 6.7593575, -8.1326199, 7.4373612, -15.6460896, 14.8919773

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
time: 0.47 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.49 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1621367
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1854254
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1642959
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1875846
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2150694
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2172286
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1642959, upper bound: 12.2150695
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.1642959, upper bound: 12.2172286
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1896169
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1896169
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.49
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159045

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -1.7720940, 3.1279130, -4.9000072, 4.9000072
1: -1.3520081, 5.2492390, -1.3520081, 5.2492390, -6.6012468, 6.6012468
2: -2.0487065, 3.6625366, -2.0487065, 3.6625366, -5.7112427, 5.7112432
3: -1.4298269, 5.8484859, -1.4298269, 5.8484859, -7.2783108, 7.2783113
4: -3.0981421, 4.2000217, -3.0981421, 4.2000217, -7.2981606, 7.2981629

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0369379, upper bound: 12.1136653
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -1.3047158, 3.5677476, -5.3398414, 4.4326286
1: -1.3520081, 5.2492390, -1.0992223, 5.5531120, -6.9051199, 6.3484612
2: -2.0487065, 3.6625366, -1.7584276, 4.1799893, -6.2286935, 5.4209623
3: -1.4298269, 5.8484859, -1.5837383, 6.1448326, -7.5746579, 7.4322228
4: -3.0981421, 4.2000217, -2.9098625, 4.7395120, -7.8376508, 7.1098833

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1136653, upper bound: 12.1174830
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0993895
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.7720940, 3.1279130, -4.4326286, 5.3398418
1: -1.0992223, 5.5531120, -1.3520081, 5.2492390, -6.3484607, 6.9051199
2: -1.7584276, 4.1799893, -2.0487065, 3.6625366, -5.4209623, 6.2286954
3: -1.5837383, 6.1448326, -1.4298269, 5.8484859, -7.4322243, 7.5746589
4: -2.9098625, 4.7395120, -3.0981421, 4.2000217, -7.1098833, 7.8376508

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1174825, upper bound: 12.1284926
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0331255
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.3047158, 3.5677476, -4.8724632, 4.8724627
1: -1.0992223, 5.5531120, -1.0992223, 5.5531120, -6.6523342, 6.6523342
2: -1.7584276, 4.1799893, -1.7584276, 4.1799893, -5.9384151, 5.9384151
3: -1.5837383, 6.1448326, -1.5837383, 6.1448326, -7.7285709, 7.7285705
4: -2.9098625, 4.7395120, -2.9098625, 4.7395120, -7.6493735, 7.6493731

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1398746, upper bound: 12.1061228
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0993895, upper bound: 12.1133035
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -4.5359459, 4.7002478, -6.4723415, 7.6638589
1: -1.3520081, 5.2492390, -3.3364797, 6.6092453, -7.9612527, 8.5857182
2: -2.0487065, 3.6625366, -4.8469982, 5.2329617, -7.2816668, 8.5095329
3: -1.4298269, 5.8484859, -2.0921249, 8.6418915, -10.0717173, 7.9406104
4: -3.0981421, 4.2000217, -6.4802928, 6.0003052, -9.0984440, 10.6803150

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0420062, upper bound: 12.1952169
time: 0.35 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 2.00 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0419371, upper bound: 12.2339812
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2632894
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -4.5359459, 4.7002478, -6.0049634, 8.1036911
1: -1.0992223, 5.5531120, -3.3364797, 6.6092453, -7.7084661, 8.8895893
2: -1.7584276, 4.1799893, -4.8469982, 5.2329617, -6.9913893, 9.0269861
3: -1.5837383, 6.1448326, -2.0921249, 8.6418915, -10.2256298, 8.2369576
4: -2.9098625, 4.7395120, -6.4802928, 6.0003052, -8.9101658, 11.2198048

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0420062, upper bound: 12.2412019
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 1.99 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0419371, upper bound: 12.2346283
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2654485
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -5.5292892, 5.9653769, -7.7374706, 8.6572018
1: -1.3520081, 5.2492390, -4.1178484, 8.0096436, -9.3616514, 9.3670864
2: -2.0487065, 3.6625366, -6.0670018, 6.5351744, -8.5838804, 9.7295380
3: -1.4298269, 5.8484859, -2.6416011, 10.8504009, -12.2802258, 8.4900866
4: -3.0981421, 4.2000217, -8.1326199, 7.4373612, -10.5355034, 12.3326416

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0420062, upper bound: 12.1765861
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 1.96 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1409591, upper bound: 12.0316506
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2150695
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -5.5292892, 5.9653769, -7.2700925, 9.0970364
1: -1.0992223, 5.5531120, -4.1178484, 8.0096436, -9.1088657, 9.6709595
2: -1.7584276, 4.1799893, -6.0670018, 6.5351744, -8.2936020, 10.2469893
3: -1.5837383, 6.1448326, -2.6416011, 10.8504009, -12.4341373, 8.7864323
4: -2.9098625, 4.7395120, -8.1326199, 7.4373612, -10.3472233, 12.8721294

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0420062, upper bound: 12.1917490
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 2.03 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2084575
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2084575
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -1.7720940, 3.1279130, -7.6638584, 6.4723415
1: -3.3364797, 6.6092453, -1.3520081, 5.2492390, -8.5857182, 7.9612513
2: -4.8469982, 5.2329617, -2.0487065, 3.6625366, -8.5095329, 7.2816682
3: -2.0921249, 8.6418915, -1.4298269, 5.8484859, -7.9406109, 10.0717182
4: -6.4802928, 6.0003052, -3.0981421, 4.2000217, -10.6803150, 9.0984478

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2632894, upper bound: 12.1626060
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1136653, upper bound: 12.0420062
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 2.52 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0419371
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2632894, upper bound: 12.1663283
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -1.3047158, 3.5677476, -8.1036930, 6.0049634
1: -3.3364797, 6.6092453, -1.0992223, 5.5531120, -8.8895912, 7.7084675
2: -4.8469982, 5.2329617, -1.7584276, 4.1799893, -9.0269852, 6.9913888
3: -2.0921249, 8.6418915, -1.5837383, 6.1448326, -8.2369576, 10.2256269
4: -6.4802928, 6.0003052, -2.9098625, 4.7395120, -11.2198048, 8.9101677

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2632894, upper bound: 12.1858947
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1952169, upper bound: 12.1246018
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 2.41 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0419371
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2632894, upper bound: 12.1896169
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -1.7720940, 3.1279130, -8.6572018, 7.7374697
1: -4.1178484, 8.0096436, -1.3520081, 5.2492390, -9.3670874, 9.3616514
2: -6.0670018, 6.5351744, -2.0487065, 3.6625366, -9.7295361, 8.5838814
3: -2.6416011, 10.8504009, -1.4298269, 5.8484859, -8.4900875, 12.2802277
4: -8.1326199, 7.4373612, -3.0981421, 4.2000217, -12.3326416, 10.5355034

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1765861, upper bound: 12.0422377
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 1.91 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0316506, upper bound: 12.1409591
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -1.3047158, 3.5677476, -9.0970364, 7.2700925
1: -4.1178484, 8.0096436, -1.0992223, 5.5531120, -9.6709595, 9.1088657
2: -6.0670018, 6.5351744, -1.7584276, 4.1799893, -10.2469873, 8.2936020
3: -2.6416011, 10.8504009, -1.5837383, 6.1448326, -8.7864342, 12.4341345
4: -8.1326199, 7.4373612, -2.9098625, 4.7395120, -12.8721313, 10.3472233

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1765861, upper bound: 12.0788842
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 1.86 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802070
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775646
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -4.5359459, 4.7002478, -9.2361937, 9.2361937
1: -3.3364797, 6.6092453, -3.3364797, 6.6092453, -9.9457207, 9.9457197
2: -4.8469982, 5.2329617, -4.8469982, 5.2329617, -10.0799589, 10.0799580
3: -2.0921249, 8.6418915, -2.0921249, 8.6418915, -10.7340164, 10.7340164
4: -6.4802928, 6.0003052, -6.4802928, 6.0003052, -12.4805984, 12.4805984

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1511806, upper bound: 12.1845915
time: 0.42 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 1.98 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0316617, upper bound: 12.2037774
time: 0.40 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2147626
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -4.5359459, 4.7002478, -10.2295370, 10.5013218
1: -4.1178484, 8.0096436, -3.3364797, 6.6092453, -10.7270889, 11.3461227
2: -6.0670018, 6.5351744, -4.8469982, 5.2329617, -11.2999582, 11.3821697
3: -2.6416011, 10.8504009, -2.0921249, 8.6418915, -11.2834902, 12.9425220
4: -8.1326199, 7.4373612, -6.4802928, 6.0003052, -14.1329250, 13.9176540

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1511806, upper bound: 12.1845915
time: 0.43 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38

Time for candidate selection: 1.92 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0316617, upper bound: 12.2037774
time: 0.38 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2147623
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -5.5292892, 5.9653769, -10.5013227, 10.2295370
1: -3.3364797, 6.6092453, -4.1178484, 8.0096436, -11.3461227, 10.7270927
2: -4.8469982, 5.2329617, -6.0670018, 6.5351744, -11.3821726, 11.2999611
3: -2.0921249, 8.6418915, -2.6416011, 10.8504009, -12.9425259, 11.2834911
4: -6.4802928, 6.0003052, -8.1326199, 7.4373612, -13.9176540, 14.1329250

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38

Time for candidate selection: 1.56 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0358421
time: 0.43 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159043
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -5.3697109, 5.9225302, -5.5292892, 5.9653769, -11.3350878, 11.4518166
1: -4.0038757, 7.9742546, -4.1178484, 8.0096436, -12.0135193, 12.0921021
2: -5.9117098, 6.4834294, -6.0670018, 6.5351744, -12.4468832, 12.5504313
3: -2.6065249, 10.7615910, -2.6416011, 10.8504009, -13.4569263, 13.4031925
4: -7.9371738, 7.3692279, -8.1326199, 7.4373612, -15.3745327, 15.5018482

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22

Time for candidate selection: 1.48 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332771, upper bound: 12.0358421
time: 0.44 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145005
time: 0.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.88 seconds
IS_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0369379, upper bound: 12.1136653
IS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1136653, upper bound: 12.1174830
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0993895
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1174825, upper bound: 12.1284926
IS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0331255
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1398746, upper bound: 12.1061228
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0993895, upper bound: 12.1133035
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0419371, upper bound: 12.2339812
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2632894
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0419371, upper bound: 12.2346283
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2654485
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1409591, upper bound: 12.0316506
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2150695
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2084575
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2084575
IS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0419371
IS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2632894, upper bound: 12.1663283
IS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0419371
IS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2632894, upper bound: 12.1896169
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0316506, upper bound: 12.1409591
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802070
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775646
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0316617, upper bound: 12.2037774
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2147626
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.0316617, upper bound: 12.2037774
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2147623
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0358421
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2159043
IS_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.1332771, upper bound: 12.0358421
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145005

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -4.5359459, 4.7002478, -4.9122152, 6.2001476
1: -0.2774035, 2.7060747, -3.3364797, 6.6092453, -6.8866448, 6.0425544
2: -0.6758080, 2.1488650, -4.8469982, 5.2329617, -5.9087696, 6.9958630
3: -0.9214749, 2.6601696, -2.0921249, 8.6418915, -9.5633659, 4.7522945
4: -1.2530408, 2.4732645, -6.4802928, 6.0003052, -7.2533436, 8.9535570

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2339810
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0568037, upper bound: 12.2337273
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.9886395, 2.7079735, -4.5359459, 4.7002478, -5.6888871, 7.2439189
1: -0.8401878, 4.6642990, -3.3364797, 6.6092453, -7.4494309, 8.0007782
2: -1.3668380, 3.2843196, -4.8469982, 5.2329617, -6.5997982, 8.1313162
3: -1.3257837, 5.0050430, -2.0921249, 8.6418915, -9.9676733, 7.0971680
4: -2.3902826, 3.7804737, -6.4802928, 6.0003052, -8.3905849, 10.2607670

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1591965, upper bound: 12.2595700
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1591965, upper bound: 12.2511791
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.5359459, 4.7002478, -4.9145575, 6.3155255
1: -0.2778803, 2.8850155, -3.3364797, 6.6092453, -6.8871236, 6.2214952
2: -0.7044127, 2.2803013, -4.8469982, 5.2329617, -5.9373741, 7.1272993
3: -0.9286744, 2.8235273, -2.0921249, 8.6418915, -9.5705662, 4.9156523
4: -1.3212409, 2.5750978, -6.4802928, 6.0003052, -7.3215446, 9.0553904

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346283
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.2344839
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.5359459, 4.7002478, -5.2261467, 7.6669259
1: -0.5956205, 4.9794006, -3.3364797, 6.6092453, -7.2048650, 8.3158770
2: -1.2669206, 3.7673125, -4.8469982, 5.2329617, -6.4998822, 8.6143103
3: -1.4658637, 5.3376389, -2.0921249, 8.6418915, -10.1077528, 7.4297638
4: -2.3321438, 4.3155594, -6.4802928, 6.0003052, -8.3324471, 10.7958527

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863545, upper bound: 12.2565418
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824850, upper bound: 12.2560777
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -4.1621442, 5.3945312, -7.1666250, 7.2900572
1: -1.3520081, 5.2492390, -3.1508241, 7.2959442, -8.6479521, 8.4000626
2: -2.0487065, 3.6625366, -4.7026148, 5.9927139, -8.0414190, 8.3651495
3: -1.4298269, 5.8484859, -2.4236612, 9.6652107, -11.0950346, 8.2721462
4: -3.0981421, 4.2000217, -6.4547949, 6.8532066, -9.9513483, 10.6548147

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2035060
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2035059
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -4.9344602, 5.2041121, -6.5088267, 8.5022078
1: -1.0992223, 5.5531120, -3.7369466, 6.8882389, -7.9874606, 9.2900581
2: -1.7584276, 4.1799893, -5.4744320, 5.6376672, -7.3960948, 9.6544189
3: -1.5837383, 6.1448326, -2.3689766, 9.8232098, -11.4069481, 8.5138092
4: -2.9098625, 4.7395120, -7.3408518, 6.5052090, -9.4150705, 12.0803642

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868742
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2084575
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -4.9817657, 5.6338024, -6.9385180, 8.5495110
1: -1.0992223, 5.5531120, -3.7128515, 7.5261250, -8.6253452, 9.2659607
2: -1.7584276, 4.1799893, -5.4640889, 6.2225499, -7.9809761, 9.6440773
3: -1.5837383, 6.1448326, -2.5014567, 10.0730762, -11.6568146, 8.6462898
4: -2.9098625, 4.7395120, -7.3761001, 7.0715237, -9.9813862, 12.1156101

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0346447, upper bound: 12.1868741
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2084575
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2119676, 1.6642017, -6.2001476, 4.9122152
1: -3.3364797, 6.6092453, -0.2774035, 2.7060747, -6.0425544, 6.8866482
2: -4.8469982, 5.2329617, -0.6758080, 2.1488650, -6.9958630, 5.9087696
3: -2.0921249, 8.6418915, -0.9214749, 2.6601696, -4.7522945, 9.5633659
4: -6.4802928, 6.0003052, -1.2530408, 2.4732645, -8.9535570, 7.2533455

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0643250
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2337273, upper bound: 12.0603139
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.9886395, 2.7079735, -7.2439184, 5.6888871
1: -3.3364797, 6.6092453, -0.8401878, 4.6642990, -8.0007772, 7.4494324
2: -4.8469982, 5.2329617, -1.3668380, 3.2843196, -8.1313162, 6.5997996
3: -2.0921249, 8.6418915, -1.3257837, 5.0050430, -7.0971680, 9.9676752
4: -6.4802928, 6.0003052, -2.3902826, 3.7804737, -10.2607651, 8.3905859

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595700, upper bound: 12.1591965
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2511791, upper bound: 12.1591965
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2143096, 1.7795796, -6.3155255, 4.9145575
1: -3.3364797, 6.6092453, -0.2778803, 2.8850155, -6.2214952, 6.8871250
2: -4.8469982, 5.2329617, -0.7044127, 2.2803013, -7.1272993, 5.9373741
3: -2.0921249, 8.6418915, -0.9286744, 2.8235273, -4.9156523, 9.5705662
4: -6.4802928, 6.0003052, -1.3212409, 2.5750978, -9.0553904, 7.3215446

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.5258994, 3.1309829, -7.6669254, 5.2261472
1: -3.3364797, 6.6092453, -0.5956205, 4.9794006, -8.3158798, 7.2048659
2: -4.8469982, 5.2329617, -1.2669206, 3.7673125, -8.6143074, 6.4998817
3: -2.0921249, 8.6418915, -1.4658637, 5.3376389, -7.4297638, 10.1077557
4: -6.4802928, 6.0003052, -2.3321438, 4.3155594, -10.7958527, 8.3324471

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565418, upper bound: 12.1863545
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2560777, upper bound: 12.1824846
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -1.7720940, 3.1279130, -7.2900572, 7.1666250
1: -3.1508241, 7.2959442, -1.3520081, 5.2492390, -8.4000626, 8.6479521
2: -4.7026148, 5.9927139, -2.0487065, 3.6625366, -8.3651476, 8.0414181
3: -2.4236612, 9.6652107, -1.4298269, 5.8484859, -8.2721462, 11.0950375
4: -6.4547949, 6.8532066, -3.0981421, 4.2000217, -10.6548166, 9.9513474

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1630514
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1590360
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -1.3047158, 3.5677476, -8.5022078, 6.5088267
1: -3.7369466, 6.8882389, -1.0992223, 5.5531120, -9.2900581, 7.9874611
2: -5.4744320, 5.6376672, -1.7584276, 4.1799893, -9.6544189, 7.3960929
3: -2.3689766, 9.8232098, -1.5837383, 6.1448326, -8.5138092, 11.4069443
4: -7.3408518, 6.5052090, -2.9098625, 4.7395120, -12.0803642, 9.4150696

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -1.3047158, 3.5677476, -8.5495110, 6.9385180
1: -3.7128515, 7.5261250, -1.0992223, 5.5531120, -9.2659607, 8.6253471
2: -5.4640889, 6.2225499, -1.7584276, 4.1799893, -9.6440783, 7.9809775
3: -2.5014567, 10.0730762, -1.5837383, 6.1448326, -8.6462898, 11.6568136
4: -7.3761001, 7.0715237, -2.9098625, 4.7395120, -12.1156101, 9.9813862

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346447
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775644
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -4.5359459, 4.7002478, -4.9744987, 6.8553329
1: -0.3634766, 3.2789097, -3.3364797, 6.6092453, -6.9727197, 6.6153893
2: -0.7444794, 2.9443834, -4.8469982, 5.2329617, -5.9774408, 7.7913818
3: -1.1945589, 3.2639122, -2.0921249, 8.6418915, -9.8364496, 5.3560371
4: -1.3846707, 3.3782203, -6.4802928, 6.0003052, -7.3849754, 9.8585129

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0888682, upper bound: 12.2414639
time: 0.45 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0840604, upper bound: 12.2409575
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -3.8283892, 4.3033652, -4.5359459, 4.7002478, -8.5286369, 8.8393097
1: -2.8217945, 6.1582298, -3.3364797, 6.6092453, -9.4310341, 9.4947062
2: -4.1151338, 4.8628931, -4.8469982, 5.2329617, -9.3480949, 9.7098885
3: -1.9461601, 7.8930817, -2.0921249, 8.6418915, -10.5880499, 9.9852066
4: -5.5757437, 5.5972967, -6.4802928, 6.0003052, -11.5760489, 12.0775890

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2603492, upper bound: 12.2583902
time: 0.38 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2598024, upper bound: 12.2590366
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.5359459, 4.7002478, -4.9894433, 7.2741680
1: -0.3822187, 3.9519520, -3.3364797, 6.6092453, -6.9914594, 7.2884316
2: -0.8797669, 3.3527915, -4.8469982, 5.2329617, -6.1127286, 8.1997900
3: -1.2386876, 3.9645653, -2.0921249, 8.6418915, -9.8805790, 6.0566902
4: -1.6498394, 3.7471557, -6.4802928, 6.0003052, -7.6501431, 10.2274485

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2037779
time: 0.41 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2037779
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -4.5359459, 4.7002478, -8.8623924, 9.9304752
1: -3.1508241, 7.2959442, -3.3364797, 6.6092453, -9.7600651, 10.6324234
2: -4.7026148, 5.9927139, -4.8469982, 5.2329617, -9.9355726, 10.8397083
3: -2.4236612, 9.6652107, -2.0921249, 8.6418915, -11.0655508, 11.7573357
4: -6.4547949, 6.8532066, -6.4802928, 6.0003052, -12.4551001, 13.3334999

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119958, upper bound: 12.2115745
time: 0.93 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.2122766
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2891956, 2.7382221, -7.2741680, 4.9894433
1: -3.3364797, 6.6092453, -0.3822187, 3.9519520, -7.2884316, 6.9914627
2: -4.8469982, 5.2329617, -0.8797669, 3.3527915, -8.1997900, 6.1127281
3: -2.0921249, 8.6418915, -1.2386876, 3.9645653, -6.0566902, 9.8805790
4: -6.4802928, 6.0003052, -1.6498394, 3.7471557, -10.2274485, 7.6501431

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0325799
time: 0.55 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2415651, upper bound: 12.0287104
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -4.1621442, 5.3945312, -9.9304771, 8.8623924
1: -3.3364797, 6.6092453, -3.1508241, 7.2959442, -10.6324234, 9.7600689
2: -4.8469982, 5.2329617, -4.7026148, 5.9927139, -10.8397112, 9.9355745
3: -2.0921249, 8.6418915, -2.4236612, 9.6652107, -11.7573347, 11.0655527
4: -6.4802928, 6.0003052, -6.4547949, 6.8532066, -13.3334990, 12.4551001

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2636573, upper bound: 12.2121292
time: 0.42 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2596419, upper bound: 12.2121292
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -5.3697109, 5.9225302, -4.1621442, 5.3945312, -10.7642412, 10.0846748
1: -4.0038757, 7.9742546, -3.1508241, 7.2959442, -11.2998199, 11.1250782
2: -5.9117098, 6.4834294, -4.7026148, 5.9927139, -11.9044199, 11.1860437
3: -2.6065249, 10.7615910, -2.4236612, 9.6652107, -12.2717352, 13.1852522
4: -7.9371738, 7.3692279, -6.4547949, 6.8532066, -14.7903795, 13.8240223

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119686, upper bound: 12.2110739
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119686, upper bound: 12.2108905
time: 0.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.47 seconds
IS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2339810
IS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0568037, upper bound: 12.2337273
IS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1591965, upper bound: 12.2595700
IS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1591965, upper bound: 12.2511791
IS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346283
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0261820, upper bound: 12.2344839
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1863545, upper bound: 12.2565418
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1824850, upper bound: 12.2560777
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2035060
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2035059
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868742
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2084575
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0346447, upper bound: 12.1868741
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2084575
IS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0643250
IS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2337273, upper bound: 12.0603139
IS_A2_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2595700, upper bound: 12.1591965
IS_A2_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2511791, upper bound: 12.1591965
IS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
IS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
IS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2565418, upper bound: 12.1863545
IS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2560777, upper bound: 12.1824846
IS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1630514
IS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1590360
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346447
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775644
IS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0888682, upper bound: 12.2414639
IS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0840604, upper bound: 12.2409575
IS_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2603492, upper bound: 12.2583902
IS_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2598024, upper bound: 12.2590366
IS_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2037779
IS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2037779
IS_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2119958, upper bound: 12.2115745
IS_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2035059, upper bound: 12.2122766
IS_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0325799
IS_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2415651, upper bound: 12.0287104
IS_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2636573, upper bound: 12.2121292
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2596419, upper bound: 12.2121292
IS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2119686, upper bound: 12.2110739
IS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -12.2119686, upper bound: 12.2108905

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -2.6754963, 4.0768895, -4.2888570, 4.3396978
1: -0.2774035, 2.7060747, -2.0858450, 5.7528667, -6.0302687, 4.7919197
2: -0.6758080, 2.1488650, -3.1665211, 4.5292182, -5.2050261, 5.3153858
3: -0.9214749, 2.6601696, -1.8468850, 7.5567837, -8.4782581, 4.5070543
4: -1.2530408, 2.4732645, -4.4865170, 5.2266645, -6.4797049, 6.9597816

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2339810
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0643250, upper bound: 12.2339812
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -4.0283499, 4.3438158, -4.5557833, 5.6925516
1: -0.2774035, 2.7060747, -2.9591291, 6.0517540, -6.3291550, 5.6652040
2: -0.6758080, 2.1488650, -4.2841620, 4.9001317, -5.5759397, 6.4330273
3: -0.9214749, 2.6601696, -1.9467452, 7.8118000, -8.7332745, 4.6069145
4: -1.2530408, 2.4732645, -5.7461939, 5.6184468, -6.8714871, 8.2194586

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0450277, upper bound: 12.0574094
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0450277, upper bound: 12.2337273
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -4.5359459, 4.7002478, -4.9765196, 6.7912307
1: -0.3717489, 3.9332523, -3.3364797, 6.6092453, -6.9809937, 7.2697320
2: -1.0147076, 2.7846820, -4.8469982, 5.2329617, -6.2476692, 7.6316805
3: -1.1808519, 4.1503563, -2.0921249, 8.6418915, -9.8227434, 6.2424812
4: -1.9016094, 3.2297649, -6.4802928, 6.0003052, -7.9019132, 9.7100582

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1409591, upper bound: 12.0877423
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1409591, upper bound: 12.2595700
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5457935, 2.4314504, -4.5359459, 4.7002478, -5.2460413, 6.9673963
1: -0.5560611, 4.1216106, -3.3364797, 6.6092453, -7.1653047, 7.4580903
2: -1.0841694, 2.9876790, -4.8469982, 5.2329617, -6.3171310, 7.8346772
3: -1.2315452, 4.3375187, -2.0921249, 8.6418915, -9.8734369, 6.4296436
4: -1.9889984, 3.4472027, -6.4802928, 6.0003052, -7.9893022, 9.9274960

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0603139, upper bound: 12.2337274
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1591965, upper bound: 12.2511791
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.6754963, 4.0768895, -4.2911992, 4.4550762
1: -0.2778803, 2.8850155, -2.0858450, 5.7528667, -6.0307469, 4.9708605
2: -0.7044127, 2.2803013, -3.1665211, 4.5292182, -5.2336311, 5.4468222
3: -0.9286744, 2.8235273, -1.8468850, 7.5567837, -8.4854584, 4.6704121
4: -1.3212409, 2.5750978, -4.4865170, 5.2266645, -6.5479054, 7.0616150

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.0283499, 4.3438158, -4.5581255, 5.8079295
1: -0.2778803, 2.8850155, -2.9591291, 6.0517540, -6.3296342, 5.8441448
2: -0.7044127, 2.2803013, -4.2841620, 4.9001317, -5.6045446, 6.5644636
3: -0.9286744, 2.8235273, -1.9467452, 7.8118000, -8.7404747, 4.7702723
4: -1.3212409, 2.5750978, -5.7461939, 5.6184468, -6.9396877, 8.3212919

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.2344839
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348053, upper bound: 12.2344839
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -2.6754963, 4.0768895, -4.6027875, 5.8064795
1: -0.5956205, 4.9794006, -2.0858450, 5.7528667, -6.3484869, 7.0652428
2: -1.2669206, 3.7673125, -3.1665211, 4.5292182, -5.7961388, 6.9338322
3: -1.4658637, 5.3376389, -1.8468850, 7.5567837, -9.0226450, 7.1845231
4: -2.3321438, 4.3155594, -4.4865170, 5.2266645, -7.5588083, 8.8020763

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863545, upper bound: 12.2565418
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.0283499, 4.3438158, -4.8697147, 7.1593285
1: -0.5956205, 4.9794006, -2.9591291, 6.0517540, -6.6473746, 7.9385290
2: -1.2669206, 3.7673125, -4.2841620, 4.9001317, -6.1670518, 8.0514736
3: -1.4658637, 5.3376389, -1.9467452, 7.8118000, -9.2776604, 7.2843838
4: -2.3321438, 4.3155594, -5.7461939, 5.6184468, -7.9505906, 10.0617514

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348053, upper bound: 12.2344839
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824850, upper bound: 12.2560777
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -3.6782563, 4.7223377, -6.4944315, 6.8061695
1: -1.3520081, 5.2492390, -2.8371258, 6.2889423, -7.6409502, 8.0863647
2: -2.0487065, 3.6625366, -4.2119579, 5.1972218, -7.2459269, 7.8744946
3: -1.4298269, 5.8484859, -2.1570172, 8.6666470, -10.0964718, 8.0055027
4: -3.0981421, 4.2000217, -5.7941303, 6.0009980, -9.0991402, 9.9941521

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1555248, upper bound: 12.2035059
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1555248, upper bound: 12.2035059
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -3.4619281, 4.9763918, -6.7484856, 6.5898409
1: -1.3520081, 5.2492390, -2.6407366, 6.6547775, -8.0067844, 7.8899755
2: -2.0487065, 3.6625366, -3.9357042, 5.6090250, -7.6577311, 7.5982380
3: -1.4298269, 5.8484859, -2.2358243, 8.6746998, -10.1045265, 8.0843096
4: -3.0981421, 4.2000217, -5.4861760, 6.4149227, -9.5130653, 9.6861973

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1409592, upper bound: 12.0200871
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2035060
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9344602, 5.2041121, -5.4184213, 6.7140398
1: -0.2778803, 2.8850155, -3.7369466, 6.8882389, -7.1661191, 6.6219621
2: -0.7044127, 2.2803013, -5.4744320, 5.6376672, -6.3420801, 7.7547331
3: -0.9286744, 2.8235273, -2.3689766, 9.8232098, -10.7518845, 5.1925039
4: -1.3212409, 2.5750978, -7.3408518, 6.5052090, -7.8264494, 9.9159498

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868741
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1868742
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9344602, 5.2041121, -5.7300105, 8.0654411
1: -0.5956205, 4.9794006, -3.7369466, 6.8882389, -7.4838595, 8.7163448
2: -1.2669206, 3.7673125, -5.4744320, 5.6376672, -6.9045873, 9.2417421
3: -1.4658637, 5.3376389, -2.3689766, 9.8232098, -11.2890701, 7.7066154
4: -2.3321438, 4.3155594, -7.3408518, 6.5052090, -8.8373518, 11.6564112

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2084575
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2084575
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9817657, 5.6338024, -5.8481121, 6.7613454
1: -0.2778803, 2.8850155, -3.7128515, 7.5261250, -7.8040028, 6.5978670
2: -0.7044127, 2.2803013, -5.4640889, 6.2225499, -6.9269624, 7.7443905
3: -0.9286744, 2.8235273, -2.5014567, 10.0730762, -11.0017509, 5.3249841
4: -1.3212409, 2.5750978, -7.3761001, 7.0715237, -8.3927650, 9.9511976

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1868742
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0346447, upper bound: 12.1868741
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9817657, 5.6338024, -6.1597018, 8.1127453
1: -0.5956205, 4.9794006, -3.7128515, 7.5261250, -8.1217442, 8.6922503
2: -1.2669206, 3.7673125, -5.4640889, 6.2225499, -7.4894691, 9.2314014
3: -1.4658637, 5.3376389, -2.5014567, 10.0730762, -11.5389366, 7.8390956
4: -2.3321438, 4.3155594, -7.3761001, 7.0715237, -9.4036646, 11.6916599

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2084575
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2084575
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2119676, 1.6642017, -4.3396978, 4.2888570
1: -2.0858450, 5.7528667, -0.2774035, 2.7060747, -4.7919197, 6.0302701
2: -3.1665211, 4.5292182, -0.6758080, 2.1488650, -5.3153858, 5.2050261
3: -1.8468850, 7.5567837, -0.9214749, 2.6601696, -4.5070543, 8.4782581
4: -4.4865170, 5.2266645, -1.2530408, 2.4732645, -6.9597816, 6.4797049

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0643250
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1332771, upper bound: 12.0643250
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2119676, 1.6642017, -5.6925516, 4.5557833
1: -2.9591291, 6.0517540, -0.2774035, 2.7060747, -5.6652040, 6.3291550
2: -4.2841620, 4.9001317, -0.6758080, 2.1488650, -6.4330273, 5.5759397
3: -1.9467452, 7.8118000, -0.9214749, 2.6601696, -4.6069145, 8.7332735
4: -5.7461939, 5.6184468, -1.2530408, 2.4732645, -8.2194586, 6.8714876

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0574094, upper bound: 12.0450277
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0574094, upper bound: 12.0603139
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2762722, 2.2552848, -6.7912307, 4.9765201
1: -3.3364797, 6.6092453, -0.3717489, 3.9332523, -7.2697320, 6.9809942
2: -4.8469982, 5.2329617, -1.0147076, 2.7846820, -7.6316805, 6.2476692
3: -2.0921249, 8.6418915, -1.1808519, 4.1503563, -6.2424812, 9.8227434
4: -6.4802928, 6.0003052, -1.9016094, 3.2297649, -9.7100582, 7.9019122

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1505732, upper bound: 12.1591296
time: 0.50 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1505732, upper bound: 12.1591965
time: 0.49 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.5457935, 2.4314504, -6.9673963, 5.2460413
1: -3.3364797, 6.6092453, -0.5560611, 4.1216106, -7.4580903, 7.1653066
2: -4.8469982, 5.2329617, -1.0841694, 2.9876790, -7.8346772, 6.3171310
3: -2.0921249, 8.6418915, -1.2315452, 4.3375187, -6.4296436, 9.8734369
4: -6.4802928, 6.0003052, -1.9889984, 3.4472027, -9.9274960, 7.9893031

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2337273, upper bound: 12.0603139
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2511791, upper bound: 12.1591965
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2143096, 1.7795796, -4.4550762, 4.2911992
1: -2.0858450, 5.7528667, -0.2778803, 2.8850155, -4.9708605, 6.0307469
2: -3.1665211, 4.5292182, -0.7044127, 2.2803013, -5.4468222, 5.2336307
3: -1.8468850, 7.5567837, -0.9286744, 2.8235273, -4.6704121, 8.4854574
4: -4.4865170, 5.2266645, -1.3212409, 2.5750978, -7.0616150, 6.5479054

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2143096, 1.7795796, -5.8079295, 4.5581255
1: -2.9591291, 6.0517540, -0.2778803, 2.8850155, -5.8441448, 6.3296337
2: -4.2841620, 4.9001317, -0.7044127, 2.2803013, -6.5644636, 5.6045446
3: -1.9467452, 7.8118000, -0.9286744, 2.8235273, -4.7702723, 8.7404747
4: -5.7461939, 5.6184468, -1.3212409, 2.5750978, -8.3212919, 6.9396877

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.5258994, 3.1309829, -5.8064790, 4.6027880
1: -2.0858450, 5.7528667, -0.5956205, 4.9794006, -7.0652442, 6.3484874
2: -3.1665211, 4.5292182, -1.2669206, 3.7673125, -6.9338326, 5.7961388
3: -1.8468850, 7.5567837, -1.4658637, 5.3376389, -7.1845236, 9.0226450
4: -4.4865170, 5.2266645, -2.3321438, 4.3155594, -8.8020763, 7.5588083

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565418, upper bound: 12.1863545
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.5258994, 3.1309829, -7.1593299, 4.8697152
1: -2.9591291, 6.0517540, -0.5956205, 4.9794006, -7.9385295, 6.6473746
2: -4.2841620, 4.9001317, -1.2669206, 3.7673125, -8.0514736, 6.1670523
3: -1.9467452, 7.8118000, -1.4658637, 5.3376389, -7.2843833, 9.2776623
4: -5.7461939, 5.6184468, -2.3321438, 4.3155594, -10.0617533, 7.9505906

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2560777, upper bound: 12.1824846
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -1.7720940, 3.1279130, -6.8061695, 6.4944315
1: -2.8371258, 6.2889423, -1.3520081, 5.2492390, -8.0863647, 7.6409502
2: -4.2119579, 5.1972218, -2.0487065, 3.6625366, -7.8744946, 7.2459283
3: -2.1570172, 8.6666470, -1.4298269, 5.8484859, -8.0055027, 10.0964718
4: -5.7941303, 6.0009980, -3.0981421, 4.2000217, -9.9941511, 9.0991402

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0642829
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.1630514
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.4619281, 4.9763918, -1.7720940, 3.1279130, -6.5898409, 6.7484856
1: -2.6407366, 6.6547775, -1.3520081, 5.2492390, -7.8899755, 8.0067844
2: -3.9357042, 5.6090250, -2.0487065, 3.6625366, -7.5982380, 7.6577306
3: -2.2358243, 8.6746998, -1.4298269, 5.8484859, -8.0843105, 10.1045256
4: -5.4861760, 6.4149227, -3.0981421, 4.2000217, -9.6861973, 9.5130653

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0200871, upper bound: 12.1409591
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1590360
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.2143096, 1.7795796, -6.7140398, 5.4184217
1: -3.7369466, 6.8882389, -0.2778803, 2.8850155, -6.6219621, 7.1661191
2: -5.4744320, 5.6376672, -0.7044127, 2.2803013, -7.7547331, 6.3420801
3: -2.3689766, 9.8232098, -0.9286744, 2.8235273, -5.1925039, 10.7518835
4: -7.3408518, 6.5052090, -1.3212409, 2.5750978, -9.9159498, 7.8264499

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346152
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.5258994, 3.1309829, -8.0654411, 5.7300110
1: -3.7369466, 6.8882389, -0.5956205, 4.9794006, -8.7163467, 7.4838595
2: -5.4744320, 5.6376672, -1.2669206, 3.7673125, -9.2417421, 6.9045877
3: -2.3689766, 9.8232098, -1.4658637, 5.3376389, -7.7066150, 11.2890692
4: -7.3408518, 6.5052090, -2.3321438, 4.3155594, -11.6564112, 8.8373508

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1569524, upper bound: 12.1775644
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.2143096, 1.7795796, -6.7613454, 5.8481121
1: -3.7128515, 7.5261250, -0.2778803, 2.8850155, -6.5978670, 7.8040037
2: -5.4640889, 6.2225499, -0.7044127, 2.2803013, -7.7443905, 6.9269629
3: -2.5014567, 10.0730762, -0.9286744, 2.8235273, -5.3249841, 11.0017509
4: -7.3761001, 7.0715237, -1.3212409, 2.5750978, -9.9511976, 8.3927650

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346152
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346447
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.5258994, 3.1309829, -8.1127415, 6.1597018
1: -3.7128515, 7.5261250, -0.5956205, 4.9794006, -8.6922522, 8.1217451
2: -5.4640889, 6.2225499, -1.2669206, 3.7673125, -9.2314014, 7.4894705
3: -2.5014567, 10.0730762, -1.4658637, 5.3376389, -7.8390956, 11.5389404
4: -7.3761001, 7.0715237, -2.3321438, 4.3155594, -11.6916599, 9.4036636

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1569524, upper bound: 12.1775644
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775644
time: 0.49 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -2.6754963, 4.0768895, -4.3511410, 4.9948835
1: -0.3634766, 3.2789097, -2.0858450, 5.7528667, -6.1163435, 5.3647547
2: -0.7444794, 2.9443834, -3.1665211, 4.5292182, -5.2736979, 6.1109047
3: -1.1945589, 3.2639122, -1.8468850, 7.5567837, -8.7513418, 5.1107969
4: -1.3846707, 3.3782203, -4.4865170, 5.2266645, -6.6113353, 7.8647375

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0888682, upper bound: 12.2414639
time: 0.58 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2414639
time: 0.52 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -4.0283499, 4.3438158, -4.6180677, 6.3477368
1: -0.3634766, 3.2789097, -2.9591291, 6.0517540, -6.4152298, 6.2380390
2: -0.7444794, 2.9443834, -4.2841620, 4.9001317, -5.6446109, 7.2285452
3: -1.1945589, 3.2639122, -1.9467452, 7.8118000, -9.0063591, 5.2106571
4: -1.3846707, 3.3782203, -5.7461939, 5.6184468, -7.0031176, 9.1244144

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0704842, upper bound: 12.0705026
time: 0.55 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0704842, upper bound: 12.2409575
time: 0.54 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -1.9736005, 3.7326164, -4.5359459, 4.7002478, -6.6738482, 8.2685623
1: -1.5754695, 5.3457413, -3.3364797, 6.6092453, -8.1847095, 8.6822195
2: -2.4574776, 4.2253923, -4.8469982, 5.2329617, -7.6904378, 9.0723877
3: -1.7051358, 6.8608799, -2.0921249, 8.6418915, -10.3470259, 8.9530048
4: -3.6110172, 4.8835225, -6.4802928, 6.0003052, -9.6113224, 11.3638153

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0885848
time: 0.41 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2420654, upper bound: 12.2583900
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -3.2853987, 3.9233079, -4.5359459, 4.7002478, -7.9856462, 8.4592514
1: -2.4223030, 5.5470581, -3.3364797, 6.6092453, -9.0315456, 8.8835373
2: -3.5144036, 4.5003858, -4.8469982, 5.2329617, -8.7473650, 9.3473835
3: -1.7845080, 6.9990902, -2.0921249, 8.6418915, -10.4263992, 9.0912151
4: -4.7958260, 5.2002707, -6.4802928, 6.0003052, -10.7961292, 11.6805630

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2415650, upper bound: 12.0841362
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2415650, upper bound: 12.2582810
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -2.6754963, 4.0768895, -4.3660846, 5.4137182
1: -0.3822187, 3.9519520, -2.0858450, 5.7528667, -6.1350832, 6.0377970
2: -0.8797669, 3.3527915, -3.1665211, 4.5292182, -5.4089847, 6.5193129
3: -1.2386876, 3.9645653, -1.8468850, 7.5567837, -8.7954712, 5.8114500
4: -1.6498394, 3.7471557, -4.4865170, 5.2266645, -6.8765030, 8.2336731

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2037779
time: 0.39 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2037779
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.0283499, 4.3438158, -4.6330113, 6.7665720
1: -0.3822187, 3.9519520, -2.9591291, 6.0517540, -6.4339714, 6.9110813
2: -0.8797669, 3.3527915, -4.2841620, 4.9001317, -5.7798982, 7.6369534
3: -1.2386876, 3.9645653, -1.9467452, 7.8118000, -9.0504875, 5.9113102
4: -1.6498394, 3.7471557, -5.7461939, 5.6184468, -7.2682853, 9.4933491

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2037779
time: 0.46 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2037779
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -4.5359459, 4.7002478, -8.3785038, 9.2582836
1: -2.8371258, 6.2889423, -3.3364797, 6.6092453, -9.4463692, 9.6254215
2: -4.2119579, 5.1972218, -4.8469982, 5.2329617, -9.4449186, 10.0442152
3: -2.1570172, 8.6666470, -2.0921249, 8.6418915, -10.7989082, 10.7587700
4: -5.7941303, 6.0009980, -6.4802928, 6.0003052, -11.7944355, 12.4812908

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0657663
time: 0.47 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1934089, upper bound: 12.2115748
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -3.4619281, 4.9763918, -4.5359459, 4.7002478, -8.1621761, 9.5123358
1: -2.6407366, 6.6547775, -3.3364797, 6.6092453, -9.2499790, 9.9912548
2: -3.9357042, 5.6090250, -4.8469982, 5.2329617, -9.1686621, 10.4560223
3: -2.2358243, 8.6746998, -2.0921249, 8.6418915, -10.8777151, 10.7668247
4: -5.4861760, 6.4149227, -6.4802928, 6.0003052, -11.4864807, 12.8952160

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0616089
time: 0.49 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1934089, upper bound: 12.2119442
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2891956, 2.7382221, -5.4137182, 4.3660851
1: -2.0858450, 5.7528667, -0.3822187, 3.9519520, -6.0377970, 6.1350851
2: -3.1665211, 4.5292182, -0.8797669, 3.3527915, -6.5193129, 5.4089851
3: -1.8468850, 7.5567837, -1.2386876, 3.9645653, -5.8114500, 8.7954712
4: -4.4865170, 5.2266645, -1.6498394, 3.7471557, -8.2336731, 6.8765025

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0325799
time: 0.42 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0325799
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2891956, 2.7382221, -6.7665720, 4.6330113
1: -2.9591291, 6.0517540, -0.3822187, 3.9519520, -6.9110813, 6.4339728
2: -4.2841620, 4.9001317, -0.8797669, 3.3527915, -7.6369534, 5.7798986
3: -1.9467452, 7.8118000, -1.2386876, 3.9645653, -5.9113102, 9.0504875
4: -5.7461939, 5.6184468, -1.6498394, 3.7471557, -9.4933491, 7.2682862

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2415651, upper bound: 12.0287104
time: 0.44 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2415650, upper bound: 12.0287104
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.6782563, 4.7223377, -9.2582836, 8.3785038
1: -3.3364797, 6.6092453, -2.8371258, 6.2889423, -9.6254196, 9.4463701
2: -4.8469982, 5.2329617, -4.2119579, 5.1972218, -10.0442190, 9.4449196
3: -2.0921249, 8.6418915, -2.1570172, 8.6666470, -10.7587719, 10.7989082
4: -6.4802928, 6.0003052, -5.7941303, 6.0009980, -12.4812908, 11.7944355

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2596123, upper bound: 12.2120622
time: 0.43 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2596123, upper bound: 12.2121292
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.4619281, 4.9763918, -9.5123377, 8.1621761
1: -3.3364797, 6.6092453, -2.6407366, 6.6547775, -9.9912548, 9.2499819
2: -4.8469982, 5.2329617, -3.9357042, 5.6090250, -10.4560213, 9.1686630
3: -2.0921249, 8.6418915, -2.2358243, 8.6746998, -10.7668247, 10.8777161
4: -6.4802928, 6.0003052, -5.4861760, 6.4149227, -12.8952160, 11.4864807

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0546015, upper bound: 12.1938918
time: 0.45 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0546015, upper bound: 12.2121292
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.8830261, 5.1888895, -4.1621442, 5.3945312, -10.2775564, 9.3510342
1: -3.6997132, 6.8758278, -3.1508241, 7.2959442, -10.9956570, 10.0266514
2: -5.4240475, 5.6200366, -4.7026148, 5.9927139, -11.4167604, 10.3226509
3: -2.3564258, 9.7877512, -2.4236612, 9.6652107, -12.0216370, 12.2114124
4: -7.2774620, 6.4821172, -6.4547949, 6.8532066, -14.1306686, 12.9369125

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0325652
time: 0.44 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2119686, upper bound: 12.2110739
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.8599315, 5.6000676, -4.1621442, 5.3945312, -10.2544632, 9.7622108
1: -3.6255574, 7.4982872, -3.1508241, 7.2959442, -10.9215012, 10.6491108
2: -5.3452921, 6.1820874, -4.7026148, 5.9927139, -11.3380022, 10.8846998
3: -2.4739380, 10.0032158, -2.4236612, 9.6652107, -12.1391478, 12.4268770
4: -7.2256942, 7.0176010, -6.4547949, 6.8532066, -14.0789003, 13.4723959

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0042798, upper bound: 12.1938918
time: 0.44 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0042798, upper bound: 12.2108540
time: 0.49 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.69 seconds
IS_A1_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2339810
IS_A1_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0643250, upper bound: 12.2339812
IS_A1_B2_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0450277, upper bound: 12.0574094
IS_A1_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0450277, upper bound: 12.2337273
IS_A1_B2_B1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1409591, upper bound: 12.0877423
IS_A1_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1409591, upper bound: 12.2595700
IS_A1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0603139, upper bound: 12.2337274
IS_A1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1591965, upper bound: 12.2511791
IS_A1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
IS_A1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
IS_A1_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0261820, upper bound: 12.2344839
IS_A1_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0348053, upper bound: 12.2344839
IS_A1_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
IS_A1_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1863545, upper bound: 12.2565418
IS_A1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0348053, upper bound: 12.2344839
IS_A1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1824850, upper bound: 12.2560777
IS_A1_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1555248, upper bound: 12.2035059
IS_A1_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1555248, upper bound: 12.2035059
IS_A1_B2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1409592, upper bound: 12.0200871
IS_A1_B2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2035060
IS_A1_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868741
IS_A1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1868742
IS_A1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2084575
IS_A1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2084575
IS_A1_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1868742
IS_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0346447, upper bound: 12.1868741
IS_A1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2084575
IS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2084575
IS_A2_B1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0643250
IS_A2_B1_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1332771, upper bound: 12.0643250
IS_A2_B1_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0574094, upper bound: 12.0450277
IS_A2_B1_A1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0574094, upper bound: 12.0603139
IS_A2_B1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1505732, upper bound: 12.1591296
IS_A2_B1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1505732, upper bound: 12.1591965
IS_A2_B1_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2337273, upper bound: 12.0603139
IS_A2_B1_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2511791, upper bound: 12.1591965
IS_A2_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
IS_A2_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
IS_A2_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
IS_A2_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
IS_A2_B1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
IS_A2_B1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2565418, upper bound: 12.1863545
IS_A2_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
IS_A2_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2560777, upper bound: 12.1824846
IS_A2_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0642829
IS_A2_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1862098, upper bound: 12.1630514
IS_A2_B1_A2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0200871, upper bound: 12.1409591
IS_A2_B1_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1590360
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346152
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1569524, upper bound: 12.1775644
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346152
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346447
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1569524, upper bound: 12.1775644
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775644
IS_A2_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0888682, upper bound: 12.2414639
IS_A2_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2414639
IS_A2_B2_B1_A1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0704842, upper bound: 12.0705026
IS_A2_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0704842, upper bound: 12.2409575
IS_A2_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0885848
IS_A2_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2420654, upper bound: 12.2583900
IS_A2_B2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2415650, upper bound: 12.0841362
IS_A2_B2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2415650, upper bound: 12.2582810
IS_A2_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2037779
IS_A2_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2037779
IS_A2_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2037779
IS_A2_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2037779
IS_A2_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0657663
IS_A2_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1934089, upper bound: 12.2115748
IS_A2_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0616089
IS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1934089, upper bound: 12.2119442
IS_A2_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0325799
IS_A2_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0325799
IS_A2_B2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2415651, upper bound: 12.0287104
IS_A2_B2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2415650, upper bound: 12.0287104
IS_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2596123, upper bound: 12.2120622
IS_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2596123, upper bound: 12.2121292
IS_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0546015, upper bound: 12.1938918
IS_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0546015, upper bound: 12.2121292
IS_A2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.1934089, upper bound: 12.0325652
IS_A2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.2119686, upper bound: 12.2110739
IS_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0042798, upper bound: 12.1938918
IS_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -12.0042798, upper bound: 12.2108540

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -2.6754963, 4.0768895, -4.2888570, 4.3396978
1: -0.2774035, 2.7060747, -2.0858450, 5.7528667, -6.0302687, 4.7919197
2: -0.6758080, 2.1488650, -3.1665211, 4.5292182, -5.2050261, 5.3153858
3: -0.9214749, 2.6601696, -1.8468850, 7.5567837, -8.4782581, 4.5070543
4: -1.2530408, 2.4732645, -4.4865170, 5.2266645, -6.4797049, 6.9597816

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2339810
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0568037, upper bound: 12.2337273
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2864694, 2.2201376, -2.6754963, 4.0768895, -4.3633580, 4.8956337
1: -0.3822893, 4.2234097, -2.0858450, 5.7528667, -6.1351557, 6.3092546
2: -1.1075993, 2.8155358, -3.1665211, 4.5292182, -5.6368175, 5.9820566
3: -1.2409499, 4.3273754, -1.8468850, 7.5567837, -8.7977324, 6.1742601
4: -2.0756645, 3.2417476, -4.4865170, 5.2266645, -7.3023291, 7.7282648

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0643250, upper bound: 12.2339812
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0568037, upper bound: 12.2337273
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -3.2853987, 3.9233079, -4.1352754, 4.9496002
1: -0.2774035, 2.7060747, -2.4223030, 5.5470581, -5.8244615, 5.1283779
2: -0.6758080, 2.1488650, -3.5144036, 4.5003858, -5.1761937, 5.6632686
3: -0.9214749, 2.6601696, -1.7845080, 6.9990902, -7.9205651, 4.4446774
4: -1.2530408, 2.4732645, -4.7958260, 5.2002707, -6.4533110, 7.2690907

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0054437, upper bound: 12.2206340
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0142377, upper bound: 12.1235070
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -3.8283892, 4.3033652, -4.5796361, 6.0836740
1: -0.3717489, 3.9332523, -2.8217945, 6.1582298, -6.5299768, 6.7550468
2: -1.0147076, 2.7846820, -4.1151338, 4.8628931, -5.8775988, 6.8998156
3: -1.1808519, 4.1503563, -1.9461601, 7.8930817, -9.0739336, 6.0965166
4: -1.9016094, 3.2297649, -5.5757437, 5.5972967, -7.4989057, 8.8055086

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1332771, upper bound: 12.2595700
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1409591, upper bound: 12.2511791
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1978316, 1.5574791, -4.5359459, 4.7002478, -4.8980794, 6.0934248
1: -0.2591586, 2.4123363, -3.3364797, 6.6092453, -6.8684030, 5.7488160
2: -0.5880604, 2.0356865, -4.8469982, 5.2329617, -5.8210220, 6.8826847
3: -0.8728676, 2.3451891, -2.0921249, 8.6418915, -9.5147591, 4.4373140
4: -1.0882773, 2.3351970, -6.4802928, 6.0003052, -7.0885811, 8.8154898

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0600639, upper bound: 12.2337274
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0600639, upper bound: 12.2337273
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5457935, 2.4314504, -4.5359459, 4.7002478, -5.2460413, 6.9673963
1: -0.5560611, 4.1216106, -3.3364797, 6.6092453, -7.1653047, 7.4580903
2: -1.0841694, 2.9876790, -4.8469982, 5.2329617, -6.3171310, 7.8346772
3: -1.2315452, 4.3375187, -2.0921249, 8.6418915, -9.8734369, 6.4296436
4: -1.9889984, 3.4472027, -6.4802928, 6.0003052, -7.9893022, 9.9274960

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1591965, upper bound: 12.2511791
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1505732, upper bound: 12.2511793
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.6754963, 4.0768895, -4.2911992, 4.4550762
1: -0.2778803, 2.8850155, -2.0858450, 5.7528667, -6.0307469, 4.9708605
2: -0.7044127, 2.2803013, -3.1665211, 4.5292182, -5.2336311, 5.4468222
3: -0.9286744, 2.8235273, -1.8468850, 7.5567837, -8.4854584, 4.6704121
4: -1.3212409, 2.5750978, -4.4865170, 5.2266645, -6.5479054, 7.0616150

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0347383, upper bound: 12.2344841
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3161098, 2.5077143, -2.6754963, 4.0768895, -4.3929992, 5.1832104
1: -0.4254482, 4.4234543, -2.0858450, 5.7528667, -6.1783152, 6.5092993
2: -1.1336632, 3.1795123, -3.1665211, 4.5292182, -5.6628814, 6.3460331
3: -1.3724446, 4.5201435, -1.8468850, 7.5567837, -8.9292278, 6.3670282
4: -2.1406889, 3.6535268, -4.4865170, 5.2266645, -7.3673534, 8.1400433

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346283
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0347383, upper bound: 12.2344841
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.0283499, 4.3438158, -4.5581255, 5.8079295
1: -0.2778803, 2.8850155, -2.9591291, 6.0517540, -6.3296342, 5.8441448
2: -0.7044127, 2.2803013, -4.2841620, 4.9001317, -5.6045446, 6.5644636
3: -0.9286744, 2.8235273, -1.9467452, 7.8118000, -8.7404747, 4.7702723
4: -1.3212409, 2.5750978, -5.7461939, 5.6184468, -6.9396877, 8.3212919

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0347383, upper bound: 12.2344841
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.2344839
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3161098, 2.5077143, -4.0283499, 4.3438158, -4.6599255, 6.5360641
1: -0.4254482, 4.4234543, -2.9591291, 6.0517540, -6.4772024, 7.3825836
2: -1.1336632, 3.1795123, -4.2841620, 4.9001317, -6.0337949, 7.4636745
3: -1.3724446, 4.5201435, -1.9467452, 7.8118000, -9.1842442, 6.4668884
4: -2.1406889, 3.6535268, -5.7461939, 5.6184468, -7.7591357, 9.3997211

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0347383, upper bound: 12.2344841
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348053, upper bound: 12.2344841
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.6754963, 4.0768895, -4.2911992, 4.4550762
1: -0.2778803, 2.8850155, -2.0858450, 5.7528667, -6.0307469, 4.9708605
2: -0.7044127, 2.2803013, -3.1665211, 4.5292182, -5.2336311, 5.4468222
3: -0.9286744, 2.8235273, -1.8468850, 7.5567837, -8.4854584, 4.6704121
4: -1.3212409, 2.5750978, -4.4865170, 5.2266645, -6.5479054, 7.0616150

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0345729, upper bound: 12.2346281
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0347383, upper bound: 12.2344839
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -2.6754963, 4.0768895, -4.6027875, 5.8064795
1: -0.5956205, 4.9794006, -2.0858450, 5.7528667, -6.3484869, 7.0652428
2: -1.2669206, 3.7673125, -3.1665211, 4.5292182, -5.7961388, 6.9338322
3: -1.4658637, 5.3376389, -1.8468850, 7.5567837, -9.0226450, 7.1845231
4: -2.3321438, 4.3155594, -4.4865170, 5.2266645, -7.5588083, 8.8020763

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863545, upper bound: 12.2565418
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1738617, upper bound: 12.2560777
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.0283499, 4.3438158, -4.5581255, 5.8079295
1: -0.2778803, 2.8850155, -2.9591291, 6.0517540, -6.3296342, 5.8441448
2: -0.7044127, 2.2803013, -4.2841620, 4.9001317, -5.6045446, 6.5644636
3: -0.9286744, 2.8235273, -1.9467452, 7.8118000, -8.7404747, 4.7702723
4: -1.3212409, 2.5750978, -5.7461939, 5.6184468, -6.9396877, 8.3212919

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0347383, upper bound: 12.2344839
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.2344839
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.0283499, 4.3438158, -4.8697147, 7.1593285
1: -0.5956205, 4.9794006, -2.9591291, 6.0517540, -6.6473746, 7.9385290
2: -1.2669206, 3.7673125, -4.2841620, 4.9001317, -6.1670518, 8.0514736
3: -1.4658637, 5.3376389, -1.9467452, 7.8118000, -9.2776604, 7.2843838
4: -2.3321438, 4.3155594, -5.7461939, 5.6184468, -7.9505906, 10.0617514

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.2560778
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824850, upper bound: 12.2560778
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3081400, 2.6606574, -3.6782563, 4.7223377, -5.0304775, 6.3389130
1: -0.4188291, 4.5758705, -2.8371258, 6.2889423, -6.7077713, 7.4129963
2: -1.1984115, 3.1719122, -4.2119579, 5.1972218, -6.3956332, 7.3838682
3: -1.2833123, 5.0121536, -2.1570172, 8.6666470, -9.9499569, 7.1691709
4: -2.2292862, 3.6357126, -5.7941303, 6.0009980, -8.2302837, 9.4298420

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1608059, upper bound: 12.2035059
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1555248, upper bound: 12.2035059
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.2891755, 2.8109655, -3.6782563, 4.7223377, -6.0115132, 6.4892211
1: -1.0236262, 4.7065582, -2.8371258, 6.2889423, -7.3125687, 7.5436840
2: -1.5639451, 3.3640735, -4.2119579, 5.1972218, -6.7611666, 7.5760298
3: -1.3348734, 5.0749187, -2.1570172, 8.6666470, -10.0015202, 7.2319360
4: -2.5174713, 3.8469820, -5.7941303, 6.0009980, -8.5184689, 9.6411095

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2035060
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1555248, upper bound: 12.2035059
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -3.4619281, 4.9763918, -6.7484856, 6.5898409
1: -1.3520081, 5.2492390, -2.6407366, 6.6547775, -8.0067844, 7.8899755
2: -2.0487065, 3.6625366, -3.9357042, 5.6090250, -7.6577311, 7.5982380
3: -1.4298269, 5.8484859, -2.2358243, 8.6746998, -10.1045265, 8.0843096
4: -3.0981421, 4.2000217, -5.4861760, 6.4149227, -9.5130653, 9.6861973

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1555248, upper bound: 12.2035059
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590360, upper bound: 12.2035059
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9344602, 5.2041121, -5.4184213, 6.7140398
1: -0.2778803, 2.8850155, -3.7369466, 6.8882389, -7.1661191, 6.6219621
2: -0.7044127, 2.2803013, -5.4744320, 5.6376672, -6.3420801, 7.7547331
3: -0.9286744, 2.8235273, -2.3689766, 9.8232098, -10.7518845, 5.1925039
4: -1.3212409, 2.5750978, -7.3408518, 6.5052090, -7.8264494, 9.9159498

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868741
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868742
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9042511, 5.5796232, -5.7939329, 6.6838307
1: -0.2778803, 2.8850155, -3.6561935, 7.4705601, -7.7484403, 6.5412092
2: -0.7044127, 2.2803013, -5.3807411, 6.1650767, -6.8694897, 7.6610422
3: -0.9286744, 2.8235273, -2.4826717, 9.9851646, -10.9138393, 5.3061991
4: -1.3212409, 2.5750978, -7.2726564, 7.0153542, -8.3365946, 9.8477545

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1868741
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0346152, upper bound: 12.1868742
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9344602, 5.2041121, -5.7300105, 8.0654411
1: -0.5956205, 4.9794006, -3.7369466, 6.8882389, -7.4838595, 8.7163448
2: -1.2669206, 3.7673125, -5.4744320, 5.6376672, -6.9045873, 9.2417421
3: -1.4658637, 5.3376389, -2.3689766, 9.8232098, -11.2890701, 7.7066154
4: -2.3321438, 4.3155594, -7.3408518, 6.5052090, -8.8373518, 11.6564112

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868741
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2084575
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9042511, 5.5796232, -6.1055222, 8.0352306
1: -0.5956205, 4.9794006, -3.6561935, 7.4705601, -8.0661793, 8.6355925
2: -1.2669206, 3.7673125, -5.3807411, 6.1650767, -7.4319968, 9.1480541
3: -1.4658637, 5.3376389, -2.4826717, 9.9851646, -11.4510288, 7.8203106
4: -2.3321438, 4.3155594, -7.2726564, 7.0153542, -9.3474932, 11.5882158

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0346152, upper bound: 12.1868741
time: 0.74 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2084575
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9344602, 5.2041121, -5.4184213, 6.7140398
1: -0.2778803, 2.8850155, -3.7369466, 6.8882389, -7.1661191, 6.6219621
2: -0.7044127, 2.2803013, -5.4744320, 5.6376672, -6.3420801, 7.7547331
3: -0.9286744, 2.8235273, -2.3689766, 9.8232098, -10.7518845, 5.1925039
4: -1.3212409, 2.5750978, -7.3408518, 6.5052090, -7.8264494, 9.9159498

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868741
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868742
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9817657, 5.6338024, -5.8481121, 6.7613454
1: -0.2778803, 2.8850155, -3.7128515, 7.5261250, -7.8040028, 6.5978670
2: -0.7044127, 2.2803013, -5.4640889, 6.2225499, -6.9269624, 7.7443905
3: -0.9286744, 2.8235273, -2.5014567, 10.0730762, -11.0017509, 5.3249841
4: -1.3212409, 2.5750978, -7.3761001, 7.0715237, -8.3927650, 9.9511976

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0346447, upper bound: 12.1868742
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0346447, upper bound: 12.1868742
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9344602, 5.2041121, -5.7300105, 8.0654411
1: -0.5956205, 4.9794006, -3.7369466, 6.8882389, -7.4838595, 8.7163448
2: -1.2669206, 3.7673125, -5.4744320, 5.6376672, -6.9045873, 9.2417421
3: -1.4658637, 5.3376389, -2.3689766, 9.8232098, -11.2890701, 7.7066154
4: -2.3321438, 4.3155594, -7.3408518, 6.5052090, -8.8373518, 11.6564112

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0386602, upper bound: 12.1868741
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2084575
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9817657, 5.6338024, -6.1597018, 8.1127453
1: -0.5956205, 4.9794006, -3.7128515, 7.5261250, -8.1217442, 8.6922503
2: -1.2669206, 3.7673125, -5.4640889, 6.2225499, -7.4894691, 9.2314014
3: -1.4658637, 5.3376389, -2.5014567, 10.0730762, -11.5389366, 7.8390956
4: -2.3321438, 4.3155594, -7.3761001, 7.0715237, -9.4036646, 11.6916599

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642476, upper bound: 12.0250386
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642476, upper bound: 12.2084575
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2119676, 1.6642017, -4.3396978, 4.2888570
1: -2.0858450, 5.7528667, -0.2774035, 2.7060747, -4.7919197, 6.0302701
2: -3.1665211, 4.5292182, -0.6758080, 2.1488650, -5.3153858, 5.2050261
3: -1.8468850, 7.5567837, -0.9214749, 2.6601696, -4.5070543, 8.4782581
4: -4.4865170, 5.2266645, -1.2530408, 2.4732645, -6.9597816, 6.4797049

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2339810, upper bound: 12.0643250
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2337273, upper bound: 12.0600639
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2762722, 2.2552848, -4.9307814, 4.3531604
1: -2.0858450, 5.7528667, -0.3717489, 3.9332523, -6.0190973, 6.1246157
2: -3.1665211, 4.5292182, -1.0147076, 2.7846820, -5.9512033, 5.5439258
3: -1.8468850, 7.5567837, -1.1808519, 4.1503563, -5.9972410, 8.7376356
4: -4.4865170, 5.2266645, -1.9016094, 3.2297649, -7.7162819, 7.1282740

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595700, upper bound: 12.1591289
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1505732, upper bound: 12.1591296
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2762722, 2.2552848, -6.2836347, 4.6200881
1: -2.9591291, 6.0517540, -0.3717489, 3.9332523, -6.8923817, 6.4235029
2: -4.2841620, 4.9001317, -1.0147076, 2.7846820, -7.0688438, 5.9148393
3: -1.9467452, 7.8118000, -1.1808519, 4.1503563, -6.0971012, 8.9926519
4: -5.7461939, 5.6184468, -1.9016094, 3.2297649, -8.9759588, 7.5200562

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595700, upper bound: 12.1591965
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2511791, upper bound: 12.1591965
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.1978316, 1.5574791, -6.0934248, 4.8980794
1: -3.3364797, 6.6092453, -0.2591586, 2.4123363, -5.7488160, 6.8684025
2: -4.8469982, 5.2329617, -0.5880604, 2.0356865, -6.8826847, 5.8210220
3: -2.0921249, 8.6418915, -0.8728676, 2.3451891, -4.4373140, 9.5147591
4: -6.4802928, 6.0003052, -1.0882773, 2.3351970, -8.8154898, 7.0885825

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2078200, upper bound: 12.0450988
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B1_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2137513, upper bound: 12.0552010
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.5457935, 2.4314504, -6.9673963, 5.2460413
1: -3.3364797, 6.6092453, -0.5560611, 4.1216106, -7.4580903, 7.1653066
2: -4.8469982, 5.2329617, -1.0841694, 2.9876790, -7.8346772, 6.3171310
3: -2.0921249, 8.6418915, -1.2315452, 4.3375187, -6.4296436, 9.8734369
4: -6.4802928, 6.0003052, -1.9889984, 3.4472027, -9.9274960, 7.9893031

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2511791, upper bound: 12.1591965
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2_B2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1505732, upper bound: 12.1591965
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2143096, 1.7795796, -4.4550762, 4.2911992
1: -2.0858450, 5.7528667, -0.2778803, 2.8850155, -4.9708605, 6.0307469
2: -3.1665211, 4.5292182, -0.7044127, 2.2803013, -5.4468222, 5.2336307
3: -1.8468850, 7.5567837, -0.9286744, 2.8235273, -4.6704121, 8.4854574
4: -4.4865170, 5.2266645, -1.3212409, 2.5750978, -7.0616150, 6.5479054

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
time: 0.49 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0347383
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.3161098, 2.5077143, -5.1832104, 4.3929992
1: -2.0858450, 5.7528667, -0.4254482, 4.4234543, -6.5092993, 6.1783152
2: -3.1665211, 4.5292182, -1.1336632, 3.1795123, -6.3460331, 5.6628814
3: -1.8468850, 7.5567837, -1.3724446, 4.5201435, -6.3670282, 8.9292278
4: -4.4865170, 5.2266645, -2.1406889, 3.6535268, -8.1400433, 7.3673534

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0347383
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2143096, 1.7795796, -5.8079295, 4.5581255
1: -2.9591291, 6.0517540, -0.2778803, 2.8850155, -5.8441448, 6.3296337
2: -4.2841620, 4.9001317, -0.7044127, 2.2803013, -6.5644636, 5.6045446
3: -1.9467452, 7.8118000, -0.9286744, 2.8235273, -4.7702723, 8.7404747
4: -5.7461939, 5.6184468, -1.3212409, 2.5750978, -8.3212919, 6.9396877

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0347383
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.3161098, 2.5077143, -6.5360641, 4.6599255
1: -2.9591291, 6.0517540, -0.4254482, 4.4234543, -7.3825836, 6.4772024
2: -4.2841620, 4.9001317, -1.1336632, 3.1795123, -7.4636745, 6.0337949
3: -1.9467452, 7.8118000, -1.3724446, 4.5201435, -6.4668884, 9.1842422
4: -5.7461939, 5.6184468, -2.1406889, 3.6535268, -9.3997211, 7.7591357

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0347383
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2143096, 1.7795796, -4.4550762, 4.2911992
1: -2.0858450, 5.7528667, -0.2778803, 2.8850155, -4.9708605, 6.0307469
2: -3.1665211, 4.5292182, -0.7044127, 2.2803013, -5.4468222, 5.2336307
3: -1.8468850, 7.5567837, -0.9286744, 2.8235273, -4.6704121, 8.4854574
4: -4.4865170, 5.2266645, -1.3212409, 2.5750978, -7.0616150, 6.5479054

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346281, upper bound: 12.0386749
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0347383
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.5258994, 3.1309829, -5.8064790, 4.6027880
1: -2.0858450, 5.7528667, -0.5956205, 4.9794006, -7.0652442, 6.3484874
2: -3.1665211, 4.5292182, -1.2669206, 3.7673125, -6.9338326, 5.7961388
3: -1.8468850, 7.5567837, -1.4658637, 5.3376389, -7.1845236, 9.0226450
4: -4.4865170, 5.2266645, -2.3321438, 4.3155594, -8.8020763, 7.5588083

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2565418, upper bound: 12.1863545
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2560777, upper bound: 12.1824176
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2143096, 1.7795796, -5.8079295, 4.5581255
1: -2.9591291, 6.0517540, -0.2778803, 2.8850155, -5.8441448, 6.3296337
2: -4.2841620, 4.9001317, -0.7044127, 2.2803013, -6.5644636, 5.6045446
3: -1.9467452, 7.8118000, -0.9286744, 2.8235273, -4.7702723, 8.7404747
4: -5.7461939, 5.6184468, -1.3212409, 2.5750978, -8.3212919, 6.9396877

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0347383
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2344839, upper bound: 12.0348053
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.5258994, 3.1309829, -7.1593299, 4.8697152
1: -2.9591291, 6.0517540, -0.5956205, 4.9794006, -7.9385295, 6.6473746
2: -4.2841620, 4.9001317, -1.2669206, 3.7673125, -8.0514736, 6.1670523
3: -1.9467452, 7.8118000, -1.4658637, 5.3376389, -7.2843833, 9.2776623
4: -5.7461939, 5.6184468, -2.3321438, 4.3155594, -10.0617533, 7.9505906

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2560777, upper bound: 12.1824176
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2560777, upper bound: 12.1824850
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -0.2119676, 1.6642017, -5.3424578, 4.9343052
1: -2.8371258, 6.2889423, -0.2774035, 2.7060747, -5.5432005, 6.5663457
2: -4.2119579, 5.1972218, -0.6758080, 2.1488650, -6.3608227, 5.8730297
3: -2.1570172, 8.6666470, -0.9214749, 2.6601696, -4.8171868, 9.5881205
4: -5.7941303, 6.0009980, -1.2530408, 2.4732645, -8.2673950, 7.2540379

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0642829
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0587775
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -0.9886395, 2.7079735, -6.3862295, 5.7109771
1: -2.8371258, 6.2889423, -0.8401878, 4.6642990, -7.5014248, 7.1291304
2: -4.2119579, 5.1972218, -1.3668380, 3.2843196, -7.4962759, 6.5640597
3: -2.1570172, 8.6666470, -1.3257837, 5.0050430, -7.1620593, 9.9924288
4: -5.7941303, 6.0009980, -2.3902826, 3.7804737, -9.5746021, 8.3912802

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.1630514
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.1590064
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -3.4619281, 4.9763918, -1.7720940, 3.1279130, -6.5898409, 6.7484856
1: -2.6407366, 6.6547775, -1.3520081, 5.2492390, -7.8899755, 8.0067844
2: -3.9357042, 5.6090250, -2.0487065, 3.6625366, -7.5982380, 7.6577306
3: -2.2358243, 8.6746998, -1.4298269, 5.8484859, -8.0843105, 10.1045256
4: -5.4861760, 6.4149227, -3.0981421, 4.2000217, -9.6861973, 9.5130653

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1590064
time: 0.48 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035058, upper bound: 12.1590355
time: 0.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.2143096, 1.7795796, -6.7140398, 5.4184217
1: -3.7369466, 6.8882389, -0.2778803, 2.8850155, -6.6219621, 7.1661191
2: -5.4744320, 5.6376672, -0.7044127, 2.2803013, -7.7547331, 6.3420801
3: -2.3689766, 9.8232098, -0.9286744, 2.8235273, -5.1925039, 10.7518835
4: -7.3408518, 6.5052090, -1.3212409, 2.5750978, -9.9159498, 7.8264499

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.47 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -4.9042511, 5.5796232, -0.2143096, 1.7795796, -6.6838307, 5.7939324
1: -3.6561935, 7.4705601, -0.2778803, 2.8850155, -6.5412092, 7.7484403
2: -5.3807411, 6.1650767, -0.7044127, 2.2803013, -7.6610422, 6.8694892
3: -2.4826717, 9.9851646, -0.9286744, 2.8235273, -5.3061991, 10.9138393
4: -7.2726564, 7.0153542, -1.3212409, 2.5750978, -9.8477545, 8.3365955

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346152
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346152
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.5258994, 3.1309829, -8.0654411, 5.7300110
1: -3.7369466, 6.8882389, -0.5956205, 4.9794006, -8.7163467, 7.4838595
2: -5.4744320, 5.6376672, -1.2669206, 3.7673125, -9.2417421, 6.9045877
3: -2.3689766, 9.8232098, -1.4658637, 5.3376389, -7.7066150, 11.2890692
4: -7.3408518, 6.5052090, -2.3321438, 4.3155594, -11.6564112, 8.8373508

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
time: 0.55 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -4.9042511, 5.5796232, -0.5258994, 3.1309829, -8.0352287, 6.1055226
1: -3.6561935, 7.4705601, -0.5956205, 4.9794006, -8.6355944, 8.0661802
2: -5.3807411, 6.1650767, -1.2669206, 3.7673125, -9.1480532, 7.4319973
3: -2.4826717, 9.9851646, -1.4658637, 5.3376389, -7.8203106, 11.4510288
4: -7.2726564, 7.0153542, -2.3321438, 4.3155594, -11.5882158, 9.3474941

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346152
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1569524, upper bound: 12.1775644
time: 0.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.2143096, 1.7795796, -6.7140398, 5.4184217
1: -3.7369466, 6.8882389, -0.2778803, 2.8850155, -6.6219621, 7.1661191
2: -5.4744320, 5.6376672, -0.7044127, 2.2803013, -7.7547331, 6.3420801
3: -2.3689766, 9.8232098, -0.9286744, 2.8235273, -5.1925039, 10.7518835
4: -7.3408518, 6.5052090, -1.3212409, 2.5750978, -9.9159498, 7.8264499

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.2143096, 1.7795796, -6.7613454, 5.8481121
1: -3.7128515, 7.5261250, -0.2778803, 2.8850155, -6.5978670, 7.8040037
2: -5.4640889, 6.2225499, -0.7044127, 2.2803013, -7.7443905, 6.9269629
3: -2.5014567, 10.0730762, -0.9286744, 2.8235273, -5.3249841, 11.0017509
4: -7.3761001, 7.0715237, -1.3212409, 2.5750978, -9.9511976, 8.3927650

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346447
time: 0.53 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0346447
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.5258994, 3.1309829, -8.0654411, 5.7300110
1: -3.7369466, 6.8882389, -0.5956205, 4.9794006, -8.7163467, 7.4838595
2: -5.4744320, 5.6376672, -1.2669206, 3.7673125, -9.2417421, 6.9045877
3: -2.3689766, 9.8232098, -1.4658637, 5.3376389, -7.7066150, 11.2890692
4: -7.3408518, 6.5052090, -2.3321438, 4.3155594, -11.6564112, 8.8373508

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1862098, upper bound: 12.0386602
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.5258994, 3.1309829, -8.1127415, 6.1597018
1: -3.7128515, 7.5261250, -0.5956205, 4.9794006, -8.6922522, 8.1217451
2: -5.4640889, 6.2225499, -1.2669206, 3.7673125, -9.2314014, 7.4894705
3: -2.5014567, 10.0730762, -1.4658637, 5.3376389, -7.8390956, 11.5389404
4: -7.3761001, 7.0715237, -2.3321438, 4.3155594, -11.6916599, 9.4036636

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0027910, upper bound: 12.1528482
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0027910, upper bound: 12.1775644
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -2.6754963, 4.0768895, -4.3511410, 4.9948835
1: -0.3634766, 3.2789097, -2.0858450, 5.7528667, -6.1163435, 5.3647547
2: -0.7444794, 2.9443834, -3.1665211, 4.5292182, -5.2736979, 6.1109047
3: -1.1945589, 3.2639122, -1.8468850, 7.5567837, -8.7513418, 5.1107969
4: -1.3846707, 3.3782203, -4.4865170, 5.2266645, -6.6113353, 7.8647375

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0888682, upper bound: 12.2414639
time: 0.59 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0840578, upper bound: 12.2409575
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3850740, 3.6355610, -2.6754963, 4.0768895, -4.4619632, 6.3110571
1: -0.5207902, 5.4637089, -2.0858450, 5.7528667, -6.2736559, 7.5495539
2: -1.4221835, 4.2520247, -3.1665211, 4.5292182, -5.9514017, 7.4185457
3: -1.6142063, 6.3127098, -1.8468850, 7.5567837, -9.1709890, 8.1595945
4: -2.5975704, 4.8423157, -4.4865170, 5.2266645, -7.8242331, 9.3288326

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0642498, upper bound: 12.2414639
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0840578, upper bound: 12.2409575
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -3.2853987, 3.9233079, -4.1975589, 5.6047859
1: -0.3634766, 3.2789097, -2.4223030, 5.5470581, -5.9105349, 5.7012129
2: -0.7444794, 2.9443834, -3.5144036, 4.5003858, -5.2448649, 6.4587870
3: -1.1945589, 3.2639122, -1.7845080, 6.9990902, -8.1936493, 5.0484200
4: -1.3846707, 3.3782203, -4.7958260, 5.2002707, -6.5849414, 8.1740465

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0278283, upper bound: 12.1310807
time: 0.54 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0309942, upper bound: 12.1226384
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -1.9736005, 3.7326164, -0.2742517, 2.3193870, -4.2929873, 4.0068679
1: -1.5754695, 5.3457413, -0.3634766, 3.2789097, -4.8543792, 5.7092166
2: -2.4574776, 4.2253923, -0.7444794, 2.9443834, -5.4018612, 4.9698715
3: -1.7051358, 6.8608799, -1.1945589, 3.2639122, -4.9690480, 8.0554380
4: -3.6110172, 4.8835225, -1.3846707, 3.3782203, -6.9892378, 6.2681932

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2420654, upper bound: 12.0885848
time: 0.45 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2415650, upper bound: 12.0841173
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -1.9736005, 3.7326164, -3.8283892, 4.3033652, -6.2769637, 7.5610056
1: -1.5754695, 5.3457413, -2.8217945, 6.1582298, -7.7336950, 8.1675348
2: -2.4574776, 4.2253923, -4.1151338, 4.8628931, -7.3203707, 8.3405266
3: -1.7051358, 6.8608799, -1.9461601, 7.8930817, -9.5982170, 8.8070393
4: -3.6110172, 4.8835225, -5.5757437, 5.5972967, -9.2083139, 10.4592648

Time for backsubstitution: 1.85 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=13.458332061767578
rel_dist={0: [-12.272072008240848, 12.272072008240851]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2030398, upper bound: 12.2562321
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2030398, upper bound: 12.2674811
time: 0.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -12.2030398, upper bound: 12.2562321
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -12.2030398, upper bound: 12.2674811

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -6.5181665, 5.6813068, -9.0809174, 10.3117990
1: -2.4989076, 6.0682201, -4.7726703, 7.8300819, -10.3289890, 10.8408871
2: -3.6286805, 4.3000150, -6.9090786, 6.1645494, -9.7932301, 11.2090931
3: -1.6621141, 7.2208080, -2.4828591, 10.7901459, -12.4522600, 9.7036667
4: -4.9867439, 4.9192414, -9.0542297, 7.0528383, -12.0395823, 13.9734707

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2010074
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2562321
time: 0.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2199450, 5.5384865, -7.0578270, 5.9997749, -12.2197199, 12.5963135
1: -4.5570164, 7.5879345, -5.1647902, 8.0923862, -12.6494017, 12.7527246
2: -6.5976086, 6.0316262, -7.4707565, 6.4741650, -13.0717735, 13.5023775
3: -2.4391820, 10.4356041, -2.6167192, 11.4067554, -13.8459377, 13.0523233
4: -8.6541815, 6.9089074, -9.7389002, 7.4115000, -16.0656815, 16.6478081

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2674809, upper bound: 12.2192610
time: 0.40 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2192610
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2010074
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -12.2010074, upper bound: 12.2562321
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -12.2674809, upper bound: 12.2192610
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2192610

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -3.3996100, 3.7936325, -7.1932425, 7.1932421
1: -2.4989076, 6.0682201, -2.4989076, 6.0682201, -8.5671263, 8.5671253
2: -3.6286805, 4.3000150, -3.6286805, 4.3000150, -7.9286952, 7.9286957
3: -1.6621141, 7.2208080, -1.6621141, 7.2208080, -8.8829193, 8.8829212
4: -4.9867439, 4.9192414, -4.9867439, 4.9192414, -9.9059830, 9.9059849

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642959, upper bound: 12.1854254
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875843, upper bound: 12.1875846
time: 0.36 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -6.2199450, 5.5384865, -8.9380951, 10.0135775
1: -2.4989076, 6.0682201, -4.5570164, 7.5879345, -10.0868416, 10.6252365
2: -3.6286805, 4.3000150, -6.5976086, 6.0316262, -9.6603069, 10.8976231
3: -1.6621141, 7.2208080, -2.4391820, 10.4356041, -12.0977182, 9.6599903
4: -4.9867439, 4.9192414, -8.6541815, 6.9089074, -11.8956509, 13.5734205

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1854254, upper bound: 12.2536414
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875846, upper bound: 12.2172286
time: 0.37 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -6.4802032, 5.7360954, -10.2720413, 11.1804504
1: -3.3364797, 6.6092453, -4.7478151, 7.7909470, -11.1274242, 11.3570557
2: -4.8469982, 5.2329617, -6.8815036, 6.2220130, -11.0690117, 12.1144629
3: -2.0921249, 8.6418915, -2.5057795, 10.8194199, -12.9115429, 11.1476698
4: -6.4802928, 6.0003052, -9.0135841, 7.1220121, -13.6023045, 15.0138874

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2536412, upper bound: 12.1896169
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2536414, upper bound: 12.2190257
time: 0.63 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -5.7025137, 5.4954720, -11.0247602, 11.6678905
1: -4.1178484, 8.0096436, -4.1966558, 7.5479527, -11.6657963, 12.2062988
2: -6.0670018, 6.5351744, -6.1181793, 6.0015764, -12.0685730, 12.6533537
3: -2.6416011, 10.8504009, -2.4031558, 10.2496758, -12.8912725, 13.2535534
4: -8.1326199, 7.4373612, -8.1085358, 6.8564520, -14.9890718, 15.5458956

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2172286, upper bound: 12.1896169
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2172286, upper bound: 12.2159045
time: 0.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.21 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.1642959, upper bound: 12.1854254
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.1875843, upper bound: 12.1875846
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.1854254, upper bound: 12.2536414
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.1875846, upper bound: 12.2172286
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.2536412, upper bound: 12.1896169
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.2536414, upper bound: 12.2190257
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.2172286, upper bound: 12.1896169
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -12.2172286, upper bound: 12.2159045

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -2.9074416, 3.5983777, -5.3704705, 6.0353546
1: -1.3520081, 5.2492390, -2.1505136, 5.8420887, -7.1940966, 7.3997526
2: -2.0487065, 3.6625366, -3.1339927, 4.1135736, -6.1622796, 6.7965288
3: -1.4298269, 5.8484859, -1.5804834, 6.8212261, -8.2510519, 7.4289689
4: -3.0981421, 4.2000217, -4.3847470, 4.7071581, -7.8052998, 8.5847683

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1621367
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1854254
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -2.1033995, 3.3071895, -4.6119037, 5.6711469
1: -1.0992223, 5.5531120, -1.5845429, 5.5075436, -6.6067657, 7.1376529
2: -1.7584276, 4.1799893, -2.3702707, 3.8270335, -5.5854592, 6.5502582
3: -1.5837383, 6.1448326, -1.4584694, 6.2334538, -7.8171916, 7.6033006
4: -2.9098625, 4.7395120, -3.4860649, 4.3662286, -7.2760892, 8.2255764

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1642959
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1875846
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2.9074416, 3.5983777, -4.5359459, 4.7002478, -7.6076894, 8.1343231
1: -2.1505136, 5.8420887, -3.3364797, 6.6092453, -8.7597580, 9.1785679
2: -3.1339927, 4.1135736, -4.8469982, 5.2329617, -8.3669538, 8.9605713
3: -1.5804834, 6.8212261, -2.0921249, 8.6418915, -10.2223749, 8.9133511
4: -4.3847470, 4.7071581, -6.4802928, 6.0003052, -10.3850517, 11.1874504

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1699647, upper bound: 12.2536414
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2150694
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2172286
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -2.1033995, 3.3071895, -5.5292892, 5.9653769, -8.0687742, 8.8364773
1: -1.5845429, 5.5075436, -4.1178484, 8.0096436, -9.5941868, 9.6253910
2: -2.3702707, 3.8270335, -6.0670018, 6.5351744, -8.9054451, 9.8940334
3: -1.4584694, 6.2334538, -2.6416011, 10.8504009, -12.3088684, 8.8750544
4: -3.4860649, 4.3662286, -8.1326199, 7.4373612, -10.9234257, 12.4988461

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642959, upper bound: 12.2150694
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2172286
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -2.9074416, 3.5983777, -8.1343231, 7.6076894
1: -3.3364797, 6.6092453, -2.1505136, 5.8420887, -9.1785679, 8.7597589
2: -4.8469982, 5.2329617, -3.1339927, 4.1135736, -8.9605713, 8.3669519
3: -2.0921249, 8.6418915, -1.5804834, 6.8212261, -8.9133511, 10.2223749
4: -6.4802928, 6.0003052, -4.3847470, 4.7071581, -11.1874504, 10.3850517

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2536412, upper bound: 12.1699647
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1663283
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1896168
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -5.6585217, 5.2859392, -9.8218851, 10.3587694
1: -3.3364797, 6.6092453, -4.1522350, 7.2949734, -10.6314507, 10.7614765
2: -4.8469982, 5.2329617, -6.0279489, 5.7889481, -10.6359463, 11.2609072
3: -2.0921249, 8.6418915, -2.3335843, 9.8811035, -11.9732285, 10.9754753
4: -6.4802928, 6.0003052, -7.9570732, 6.6318111, -13.1121044, 13.9573784

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427873, upper bound: 12.2190257
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427873, upper bound: 12.2190257
time: 0.77 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -2.1033995, 3.3071895, -8.8364763, 8.0687752
1: -4.1178484, 8.0096436, -1.5845429, 5.5075436, -9.6253910, 9.5941858
2: -6.0670018, 6.5351744, -2.3702707, 3.8270335, -9.8940344, 8.9054451
3: -2.6416011, 10.8504009, -1.4584694, 6.2334538, -8.8750534, 12.3088675
4: -8.1326199, 7.4373612, -3.4860649, 4.3662286, -12.4988461, 10.9234257

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1834686
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -4.9113054, 5.0598536, -10.5891418, 10.8766813
1: -4.1178484, 8.0096436, -3.6233335, 7.0735130, -11.1913614, 11.6329765
2: -6.0670018, 6.5351744, -5.2905264, 5.5907722, -11.6577702, 11.8257008
3: -2.6416011, 10.8504009, -2.2410431, 9.3554363, -11.9970379, 13.0914440
4: -8.1326199, 7.4373612, -7.0852556, 6.3898368, -14.5224571, 14.5226173

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2145008
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2145008
time: 0.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.30 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1621367
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1621367, upper bound: 12.1854254
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1642959
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1854254, upper bound: 12.1875846
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2150694
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1621367, upper bound: 12.2172286
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1642959, upper bound: 12.2150694
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2172286
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1663283
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1896168
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2427873, upper bound: 12.2190257
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2427873, upper bound: 12.2190257
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1834686
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2145008
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -12.2150695, upper bound: 12.2145008

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -1.7720940, 3.1279130, -4.9000072, 4.9000072
1: -1.3520081, 5.2492390, -1.3520081, 5.2492390, -6.6012468, 6.6012468
2: -2.0487065, 3.6625366, -2.0487065, 3.6625366, -5.7112427, 5.7112432
3: -1.4298269, 5.8484859, -1.4298269, 5.8484859, -7.2783108, 7.2783113
4: -3.0981421, 4.2000217, -3.0981421, 4.2000217, -7.2981606, 7.2981629

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0307843, upper bound: 12.1136653
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -1.3047158, 3.5677476, -5.3398414, 4.4326286
1: -1.3520081, 5.2492390, -1.0992223, 5.5531120, -6.9051199, 6.3484612
2: -2.0487065, 3.6625366, -1.7584276, 4.1799893, -6.2286935, 5.4209623
3: -1.4298269, 5.8484859, -1.5837383, 6.1448326, -7.5746579, 7.4322228
4: -3.0981421, 4.2000217, -2.9098625, 4.7395120, -7.8376508, 7.1098833

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1136653, upper bound: 12.1174830
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0776156
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.7360146, 3.0926361, -4.3973513, 5.3037624
1: -1.0992223, 5.5531120, -1.3263460, 5.2044163, -6.3036385, 6.8794580
2: -1.7584276, 4.1799893, -2.0131476, 3.6256113, -5.3840389, 6.1931353
3: -1.5837383, 6.1448326, -1.4237120, 5.7915325, -7.3752708, 7.5685444
4: -2.9098625, 4.7395120, -3.0545263, 4.1621399, -7.0720024, 7.7940354

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1174823, upper bound: 12.1284926
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0776152, upper bound: 12.0264942
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.3006983, 3.5677476, -4.8724632, 4.8684459
1: -1.0992223, 5.5531120, -1.0944859, 5.5531120, -6.6523342, 6.6475973
2: -1.7584276, 4.1799893, -1.7523882, 4.1799893, -5.9384151, 5.9323750
3: -1.5837383, 6.1448326, -1.5788388, 6.1448326, -7.7285709, 7.7236714
4: -2.9098625, 4.7395120, -2.9050078, 4.7395120, -7.6493735, 7.6445184

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1174830, upper bound: 12.1547019
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0776156, upper bound: 12.1133036
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -4.5359459, 4.7002478, -6.4723415, 7.6638589
1: -1.3520081, 5.2492390, -3.3364797, 6.6092453, -7.9612527, 8.5857182
2: -2.0487065, 3.6625366, -4.8469982, 5.2329617, -7.2816668, 8.5095329
3: -1.4298269, 5.8484859, -2.0921249, 8.6418915, -10.0717173, 7.9406104
4: -3.0981421, 4.2000217, -6.4802928, 6.0003052, -9.0984440, 10.6803150

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1559032, upper bound: 12.2427869
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0223006, upper bound: 12.1358232
time: 0.35 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 2.24 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0379964, upper bound: 12.1858748
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2427869
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -4.5359459, 4.7002478, -6.0049634, 8.1036911
1: -1.0992223, 5.5531120, -3.3364797, 6.6092453, -7.7084661, 8.8895893
2: -1.7584276, 4.1799893, -4.8469982, 5.2329617, -6.9913893, 9.0269861
3: -1.5837383, 6.1448326, -2.0921249, 8.6418915, -10.2256298, 8.2369576
4: -2.9098625, 4.7395120, -6.4802928, 6.0003052, -8.9101658, 11.2198048

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1559032, upper bound: 12.2536412
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0223006, upper bound: 12.1884490
time: 0.35 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 2.20 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0379964, upper bound: 12.1873944
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2536414
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.7360146, 3.0926361, -5.5292892, 5.9653769, -7.7013912, 8.6219254
1: -1.3263460, 5.2044163, -4.1178484, 8.0096436, -9.3359900, 9.3222647
2: -2.0131476, 3.6256113, -6.0670018, 6.5351744, -8.5483217, 9.6926098
3: -1.4237120, 5.7915325, -2.6416011, 10.8504009, -12.2741127, 8.4331331
4: -3.0545263, 4.1621399, -8.1326199, 7.4373612, -10.4918861, 12.2947598

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0035847, upper bound: 12.1154857
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 1.89 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1327170, upper bound: 12.0316506
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2150694
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.3006983, 3.5677476, -5.5292892, 5.9653769, -7.2660751, 9.0970364
1: -1.0944859, 5.5531120, -4.1178484, 8.0096436, -9.1041298, 9.6709595
2: -1.7523882, 4.1799893, -6.0670018, 6.5351744, -8.2875624, 10.2469893
3: -1.5788388, 6.1448326, -2.6416011, 10.8504009, -12.4292393, 8.7864323
4: -2.9050078, 4.7395120, -8.1326199, 7.4373612, -10.3423672, 12.8721294

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0035847, upper bound: 12.1794098
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 1.75 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2079222
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2079222
time: 0.58 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -1.7720940, 3.1279130, -7.6638584, 6.4723415
1: -3.3364797, 6.6092453, -1.3520081, 5.2492390, -8.5857182, 7.9612513
2: -4.8469982, 5.2329617, -2.0487065, 3.6625366, -8.5095329, 7.2816682
3: -2.0921249, 8.6418915, -1.4298269, 5.8484859, -7.9406109, 10.0717182
4: -6.4802928, 6.0003052, -3.0981421, 4.2000217, -10.6803150, 9.0984478

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1559032
time: 0.38 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1136653, upper bound: 12.0223006
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 2.22 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0379964
time: 0.39 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1663283
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -1.3047158, 3.5677476, -8.1036930, 6.0049634
1: -3.3364797, 6.6092453, -1.0992223, 5.5531120, -8.8895912, 7.7084675
2: -4.8469982, 5.2329617, -1.7584276, 4.1799893, -9.0269852, 6.9913888
3: -2.0921249, 8.6418915, -1.5837383, 6.1448326, -8.2369576, 10.2256269
4: -6.4802928, 6.0003052, -2.9098625, 4.7395120, -11.2198048, 8.9101677

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1699647
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1136653, upper bound: 12.1185458
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 2.33 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0379964
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1896169
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -4.5359459, 4.7002478, -9.2361937, 9.2361937
1: -3.3364797, 6.6092453, -3.3364797, 6.6092453, -9.9457207, 9.9457197
2: -4.8469982, 5.2329617, -4.8469982, 5.2329617, -10.0799589, 10.0799580
3: -2.0921249, 8.6418915, -2.0921249, 8.6418915, -10.7340164, 10.7340164
4: -6.4802928, 6.0003052, -6.4802928, 6.0003052, -12.4805984, 12.4805984

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2674809, upper bound: 12.2155387
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 1.84 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0351219
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2674810, upper bound: 12.2190257
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -5.3601141, 5.9159899, -10.4519348, 10.0603619
1: -3.3364797, 6.6092453, -3.9965582, 7.9697895, -11.3062687, 10.6058025
2: -4.8469982, 5.2329617, -5.9000158, 6.4758339, -11.3228321, 11.1329775
3: -2.0921249, 8.6418915, -2.6022196, 10.7416420, -12.8337669, 11.2441101
4: -6.4802928, 6.0003052, -7.9224205, 7.3602033, -13.8404961, 13.9227257

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2674809, upper bound: 12.2155387
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38

Time for candidate selection: 1.80 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0351219
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2674810, upper bound: 12.2190257
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -1.7360146, 3.0926361, -8.6219254, 7.7013912
1: -4.1178484, 8.0096436, -1.3263460, 5.2044163, -9.3222637, 9.3359900
2: -6.0670018, 6.5351744, -2.0131476, 3.6256113, -9.6926107, 8.5483217
3: -2.6416011, 10.8504009, -1.4237120, 5.7915325, -8.4331341, 12.2741098
4: -8.1326199, 7.4373612, -3.0545263, 4.1621399, -12.2947598, 10.4918880

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1154857, upper bound: 12.0035847
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 1.79 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0316506, upper bound: 12.1403412
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -1.3006983, 3.5677476, -9.0970364, 7.2660751
1: -4.1178484, 8.0096436, -1.0944859, 5.5531120, -9.6709595, 9.1041298
2: -6.0670018, 6.5351744, -1.7523882, 4.1799893, -10.2469873, 8.2875624
3: -2.6416011, 10.8504009, -1.5788388, 6.1448326, -8.7864342, 12.4292393
4: -8.1326199, 7.4373612, -2.9050078, 4.7395120, -12.8721313, 10.3423691

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1154857, upper bound: 12.0710651
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 1.80 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802070
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775618
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -4.4856672, 4.6578822, -10.1871710, 10.4510441
1: -4.1178484, 8.0096436, -3.2995057, 6.5600414, -10.6778851, 11.3091488
2: -6.0670018, 6.5351744, -4.7939720, 5.1907496, -11.2577477, 11.3291464
3: -2.6416011, 10.8504009, -2.0794299, 8.5718803, -11.2134819, 12.9298306
4: -8.1326199, 7.4373612, -6.4134078, 5.9567304, -14.0893459, 13.8507690

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38

Time for candidate selection: 1.42 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0316617, upper bound: 12.1922919
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -5.3601141, 5.9159899, -11.4452782, 11.3254910
1: -4.1178484, 8.0096436, -3.9965582, 7.9697895, -12.0876369, 12.0062017
2: -6.0670018, 6.5351744, -5.9000158, 6.4758339, -12.5428324, 12.4351902
3: -2.6416011, 10.8504009, -2.6022196, 10.7416420, -13.3832407, 13.4526205
4: -8.1326199, 7.4373612, -7.9224205, 7.3602033, -15.4928226, 15.3597813

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 1.42 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0316617, upper bound: 12.1922919
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
time: 0.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.56 seconds
IS_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0307843, upper bound: 12.1136653
IS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1136653, upper bound: 12.1174830
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0776156
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1174823, upper bound: 12.1284926
IS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0776152, upper bound: 12.0264942
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1174830, upper bound: 12.1547019
IS_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0776156, upper bound: 12.1133036
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0379964, upper bound: 12.1858748
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2427869
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0379964, upper bound: 12.1873944
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2536414
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1327170, upper bound: 12.0316506
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1663283, upper bound: 12.2150694
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2079222
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2079222
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0379964
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1663283
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0379964
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2427869, upper bound: 12.1896169
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0351219
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2674810, upper bound: 12.2190257
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0351219
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2674810, upper bound: 12.2190257
IS_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0316506, upper bound: 12.1403412
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2150694, upper bound: 12.1663283
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802070
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775618
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0316617, upper bound: 12.1922919
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.0316617, upper bound: 12.1922919
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -12.2192610, upper bound: 12.2145008

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.1397668, 3.3955503, -4.7002659, 4.7075143
1: -1.0992223, 5.5531120, -0.9875312, 5.3858223, -6.4850445, 6.5406423
2: -1.7584276, 4.1799893, -1.6082304, 4.0064540, -5.7648807, 5.7882175
3: -1.5837383, 6.1448326, -1.5311539, 5.8780403, -7.4617786, 7.6759863
4: -2.9098625, 4.7395120, -2.7573147, 4.5431938, -7.4530525, 7.4968243

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142164, upper bound: 12.1061224
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142164, upper bound: 12.1133032
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -4.5359459, 4.7002478, -4.9122152, 6.2001476
1: -0.2774035, 2.7060747, -3.3364797, 6.6092453, -6.8866448, 6.0425544
2: -0.6758080, 2.1488650, -4.8469982, 5.2329617, -5.9087696, 6.9958630
3: -0.9214749, 2.6601696, -2.0921249, 8.6418915, -9.5633659, 4.7522945
4: -1.2530408, 2.4732645, -6.4802928, 6.0003052, -7.2533436, 8.9535570

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0584438, upper bound: 12.1858746
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0493919, upper bound: 12.1327170
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.9886395, 2.7079735, -4.5359459, 4.7002478, -5.6888871, 7.2439189
1: -0.8401878, 4.6642990, -3.3364797, 6.6092453, -7.4494309, 8.0007782
2: -1.3668380, 3.2843196, -4.8469982, 5.2329617, -6.5997982, 8.1313162
3: -1.3257837, 5.0050430, -2.0921249, 8.6418915, -9.9676733, 7.0971680
4: -2.3902826, 3.7804737, -6.4802928, 6.0003052, -8.3905849, 10.2607670

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2398629
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2196475
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.5359459, 4.7002478, -4.9145575, 6.3155255
1: -0.2778803, 2.8850155, -3.3364797, 6.6092453, -6.8871236, 6.2214952
2: -0.7044127, 2.2803013, -4.8469982, 5.2329617, -5.9373741, 7.1272993
3: -0.9286744, 2.8235273, -2.0921249, 8.6418915, -9.5705662, 4.9156523
4: -1.3212409, 2.5750978, -6.4802928, 6.0003052, -7.3215446, 9.0553904

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873941
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.5359459, 4.7002478, -5.2261467, 7.6669259
1: -0.5956205, 4.9794006, -3.3364797, 6.6092453, -7.2048650, 8.3158770
2: -1.2669206, 3.7673125, -4.8469982, 5.2329617, -6.4998822, 8.6143103
3: -1.4658637, 5.3376389, -2.0921249, 8.6418915, -10.1077528, 7.4297638
4: -2.3321438, 4.3155594, -6.4802928, 6.0003052, -8.3324471, 10.7958527

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863544, upper bound: 12.2345203
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824820, upper bound: 12.2345203
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.7360146, 3.0926361, -4.1621442, 5.3945312, -7.1305456, 7.2547798
1: -1.3263460, 5.2044163, -3.1508241, 7.2959442, -8.6222906, 8.3552399
2: -2.0131476, 3.6256113, -4.7026148, 5.9927139, -8.0058613, 8.3282232
3: -1.4237120, 5.7915325, -2.4236612, 9.6652107, -11.0889225, 8.2151928
4: -3.0545263, 4.1621399, -6.4547949, 6.8532066, -9.9077291, 10.6169348

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2035059
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589979, upper bound: 12.2035060
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.3006983, 3.5677476, -4.9344602, 5.2041121, -6.5048103, 8.5022078
1: -1.0944859, 5.5531120, -3.7369466, 6.8882389, -7.9827247, 9.2900581
2: -1.7523882, 4.1799893, -5.4744320, 5.6376672, -7.3900542, 9.6544189
3: -1.5788388, 6.1448326, -2.3689766, 9.8232098, -11.4020481, 8.5138092
4: -2.9050078, 4.7395120, -7.3408518, 6.5052090, -9.4102144, 12.0803642

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2079222
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.3006983, 3.5677476, -4.9817657, 5.6338024, -6.9345007, 8.5495110
1: -1.0944859, 5.5531120, -3.7128515, 7.5261250, -8.6206112, 9.2659607
2: -1.7523882, 4.1799893, -5.4640889, 6.2225499, -7.9749374, 9.6440773
3: -1.5788388, 6.1448326, -2.5014567, 10.0730762, -11.6519146, 8.6462898
4: -2.9050078, 4.7395120, -7.3761001, 7.0715237, -9.9765272, 12.1156101

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1509753
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1823244, upper bound: 12.2079223
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2119676, 1.6642017, -6.2001476, 4.9122152
1: -3.3364797, 6.6092453, -0.2774035, 2.7060747, -6.0425544, 6.8866482
2: -4.8469982, 5.2329617, -0.6758080, 2.1488650, -6.9958630, 5.9087696
3: -2.0921249, 8.6418915, -0.9214749, 2.6601696, -4.7522945, 9.5633659
4: -6.4802928, 6.0003052, -1.2530408, 2.4732645, -8.9535570, 7.2533455

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.9886395, 2.7079735, -7.2439184, 5.6888871
1: -3.3364797, 6.6092453, -0.8401878, 4.6642990, -8.0007772, 7.4494324
2: -4.8469982, 5.2329617, -1.3668380, 3.2843196, -8.1313162, 6.5997996
3: -2.0921249, 8.6418915, -1.3257837, 5.0050430, -7.0971680, 9.9676752
4: -6.4802928, 6.0003052, -2.3902826, 3.7804737, -10.2607651, 8.3905859

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2398624, upper bound: 12.1589889
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2143096, 1.7795796, -6.3155255, 4.9145575
1: -3.3364797, 6.6092453, -0.2778803, 2.8850155, -6.2214952, 6.8871250
2: -4.8469982, 5.2329617, -0.7044127, 2.2803013, -7.1272993, 5.9373741
3: -2.0921249, 8.6418915, -0.9286744, 2.8235273, -4.9156523, 9.5705662
4: -6.4802928, 6.0003052, -1.3212409, 2.5750978, -9.0553904, 7.3215446

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.5258994, 3.1309829, -7.6669254, 5.2261472
1: -3.3364797, 6.6092453, -0.5956205, 4.9794006, -8.3158798, 7.2048659
2: -4.8469982, 5.2329617, -1.2669206, 3.7673125, -8.6143074, 6.4998817
3: -2.0921249, 8.6418915, -1.4658637, 5.3376389, -7.4297638, 10.1077557
4: -6.4802928, 6.0003052, -2.3321438, 4.3155594, -10.7958527, 8.3324471

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1863544
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1824820
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2742517, 2.3193870, -6.8553329, 4.9744997
1: -3.3364797, 6.6092453, -0.3634766, 3.2789097, -6.6153893, 6.9727206
2: -4.8469982, 5.2329617, -0.7444794, 2.9443834, -7.7913818, 5.9774408
3: -2.0921249, 8.6418915, -1.1945589, 3.2639122, -5.3560371, 9.8364506
4: -6.4802928, 6.0003052, -1.3846707, 3.3782203, -9.8585129, 7.3849754

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2373023, upper bound: 12.0756598
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.8283892, 4.3033652, -8.8393116, 8.5286369
1: -3.3364797, 6.6092453, -2.8217945, 6.1582298, -9.4947071, 9.4310369
2: -4.8469982, 5.2329617, -4.1151338, 4.8628931, -9.7098885, 9.3480949
3: -2.0921249, 8.6418915, -1.9461601, 7.8930817, -9.9852066, 10.5880508
4: -6.4802928, 6.0003052, -5.5757437, 5.5972967, -12.0775890, 11.5760479

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2636720, upper bound: 12.2588644
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2588644
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2891956, 2.7382221, -7.2741680, 4.9894433
1: -3.3364797, 6.6092453, -0.3822187, 3.9519520, -7.2884316, 6.9914627
2: -4.8469982, 5.2329617, -0.8797669, 3.3527915, -8.1997900, 6.1127281
3: -2.0921249, 8.6418915, -1.2386876, 3.9645653, -6.0566902, 9.8805790
4: -6.4802928, 6.0003052, -1.6498394, 3.7471557, -10.2274485, 7.6501431

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0316612
time: 0.60 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0274242
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -4.1076355, 5.3777466, -9.9136925, 8.8078833
1: -3.3364797, 6.6092453, -3.1126299, 7.2822380, -10.6187153, 9.7218742
2: -4.8469982, 5.2329617, -4.6482496, 5.9726243, -10.8196220, 9.8812094
3: -2.0921249, 8.6418915, -2.4097505, 9.6314602, -11.7235851, 11.0516415
4: -6.4802928, 6.0003052, -6.3888316, 6.8270602, -13.3073530, 12.3891354

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2636573, upper bound: 12.2120318
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2590806, upper bound: 12.2120318
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -1.7360146, 3.0926361, -7.2547798, 7.1305451
1: -3.1508241, 7.2959442, -1.3263460, 5.2044163, -8.3552399, 8.6222906
2: -4.7026148, 5.9927139, -2.0131476, 3.6256113, -8.3282232, 8.0058603
3: -2.4236612, 9.6652107, -1.4237120, 5.7915325, -8.2151928, 11.0889225
4: -6.4547949, 6.8532066, -3.0545263, 4.1621399, -10.6169338, 9.9077301

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1630514
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1589979
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -1.3006983, 3.5677476, -8.5022078, 6.5048103
1: -3.7369466, 6.8882389, -1.0944859, 5.5531120, -9.2900581, 7.9827247
2: -5.4744320, 5.6376672, -1.7523882, 4.1799893, -9.6544189, 7.3900547
3: -2.3689766, 9.8232098, -1.5788388, 6.1448326, -8.5138092, 11.4020472
4: -7.3408518, 6.5052090, -2.9050078, 4.7395120, -12.0803642, 9.4102154

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0349293
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -1.3006983, 3.5677476, -8.5495110, 6.9345007
1: -3.7128515, 7.5261250, -1.0944859, 5.5531120, -9.2659607, 8.6206093
2: -5.4640889, 6.2225499, -1.7523882, 4.1799893, -9.6440783, 7.9749384
3: -2.5014567, 10.0730762, -1.5788388, 6.1448326, -8.6462898, 11.6519146
4: -7.3761001, 7.0715237, -2.9050078, 4.7395120, -12.1156101, 9.9765301

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810497, upper bound: 12.0294756
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775616
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.4856672, 4.6578822, -4.9470778, 7.2238894
1: -0.3822187, 3.9519520, -3.2995057, 6.5600414, -6.9422574, 7.2514577
2: -0.8797669, 3.3527915, -4.7939720, 5.1907496, -6.0705152, 8.1467638
3: -1.2386876, 3.9645653, -2.0794299, 8.5718803, -9.8105679, 6.0439949
4: -1.6498394, 3.7471557, -6.4134078, 5.9567304, -7.6065693, 10.1605635

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2020829
time: 0.65 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2020825
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -4.4856672, 4.6578822, -8.8200264, 9.8801966
1: -3.1508241, 7.2959442, -3.2995057, 6.5600414, -9.7108612, 10.5954494
2: -4.7026148, 5.9927139, -4.7939720, 5.1907496, -9.8933601, 10.7866840
3: -2.4236612, 9.6652107, -2.0794299, 8.5718803, -10.9955406, 11.7446404
4: -6.4547949, 6.8532066, -6.4134078, 5.9567304, -12.4115219, 13.2666111

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118832, upper bound: 12.2115748
time: 0.55 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035060, upper bound: 12.2122770
time: 0.52 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -5.3601141, 5.9159899, -6.2051854, 8.0983362
1: -0.3822187, 3.9519520, -3.9965582, 7.9697895, -8.3520050, 7.9485102
2: -0.8797669, 3.3527915, -5.9000158, 6.4758339, -7.3555994, 9.2528076
3: -1.2386876, 3.9645653, -2.6022196, 10.7416420, -11.9803295, 6.5667849
4: -1.6498394, 3.7471557, -7.9224205, 7.3602033, -9.0100422, 11.6695766

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.1922919
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -5.3601141, 5.9159899, -10.0781345, 10.7546453
1: -3.1508241, 7.2959442, -3.9965582, 7.9697895, -11.1206093, 11.2925024
2: -4.7026148, 5.9927139, -5.9000158, 6.4758339, -11.1784468, 11.8927298
3: -2.4236612, 9.6652107, -2.6022196, 10.7416420, -13.1653032, 12.2674303
4: -6.4547949, 6.8532066, -7.9224205, 7.3602033, -13.8149986, 14.7756271

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159841, upper bound: 12.2108905
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2118426, upper bound: 12.2108905
time: 0.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.42 seconds
IS_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1142164, upper bound: 12.1061224
IS_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1142164, upper bound: 12.1133032
IS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0584438, upper bound: 12.1858746
IS_A1_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0493919, upper bound: 12.1327170
IS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2398629
IS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2196475
IS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873941
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1863544, upper bound: 12.2345203
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1824820, upper bound: 12.2345203
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2035059
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1589979, upper bound: 12.2035060
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2079222
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1509753
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1823244, upper bound: 12.2079223
IS_A2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
IS_A2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
IS_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2398624, upper bound: 12.1589889
IS_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
IS_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
IS_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
IS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1863544
IS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1824820
IS_A2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
IS_A2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2373023, upper bound: 12.0756598
IS_A2_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2636720, upper bound: 12.2588644
IS_A2_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2588644
IS_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0316612
IS_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0274242
IS_A2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2636573, upper bound: 12.2120318
IS_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2590806, upper bound: 12.2120318
IS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1630514
IS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1589979
IS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0349293
IS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
IS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.1810497, upper bound: 12.0294756
IS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1775616
IS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2020829
IS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2020825
IS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2118832, upper bound: 12.2115748
IS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2035060, upper bound: 12.2122770
IS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
IS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.0217164, upper bound: 12.1922919
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2159841, upper bound: 12.2108905
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -12.2118426, upper bound: 12.2108905

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -2.6754963, 4.0768895, -4.2888570, 4.3396978
1: -0.2774035, 2.7060747, -2.0858450, 5.7528667, -6.0302687, 4.7919197
2: -0.6758080, 2.1488650, -3.1665211, 4.5292182, -5.2050261, 5.3153858
3: -0.9214749, 2.6601696, -1.8468850, 7.5567837, -8.4782581, 4.5070543
4: -1.2530408, 2.4732645, -4.4865170, 5.2266645, -6.4797049, 6.9597816

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0584438, upper bound: 12.1858748
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0577193, upper bound: 12.1327170
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -4.5359459, 4.7002478, -4.9765196, 6.7912307
1: -0.3717489, 3.9332523, -3.3364797, 6.6092453, -6.9809937, 7.2697320
2: -1.0147076, 2.7846820, -4.8469982, 5.2329617, -6.2476692, 7.6316805
3: -1.1808519, 4.1503563, -2.0921249, 8.6418915, -9.8227434, 6.2424812
4: -1.9016094, 3.2297649, -6.4802928, 6.0003052, -7.9019132, 9.7100582

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1403413, upper bound: 12.0792428
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589783, upper bound: 12.2196471
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589783, upper bound: 12.2196471
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5457935, 2.4314504, -4.5359459, 4.7002478, -5.2460413, 6.9673963
1: -0.5560611, 4.1216106, -3.3364797, 6.6092453, -7.1653047, 7.4580903
2: -1.0841694, 2.9876790, -4.8469982, 5.2329617, -6.3171310, 7.8346772
3: -1.2315452, 4.3375187, -2.0921249, 8.6418915, -9.8734369, 6.4296436
4: -1.9889984, 3.4472027, -6.4802928, 6.0003052, -7.9893022, 9.9274960

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0530349, upper bound: 12.1858746
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1505732, upper bound: 12.2196475
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.6754963, 4.0768895, -4.2911992, 4.4550762
1: -0.2778803, 2.8850155, -2.0858450, 5.7528667, -6.0307469, 4.9708605
2: -0.7044127, 2.2803013, -3.1665211, 4.5292182, -5.2336311, 5.4468222
3: -0.9286744, 2.8235273, -1.8468850, 7.5567837, -8.4854584, 4.6704121
4: -1.3212409, 2.5750978, -4.4865170, 5.2266645, -6.5479054, 7.0616150

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873941
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.0283499, 4.3438158, -4.5581255, 5.8079295
1: -0.2778803, 2.8850155, -2.9591291, 6.0517540, -6.3296342, 5.8441448
2: -0.7044127, 2.2803013, -4.2841620, 4.9001317, -5.6045446, 6.5644636
3: -0.9286744, 2.8235273, -1.9467452, 7.8118000, -8.7404747, 4.7702723
4: -1.3212409, 2.5750978, -5.7461939, 5.6184468, -6.9396877, 8.3212919

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873941
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1873944
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -2.6754963, 4.0768895, -4.6027875, 5.8064795
1: -0.5956205, 4.9794006, -2.0858450, 5.7528667, -6.3484869, 7.0652428
2: -1.2669206, 3.7673125, -3.1665211, 4.5292182, -5.7961388, 6.9338322
3: -1.4658637, 5.3376389, -1.8468850, 7.5567837, -9.0226450, 7.1845231
4: -2.3321438, 4.3155594, -4.4865170, 5.2266645, -7.5588083, 8.8020763

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863544, upper bound: 12.2345207
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.0283499, 4.3438158, -4.8697147, 7.1593285
1: -0.5956205, 4.9794006, -2.9591291, 6.0517540, -6.6473746, 7.9385290
2: -1.2669206, 3.7673125, -4.2841620, 4.9001317, -6.1670518, 8.0514736
3: -1.4658637, 5.3376389, -1.9467452, 7.8118000, -9.2776604, 7.2843838
4: -2.3321438, 4.3155594, -5.7461939, 5.6184468, -7.9505906, 10.0617514

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873944
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824820, upper bound: 12.2345203
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.7360146, 3.0926361, -3.6782563, 4.7223377, -6.4583521, 6.7708907
1: -1.3263460, 5.2044163, -2.8371258, 6.2889423, -7.6152883, 8.0415421
2: -2.0131476, 3.6256113, -4.2119579, 5.1972218, -7.2103691, 7.8375688
3: -1.4237120, 5.7915325, -2.1570172, 8.6666470, -10.0903587, 7.9485497
4: -3.0545263, 4.1621399, -5.7941303, 6.0009980, -9.0555239, 9.9562683

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589869, upper bound: 12.2035059
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589869, upper bound: 12.2035059
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.7360146, 3.0926361, -3.4619281, 4.9763918, -6.7124062, 6.5545640
1: -1.3263460, 5.2044163, -2.6407366, 6.6547775, -7.9811211, 7.8451529
2: -2.0131476, 3.6256113, -3.9357042, 5.6090250, -7.6221704, 7.5613117
3: -1.4237120, 5.7915325, -2.2358243, 8.6746998, -10.0984116, 8.0273571
4: -3.0545263, 4.1621399, -5.4861760, 6.4149227, -9.4694490, 9.6483135

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1328502, upper bound: 12.0200871
time: 0.48 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589979, upper bound: 12.2035060
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9344602, 5.2041121, -5.4184213, 6.7140398
1: -0.2778803, 2.8850155, -3.7369466, 6.8882389, -7.1661191, 6.6219621
2: -0.7044127, 2.2803013, -5.4744320, 5.6376672, -6.3420801, 7.7547331
3: -0.9286744, 2.8235273, -2.3689766, 9.8232098, -10.7518845, 5.1925039
4: -1.3212409, 2.5750978, -7.3408518, 6.5052090, -7.8264494, 9.9159498

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1507419
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9344602, 5.2041121, -5.7300105, 8.0654411
1: -0.5956205, 4.9794006, -3.7369466, 6.8882389, -7.4838595, 8.7163448
2: -1.2669206, 3.7673125, -5.4744320, 5.6376672, -6.9045873, 9.2417421
3: -1.4658637, 5.3376389, -2.3689766, 9.8232098, -11.2890701, 7.7066154
4: -2.3321438, 4.3155594, -7.3408518, 6.5052090, -8.8373518, 11.6564112

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2079222
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2079222
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9817657, 5.6338024, -5.8481121, 6.7613454
1: -0.2778803, 2.8850155, -3.7128515, 7.5261250, -7.8040028, 6.5978670
2: -0.7044127, 2.2803013, -5.4640889, 6.2225499, -6.9269624, 7.7443905
3: -0.9286744, 2.8235273, -2.5014567, 10.0730762, -11.0017509, 5.3249841
4: -1.3212409, 2.5750978, -7.3761001, 7.0715237, -8.3927650, 9.9511976

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1507419
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1509753
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9817657, 5.6338024, -6.1597018, 8.1127453
1: -0.5956205, 4.9794006, -3.7128515, 7.5261250, -8.1217442, 8.6922503
2: -1.2669206, 3.7673125, -5.4640889, 6.2225499, -7.4894691, 9.2314014
3: -1.4658637, 5.3376389, -2.5014567, 10.0730762, -11.5389366, 7.8390956
4: -2.3321438, 4.3155594, -7.3761001, 7.0715237, -9.4036646, 11.6916599

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2079222
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1823244, upper bound: 12.2079222
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2119676, 1.6642017, -4.3396978, 4.2888570
1: -2.0858450, 5.7528667, -0.2774035, 2.7060747, -4.7919197, 6.0302701
2: -3.1665211, 4.5292182, -0.6758080, 2.1488650, -5.3153858, 5.2050261
3: -1.8468850, 7.5567837, -0.9214749, 2.6601696, -4.5070543, 8.4782581
4: -4.4865170, 5.2266645, -1.2530408, 2.4732645, -6.9597816, 6.4797049

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2119676, 1.6642017, -5.6925516, 4.5557833
1: -2.9591291, 6.0517540, -0.2774035, 2.7060747, -5.6652040, 6.3291550
2: -4.2841620, 4.9001317, -0.6758080, 2.1488650, -6.4330273, 5.5759397
3: -1.9467452, 7.8118000, -0.9214749, 2.6601696, -4.6069145, 8.7332735
4: -5.7461939, 5.6184468, -1.2530408, 2.4732645, -8.2194586, 6.8714876

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2762722, 2.2552848, -6.7912307, 4.9765201
1: -3.3364797, 6.6092453, -0.3717489, 3.9332523, -7.2697320, 6.9809942
2: -4.8469982, 5.2329617, -1.0147076, 2.7846820, -7.6316805, 6.2476692
3: -2.0921249, 8.6418915, -1.1808519, 4.1503563, -6.2424812, 9.8227434
4: -6.4802928, 6.0003052, -1.9016094, 3.2297649, -9.7100582, 7.9019122

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589783
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.5457935, 2.4314504, -6.9673963, 5.2460413
1: -3.3364797, 6.6092453, -0.5560611, 4.1216106, -7.4580903, 7.1653066
2: -4.8469982, 5.2329617, -1.0841694, 2.9876790, -7.8346772, 6.3171310
3: -2.0921249, 8.6418915, -1.2315452, 4.3375187, -6.4296436, 9.8734369
4: -6.4802928, 6.0003052, -1.9889984, 3.4472027, -9.9274960, 7.9893031

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2143096, 1.7795796, -4.4550762, 4.2911992
1: -2.0858450, 5.7528667, -0.2778803, 2.8850155, -4.9708605, 6.0307469
2: -3.1665211, 4.5292182, -0.7044127, 2.2803013, -5.4468222, 5.2336307
3: -1.8468850, 7.5567837, -0.9286744, 2.8235273, -4.6704121, 8.4854574
4: -4.4865170, 5.2266645, -1.3212409, 2.5750978, -7.0616150, 6.5479054

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2143096, 1.7795796, -5.8079295, 4.5581255
1: -2.9591291, 6.0517540, -0.2778803, 2.8850155, -5.8441448, 6.3296337
2: -4.2841620, 4.9001317, -0.7044127, 2.2803013, -6.5644636, 5.6045446
3: -1.9467452, 7.8118000, -0.9286744, 2.8235273, -4.7702723, 8.7404747
4: -5.7461939, 5.6184468, -1.3212409, 2.5750978, -8.3212919, 6.9396877

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.5258994, 3.1309829, -5.8064790, 4.6027880
1: -2.0858450, 5.7528667, -0.5956205, 4.9794006, -7.0652442, 6.3484874
2: -3.1665211, 4.5292182, -1.2669206, 3.7673125, -6.9338326, 5.7961388
3: -1.8468850, 7.5567837, -1.4658637, 5.3376389, -7.1845236, 9.0226450
4: -4.4865170, 5.2266645, -2.3321438, 4.3155594, -8.8020763, 7.5588083

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1863544
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.5258994, 3.1309829, -7.1593299, 4.8697152
1: -2.9591291, 6.0517540, -0.5956205, 4.9794006, -7.9385295, 6.6473746
2: -4.2841620, 4.9001317, -1.2669206, 3.7673125, -8.0514736, 6.1670523
3: -1.9467452, 7.8118000, -1.4658637, 5.3376389, -7.2843833, 9.2776623
4: -5.7461939, 5.6184468, -2.3321438, 4.3155594, -10.0617533, 7.9505906

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.52 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1824820
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2742517, 2.3193870, -4.9948835, 4.3511405
1: -2.0858450, 5.7528667, -0.3634766, 3.2789097, -5.3647547, 6.1163411
2: -3.1665211, 4.5292182, -0.7444794, 2.9443834, -6.1109047, 5.2736979
3: -1.8468850, 7.5567837, -1.1945589, 3.2639122, -5.1107969, 8.7513428
4: -4.4865170, 5.2266645, -1.3846707, 3.3782203, -7.8647375, 6.6113353

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2742517, 2.3193870, -6.3477368, 4.6180677
1: -2.9591291, 6.0517540, -0.3634766, 3.2789097, -6.2380390, 6.4152293
2: -4.2841620, 4.9001317, -0.7444794, 2.9443834, -7.2285452, 5.6446114
3: -1.9467452, 7.8118000, -1.1945589, 3.2639122, -5.2106571, 9.0063572
4: -5.7461939, 5.6184468, -1.3846707, 3.3782203, -9.1244144, 7.0031176

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0640623, upper bound: 12.0642879
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0640623, upper bound: 12.0756598
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -1.9736005, 3.7326164, -8.2685623, 6.6738482
1: -3.3364797, 6.6092453, -1.5754695, 5.3457413, -8.6822186, 8.1847134
2: -4.8469982, 5.2329617, -2.4574776, 4.2253923, -9.0723896, 7.6904383
3: -2.0921249, 8.6418915, -1.7051358, 6.8608799, -8.9530048, 10.3470268
4: -6.4802928, 6.0003052, -3.6110172, 4.8835225, -11.3638153, 9.6113224

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2582148
time: 0.49 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2588644
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.2853987, 3.9233079, -8.4592524, 7.9856462
1: -3.3364797, 6.6092453, -2.4223030, 5.5470581, -8.8835373, 9.0315466
2: -4.8469982, 5.2329617, -3.5144036, 4.5003858, -9.3473835, 8.7473650
3: -2.0921249, 8.6418915, -1.7845080, 6.9990902, -9.0912151, 10.4263992
4: -6.4802928, 6.0003052, -4.7958260, 5.2002707, -11.6805630, 10.7961311

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0640623, upper bound: 12.2372875
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0640623, upper bound: 12.2582810
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2891956, 2.7382221, -5.4137182, 4.3660851
1: -2.0858450, 5.7528667, -0.3822187, 3.9519520, -6.0377970, 6.1350851
2: -3.1665211, 4.5292182, -0.8797669, 3.3527915, -6.5193129, 5.4089851
3: -1.8468850, 7.5567837, -1.2386876, 3.9645653, -5.8114500, 8.7954712
4: -4.4865170, 5.2266645, -1.6498394, 3.7471557, -8.2336731, 6.8765025

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0316612
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0316612
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2891956, 2.7382221, -6.7665720, 4.6330113
1: -2.9591291, 6.0517540, -0.3822187, 3.9519520, -6.9110813, 6.4339728
2: -4.2841620, 4.9001317, -0.8797669, 3.3527915, -7.6369534, 5.7798986
3: -1.9467452, 7.8118000, -1.2386876, 3.9645653, -5.9113102, 9.0504875
4: -5.7461939, 5.6184468, -1.6498394, 3.7471557, -9.4933491, 7.2682862

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873942, upper bound: 12.0274242
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0274242
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.6782563, 4.7223377, -9.2582836, 8.3785038
1: -3.3364797, 6.6092453, -2.8371258, 6.2889423, -9.6254196, 9.4463701
2: -4.8469982, 5.2329617, -4.2119579, 5.1972218, -10.0442190, 9.4449196
3: -2.0921249, 8.6418915, -2.1570172, 8.6666470, -10.7587719, 10.7989082
4: -6.4802928, 6.0003052, -5.7941303, 6.0009980, -12.4812908, 11.7944355

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2590806, upper bound: 12.2120318
time: 0.58 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2590806, upper bound: 12.2120318
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.4619281, 4.9763918, -9.5123377, 8.1621761
1: -3.3364797, 6.6092453, -2.6407366, 6.6547775, -9.9912548, 9.2499819
2: -4.8469982, 5.2329617, -3.9357042, 5.6090250, -10.4560213, 9.1686630
3: -2.0921249, 8.6418915, -2.2358243, 8.6746998, -10.7668247, 10.8777161
4: -6.4802928, 6.0003052, -5.4861760, 6.4149227, -12.8952160, 11.4864807

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0321817, upper bound: 12.1923025
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0321817, upper bound: 12.2120318
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -1.7360146, 3.0926361, -6.7708912, 6.4583521
1: -2.8371258, 6.2889423, -1.3263460, 5.2044163, -8.0415421, 7.6152883
2: -4.2119579, 5.1972218, -2.0131476, 3.6256113, -7.8375669, 7.2103691
3: -2.1570172, 8.6666470, -1.4237120, 5.7915325, -7.9485493, 10.0903578
4: -5.7941303, 6.0009980, -3.0545263, 4.1621399, -9.9562683, 9.0555229

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1802274, upper bound: 12.0456740
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1802274, upper bound: 12.1630514
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.4619281, 4.9763918, -1.7360146, 3.0926361, -6.5545635, 6.7124062
1: -2.6407366, 6.6547775, -1.3263460, 5.2044163, -7.8451519, 7.9811234
2: -3.9357042, 5.6090250, -2.0131476, 3.6256113, -7.5613108, 7.6221724
3: -2.2358243, 8.6746998, -1.4237120, 5.7915325, -8.0273571, 10.0984116
4: -5.4861760, 6.4149227, -3.0545263, 4.1621399, -9.6483154, 9.4694481

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0200871, upper bound: 12.1403412
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035058, upper bound: 12.1589979
time: 0.54 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.2143096, 1.7795796, -6.7140398, 5.4184217
1: -3.7369466, 6.8882389, -0.2778803, 2.8850155, -6.6219621, 7.1661191
2: -5.4744320, 5.6376672, -0.7044127, 2.2803013, -7.7547331, 6.3420801
3: -2.3689766, 9.8232098, -0.9286744, 2.8235273, -5.1925039, 10.7518835
4: -7.3408518, 6.5052090, -1.3212409, 2.5750978, -9.9159498, 7.8264499

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1787380, upper bound: 12.0349293
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0294756
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.5258994, 3.1309829, -8.0654411, 5.7300110
1: -3.7369466, 6.8882389, -0.5956205, 4.9794006, -8.7163467, 7.4838595
2: -5.4744320, 5.6376672, -1.2669206, 3.7673125, -9.2417421, 6.9045877
3: -2.3689766, 9.8232098, -1.4658637, 5.3376389, -7.7066150, 11.2890692
4: -7.3408518, 6.5052090, -2.3321438, 4.3155594, -11.6564112, 8.8373508

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035058, upper bound: 12.1775616
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.2143096, 1.7795796, -6.7613454, 5.8481121
1: -3.7128515, 7.5261250, -0.2778803, 2.8850155, -6.5978670, 7.8040037
2: -5.4640889, 6.2225499, -0.7044127, 2.2803013, -7.7443905, 6.9269629
3: -2.5014567, 10.0730762, -0.9286744, 2.8235273, -5.3249841, 11.0017509
4: -7.3761001, 7.0715237, -1.3212409, 2.5750978, -9.9511976, 8.3927650

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0294756
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0294756
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.5258994, 3.1309829, -8.1127415, 6.1597018
1: -3.7128515, 7.5261250, -0.5956205, 4.9794006, -8.6922522, 8.1217451
2: -5.4640889, 6.2225499, -1.2669206, 3.7673125, -9.2314014, 7.4894705
3: -2.5014567, 10.0730762, -1.4658637, 5.3376389, -7.8390956, 11.5389404
4: -7.3761001, 7.0715237, -2.3321438, 4.3155594, -11.6916599, 9.4036636

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2035058, upper bound: 12.1775616
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810498, upper bound: 12.1775616
time: 0.48 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -2.6432900, 4.0474329, -4.3366280, 5.3815122
1: -0.3822187, 3.9519520, -2.0620241, 5.7167988, -6.0990152, 6.0139761
2: -0.8797669, 3.3527915, -3.1320014, 4.5007553, -5.3805208, 6.4847927
3: -1.2386876, 3.9645653, -1.8380175, 7.5076027, -8.7462902, 5.8025827
4: -1.6498394, 3.7471557, -4.4430809, 5.1974311, -6.8472695, 8.1902370

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2020829
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2020825
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.0207763, 4.3372831, -4.6264787, 6.7589984
1: -0.3822187, 3.9519520, -2.9535494, 6.0441322, -6.4263496, 6.9055014
2: -0.8797669, 3.3527915, -4.2761312, 4.8936901, -5.7734566, 7.6289225
3: -1.2386876, 3.9645653, -1.9447784, 7.8010159, -9.0397034, 5.9093437
4: -1.6498394, 3.7471557, -5.7360392, 5.6117840, -7.2616234, 9.4831944

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2020825
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2020825
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -4.4856672, 4.6578822, -8.3361387, 9.2080050
1: -2.8371258, 6.2889423, -3.2995057, 6.5600414, -9.3971672, 9.5884476
2: -4.2119579, 5.1972218, -4.7939720, 5.1907496, -9.4027061, 9.9911928
3: -2.1570172, 8.6666470, -2.0794299, 8.5718803, -10.7288952, 10.7460756
4: -5.7941303, 6.0009980, -6.4134078, 5.9567304, -11.7508545, 12.4144039

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1915171, upper bound: 12.0456740
time: 0.78 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1915171, upper bound: 12.2115748
time: 0.59 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.4619281, 4.9763918, -4.4856672, 4.6578822, -8.1198092, 9.4620590
1: -2.6407366, 6.6547775, -3.2995057, 6.5600414, -9.2007771, 9.9542828
2: -3.9357042, 5.6090250, -4.7939720, 5.1907496, -9.1264496, 10.4029961
3: -2.2358243, 8.6746998, -2.0794299, 8.5718803, -10.8077049, 10.7541294
4: -5.4861760, 6.4149227, -6.4134078, 5.9567304, -11.4429035, 12.8283310

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1915171, upper bound: 12.0412260
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1915171, upper bound: 12.2119442
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.8830261, 5.1888895, -5.4780850, 7.6212482
1: -0.3822187, 3.9519520, -3.6997132, 6.8758278, -7.2580462, 7.6516652
2: -0.8797669, 3.3527915, -5.4240475, 5.6200366, -6.4998021, 8.7768393
3: -1.2386876, 3.9645653, -2.3564258, 9.7877512, -11.0264387, 6.3209910
4: -1.6498394, 3.7471557, -7.2774620, 6.4821172, -8.1319561, 11.0246181

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.8599315, 5.6000676, -5.8892627, 7.5981536
1: -0.3822187, 3.9519520, -3.6255574, 7.4982872, -7.8805051, 7.5775094
2: -0.8797669, 3.3527915, -5.3452921, 6.1820874, -7.0618520, 8.6980839
3: -1.2386876, 3.9645653, -2.4739380, 10.0032158, -11.2419033, 6.4385033
4: -1.6498394, 3.7471557, -7.2256942, 7.0176010, -8.6674404, 10.9728498

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.1922919
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0217164, upper bound: 12.1922919
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -4.8830261, 5.1888895, -9.3510342, 10.2775574
1: -3.1508241, 7.2959442, -3.6997132, 6.8758278, -10.0266514, 10.9956570
2: -4.7026148, 5.9927139, -5.4240475, 5.6200366, -10.3226490, 11.4167604
3: -2.4236612, 9.6652107, -2.3564258, 9.7877512, -12.2114124, 12.0216370
4: -6.4547949, 6.8532066, -7.2774620, 6.4821172, -12.9369125, 14.1306686

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159841, upper bound: 12.2108905
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -4.8599315, 5.6000676, -9.7622118, 10.2544632
1: -3.1508241, 7.2959442, -3.6255574, 7.4982872, -10.6491108, 10.9215012
2: -4.7026148, 5.9927139, -5.3452921, 6.1820874, -10.8846960, 11.3380013
3: -2.4236612, 9.6652107, -2.4739380, 10.0032158, -12.4268770, 12.1391487
4: -6.4547949, 6.8532066, -7.2256942, 7.0176010, -13.4723959, 14.0789013

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1915171, upper bound: 12.0275994
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1915171, upper bound: 12.2108540
time: 0.47 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.57 seconds
IS_A1_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0584438, upper bound: 12.1858748
IS_A1_B2_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0577193, upper bound: 12.1327170
IS_A1_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1589783, upper bound: 12.2196471
IS_A1_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1589783, upper bound: 12.2196471
IS_A1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0530349, upper bound: 12.1858746
IS_A1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1505732, upper bound: 12.2196475
IS_A1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873941
IS_A1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
IS_A1_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873941
IS_A1_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1873944
IS_A1_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
IS_A1_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1863544, upper bound: 12.2345207
IS_A1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873944
IS_A1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1824820, upper bound: 12.2345203
IS_A1_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1589869, upper bound: 12.2035059
IS_A1_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1589869, upper bound: 12.2035059
IS_A1_B2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1328502, upper bound: 12.0200871
IS_A1_B2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1589979, upper bound: 12.2035060
IS_A1_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
IS_A1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1507419
IS_A1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2079222
IS_A1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2079222
IS_A1_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1507419
IS_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1509753
IS_A1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2079222
IS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1823244, upper bound: 12.2079222
IS_A2_A1_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
IS_A2_A1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
IS_A2_A1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
IS_A2_A1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
IS_A2_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589783
IS_A2_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
IS_A2_A1_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
IS_A2_A1_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
IS_A2_A1_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
IS_A2_A1_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
IS_A2_A1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
IS_A2_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
IS_A2_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
IS_A2_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1863544
IS_A2_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
IS_A2_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1824820
IS_A2_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
IS_A2_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
IS_A2_A1_B2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0640623, upper bound: 12.0642879
IS_A2_A1_B2_B1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0640623, upper bound: 12.0756598
IS_A2_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2582148
IS_A2_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2588644
IS_A2_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0640623, upper bound: 12.2372875
IS_A2_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0640623, upper bound: 12.2582810
IS_A2_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0316612
IS_A2_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0316612
IS_A2_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1873942, upper bound: 12.0274242
IS_A2_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0274242
IS_A2_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2590806, upper bound: 12.2120318
IS_A2_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2590806, upper bound: 12.2120318
IS_A2_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0321817, upper bound: 12.1923025
IS_A2_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0321817, upper bound: 12.2120318
IS_A2_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1802274, upper bound: 12.0456740
IS_A2_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1802274, upper bound: 12.1630514
IS_A2_A2_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0200871, upper bound: 12.1403412
IS_A2_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2035058, upper bound: 12.1589979
IS_A2_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1787380, upper bound: 12.0349293
IS_A2_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0294756
IS_A2_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2035059, upper bound: 12.1802068
IS_A2_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2035058, upper bound: 12.1775616
IS_A2_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0294756
IS_A2_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1810498, upper bound: 12.0294756
IS_A2_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2035058, upper bound: 12.1775616
IS_A2_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1810498, upper bound: 12.1775616
IS_A2_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2020829
IS_A2_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0284779, upper bound: 12.2020825
IS_A2_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2020825
IS_A2_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0217164, upper bound: 12.2020825
IS_A2_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1915171, upper bound: 12.0456740
IS_A2_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1915171, upper bound: 12.2115748
IS_A2_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1915171, upper bound: 12.0412260
IS_A2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1915171, upper bound: 12.2119442
IS_A2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
IS_A2_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
IS_A2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0217164, upper bound: 12.1922919
IS_A2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0217164, upper bound: 12.1922919
IS_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.0284779, upper bound: 12.1922919
IS_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.2159841, upper bound: 12.2108905
IS_A2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1915171, upper bound: 12.0275994
IS_A2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.57
Output dim: 0, lower bound: -12.1915171, upper bound: 12.2108540

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -2.6754963, 4.0768895, -4.2888570, 4.3396978
1: -0.2774035, 2.7060747, -2.0858450, 5.7528667, -6.0302687, 4.7919197
2: -0.6758080, 2.1488650, -3.1665211, 4.5292182, -5.2050261, 5.3153858
3: -0.9214749, 2.6601696, -1.8468850, 7.5567837, -8.4782581, 4.5070543
4: -1.2530408, 2.4732645, -4.4865170, 5.2266645, -6.4797049, 6.9597816

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0584438, upper bound: 12.1858746
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0530349, upper bound: 12.1858748
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -2.6754963, 4.0768895, -4.3531604, 4.9307814
1: -0.3717489, 3.9332523, -2.0858450, 5.7528667, -6.1246157, 6.0190973
2: -1.0147076, 2.7846820, -3.1665211, 4.5292182, -5.5439258, 5.9512033
3: -1.1808519, 4.1503563, -1.8468850, 7.5567837, -8.7376356, 5.9972410
4: -1.9016094, 3.2297649, -4.4865170, 5.2266645, -7.1282740, 7.7162819

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589783, upper bound: 12.2398624
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589783, upper bound: 12.2196475
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -4.0283499, 4.3438158, -4.6200881, 6.2836347
1: -0.3717489, 3.9332523, -2.9591291, 6.0517540, -6.4235029, 6.8923817
2: -1.0147076, 2.7846820, -4.2841620, 4.9001317, -5.9148393, 7.0688438
3: -1.1808519, 4.1503563, -1.9467452, 7.8118000, -8.9926519, 6.0971012
4: -1.9016094, 3.2297649, -5.7461939, 5.6184468, -7.5200562, 8.9759588

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2398629
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2196471
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1978316, 1.5574791, -4.5359459, 4.7002478, -4.8980794, 6.0934248
1: -0.2591586, 2.4123363, -3.3364797, 6.6092453, -6.8684030, 5.7488160
2: -0.5880604, 2.0356865, -4.8469982, 5.2329617, -5.8210220, 6.8826847
3: -0.8728676, 2.3451891, -2.0921249, 8.6418915, -9.5147591, 4.4373140
4: -1.0882773, 2.3351970, -6.4802928, 6.0003052, -7.0885811, 8.8154898

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0530349, upper bound: 12.1858748
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0530349, upper bound: 12.1858746
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5457935, 2.4314504, -4.5359459, 4.7002478, -5.2460413, 6.9673963
1: -0.5560611, 4.1216106, -3.3364797, 6.6092453, -7.1653047, 7.4580903
2: -1.0841694, 2.9876790, -4.8469982, 5.2329617, -6.3171310, 7.8346772
3: -1.2315452, 4.3375187, -2.0921249, 8.6418915, -9.8734369, 6.4296436
4: -1.9889984, 3.4472027, -6.4802928, 6.0003052, -7.9893022, 9.9274960

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2196475
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589889, upper bound: 12.2196471
time: 0.52 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.6754963, 4.0768895, -4.2911992, 4.4550762
1: -0.2778803, 2.8850155, -2.0858450, 5.7528667, -6.0307469, 4.9708605
2: -0.7044127, 2.2803013, -3.1665211, 4.5292182, -5.2336311, 5.4468222
3: -0.9286744, 2.8235273, -1.8468850, 7.5567837, -8.4854584, 4.6704121
4: -1.3212409, 2.5750978, -4.4865170, 5.2266645, -6.5479054, 7.0616150

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873941
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3161098, 2.5077143, -2.6754963, 4.0768895, -4.3929992, 5.1832104
1: -0.4254482, 4.4234543, -2.0858450, 5.7528667, -6.1783152, 6.5092993
2: -1.1336632, 3.1795123, -3.1665211, 4.5292182, -5.6628814, 6.3460331
3: -1.3724446, 4.5201435, -1.8468850, 7.5567837, -8.9292278, 6.3670282
4: -2.1406889, 3.6535268, -4.4865170, 5.2266645, -7.3673534, 8.1400433

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873941
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873941
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.0283499, 4.3438158, -4.5581255, 5.8079295
1: -0.2778803, 2.8850155, -2.9591291, 6.0517540, -6.3296342, 5.8441448
2: -0.7044127, 2.2803013, -4.2841620, 4.9001317, -5.6045446, 6.5644636
3: -0.9286744, 2.8235273, -1.9467452, 7.8118000, -8.7404747, 4.7702723
4: -1.3212409, 2.5750978, -5.7461939, 5.6184468, -6.9396877, 8.3212919

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873941
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1328502
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3161098, 2.5077143, -4.0283499, 4.3438158, -4.6599255, 6.5360641
1: -0.4254482, 4.4234543, -2.9591291, 6.0517540, -6.4772024, 7.3825836
2: -1.1336632, 3.1795123, -4.2841620, 4.9001317, -6.0337949, 7.4636745
3: -1.3724446, 4.5201435, -1.9467452, 7.8118000, -9.1842442, 6.4668884
4: -2.1406889, 3.6535268, -5.7461939, 5.6184468, -7.7591357, 9.3997211

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1873944
time: 0.50 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1873941
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.6754963, 4.0768895, -4.2911992, 4.4550762
1: -0.2778803, 2.8850155, -2.0858450, 5.7528667, -6.0307469, 4.9708605
2: -0.7044127, 2.2803013, -3.1665211, 4.5292182, -5.2336311, 5.4468222
3: -0.9286744, 2.8235273, -1.8468850, 7.5567837, -8.4854584, 4.6704121
4: -1.3212409, 2.5750978, -4.4865170, 5.2266645, -6.5479054, 7.0616150

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1873944
time: 0.46 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873944
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -2.6754963, 4.0768895, -4.6027875, 5.8064795
1: -0.5956205, 4.9794006, -2.0858450, 5.7528667, -6.3484869, 7.0652428
2: -1.2669206, 3.7673125, -3.1665211, 4.5292182, -5.7961388, 6.9338322
3: -1.4658637, 5.3376389, -1.8468850, 7.5567837, -9.0226450, 7.1845231
4: -2.3321438, 4.3155594, -4.4865170, 5.2266645, -7.5588083, 8.8020763

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863544, upper bound: 12.2345203
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824179, upper bound: 12.2345203
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.0283499, 4.3438158, -4.5581255, 5.8079295
1: -0.2778803, 2.8850155, -2.9591291, 6.0517540, -6.3296342, 5.8441448
2: -0.7044127, 2.2803013, -4.2841620, 4.9001317, -5.6045446, 6.5644636
3: -0.9286744, 2.8235273, -1.9467452, 7.8118000, -8.7404747, 4.7702723
4: -1.3212409, 2.5750978, -5.7461939, 5.6184468, -6.9396877, 8.3212919

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1873944
time: 0.49 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1328502
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.0283499, 4.3438158, -4.8697147, 7.1593285
1: -0.5956205, 4.9794006, -2.9591291, 6.0517540, -6.6473746, 7.9385290
2: -1.2669206, 3.7673125, -4.2841620, 4.9001317, -6.1670518, 8.0514736
3: -1.4658637, 5.3376389, -1.9467452, 7.8118000, -9.2776604, 7.2843838
4: -2.3321438, 4.3155594, -5.7461939, 5.6184468, -7.9505906, 10.0617514

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824179, upper bound: 12.2345203
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1824820, upper bound: 12.2345203
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3069287, 2.6366191, -3.6782563, 4.7223377, -5.0292664, 6.3148746
1: -0.4171031, 4.5447903, -2.8371258, 6.2889423, -6.7060456, 7.3819132
2: -1.1923580, 3.1466510, -4.2119579, 5.1972218, -6.3895798, 7.3586068
3: -1.2790046, 4.9741669, -2.1570172, 8.6666470, -9.9456501, 7.1311812
4: -2.2190628, 3.6107554, -5.7941303, 6.0009980, -8.2200603, 9.4048843

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1385515, upper bound: 12.1930816
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1358265, upper bound: 12.1288394
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.2866788, 2.8084860, -3.6782563, 4.7223377, -6.0090165, 6.4867420
1: -1.0219777, 4.7035484, -2.8371258, 6.2889423, -7.3109198, 7.5406742
2: -1.5617459, 3.3616722, -4.2119579, 5.1972218, -6.7589674, 7.5736299
3: -1.3344283, 5.0709953, -2.1570172, 8.6666470, -10.0010738, 7.2280107
4: -2.5152102, 3.8443499, -5.7941303, 6.0009980, -8.5162086, 9.6384792

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2035059
time: 0.51 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589869, upper bound: 12.2035059
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -1.7360146, 3.0926361, -3.4619281, 4.9763918, -6.7124062, 6.5545640
1: -1.3263460, 5.2044163, -2.6407366, 6.6547775, -7.9811211, 7.8451529
2: -2.0131476, 3.6256113, -3.9357042, 5.6090250, -7.6221704, 7.5613117
3: -1.4237120, 5.7915325, -2.2358243, 8.6746998, -10.0984116, 8.0273571
4: -3.0545263, 4.1621399, -5.4861760, 6.4149227, -9.4694490, 9.6483135

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589869, upper bound: 12.2035060
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1589979, upper bound: 12.2035059
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9344602, 5.2041121, -5.4184213, 6.7140398
1: -0.2778803, 2.8850155, -3.7369466, 6.8882389, -7.1661191, 6.6219621
2: -0.7044127, 2.2803013, -5.4744320, 5.6376672, -6.3420801, 7.7547331
3: -0.9286744, 2.8235273, -2.3689766, 9.8232098, -10.7518845, 5.1925039
4: -1.3212409, 2.5750978, -7.3408518, 6.5052090, -7.8264494, 9.9159498

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807453
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9042511, 5.5796232, -5.7939329, 6.6838307
1: -0.2778803, 2.8850155, -3.6561935, 7.4705601, -7.7484403, 6.5412092
2: -0.7044127, 2.2803013, -5.3807411, 6.1650767, -6.8694897, 7.6610422
3: -0.9286744, 2.8235273, -2.4826717, 9.9851646, -10.9138393, 5.3061991
4: -1.3212409, 2.5750978, -7.2726564, 7.0153542, -8.3365946, 9.8477545

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1507419
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0292694, upper bound: 12.1807453
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9344602, 5.2041121, -5.7300105, 8.0654411
1: -0.5956205, 4.9794006, -3.7369466, 6.8882389, -7.4838595, 8.7163448
2: -1.2669206, 3.7673125, -5.4744320, 5.6376672, -6.9045873, 9.2417421
3: -1.4658637, 5.3376389, -2.3689766, 9.8232098, -11.2890701, 7.7066154
4: -2.3321438, 4.3155594, -7.3408518, 6.5052090, -8.8373518, 11.6564112

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1630514, upper bound: 12.2079222
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9042511, 5.5796232, -6.1055222, 8.0352306
1: -0.5956205, 4.9794006, -3.6561935, 7.4705601, -8.0661793, 8.6355925
2: -1.2669206, 3.7673125, -5.3807411, 6.1650767, -7.4319968, 9.1480541
3: -1.4658637, 5.3376389, -2.4826717, 9.9851646, -11.4510288, 7.8203106
4: -2.3321438, 4.3155594, -7.2726564, 7.0153542, -9.3474932, 11.5882158

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1807454
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1822948, upper bound: 12.2079222
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9344602, 5.2041121, -5.4184213, 6.7140398
1: -0.2778803, 2.8850155, -3.7369466, 6.8882389, -7.1661191, 6.6219621
2: -0.7044127, 2.2803013, -5.4744320, 5.6376672, -6.3420801, 7.7547331
3: -0.9286744, 2.8235273, -2.3689766, 9.8232098, -10.7518845, 5.1925039
4: -1.3212409, 2.5750978, -7.3408518, 6.5052090, -7.8264494, 9.9159498

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.9817657, 5.6338024, -5.8481121, 6.7613454
1: -0.2778803, 2.8850155, -3.7128515, 7.5261250, -7.8040028, 6.5978670
2: -0.7044127, 2.2803013, -5.4640889, 6.2225499, -6.9269624, 7.7443905
3: -0.9286744, 2.8235273, -2.5014567, 10.0730762, -11.0017509, 5.3249841
4: -1.3212409, 2.5750978, -7.3761001, 7.0715237, -8.3927650, 9.9511976

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1509753
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0261820, upper bound: 12.1509753
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9344602, 5.2041121, -5.7300105, 8.0654411
1: -0.5956205, 4.9794006, -3.7369466, 6.8882389, -7.4838595, 8.7163448
2: -1.2669206, 3.7673125, -5.4744320, 5.6376672, -6.9045873, 9.2417421
3: -1.4658637, 5.3376389, -2.3689766, 9.8232098, -11.2890701, 7.7066154
4: -2.3321438, 4.3155594, -7.3408518, 6.5052090, -8.8373518, 11.6564112

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0348062, upper bound: 12.1807454
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1863398, upper bound: 12.2079222
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.9817657, 5.6338024, -6.1597018, 8.1127453
1: -0.5956205, 4.9794006, -3.7128515, 7.5261250, -8.1217442, 8.6922503
2: -1.2669206, 3.7673125, -5.4640889, 6.2225499, -7.4894691, 9.2314014
3: -1.4658637, 5.3376389, -2.5014567, 10.0730762, -11.5389366, 7.8390956
4: -2.3321438, 4.3155594, -7.3761001, 7.0715237, -9.4036646, 11.6916599

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642476, upper bound: 12.0246627
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1642476, upper bound: 12.2077610
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2119676, 1.6642017, -4.3396978, 4.2888570
1: -2.0858450, 5.7528667, -0.2774035, 2.7060747, -4.7919197, 6.0302701
2: -3.1665211, 4.5292182, -0.6758080, 2.1488650, -5.3153858, 5.2050261
3: -1.8468850, 7.5567837, -0.9214749, 2.6601696, -4.5070543, 8.4782581
4: -4.4865170, 5.2266645, -1.2530408, 2.4732645, -6.9597816, 6.4797049

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2864694, 2.2201376, -4.8956337, 4.3633590
1: -2.0858450, 5.7528667, -0.3822893, 4.2234097, -6.3092546, 6.1351562
2: -3.1665211, 4.5292182, -1.1075993, 2.8155358, -5.9820566, 5.6368175
3: -1.8468850, 7.5567837, -1.2409499, 4.3273754, -6.1742601, 8.7977314
4: -4.4865170, 5.2266645, -2.0756645, 3.2417476, -7.7282648, 7.3023291

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0584438
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2119676, 1.6642017, -5.6925516, 4.5557833
1: -2.9591291, 6.0517540, -0.2774035, 2.7060747, -5.6652040, 6.3291550
2: -4.2841620, 4.9001317, -0.6758080, 2.1488650, -6.4330273, 5.5759397
3: -1.9467452, 7.8118000, -0.9214749, 2.6601696, -4.6069145, 8.7332735
4: -5.7461939, 5.6184468, -1.2530408, 2.4732645, -8.2194586, 6.8714876

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2864694, 2.2201376, -6.2484875, 4.6302853
1: -2.9591291, 6.0517540, -0.3822893, 4.2234097, -7.1825390, 6.4340434
2: -4.2841620, 4.9001317, -1.1075993, 2.8155358, -7.0996981, 6.0077310
3: -1.9467452, 7.8118000, -1.2409499, 4.3273754, -6.2741203, 9.0527487
4: -5.7461939, 5.6184468, -2.0756645, 3.2417476, -8.9879417, 7.6941113

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1858746, upper bound: 12.0530349
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2762722, 2.2552848, -4.9307814, 4.3531604
1: -2.0858450, 5.7528667, -0.3717489, 3.9332523, -6.0190973, 6.1246157
2: -3.1665211, 4.5292182, -1.0147076, 2.7846820, -5.9512033, 5.5439258
3: -1.8468850, 7.5567837, -1.1808519, 4.1503563, -5.9972410, 8.7376356
4: -4.4865170, 5.2266645, -1.9016094, 3.2297649, -7.7162819, 7.1282740

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2398624, upper bound: 12.1589783
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589783
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2762722, 2.2552848, -6.2836347, 4.6200881
1: -2.9591291, 6.0517540, -0.3717489, 3.9332523, -6.8923817, 6.4235029
2: -4.2841620, 4.9001317, -1.0147076, 2.7846820, -7.0688438, 5.9148393
3: -1.9467452, 7.8118000, -1.1808519, 4.1503563, -6.0971012, 8.9926519
4: -5.7461939, 5.6184468, -1.9016094, 3.2297649, -8.9759588, 7.5200562

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2398624, upper bound: 12.1589889
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.1978316, 1.5574791, -6.0934248, 4.8980794
1: -3.3364797, 6.6092453, -0.2591586, 2.4123363, -5.7488160, 6.8684025
2: -4.8469982, 5.2329617, -0.5880604, 2.0356865, -6.8826847, 5.8210220
3: -2.0921249, 8.6418915, -0.8728676, 2.3451891, -4.4373140, 9.5147591
4: -6.4802928, 6.0003052, -1.0882773, 2.3351970, -8.8154898, 7.0885825

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1654904, upper bound: 12.0372494
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1716703, upper bound: 12.0198143
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1220548, upper bound: 12.0198143
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.5457935, 2.4314504, -6.9673963, 5.2460413
1: -3.3364797, 6.6092453, -0.5560611, 4.1216106, -7.4580903, 7.1653066
2: -4.8469982, 5.2329617, -1.0841694, 2.9876790, -7.8346772, 6.3171310
3: -2.0921249, 8.6418915, -1.2315452, 4.3375187, -6.4296436, 9.8734369
4: -6.4802928, 6.0003052, -1.9889984, 3.4472027, -9.9274960, 7.9893031

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196471, upper bound: 12.1589889
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2143096, 1.7795796, -4.4550762, 4.2911992
1: -2.0858450, 5.7528667, -0.2778803, 2.8850155, -4.9708605, 6.0307469
2: -3.1665211, 4.5292182, -0.7044127, 2.2803013, -5.4468222, 5.2336307
3: -1.8468850, 7.5567837, -0.9286744, 2.8235273, -4.6704121, 8.4854574
4: -4.4865170, 5.2266645, -1.3212409, 2.5750978, -7.0616150, 6.5479054

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.3161098, 2.5077143, -5.1832104, 4.3929992
1: -2.0858450, 5.7528667, -0.4254482, 4.4234543, -6.5092993, 6.1783152
2: -3.1665211, 4.5292182, -1.1336632, 3.1795123, -6.3460331, 5.6628814
3: -1.8468850, 7.5567837, -1.3724446, 4.5201435, -6.3670282, 8.9292278
4: -4.4865170, 5.2266645, -2.1406889, 3.6535268, -8.1400433, 7.3673534

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2143096, 1.7795796, -5.8079295, 4.5581255
1: -2.9591291, 6.0517540, -0.2778803, 2.8850155, -5.8441448, 6.3296337
2: -4.2841620, 4.9001317, -0.7044127, 2.2803013, -6.5644636, 5.6045446
3: -1.9467452, 7.8118000, -0.9286744, 2.8235273, -4.7702723, 8.7404747
4: -5.7461939, 5.6184468, -1.3212409, 2.5750978, -8.3212919, 6.9396877

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1328502, upper bound: 12.0292694
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.3161098, 2.5077143, -6.5360641, 4.6599255
1: -2.9591291, 6.0517540, -0.4254482, 4.4234543, -7.3825836, 6.4772024
2: -4.2841620, 4.9001317, -1.1336632, 3.1795123, -7.4636745, 6.0337949
3: -1.9467452, 7.8118000, -1.3724446, 4.5201435, -6.4668884, 9.1842422
4: -5.7461939, 5.6184468, -2.1406889, 3.6535268, -9.3997211, 7.7591357

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1328502, upper bound: 12.0292694
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2143096, 1.7795796, -4.4550762, 4.2911992
1: -2.0858450, 5.7528667, -0.2778803, 2.8850155, -4.9708605, 6.0307469
2: -3.1665211, 4.5292182, -0.7044127, 2.2803013, -5.4468222, 5.2336307
3: -1.8468850, 7.5567837, -0.9286744, 2.8235273, -4.6704121, 8.4854574
4: -4.4865170, 5.2266645, -1.3212409, 2.5750978, -7.0616150, 6.5479054

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0348062
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.5258994, 3.1309829, -5.8064790, 4.6027880
1: -2.0858450, 5.7528667, -0.5956205, 4.9794006, -7.0652442, 6.3484874
2: -3.1665211, 4.5292182, -1.2669206, 3.7673125, -6.9338326, 5.7961388
3: -1.8468850, 7.5567837, -1.4658637, 5.3376389, -7.1845236, 9.0226450
4: -4.4865170, 5.2266645, -2.3321438, 4.3155594, -8.8020763, 7.5588083

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1863544
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1824179
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2143096, 1.7795796, -5.8079295, 4.5581255
1: -2.9591291, 6.0517540, -0.2778803, 2.8850155, -5.8441448, 6.3296337
2: -4.2841620, 4.9001317, -0.7044127, 2.2803013, -6.5644636, 5.6045446
3: -1.9467452, 7.8118000, -0.9286744, 2.8235273, -4.7702723, 8.7404747
4: -5.7461939, 5.6184468, -1.3212409, 2.5750978, -8.3212919, 6.9396877

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873941, upper bound: 12.0292694
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1328502, upper bound: 12.0292694
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.5258994, 3.1309829, -7.1593299, 4.8697152
1: -2.9591291, 6.0517540, -0.5956205, 4.9794006, -7.9385295, 6.6473746
2: -4.2841620, 4.9001317, -1.2669206, 3.7673125, -8.0514736, 6.1670523
3: -1.9467452, 7.8118000, -1.4658637, 5.3376389, -7.2843833, 9.2776623
4: -5.7461939, 5.6184468, -2.3321438, 4.3155594, -10.0617533, 7.9505906

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1824179
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345203, upper bound: 12.1824820
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2742517, 2.3193870, -4.9948835, 4.3511405
1: -2.0858450, 5.7528667, -0.3634766, 3.2789097, -5.3647547, 6.1163411
2: -3.1665211, 4.5292182, -0.7444794, 2.9443834, -6.1109047, 5.2736979
3: -1.8468850, 7.5567837, -1.1945589, 3.2639122, -5.1107969, 8.7513428
4: -4.4865170, 5.2266645, -1.3846707, 3.3782203, -7.8647375, 6.6113353

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2373023, upper bound: 12.0756598
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.3850740, 3.6355610, -6.3110571, 4.4619632
1: -2.0858450, 5.7528667, -0.5207902, 5.4637089, -7.5495539, 6.2736568
2: -3.1665211, 4.5292182, -1.4221835, 4.2520247, -7.4185457, 5.9514017
3: -1.8468850, 7.5567837, -1.6142063, 6.3127098, -8.1595945, 9.1709900
4: -4.4865170, 5.2266645, -2.5975704, 4.8423157, -9.3288326, 7.8242350

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2374988, upper bound: 12.0811221
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2373023, upper bound: 12.0756598
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -1.9736005, 3.7326164, -6.4081125, 6.0504880
1: -2.0858450, 5.7528667, -1.5754695, 5.3457413, -7.4315858, 7.3283358
2: -3.1665211, 4.5292182, -2.4574776, 4.2253923, -7.3919125, 6.9866953
3: -1.8468850, 7.5567837, -1.7051358, 6.8608799, -8.7077646, 9.2619190
4: -4.4865170, 5.2266645, -3.6110172, 4.8835225, -9.3700390, 8.8376818

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2636720, upper bound: 12.2582148
time: 0.51 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2582145
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -1.9736005, 3.7326164, -7.7609663, 6.3174162
1: -2.9591291, 6.0517540, -1.5754695, 5.3457413, -8.3048706, 7.6272235
2: -4.2841620, 4.9001317, -2.4574776, 4.2253923, -8.5095539, 7.3576093
3: -1.9467452, 7.8118000, -1.7051358, 6.8608799, -8.8076248, 9.5169334
4: -5.7461939, 5.6184468, -3.6110172, 4.8835225, -10.6297169, 9.2294636

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2636720, upper bound: 12.2588642
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2595147, upper bound: 12.2588644
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -3.2853987, 3.9233079, -4.1975589, 5.6047859
1: -0.3634766, 3.2789097, -2.4223030, 5.5470581, -5.9105349, 5.7012129
2: -0.7444794, 2.9443834, -3.5144036, 4.5003858, -5.2448649, 6.4587870
3: -1.1945589, 3.2639122, -1.7845080, 6.9990902, -8.1936493, 5.0484200
4: -1.3846707, 3.3782203, -4.7958260, 5.2002707, -6.5849414, 8.1740465

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0756666, upper bound: 12.2372875
time: 0.49 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0756666, upper bound: 12.2372875
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -3.8283892, 4.3033652, -3.2853987, 3.9233079, -7.7516971, 7.5887623
1: -2.8217945, 6.1582298, -2.4223030, 5.5470581, -8.3688526, 8.5805311
2: -4.1151338, 4.8628931, -3.5144036, 4.5003858, -8.6155195, 8.3772964
3: -1.9461601, 7.8930817, -1.7845080, 6.9990902, -8.9452505, 9.6775894
4: -5.5757437, 5.5972967, -4.7958260, 5.2002707, -10.7760115, 10.3931217

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0756666, upper bound: 12.2588642
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0756666, upper bound: 12.2588642
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2891956, 2.7382221, -5.4137182, 4.3660851
1: -2.0858450, 5.7528667, -0.3822187, 3.9519520, -6.0377970, 6.1350851
2: -3.1665211, 4.5292182, -0.8797669, 3.3527915, -6.5193129, 5.4089851
3: -1.8468850, 7.5567837, -1.2386876, 3.9645653, -5.8114500, 8.7954712
4: -4.4865170, 5.2266645, -1.6498394, 3.7471557, -8.2336731, 6.8765025

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0316612
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0273909
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.4763252, 4.5747962, -7.2502928, 4.5532126
1: -2.0858450, 5.7528667, -0.6584027, 6.4896393, -8.5754843, 6.4112687
2: -3.1665211, 4.5292182, -1.7513924, 5.2737622, -8.4402828, 6.2806106
3: -1.8468850, 7.5567837, -1.9422446, 7.8955307, -9.7424154, 9.4990273
4: -4.4865170, 5.2266645, -3.1807480, 5.9438157, -10.4303322, 8.4074125

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1873942, upper bound: 12.0316612
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2360305, upper bound: 12.0273909
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2891956, 2.7382221, -6.7665720, 4.6330113
1: -2.9591291, 6.0517540, -0.3822187, 3.9519520, -6.9110813, 6.4339728
2: -4.2841620, 4.9001317, -0.8797669, 3.3527915, -7.6369534, 5.7798986
3: -1.9467452, 7.8118000, -1.2386876, 3.9645653, -5.9113102, 9.0504875
4: -5.7461939, 5.6184468, -1.6498394, 3.7471557, -9.4933491, 7.2682862

Time for backsubstitution: 1.96 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=13.458332061767578
rel_dist={0: [-12.27133016030539, 12.27133016030539]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2011499, upper bound: 12.2346717
time: 0.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2667180, upper bound: 12.2667182
time: 0.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -12.2011499, upper bound: 12.2346717
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.86
Output dim: 0, lower bound: -12.2667180, upper bound: 12.2667182

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -5.9727926, 5.3409023, -8.7405119, 9.7664251
1: -2.4989076, 6.0682201, -4.3751678, 7.5285034, -10.0274105, 10.4433851
2: -3.6286805, 4.3000150, -6.3356962, 5.8320880, -9.4607687, 10.6357117
3: -1.6621141, 7.2208080, -2.3450775, 10.1313782, -11.7934923, 9.5658846
4: -4.9867439, 4.9192414, -8.3474159, 6.6734519, -11.6601954, 13.2666540

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2006907, upper bound: 12.2006907
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2006907, upper bound: 12.2346717
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.2199450, 5.5384865, -6.8069258, 5.8532448, -12.0731888, 12.3454123
1: -4.5570164, 7.5879345, -4.9823050, 7.9284000, -12.4854164, 12.5702400
2: -6.5976086, 6.0316262, -7.2073011, 6.3334384, -12.9310455, 13.2389269
3: -2.4391820, 10.4356041, -2.5603251, 11.1020756, -13.5412579, 12.9959297
4: -8.6541815, 6.9089074, -9.4078369, 7.2520056, -15.9061861, 16.3167439

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1636282, upper bound: 12.2192263
time: 0.37 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192262, upper bound: 12.2192263
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.07 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -12.2006907, upper bound: 12.2006907
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -12.2006907, upper bound: 12.2346717
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -12.1636282, upper bound: 12.2192263
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -12.2192262, upper bound: 12.2192263

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -3.3996100, 3.7936325, -7.1932425, 7.1932421
1: -2.4989076, 6.0682201, -2.4989076, 6.0682201, -8.5671263, 8.5671253
2: -3.6286805, 4.3000150, -3.6286805, 4.3000150, -7.9286952, 7.9286957
3: -1.6621141, 7.2208080, -1.6621141, 7.2208080, -8.8829193, 8.8829212
4: -4.9867439, 4.9192414, -4.9867439, 4.9192414, -9.9059830, 9.9059849

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1628301, upper bound: 12.1808159
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875843, upper bound: 12.1875846
time: 0.47 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -3.3996100, 3.7936325, -6.1691380, 5.5214491, -8.9210587, 9.9627705
1: -2.4989076, 6.0682201, -4.5203118, 7.5722961, -10.0712032, 10.5885305
2: -3.6286805, 4.3000150, -6.5458431, 6.0171990, -9.6458797, 10.8458576
3: -1.6621141, 7.2208080, -2.4296532, 10.3868408, -12.0489550, 9.6504612
4: -4.9867439, 4.9192414, -8.5917435, 6.8908434, -11.8775873, 13.5109835

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1808159, upper bound: 12.2196978
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875846, upper bound: 12.2144311
time: 0.35 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -5.5601344, 5.2747917, -9.8107376, 10.2603817
1: -3.3364797, 6.6092453, -4.0815969, 7.2566833, -10.5931606, 10.6908379
2: -4.8469982, 5.2329617, -5.9280052, 5.7745790, -10.6215773, 11.1609669
3: -2.0921249, 8.6418915, -2.3175328, 9.8255882, -11.9177132, 10.9594231
4: -6.4802928, 6.0003052, -7.8366842, 6.6154847, -13.0957775, 13.8369894

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196976, upper bound: 12.1884568
time: 0.37 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2196978, upper bound: 12.2189792
time: 0.40 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -4.8974800, 5.1680098, -10.6972990, 10.8628559
1: -4.1178484, 8.0096436, -3.6206856, 7.2120571, -11.3299055, 11.6303291
2: -6.0670018, 6.5351744, -5.3062572, 5.6994610, -11.7664585, 11.8414316
3: -2.6416011, 10.8504009, -2.2697003, 9.5387630, -12.1803646, 13.1200991
4: -8.1326199, 7.4373612, -7.1292849, 6.5024023, -14.6350222, 14.5666466

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2144309, upper bound: 12.1888249
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2144311, upper bound: 12.2157844
time: 0.44 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.18 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.1628301, upper bound: 12.1808159
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.1875843, upper bound: 12.1875846
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.1808159, upper bound: 12.2196978
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.1875846, upper bound: 12.2144311
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.2196976, upper bound: 12.1884568
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.2196978, upper bound: 12.2189792
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.2144309, upper bound: 12.1888249
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.18
Output dim: 0, lower bound: -12.2144311, upper bound: 12.2157844

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -2.3467333, 3.3773012, -5.1493950, 5.4746466
1: -1.3520081, 5.2492390, -1.7559460, 5.5761147, -6.9281225, 7.0051837
2: -2.0487065, 3.6625366, -2.5920372, 3.9006433, -5.9493484, 6.2545729
3: -1.4298269, 5.8484859, -1.4990642, 6.3629837, -7.7928104, 7.3475504
4: -3.0981421, 4.2000217, -3.7442846, 4.4675279, -7.5656695, 7.9443045

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1618156, upper bound: 12.1618156
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1618156, upper bound: 12.1808159
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.4815927, 3.0992455, -4.4039607, 5.0493402
1: -1.0992223, 5.5531120, -1.1635927, 5.2631302, -6.3623495, 6.7167020
2: -1.7584276, 4.1799893, -1.8075175, 3.6310673, -5.3894949, 5.9875059
3: -1.5837383, 6.1448326, -1.3910677, 5.7828999, -7.3666382, 7.5359001
4: -2.9098625, 4.7395120, -2.8426781, 4.1381412, -7.0480037, 7.5821886

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1808159, upper bound: 12.1628301
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1808159, upper bound: 12.1875846
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -2.3467333, 3.3773012, -4.4929471, 4.6846557, -7.0313892, 7.8702483
1: -1.7559460, 5.5761147, -3.3053031, 6.5950356, -8.3509817, 8.8814144
2: -2.5920372, 3.9006433, -4.8029432, 5.2199960, -7.8120303, 8.7035866
3: -1.4990642, 6.3629837, -2.0835414, 8.6005402, -10.0996037, 8.4465256
4: -3.7442846, 4.4675279, -6.4269009, 5.9843197, -9.7286005, 10.8944273

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1635481, upper bound: 12.2039305
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1618156, upper bound: 12.1628301
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.4815927, 3.0992455, -4.9395409, 5.7222033, -7.2037959, 8.0387850
1: -1.1635927, 5.2631302, -3.6858649, 7.7874050, -8.9509974, 8.9489927
2: -1.8075175, 3.6310673, -5.4481621, 6.3277917, -8.1353092, 9.0792294
3: -1.3910677, 5.7828999, -2.5053167, 10.1801186, -11.5711861, 8.2882156
4: -2.8426781, 4.1381412, -7.3651257, 7.1747870, -10.0174656, 11.5032654

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1636282, upper bound: 12.2047833
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1636282, upper bound: 12.2144310
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -2.3195224, 3.3517509, -7.8876948, 7.0197701
1: -3.3364797, 6.6092453, -1.7358882, 5.5380630, -8.8745422, 8.3451338
2: -4.8469982, 5.2329617, -2.5622509, 3.8750319, -8.7220306, 7.7952108
3: -2.0921249, 8.6418915, -1.4899143, 6.3161163, -8.4082413, 10.1318054
4: -6.4802928, 6.0003052, -3.7062016, 4.4397888, -10.9200821, 9.7065048

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1635481
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1884568
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -5.1131897, 5.0019217, -9.5378666, 9.8134375
1: -3.3364797, 6.6092453, -3.7565775, 6.9642105, -10.3006897, 10.3658199
2: -4.8469982, 5.2329617, -5.4485226, 5.5168920, -10.3638897, 10.6814823
3: -2.0921249, 8.6418915, -2.2166624, 9.2865009, -11.3786259, 10.8585529
4: -6.4802928, 6.0003052, -7.2438612, 6.3237486, -12.8040409, 13.2441664

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2141219, upper bound: 12.2189792
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2141219, upper bound: 12.2189792
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -1.4710138, 3.0851369, -8.6144247, 7.4363899
1: -4.1178484, 8.0096436, -1.1560937, 5.2474880, -9.3653364, 9.1657372
2: -6.0670018, 6.5351744, -1.7971425, 3.6160369, -9.6830330, 8.3323174
3: -2.6416011, 10.8504009, -1.3868579, 5.7629185, -8.4045200, 12.2372580
4: -8.1326199, 7.4373612, -2.8311114, 4.1218090, -12.2544260, 10.2684727

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2047831, upper bound: 12.1636282
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2047832, upper bound: 12.1834686
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -4.3354559, 4.8671846, -10.3964739, 10.3008327
1: -4.1178484, 8.0096436, -3.2136395, 6.8882504, -11.0060978, 11.2232828
2: -6.0670018, 6.5351744, -4.7205787, 5.4182024, -11.4851999, 11.2557526
3: -2.6416011, 10.8504009, -2.1576259, 8.9268188, -11.5684204, 13.0080271
4: -8.1326199, 7.4373612, -6.4084826, 6.1817856, -14.3144054, 13.8458443

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2047832, upper bound: 12.2145008
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2047833, upper bound: 12.2145008
time: 0.46 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1618156, upper bound: 12.1618156
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1618156, upper bound: 12.1808159
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1808159, upper bound: 12.1628301
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1808159, upper bound: 12.1875846
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1635481, upper bound: 12.2039305
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1618156, upper bound: 12.1628301
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1636282, upper bound: 12.2047833
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.1636282, upper bound: 12.2144310
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1635481
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1884568
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2141219, upper bound: 12.2189792
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2141219, upper bound: 12.2189792
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2047831, upper bound: 12.1636282
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2047832, upper bound: 12.1834686
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2047832, upper bound: 12.2145008
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.96
Output dim: 0, lower bound: -12.2047833, upper bound: 12.2145008

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -1.7720940, 3.1279130, -4.9000072, 4.9000072
1: -1.3520081, 5.2492390, -1.3520081, 5.2492390, -6.6012468, 6.6012468
2: -2.0487065, 3.6625366, -2.0487065, 3.6625366, -5.7112427, 5.7112432
3: -1.4298269, 5.8484859, -1.4298269, 5.8484859, -7.2783108, 7.2783113
4: -3.0981421, 4.2000217, -3.0981421, 4.2000217, -7.2981606, 7.2981629

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0941762, upper bound: 12.0231103
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -1.3047158, 3.5677476, -5.3398414, 4.4326286
1: -1.3520081, 5.2492390, -1.0992223, 5.5531120, -6.9051199, 6.3484612
2: -2.0487065, 3.6625366, -1.7584276, 4.1799893, -6.2286935, 5.4209623
3: -1.4298269, 5.8484859, -1.5837383, 6.1448326, -7.5746579, 7.4322228
4: -3.0981421, 4.2000217, -2.9098625, 4.7395120, -7.8376508, 7.1098833

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0941762, upper bound: 12.0849556
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.7353463, 3.0922532, -4.3969688, 5.3030939
1: -1.0992223, 5.5531120, -1.3258518, 5.2033310, -6.3025508, 6.8789616
2: -1.7584276, 4.1799893, -2.0123925, 3.6252692, -5.3836942, 6.1923819
3: -1.5837383, 6.1448326, -1.4236145, 5.7902241, -7.3739624, 7.5684438
4: -2.9098625, 4.7395120, -3.0535307, 4.1617336, -7.0715952, 7.7930412

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0849549, upper bound: 12.1257561
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0616992, upper bound: 12.0215723
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -1.2624179, 3.4834623, -4.7881775, 4.8301644
1: -1.0992223, 5.5531120, -1.0676382, 5.4686108, -6.5678325, 6.6207500
2: -1.7584276, 4.1799893, -1.7101583, 4.0883298, -5.8467565, 5.8901477
3: -1.5837383, 6.1448326, -1.5565034, 6.0524197, -7.6361580, 7.7013359
4: -2.9098625, 4.7395120, -2.8450947, 4.6510177, -7.5608802, 7.5846066

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0849556, upper bound: 12.1535445
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0616997, upper bound: 12.1133036
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -1.7720940, 3.1279130, -4.4929471, 4.6846557, -6.4567494, 7.6208601
1: -1.3520081, 5.2492390, -3.3053031, 6.5950356, -7.9470434, 8.5545425
2: -2.0487065, 3.6625366, -4.8029432, 5.2199960, -7.2687011, 8.4654799
3: -1.4298269, 5.8484859, -2.0835414, 8.6005402, -10.0303650, 7.9320269
4: -3.0981421, 4.2000217, -6.4269009, 5.9843197, -9.0824585, 10.6269217

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -11.9496305, upper bound: 12.0835018
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 1.88 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0328487, upper bound: 12.1540377
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1635481, upper bound: 12.2141215
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -1.3047158, 3.5677476, -4.4929471, 4.6846557, -5.9893708, 8.0606947
1: -1.0992223, 5.5531120, -3.3053031, 6.5950356, -7.6942577, 8.8584137
2: -1.7584276, 4.1799893, -4.8029432, 5.2199960, -6.9784207, 8.9829330
3: -1.5837383, 6.1448326, -2.0835414, 8.6005402, -10.1842766, 8.2283745
4: -2.9098625, 4.7395120, -6.4269009, 5.9843197, -8.8941822, 11.1664114

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -11.9496305, upper bound: 12.0835018
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 1.79 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0328487, upper bound: 12.1554049
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1635481, upper bound: 12.2141219
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.7353463, 3.0922532, -4.9395409, 5.7222033, -7.4575496, 8.0317926
1: -1.3258518, 5.2033310, -3.6858649, 7.7874050, -9.1132565, 8.8891954
2: -2.0123925, 3.6252692, -5.4481621, 6.3277917, -8.3401842, 9.0734310
3: -1.4236145, 5.7902241, -2.5053167, 10.1801186, -11.6037321, 8.2955399
4: -3.0535307, 4.1617336, -7.3651257, 7.1747870, -10.2283173, 11.5268555

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 1.61 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1340902, upper bound: 12.0309530
time: 0.37 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1340901, upper bound: 12.2047832
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.2624179, 3.4834623, -4.9395409, 5.7222033, -6.9846210, 8.4230013
1: -1.0676382, 5.4686108, -3.6858649, 7.7874050, -8.8550434, 9.1544752
2: -1.7101583, 4.0883298, -5.4481621, 6.3277917, -8.0379505, 9.5364914
3: -1.5565034, 6.0524197, -2.5053167, 10.1801186, -11.7366219, 8.5577364
4: -2.8450947, 4.6510177, -7.3651257, 7.1747870, -10.0198822, 12.0161438

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 1.68 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1601383, upper bound: 12.2009624
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1502774, upper bound: 12.2009623
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -1.7581339, 3.1115222, -7.6474662, 6.4583817
1: -3.3364797, 6.6092453, -1.3418462, 5.2277756, -8.5642519, 7.9510889
2: -4.8469982, 5.2329617, -2.0340075, 3.6463227, -8.4933195, 7.2669692
3: -2.0921249, 8.6418915, -1.4249523, 5.8213387, -7.9134636, 10.0668411
4: -6.4802928, 6.0003052, -3.0788002, 4.1824570, -10.6627502, 9.0791054

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0835018, upper bound: 11.9496305
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 1.95 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0328487
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1635481
time: 0.52 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -1.2065203, 3.4600496, -7.9959955, 5.9067678
1: -3.3364797, 6.6092453, -1.0286949, 5.4302425, -8.7667217, 7.6379395
2: -4.8469982, 5.2329617, -1.6656442, 4.0623875, -8.9093857, 6.8986058
3: -2.0921249, 8.6418915, -1.5455453, 5.9919910, -8.0841160, 10.1874371
4: -6.4802928, 6.0003052, -2.8117471, 4.6147075, -11.0950003, 8.8120499

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0835018, upper bound: 12.0521843
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 1.86 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0328487
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1884568
time: 0.51 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -4.5359459, 4.7002478, -9.2361937, 9.2361937
1: -3.3364797, 6.6092453, -3.3364797, 6.6092453, -9.9457207, 9.9457197
2: -4.8469982, 5.2329617, -4.8469982, 5.2329617, -10.0799589, 10.0799580
3: -2.0921249, 8.6418915, -2.0921249, 8.6418915, -10.7340164, 10.7340164
4: -6.4802928, 6.0003052, -6.4802928, 6.0003052, -12.4805984, 12.4805984

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 1.55 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0474759, upper bound: 12.1899355
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0474759, upper bound: 12.2189791
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -5.3601141, 5.9159899, -10.4519348, 10.0603619
1: -3.3364797, 6.6092453, -3.9965582, 7.9697895, -11.3062687, 10.6058025
2: -4.8469982, 5.2329617, -5.9000158, 6.4758339, -11.3228321, 11.1329775
3: -2.0921249, 8.6418915, -2.6022196, 10.7416420, -12.8337669, 11.2441101
4: -6.4802928, 6.0003052, -7.9224205, 7.3602033, -13.8404961, 13.9227257

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38

Time for candidate selection: 1.57 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0318079
time: 0.41 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2527111, upper bound: 12.2189791
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -1.7240775, 3.0773549, -8.6066437, 7.6894541
1: -4.1178484, 8.0096436, -1.3177787, 5.1861057, -9.3039532, 9.3274202
2: -6.0670018, 6.5351744, -2.0007284, 3.6104007, -9.6774015, 8.5359030
3: -2.6416011, 10.8504009, -1.4191301, 5.7682171, -8.4098186, 12.2695265
4: -8.1326199, 7.4373612, -3.0381684, 4.1457658, -12.2783852, 10.4755297

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 1.65 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0309530, upper bound: 12.1340901
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2047831, upper bound: 12.1636282
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -1.1708714, 3.3879943, -8.9172812, 7.1362481
1: -4.1178484, 8.0096436, -1.0038316, 5.3580608, -9.4759083, 9.0134754
2: -6.0670018, 6.5351744, -1.6267047, 3.9844241, -10.0514231, 8.1618786
3: -2.6416011, 10.8504009, -1.5247964, 5.9122162, -8.5538158, 12.3751965
4: -8.1326199, 7.4373612, -2.7564511, 4.5426917, -12.6753120, 10.1938124

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 1.64 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1818622, upper bound: 12.1802070
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1825781, upper bound: 12.1774771
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -4.4856672, 4.6578822, -10.1871710, 10.4510441
1: -4.1178484, 8.0096436, -3.2995057, 6.5600414, -10.6778851, 11.3091488
2: -6.0670018, 6.5351744, -4.7939720, 5.1907496, -11.2577477, 11.3291464
3: -2.6416011, 10.8504009, -2.0794299, 8.5718803, -11.2134819, 12.9298306
4: -8.1326199, 7.4373612, -6.4134078, 5.9567304, -14.0893459, 13.8507690

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38

Time for candidate selection: 1.52 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0311161, upper bound: 12.1890111
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192263, upper bound: 12.2145008
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -5.5292892, 5.9653769, -5.3601141, 5.9159899, -11.4452782, 11.3254910
1: -4.1178484, 8.0096436, -3.9965582, 7.9697895, -12.0876369, 12.0062017
2: -6.0670018, 6.5351744, -5.9000158, 6.4758339, -12.5428324, 12.4351902
3: -2.6416011, 10.8504009, -2.6022196, 10.7416420, -13.3832407, 13.4526205
4: -8.1326199, 7.4373612, -7.9224205, 7.3602033, -15.4928226, 15.3597813

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 1.57 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0311161, upper bound: 12.1890111
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2192263, upper bound: 12.2145008
time: 0.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.68 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0941762, upper bound: 12.0231103
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0941762, upper bound: 12.0849556
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0188443, upper bound: 12.0188443
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0849549, upper bound: 12.1257561
IS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0616992, upper bound: 12.0215723
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0849556, upper bound: 12.1535445
IS_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0616997, upper bound: 12.1133036
IS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0328487, upper bound: 12.1540377
IS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1635481, upper bound: 12.2141215
IS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0328487, upper bound: 12.1554049
IS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1635481, upper bound: 12.2141219
IS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1340902, upper bound: 12.0309530
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1340901, upper bound: 12.2047832
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1601383, upper bound: 12.2009624
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1502774, upper bound: 12.2009623
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0328487
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1635481
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0328487
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.2141215, upper bound: 12.1884568
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0474759, upper bound: 12.1899355
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0474759, upper bound: 12.2189791
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0318079
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.2527111, upper bound: 12.2189791
IS_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0309530, upper bound: 12.1340901
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.2047831, upper bound: 12.1636282
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1818622, upper bound: 12.1802070
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.1825781, upper bound: 12.1774771
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0311161, upper bound: 12.1890111
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.2192263, upper bound: 12.2145008
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.0311161, upper bound: 12.1890111
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -12.2192263, upper bound: 12.2145008

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1.2984991, 3.5614290, -1.1054896, 3.3209162, -4.6194153, 4.6669188
1: -1.0948845, 5.5469856, -0.9644426, 5.3104076, -6.4052920, 6.5114264
2: -1.7526169, 4.1736655, -1.5707686, 3.9254885, -5.6781025, 5.7444334
3: -1.5816003, 6.1347599, -1.5108141, 5.7992516, -7.3808517, 7.6455731
4: -2.9039445, 4.7323523, -2.7037435, 4.4664245, -7.3703685, 7.4360957

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142163, upper bound: 12.1061224
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1142163, upper bound: 12.1133032
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -4.4929471, 4.6846557, -4.8966231, 6.1571488
1: -0.2774035, 2.7060747, -3.3053031, 6.5950356, -6.8724384, 6.0113778
2: -0.6758080, 2.1488650, -4.8029432, 5.2199960, -5.8958039, 6.9518080
3: -0.9214749, 2.6601696, -2.0835414, 8.6005402, -9.5220146, 4.7437110
4: -1.2530408, 2.4732645, -6.4269009, 5.9843197, -7.2373605, 8.9001656

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540377
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1540380
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.9886395, 2.7079735, -4.4929471, 4.6846557, -5.6732950, 7.2009206
1: -0.8401878, 4.6642990, -3.3053031, 6.5950356, -7.4352236, 7.9696021
2: -1.3668380, 3.2843196, -4.8029432, 5.2199960, -6.5868316, 8.0872612
3: -1.3257837, 5.0050430, -2.0835414, 8.6005402, -9.9263210, 7.0885839
4: -2.3902826, 3.7804737, -6.4269009, 5.9843197, -8.3746023, 10.2073736

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1549376, upper bound: 12.2111971
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1900037
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.4929471, 4.6846557, -4.8989654, 6.2725267
1: -0.2778803, 2.8850155, -3.3053031, 6.5950356, -6.8729157, 6.1903186
2: -0.7044127, 2.2803013, -4.8029432, 5.2199960, -5.9244084, 7.0832443
3: -0.9286744, 2.8235273, -2.0835414, 8.6005402, -9.5292149, 4.9070687
4: -1.3212409, 2.5750978, -6.4269009, 5.9843197, -7.3055606, 9.0019989

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554051
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554051
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -4.4929471, 4.6846557, -5.2105551, 7.6239285
1: -0.5956205, 4.9794006, -3.3053031, 6.5950356, -7.1906562, 8.2847023
2: -1.2669206, 3.7673125, -4.8029432, 5.2199960, -6.4869151, 8.5702553
3: -1.4658637, 5.3376389, -2.0835414, 8.6005402, -10.0664005, 7.4211798
4: -2.3321438, 4.3155594, -6.4269009, 5.9843197, -8.3164606, 10.7424574

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.2011700
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1805167, upper bound: 12.2011700
time: 0.50 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.7353463, 3.0922532, -3.5947866, 5.1461210, -6.8814673, 6.6870399
1: -1.3258518, 5.2033310, -2.7206869, 7.0730000, -8.3988514, 7.9240174
2: -2.0123925, 3.6252692, -4.1022797, 5.7853966, -7.7977891, 7.7275486
3: -1.4236145, 5.7902241, -2.2889237, 8.9996395, -10.4232512, 8.0791473
4: -3.0535307, 4.1617336, -5.6985736, 6.5908761, -9.6444073, 9.8603058

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1525738, upper bound: 12.2018588
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1550489, upper bound: 12.1825782
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.2624179, 3.4834623, -3.4458966, 4.8638849, -6.1263008, 6.9293585
1: -1.0676382, 5.4686108, -2.6795650, 6.5769024, -7.6445398, 8.1481733
2: -1.7101583, 4.0883298, -4.0282803, 5.3606248, -7.0707831, 8.1166096
3: -1.5565034, 6.0524197, -2.1814013, 8.7576485, -10.3141518, 8.2338209
4: -2.8450947, 4.6510177, -5.5947256, 6.1345906, -8.9796848, 10.2457428

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1586326
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1840943, upper bound: 12.2009623
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.2624179, 3.4834623, -4.3892775, 5.3788052, -6.6412210, 7.8727398
1: -1.0676382, 5.4686108, -3.2772136, 7.2921524, -8.3597879, 8.7458229
2: -1.7101583, 4.0883298, -4.8330884, 6.0067921, -7.7169504, 8.9214182
3: -1.5565034, 6.0524197, -2.3618312, 9.3598366, -10.9163399, 8.4142494
4: -2.8450947, 4.6510177, -6.5955544, 6.8001308, -9.6452255, 11.2465725

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1589803
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009623
time: 0.47 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2110309, 1.6549163, -6.1908622, 4.9112787
1: -3.3364797, 6.6092453, -0.2760494, 2.6907148, -6.0271945, 6.8852935
2: -4.8469982, 5.2329617, -0.6732438, 2.1372545, -6.9842529, 5.9062052
3: -2.0921249, 8.6418915, -0.9174765, 2.6461020, -4.7382269, 9.5593681
4: -6.4802928, 6.0003052, -1.2481065, 2.4617484, -8.9420414, 7.2484097

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0544692
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1540368, upper bound: 12.0471736
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.9776220, 2.6941109, -7.2300568, 5.6778698
1: -3.3364797, 6.6092453, -0.8326924, 4.6479073, -7.9843855, 7.4419360
2: -4.8469982, 5.2329617, -1.3567810, 3.2700517, -8.1170483, 6.5897427
3: -2.0921249, 8.6418915, -1.3216619, 4.9836864, -7.0758114, 9.9635534
4: -6.4802928, 6.0003052, -2.3795023, 3.7652493, -10.2455425, 8.3798065

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2111971, upper bound: 12.1549376
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2081061, 1.7196934, -6.2556391, 4.9083538
1: -3.3364797, 6.6092453, -0.2689453, 2.7844934, -6.1209731, 6.8781905
2: -4.8469982, 5.2329617, -0.6878142, 2.2030742, -7.0500727, 5.9207759
3: -2.0921249, 8.6418915, -0.9015521, 2.7321863, -4.8243113, 9.5434437
4: -6.4802928, 6.0003052, -1.2857537, 2.5002146, -8.9805069, 7.2860589

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.4617273, 3.0399928, -7.5759387, 5.1619749
1: -3.3364797, 6.6092453, -0.5485357, 4.8817215, -8.2182007, 7.1577797
2: -4.8469982, 5.2329617, -1.2297997, 3.6682897, -8.5152845, 6.4627609
3: -2.0921249, 8.6418915, -1.4339643, 5.2126656, -7.3047905, 10.0758553
4: -6.4802928, 6.0003052, -2.2761354, 4.2083616, -10.6886539, 8.2764406

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2011700, upper bound: 12.1849794
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2011700, upper bound: 12.1805168
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -4.5359459, 4.7002478, -4.9744987, 6.8553329
1: -0.3634766, 3.2789097, -3.3364797, 6.6092453, -6.9727197, 6.6153893
2: -0.7444794, 2.9443834, -4.8469982, 5.2329617, -5.9774408, 7.7913818
3: -1.1945589, 3.2639122, -2.0921249, 8.6418915, -9.8364496, 5.3560371
4: -1.3846707, 3.3782203, -6.4802928, 6.0003052, -7.3849754, 9.8585129

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0540145, upper bound: 12.2349286
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0712027, upper bound: 12.2347949
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -3.8283892, 4.3033652, -4.5359459, 4.7002478, -8.5286369, 8.8393097
1: -2.8217945, 6.1582298, -3.3364797, 6.6092453, -9.4310341, 9.4947062
2: -4.1151338, 4.8628931, -4.8469982, 5.2329617, -9.3480949, 9.7098885
3: -1.9461601, 7.8930817, -2.0921249, 8.6418915, -10.5880499, 9.9852066
4: -5.5757437, 5.5972967, -6.4802928, 6.0003052, -11.5760489, 12.0775890

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2598722, upper bound: 12.2583900
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2593504, upper bound: 12.2585996
time: 0.45 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2891956, 2.7382221, -7.2741680, 4.9894433
1: -3.3364797, 6.6092453, -0.3822187, 3.9519520, -7.2884316, 6.9914627
2: -4.8469982, 5.2329617, -0.8797669, 3.3527915, -8.1997900, 6.1127281
3: -2.0921249, 8.6418915, -1.2386876, 3.9645653, -6.0566902, 9.8805790
4: -6.4802928, 6.0003052, -1.6498394, 3.7471557, -10.2274485, 7.6501431

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0283118
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0232637
time: 0.42 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -4.1076355, 5.3777466, -9.9136925, 8.8078833
1: -3.3364797, 6.6092453, -3.1126299, 7.2822380, -10.6187153, 9.7218742
2: -4.8469982, 5.2329617, -4.6482496, 5.9726243, -10.8196220, 9.8812094
3: -2.0921249, 8.6418915, -2.4097505, 9.6314602, -11.7235851, 11.0516415
4: -6.4802928, 6.0003052, -6.3888316, 6.8270602, -13.3073530, 12.3891354

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2170752, upper bound: 12.2117240
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2448737, upper bound: 12.2117240
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -1.7240775, 3.0773549, -7.2394991, 7.1186075
1: -3.1508241, 7.2959442, -1.3177787, 5.1861057, -8.3369274, 8.6137228
2: -4.7026148, 5.9927139, -2.0007284, 3.6104007, -8.3130150, 7.9934406
3: -2.4236612, 9.6652107, -1.4191301, 5.7682171, -8.1918764, 11.0843382
4: -6.4547949, 6.8532066, -3.0381684, 4.1457658, -10.6005592, 9.8913727

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1818622, upper bound: 12.1602054
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1825781, upper bound: 12.1550489
time: 0.49 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -1.1708714, 3.3879943, -8.3224535, 6.3749833
1: -3.7369466, 6.8882389, -1.0038316, 5.3580608, -9.0950069, 7.8920708
2: -5.4744320, 5.6376672, -1.6267047, 3.9844241, -9.4588556, 7.2643709
3: -2.3689766, 9.8232098, -1.5247964, 5.9122162, -8.2811928, 11.3480062
4: -7.3408518, 6.5052090, -2.7564511, 4.5426917, -11.8835430, 9.2616596

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1586363, upper bound: 12.0297241
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1802068
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -1.1708714, 3.3879943, -8.3697577, 6.8046737
1: -3.7128515, 7.5261250, -1.0038316, 5.3580608, -9.0709114, 8.5299549
2: -5.4640889, 6.2225499, -1.6267047, 3.9844241, -9.4485130, 7.8492546
3: -2.5014567, 10.0730762, -1.5247964, 5.9122162, -8.4136734, 11.5978727
4: -7.3761001, 7.0715237, -2.7564511, 4.5426917, -11.9187918, 9.8279743

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590188, upper bound: 12.0239668
time: 0.64 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1946470, upper bound: 12.1774769
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.4856672, 4.6578822, -4.9470778, 7.2238894
1: -0.3822187, 3.9519520, -3.2995057, 6.5600414, -6.9422574, 7.2514577
2: -0.8797669, 3.3527915, -4.7939720, 5.1907496, -6.0705152, 8.1467638
3: -1.2386876, 3.9645653, -2.0794299, 8.5718803, -9.8105679, 6.0439949
4: -1.6498394, 3.7471557, -6.4134078, 5.9567304, -7.6065693, 10.1605635

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1990533
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0197084, upper bound: 12.1990537
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -4.4856672, 4.6578822, -8.8200264, 9.8801966
1: -3.1508241, 7.2959442, -3.2995057, 6.5600414, -9.7108612, 10.5954494
2: -4.7026148, 5.9927139, -4.7939720, 5.1907496, -9.8933601, 10.7866840
3: -2.4236612, 9.6652107, -2.0794299, 8.5718803, -10.9955406, 11.7446404
4: -6.4547949, 6.8532066, -6.4134078, 5.9567304, -12.4115219, 13.2666111

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2114956, upper bound: 12.2114996
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2114956, upper bound: 12.2121982
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -5.3601141, 5.9159899, -6.2051854, 8.0983362
1: -0.3822187, 3.9519520, -3.9965582, 7.9697895, -8.3520050, 7.9485102
2: -0.8797669, 3.3527915, -5.9000158, 6.4758339, -7.3555994, 9.2528076
3: -1.2386876, 3.9645653, -2.6022196, 10.7416420, -11.9803295, 6.5667849
4: -1.6498394, 3.7471557, -7.9224205, 7.3602033, -9.0100422, 11.6695766

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1890111
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -5.3601141, 5.9159899, -10.0781345, 10.7546453
1: -3.1508241, 7.2959442, -3.9965582, 7.9697895, -11.1206093, 11.2925024
2: -4.7026148, 5.9927139, -5.9000158, 6.4758339, -11.1784468, 11.8927298
3: -2.4236612, 9.6652107, -2.6022196, 10.7416420, -13.1653032, 12.2674303
4: -6.4547949, 6.8532066, -7.9224205, 7.3602033, -13.8149986, 14.7756271

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.2108905
time: 0.57 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2114726, upper bound: 12.2108905
time: 0.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.68 seconds
IS_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1142163, upper bound: 12.1061224
IS_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1142163, upper bound: 12.1133032
IS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540377
IS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1540380
IS_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1549376, upper bound: 12.2111971
IS_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1900037
IS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554051
IS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554051
IS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0296825, upper bound: 12.2011700
IS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1805167, upper bound: 12.2011700
IS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1525738, upper bound: 12.2018588
IS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1550489, upper bound: 12.1825782
IS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1586326
IS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1840943, upper bound: 12.2009623
IS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1589803
IS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009623
IS_A2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0544692
IS_A2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1540368, upper bound: 12.0471736
IS_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2111971, upper bound: 12.1549376
IS_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
IS_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
IS_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
IS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2011700, upper bound: 12.1849794
IS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2011700, upper bound: 12.1805168
IS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0540145, upper bound: 12.2349286
IS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0712027, upper bound: 12.2347949
IS_A2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2598722, upper bound: 12.2583900
IS_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2593504, upper bound: 12.2585996
IS_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0283118
IS_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0232637
IS_A2_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2170752, upper bound: 12.2117240
IS_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2448737, upper bound: 12.2117240
IS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1818622, upper bound: 12.1602054
IS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1825781, upper bound: 12.1550489
IS_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1586363, upper bound: 12.0297241
IS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1802068
IS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1590188, upper bound: 12.0239668
IS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.1946470, upper bound: 12.1774769
IS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1990533
IS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0197084, upper bound: 12.1990537
IS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2114956, upper bound: 12.2114996
IS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2114956, upper bound: 12.2121982
IS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
IS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1890111
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.0276476, upper bound: 12.2108905
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -12.2114726, upper bound: 12.2108905

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -2.5198274, 4.0429096, -4.2548771, 4.1840291
1: -0.2774035, 2.7060747, -1.9749300, 5.7233124, -6.0007143, 4.6810045
2: -0.6758080, 2.1488650, -3.0195792, 4.5028224, -5.1786304, 5.1684442
3: -0.9214749, 2.6601696, -1.8268865, 7.4602890, -8.3817635, 4.4870563
4: -1.2530408, 2.4732645, -4.3079014, 5.1903458, -6.4433861, 6.7811661

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540380
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540380
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -3.9858012, 4.3280334, -4.5400009, 5.6500030
1: -0.2774035, 2.7060747, -2.9282193, 6.0375443, -6.3149467, 5.6342940
2: -0.6758080, 2.1488650, -4.2405438, 4.8869801, -5.5627880, 6.3894091
3: -0.9214749, 2.6601696, -1.9378814, 7.7693977, -8.6908722, 4.5980511
4: -1.2530408, 2.4732645, -5.6933999, 5.6019974, -6.8550358, 8.1666641

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1540377
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1540377
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -4.4929471, 4.6846557, -4.9609275, 6.7482319
1: -0.3717489, 3.9332523, -3.3053031, 6.5950356, -6.9667845, 7.2385554
2: -1.0147076, 2.7846820, -4.8029432, 5.2199960, -6.2347026, 7.5876255
3: -1.1808519, 4.1503563, -2.0835414, 8.6005402, -9.7813911, 6.2338977
4: -1.9016094, 3.2297649, -6.4269009, 5.9843197, -7.8859291, 9.6566658

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1340902, upper bound: 12.0742435
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.5457935, 2.4314504, -4.4929471, 4.6846557, -5.2304492, 6.9243975
1: -0.5560611, 4.1216106, -3.3053031, 6.5950356, -7.1510968, 7.4269137
2: -1.0841694, 2.9876790, -4.8029432, 5.2199960, -6.3041654, 7.7906222
3: -1.2315452, 4.3375187, -2.0835414, 8.6005402, -9.8320847, 6.4210601
4: -1.9889984, 3.4472027, -6.4269009, 5.9843197, -7.9733181, 9.8741035

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.5198274, 4.0429096, -4.2572193, 4.2994070
1: -0.2778803, 2.8850155, -1.9749300, 5.7233124, -6.0011926, 4.8599453
2: -0.7044127, 2.2803013, -3.0195792, 4.5028224, -5.2072339, 5.2998805
3: -0.9286744, 2.8235273, -1.8268865, 7.4602890, -8.3889637, 4.6504140
4: -1.3212409, 2.5750978, -4.3079014, 5.1903458, -6.5115852, 6.8829994

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554049
time: 0.38 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554051
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -3.9858012, 4.3280334, -4.5423431, 5.7653809
1: -0.2778803, 2.8850155, -2.9282193, 6.0375443, -6.3154235, 5.8132348
2: -0.7044127, 2.2803013, -4.2405438, 4.8869801, -5.5913925, 6.5208454
3: -0.9286744, 2.8235273, -1.9378814, 7.7693977, -8.6980715, 4.7614088
4: -1.3212409, 2.5750978, -5.6933999, 5.6019974, -6.9232368, 8.2684975

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554051
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554051
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -2.5198274, 4.0429096, -4.5688076, 5.6508074
1: -0.5956205, 4.9794006, -1.9749300, 5.7233124, -6.3189306, 6.9543304
2: -1.2669206, 3.7673125, -3.0195792, 4.5028224, -5.7697430, 6.7868900
3: -1.4658637, 5.3376389, -1.8268865, 7.4602890, -8.9261494, 7.1645255
4: -2.3321438, 4.3155594, -4.3079014, 5.1903458, -7.5224876, 8.6234608

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554051
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1778306, upper bound: 12.2011700
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.5258994, 3.1309829, -3.9858012, 4.3280334, -4.8539324, 7.1167827
1: -0.5956205, 4.9794006, -2.9282193, 6.0375443, -6.6331649, 7.9076180
2: -1.2669206, 3.7673125, -4.2405438, 4.8869801, -6.1539006, 8.0078545
3: -1.4658637, 5.3376389, -1.9378814, 7.7693977, -9.2352591, 7.2755203
4: -2.3321438, 4.3155594, -5.6933999, 5.6019974, -7.9341373, 10.0089579

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554049
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1805167, upper bound: 12.2011700
time: 0.51 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3069287, 2.6366191, -3.5947866, 5.1461210, -5.4530497, 6.2314057
1: -0.4171031, 4.5447903, -2.7206869, 7.0730000, -7.4901032, 7.2654772
2: -1.1923580, 3.1466510, -4.1022797, 5.7853966, -6.9777536, 7.2489305
3: -1.2790046, 4.9741669, -2.2889237, 8.9996395, -10.2786398, 7.2630892
4: -2.2190628, 3.6107554, -5.6985736, 6.5908761, -8.8099384, 9.3093281

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1340902, upper bound: 12.0276471
time: 0.43 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1340901, upper bound: 12.2018587
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.2866788, 2.8084860, -3.5947866, 5.1461210, -6.4327998, 6.4032726
1: -1.0219777, 4.7035484, -2.7206869, 7.0730000, -8.0949774, 7.4242349
2: -1.5617459, 3.3616722, -4.1022797, 5.7853966, -7.3471394, 7.4639521
3: -1.3344283, 5.0709953, -2.2889237, 8.9996395, -10.3340673, 7.3599186
4: -2.5152102, 3.8443499, -5.6985736, 6.5908761, -9.1060848, 9.5429230

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.1340902, upper bound: 12.0197084
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1550489, upper bound: 12.1825782
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -3.4458966, 4.8638849, -5.0781946, 5.2254763
1: -0.2778803, 2.8850155, -2.6795650, 6.5769024, -6.8547812, 5.5645804
2: -0.7044127, 2.2803013, -4.0282803, 5.3606248, -6.0650377, 6.3085814
3: -0.9286744, 2.8235273, -2.1814013, 8.7576485, -9.6863232, 5.0049286
4: -1.3212409, 2.5750978, -5.5947256, 6.1345906, -7.4558315, 8.1698236

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1586326
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0236710, upper bound: 12.1586326
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.5106525, 3.0879889, -3.4458966, 4.8638849, -5.3745375, 6.5338855
1: -0.5847617, 4.9359245, -2.6795650, 6.5769024, -7.1616640, 7.6154895
2: -1.2542624, 3.7202733, -4.0282803, 5.3606248, -6.6148872, 7.7485533
3: -1.4538450, 5.2910919, -2.1814013, 8.7576485, -10.2114935, 7.4724932
4: -2.3069859, 4.2723231, -5.5947256, 6.1345906, -8.4415760, 9.8670483

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1853963, upper bound: 12.2009624
time: 0.47 seconds

## Relational analysis of IS_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009623
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -4.3892775, 5.3788052, -5.5931139, 6.1688571
1: -0.2778803, 2.8850155, -3.2772136, 7.2921524, -7.5700293, 6.1622291
2: -0.7044127, 2.2803013, -4.8330884, 6.0067921, -6.7112045, 7.1133900
3: -0.9286744, 2.8235273, -2.3618312, 9.3598366, -10.2885113, 5.1853585
4: -1.3212409, 2.5750978, -6.5955544, 6.8001308, -8.1213722, 9.1706524

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0236710, upper bound: 12.1586326
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1589803
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.5106525, 3.0879889, -4.3892775, 5.3788052, -5.8894577, 7.4772658
1: -0.5847617, 4.9359245, -3.2772136, 7.2921524, -7.8769121, 8.2131386
2: -1.2542624, 3.7202733, -4.8330884, 6.0067921, -7.2610545, 8.5533619
3: -1.4538450, 5.2910919, -2.3618312, 9.3598366, -10.8136816, 7.6529231
4: -2.3069859, 4.2723231, -6.5955544, 6.8001308, -9.1071148, 10.8678780

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009623
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009624
time: 0.68 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2110309, 1.6549163, -4.3304129, 4.2879200
1: -2.0858450, 5.7528667, -0.2760494, 2.6907148, -4.7765598, 6.0289159
2: -3.1665211, 4.5292182, -0.6732438, 2.1372545, -5.3037758, 5.2024622
3: -1.8468850, 7.5567837, -0.9174765, 2.6461020, -4.4929867, 8.4742594
4: -4.4865170, 5.2266645, -1.2481065, 2.4617484, -6.9482651, 6.4747710

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0544692
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0544692
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2110309, 1.6549163, -5.6832662, 4.5548468
1: -2.9591291, 6.0517540, -0.2760494, 2.6907148, -5.6498442, 6.3278036
2: -4.2841620, 4.9001317, -0.6732438, 2.1372545, -6.4214163, 5.5733757
3: -1.9467452, 7.8118000, -0.9174765, 2.6461020, -4.5928469, 8.7292767
4: -5.7461939, 5.6184468, -1.2481065, 2.4617484, -8.2079420, 6.8665533

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0340547, upper bound: 12.0255795
time: 0.59 seconds

## Relational analysis of IS_A2_A1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B1_B1_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0340547, upper bound: 12.0471739
time: 0.39 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.2685398, 2.1750150, -6.7109609, 4.9687872
1: -3.3364797, 6.6092453, -0.3607419, 3.8328800, -7.1693597, 6.9699855
2: -4.8469982, 5.2329617, -0.9898968, 2.6970978, -7.5440960, 6.2228584
3: -2.0921249, 8.6418915, -1.1543262, 4.0245876, -6.1167126, 9.7962179
4: -6.4802928, 6.0003052, -1.8571100, 3.1372511, -9.6175442, 7.8574133

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
time: 0.43 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -0.5357876, 2.4171166, -6.9530625, 5.2360353
1: -3.3364797, 6.6092453, -0.5494075, 4.1040154, -7.4404950, 7.1586509
2: -4.8469982, 5.2329617, -1.0793633, 2.9711723, -7.8181705, 6.3123250
3: -2.0921249, 8.6418915, -1.2273915, 4.3189678, -6.4110928, 9.8692818
4: -6.4802928, 6.0003052, -1.9813614, 3.4312840, -9.9115772, 7.9816661

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0471739
time: 0.40 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2081061, 1.7196934, -4.3951898, 4.2849956
1: -2.0858450, 5.7528667, -0.2689453, 2.7844934, -4.8703384, 6.0218120
2: -3.1665211, 4.5292182, -0.6878142, 2.2030742, -5.3695955, 5.2170324
3: -1.8468850, 7.5567837, -0.9015521, 2.7321863, -4.5790710, 8.4583330
4: -4.4865170, 5.2266645, -1.2857537, 2.5002146, -6.9867315, 6.5124183

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
time: 0.41 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2081061, 1.7196934, -5.7480431, 4.5519218
1: -2.9591291, 6.0517540, -0.2689453, 2.7844934, -5.7436228, 6.3206992
2: -4.2841620, 4.9001317, -0.6878142, 2.2030742, -6.4872360, 5.5879459
3: -1.9467452, 7.8118000, -0.9015521, 2.7321863, -4.6789312, 8.7133503
4: -5.7461939, 5.6184468, -1.2857537, 2.5002146, -8.2464085, 6.9042006

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
time: 0.43 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.4617273, 3.0399928, -5.7154889, 4.5386157
1: -2.0858450, 5.7528667, -0.5485357, 4.8817215, -6.9675651, 6.3014002
2: -3.1665211, 4.5292182, -1.2297997, 3.6682897, -6.8348088, 5.7590179
3: -1.8468850, 7.5567837, -1.4339643, 5.2126656, -7.0595503, 8.9907475
4: -4.4865170, 5.2266645, -2.2761354, 4.2083616, -8.6948786, 7.5027990

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2011700, upper bound: 12.1849794
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.4617273, 3.0399928, -7.0683427, 4.8055429
1: -2.9591291, 6.0517540, -0.5485357, 4.8817215, -7.8408504, 6.6002884
2: -4.2841620, 4.9001317, -1.2297997, 3.6682897, -7.9524508, 6.1299305
3: -1.9467452, 7.8118000, -1.4339643, 5.2126656, -7.1594105, 9.2457619
4: -5.7461939, 5.6184468, -2.2761354, 4.2083616, -9.9545555, 7.8945823

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
time: 0.44 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1900025, upper bound: 12.1805161
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -2.6754963, 4.0768895, -4.3511410, 4.9948835
1: -0.3634766, 3.2789097, -2.0858450, 5.7528667, -6.1163435, 5.3647547
2: -0.7444794, 2.9443834, -3.1665211, 4.5292182, -5.2736979, 6.1109047
3: -1.1945589, 3.2639122, -1.8468850, 7.5567837, -8.7513418, 5.1107969
4: -1.3846707, 3.3782203, -4.4865170, 5.2266645, -6.6113353, 7.8647375

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0540145, upper bound: 12.1278519
time: 0.45 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0540145, upper bound: 12.1278519
time: 0.48 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2742517, 2.3193870, -4.0283499, 4.3438158, -4.6180677, 6.3477368
1: -0.3634766, 3.2789097, -2.9591291, 6.0517540, -6.4152298, 6.2380390
2: -0.7444794, 2.9443834, -4.2841620, 4.9001317, -5.6446109, 7.2285452
3: -1.1945589, 3.2639122, -1.9467452, 7.8118000, -9.0063591, 5.2106571
4: -1.3846707, 3.3782203, -5.7461939, 5.6184468, -7.0031176, 9.1244144

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0571018, upper bound: 12.0583974
time: 0.53 seconds

## Relational analysis of IS_A2_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0571018, upper bound: 12.2347949
time: 0.44 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1.9736005, 3.7326164, -4.5359459, 4.7002478, -6.6738482, 8.2685623
1: -1.5754695, 5.3457413, -3.3364797, 6.6092453, -8.1847095, 8.6822195
2: -2.4574776, 4.2253923, -4.8469982, 5.2329617, -7.6904378, 9.0723877
3: -1.7051358, 6.8608799, -2.0921249, 8.6418915, -10.3470259, 8.9530048
4: -3.6110172, 4.8835225, -6.4802928, 6.0003052, -9.6113224, 11.3638153

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346645, upper bound: 12.0767516
time: 0.50 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2346645, upper bound: 12.2583900
time: 0.62 seconds

## BFS IS instance: IS_A2_A1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.2853987, 3.9233079, -4.5359459, 4.7002478, -7.9856462, 8.4592514
1: -2.4223030, 5.5470581, -3.3364797, 6.6092453, -9.0315456, 8.8835373
2: -3.5144036, 4.5003858, -4.8469982, 5.2329617, -8.7473650, 9.3473835
3: -1.7845080, 6.9990902, -2.0921249, 8.6418915, -10.4263992, 9.0912151
4: -4.7958260, 5.2002707, -6.4802928, 6.0003052, -10.7961292, 11.6805630

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345671, upper bound: 12.0712046
time: 0.47 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2345671, upper bound: 12.2582813
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -2.6754963, 4.0768895, -0.2891956, 2.7382221, -5.4137182, 4.3660851
1: -2.0858450, 5.7528667, -0.3822187, 3.9519520, -6.0377970, 6.1350851
2: -3.1665211, 4.5292182, -0.8797669, 3.3527915, -6.5193129, 5.4089851
3: -1.8468850, 7.5567837, -1.2386876, 3.9645653, -5.8114500, 8.7954712
4: -4.4865170, 5.2266645, -1.6498394, 3.7471557, -8.2336731, 6.8765025

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0283118
time: 0.52 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0283118
time: 0.46 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -4.0283499, 4.3438158, -0.2891956, 2.7382221, -6.7665720, 4.6330113
1: -2.9591291, 6.0517540, -0.3822187, 3.9519520, -6.9110813, 6.4339728
2: -4.2841620, 4.9001317, -0.8797669, 3.3527915, -7.6369534, 5.7798986
3: -1.9467452, 7.8118000, -1.2386876, 3.9645653, -5.9113102, 9.0504875
4: -5.7461939, 5.6184468, -1.6498394, 3.7471557, -9.4933491, 7.2682862

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0232637
time: 0.48 seconds

## Relational analysis of IS_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0232637
time: 0.50 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.6782563, 4.7223377, -9.2582836, 8.3785038
1: -3.3364797, 6.6092453, -2.8371258, 6.2889423, -9.6254196, 9.4463701
2: -4.8469982, 5.2329617, -4.2119579, 5.1972218, -10.0442190, 9.4449196
3: -2.0921249, 8.6418915, -2.1570172, 8.6666470, -10.7587719, 10.7989082
4: -6.4802928, 6.0003052, -5.7941303, 6.0009980, -12.4812908, 11.7944355

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2448737, upper bound: 12.2117240
time: 0.42 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2448737, upper bound: 12.2117240
time: 0.49 seconds

## BFS IS instance: IS_A2_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -4.5359459, 4.7002478, -3.4619281, 4.9763918, -9.5123377, 8.1621761
1: -3.3364797, 6.6092453, -2.6407366, 6.6547775, -9.9912548, 9.2499819
2: -4.8469982, 5.2329617, -3.9357042, 5.6090250, -10.4560213, 9.1686630
3: -2.0921249, 8.6418915, -2.2358243, 8.6746998, -10.7668247, 10.8777161
4: -6.4802928, 6.0003052, -5.4861760, 6.4149227, -12.8952160, 11.4864807

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0123554, upper bound: 12.1899355
time: 0.46 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0123554, upper bound: 12.1626593
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -1.7240775, 3.0773549, -6.7556114, 6.4464149
1: -2.8371258, 6.2889423, -1.3177787, 5.1861057, -8.0232315, 7.6067185
2: -4.2119579, 5.1972218, -2.0007284, 3.6104007, -7.8223586, 7.1979494
3: -2.1570172, 8.6666470, -1.4191301, 5.7682171, -7.9252310, 10.0857763
4: -5.7941303, 6.0009980, -3.0381684, 4.1457658, -9.9398918, 9.0391655

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1494798, upper bound: 12.0323726
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1494798, upper bound: 12.1602054
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.4619281, 4.9763918, -1.7240775, 3.0773549, -6.5392823, 6.7004690
1: -2.6407366, 6.6547775, -1.3177787, 5.1861057, -7.8268409, 7.9725518
2: -3.9357042, 5.6090250, -2.0007284, 3.6104007, -7.5461035, 7.6097527
3: -2.2358243, 8.6746998, -1.4191301, 5.7682171, -8.0040407, 10.0938301
4: -5.4861760, 6.4149227, -3.0381684, 4.1457658, -9.6319408, 9.4530907

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -12.0197084, upper bound: 12.1340901
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1825781, upper bound: 12.1550489
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.2081061, 1.7196934, -6.6541538, 5.4122176
1: -3.7369466, 6.8882389, -0.2689453, 2.7844934, -6.5214400, 7.1571841
2: -5.4744320, 5.6376672, -0.6878142, 2.2030742, -7.6775064, 6.3254814
3: -2.3689766, 9.8232098, -0.9015521, 2.7321863, -5.1011629, 10.7247610
4: -7.3408518, 6.5052090, -1.2857537, 2.5002146, -9.8410664, 7.7909627

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1586363, upper bound: 12.0297241
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1586362, upper bound: 12.0239668
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -4.9344602, 5.2041121, -0.4490069, 3.0029449, -7.9374027, 5.6531181
1: -3.7369466, 6.8882389, -0.5394872, 4.8438778, -8.5808210, 7.4277263
2: -5.4744320, 5.6376672, -1.2191920, 3.6275666, -9.1019974, 6.8568592
3: -2.3689766, 9.8232098, -1.4235640, 5.1721821, -7.5411587, 11.2467718
4: -7.3408518, 6.5052090, -2.2547932, 4.1710539, -11.5119057, 8.7600012

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1802068
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1774769
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.2081061, 1.7196934, -6.7014589, 5.8419085
1: -3.7128515, 7.5261250, -0.2689453, 2.7844934, -6.4973450, 7.7950702
2: -5.4640889, 6.2225499, -0.6878142, 2.2030742, -7.6671629, 6.9103632
3: -2.5014567, 10.0730762, -0.9015521, 2.7321863, -5.2336431, 10.9746284
4: -7.3761001, 7.0715237, -1.2857537, 2.5002146, -9.8763142, 8.3572769

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1586362, upper bound: 12.0239668
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1590188, upper bound: 12.0239668
time: 0.60 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.9817657, 5.6338024, -0.4490069, 3.0029449, -7.9847050, 6.0828094
1: -3.7128515, 7.5261250, -0.5394872, 4.8438778, -8.5567274, 8.0656118
2: -5.4640889, 6.2225499, -1.2191920, 3.6275666, -9.0916557, 7.4417419
3: -2.5014567, 10.0730762, -1.4235640, 5.1721821, -7.6736388, 11.4966393
4: -7.3761001, 7.0715237, -2.2547932, 4.1710539, -11.5471535, 9.3263140

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1774769
time: 0.59 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1946474, upper bound: 12.1774769
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -2.6432900, 4.0474329, -4.3366280, 5.3815122
1: -0.3822187, 3.9519520, -2.0620241, 5.7167988, -6.0990152, 6.0139761
2: -0.8797669, 3.3527915, -3.1320014, 4.5007553, -5.3805208, 6.4847927
3: -1.2386876, 3.9645653, -1.8380175, 7.5076027, -8.7462902, 5.8025827
4: -1.6498394, 3.7471557, -4.4430809, 5.1974311, -6.8472695, 8.1902370

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1990533
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1990537
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.0207763, 4.3372831, -4.6264787, 6.7589984
1: -0.3822187, 3.9519520, -2.9535494, 6.0441322, -6.4263496, 6.9055014
2: -0.8797669, 3.3527915, -4.2761312, 4.8936901, -5.7734566, 7.6289225
3: -1.2386876, 3.9645653, -1.9447784, 7.8010159, -9.0397034, 5.9093437
4: -1.6498394, 3.7471557, -5.7360392, 5.6117840, -7.2616234, 9.4831944

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1990533
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1990533
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -3.6782563, 4.7223377, -4.4856672, 4.6578822, -8.3361387, 9.2080050
1: -2.8371258, 6.2889423, -3.2995057, 6.5600414, -9.3971672, 9.5884476
2: -4.2119579, 5.1972218, -4.7939720, 5.1907496, -9.4027061, 9.9911928
3: -2.1570172, 8.6666470, -2.0794299, 8.5718803, -10.7288952, 10.7460756
4: -5.7941303, 6.0009980, -6.4134078, 5.9567304, -11.7508545, 12.4144039

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875251, upper bound: 12.0323726
time: 0.50 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875251, upper bound: 12.2115000
time: 0.51 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -3.4619281, 4.9763918, -4.4856672, 4.6578822, -8.1198092, 9.4620590
1: -2.6407366, 6.6547775, -3.2995057, 6.5600414, -9.2007771, 9.9542828
2: -3.9357042, 5.6090250, -4.7939720, 5.1907496, -9.1264496, 10.4029961
3: -2.2358243, 8.6746998, -2.0794299, 8.5718803, -10.8077049, 10.7541294
4: -5.4861760, 6.4149227, -6.4134078, 5.9567304, -11.4429035, 12.8283310

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875251, upper bound: 12.0283341
time: 0.52 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875251, upper bound: 12.2117553
time: 0.54 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.8830261, 5.1888895, -5.4780850, 7.6212482
1: -0.3822187, 3.9519520, -3.6997132, 6.8758278, -7.2580462, 7.6516652
2: -0.8797669, 3.3527915, -5.4240475, 5.6200366, -6.4998021, 8.7768393
3: -1.2386876, 3.9645653, -2.3564258, 9.7877512, -11.0264387, 6.3209910
4: -1.6498394, 3.7471557, -7.2774620, 6.4821172, -8.1319561, 11.0246181

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
time: 0.56 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2891956, 2.7382221, -4.8599315, 5.6000676, -5.8892627, 7.5981536
1: -0.3822187, 3.9519520, -3.6255574, 7.4982872, -7.8805051, 7.5775094
2: -0.8797669, 3.3527915, -5.3452921, 6.1820874, -7.0618520, 8.6980839
3: -1.2386876, 3.9645653, -2.4739380, 10.0032158, -11.2419033, 6.4385033
4: -1.6498394, 3.7471557, -7.2256942, 7.0176010, -8.6674404, 10.9728498

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1890111
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1890111
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -4.8830261, 5.1888895, -9.3510342, 10.2775574
1: -3.1508241, 7.2959442, -3.6997132, 6.8758278, -10.0266514, 10.9956570
2: -4.7026148, 5.9927139, -5.4240475, 5.6200366, -10.3226490, 11.4167604
3: -2.4236612, 9.6652107, -2.3564258, 9.7877512, -12.2114124, 12.0216370
4: -6.4547949, 6.8532066, -7.2774620, 6.4821172, -12.9369125, 14.1306686

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.2159520, upper bound: 12.2108905
time: 0.50 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -4.1621442, 5.3945312, -4.8599315, 5.6000676, -9.7622118, 10.2544632
1: -3.1508241, 7.2959442, -3.6255574, 7.4982872, -10.6491108, 10.9215012
2: -4.7026148, 5.9927139, -5.3452921, 6.1820874, -10.8846960, 11.3380013
3: -2.4236612, 9.6652107, -2.4739380, 10.0032158, -12.4268770, 12.1391487
4: -6.4547949, 6.8532066, -7.2256942, 7.0176010, -13.4723959, 14.0789013

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875251, upper bound: 12.0233714
time: 0.56 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1875251, upper bound: 12.2108540
time: 0.60 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.91 seconds
IS_A1_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540380
IS_A1_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540380
IS_A1_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1540377
IS_A1_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1540377
IS_A1_B2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
IS_A1_B2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
IS_A1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
IS_A1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
IS_A1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554049
IS_A1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554051
IS_A1_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554051
IS_A1_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554051
IS_A1_B2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554051
IS_A1_B2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1778306, upper bound: 12.2011700
IS_A1_B2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554049
IS_A1_B2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1805167, upper bound: 12.2011700
IS_A1_B2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1340902, upper bound: 12.0276471
IS_A1_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1340901, upper bound: 12.2018587
IS_A1_B2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1340902, upper bound: 12.0197084
IS_A1_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1550489, upper bound: 12.1825782
IS_A1_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1586326
IS_A1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0236710, upper bound: 12.1586326
IS_A1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1853963, upper bound: 12.2009624
IS_A1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009623
IS_A1_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0236710, upper bound: 12.1586326
IS_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1589803
IS_A1_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009623
IS_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1810623, upper bound: 12.2009624
IS_A2_A1_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0544692
IS_A2_A1_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0544692
IS_A2_A1_B1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0340547, upper bound: 12.0255795
IS_A2_A1_B1_B1_B1_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0340547, upper bound: 12.0471739
IS_A2_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
IS_A2_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
IS_A2_A1_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1540377, upper bound: 12.0471739
IS_A2_A1_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1900037, upper bound: 12.1549376
IS_A2_A1_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
IS_A2_A1_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
IS_A2_A1_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
IS_A2_A1_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
IS_A2_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0296825
IS_A2_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2011700, upper bound: 12.1849794
IS_A2_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1554049, upper bound: 12.0238345
IS_A2_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1900025, upper bound: 12.1805161
IS_A2_A1_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0540145, upper bound: 12.1278519
IS_A2_A1_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0540145, upper bound: 12.1278519
IS_A2_A1_B2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0571018, upper bound: 12.0583974
IS_A2_A1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0571018, upper bound: 12.2347949
IS_A2_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2346645, upper bound: 12.0767516
IS_A2_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2346645, upper bound: 12.2583900
IS_A2_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2345671, upper bound: 12.0712046
IS_A2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2345671, upper bound: 12.2582813
IS_A2_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0283118
IS_A2_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0283118
IS_A2_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0232637
IS_A2_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2190772, upper bound: 12.0232637
IS_A2_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2448737, upper bound: 12.2117240
IS_A2_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2448737, upper bound: 12.2117240
IS_A2_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0123554, upper bound: 12.1899355
IS_A2_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0123554, upper bound: 12.1626593
IS_A2_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1494798, upper bound: 12.0323726
IS_A2_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1494798, upper bound: 12.1602054
IS_A2_A2_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0197084, upper bound: 12.1340901
IS_A2_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1825781, upper bound: 12.1550489
IS_A2_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1586363, upper bound: 12.0297241
IS_A2_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1586362, upper bound: 12.0239668
IS_A2_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1802068
IS_A2_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1774769
IS_A2_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1586362, upper bound: 12.0239668
IS_A2_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1590188, upper bound: 12.0239668
IS_A2_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1945263, upper bound: 12.1774769
IS_A2_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1946474, upper bound: 12.1774769
IS_A2_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1990533
IS_A2_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1990537
IS_A2_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1990533
IS_A2_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1990533
IS_A2_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1875251, upper bound: 12.0323726
IS_A2_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1875251, upper bound: 12.2115000
IS_A2_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1875251, upper bound: 12.0283341
IS_A2_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1875251, upper bound: 12.2117553
IS_A2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
IS_A2_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
IS_A2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1890111
IS_A2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0212001, upper bound: 12.1890111
IS_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.0276476, upper bound: 12.1890111
IS_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.2159520, upper bound: 12.2108905
IS_A2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1875251, upper bound: 12.0233714
IS_A2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.91
Output dim: 0, lower bound: -12.1875251, upper bound: 12.2108540

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -2.5198274, 4.0429096, -4.2548771, 4.1840291
1: -0.2774035, 2.7060747, -1.9749300, 5.7233124, -6.0007143, 4.6810045
2: -0.6758080, 2.1488650, -3.0195792, 4.5028224, -5.1786304, 5.1684442
3: -0.9214749, 2.6601696, -1.8268865, 7.4602890, -8.3817635, 4.4870563
4: -1.2530408, 2.4732645, -4.3079014, 5.1903458, -6.4433861, 6.7811661

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540377
time: 0.44 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2864694, 2.2201376, -2.5198274, 4.0429096, -4.3293781, 4.7399650
1: -0.3822893, 4.2234097, -1.9749300, 5.7233124, -6.1056013, 6.1983395
2: -1.1075993, 2.8155358, -3.0195792, 4.5028224, -5.6104217, 5.8351150
3: -1.2409499, 4.3273754, -1.8268865, 7.4602890, -8.7012348, 6.1542621
4: -2.0756645, 3.2417476, -4.3079014, 5.1903458, -7.2660103, 7.5496492

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0544692, upper bound: 12.1540380
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2119676, 1.6642017, -3.9858012, 4.3280334, -4.5400009, 5.6500030
1: -0.2774035, 2.7060747, -2.9282193, 6.0375443, -6.3149467, 5.6342940
2: -0.6758080, 2.1488650, -4.2405438, 4.8869801, -5.5627880, 6.3894091
3: -0.9214749, 2.6601696, -1.9378814, 7.7693977, -8.6908722, 4.5980511
4: -1.2530408, 2.4732645, -5.6933999, 5.6019974, -6.8550358, 8.1666641

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1540377
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2864694, 2.2201376, -3.9858012, 4.3280334, -4.6145020, 6.2059388
1: -0.3822893, 4.2234097, -2.9282193, 6.0375443, -6.4198337, 7.1516290
2: -1.1075993, 2.8155358, -4.2405438, 4.8869801, -5.9945793, 7.0560799
3: -1.2409499, 4.3273754, -1.9378814, 7.7693977, -9.0103464, 6.2652569
4: -2.0756645, 3.2417476, -5.6933999, 5.6019974, -7.6776619, 8.9351473

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.40 seconds

## Relational analysis of IS_A1_B2_B1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -2.5198274, 4.0429096, -4.3191814, 4.7751122
1: -0.3717489, 3.9332523, -1.9749300, 5.7233124, -6.0950613, 5.9081821
2: -1.0147076, 2.7846820, -3.0195792, 4.5028224, -5.5175300, 5.8042612
3: -1.1808519, 4.1503563, -1.8268865, 7.4602890, -8.6411400, 5.9772429
4: -1.9016094, 3.2297649, -4.3079014, 5.1903458, -7.0919533, 7.5376663

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1549376, upper bound: 12.2111971
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2762722, 2.2552848, -3.9858012, 4.3280334, -4.6043057, 6.2410860
1: -0.3717489, 3.9332523, -2.9282193, 6.0375443, -6.4092932, 6.8614717
2: -1.0147076, 2.7846820, -4.2405438, 4.8869801, -5.9016876, 7.0252256
3: -1.1808519, 4.1503563, -1.9378814, 7.7693977, -8.9502497, 6.0882378
4: -1.9016094, 3.2297649, -5.6933999, 5.6019974, -7.5036030, 8.9231644

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 1
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 1

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1549376, upper bound: 12.2111975
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1549376, upper bound: 12.1900040
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.1978316, 1.5574791, -4.4929471, 4.6846557, -4.8824868, 6.0504265
1: -0.2591586, 2.4123363, -3.3053031, 6.5950356, -6.8541942, 5.7176394
2: -0.5880604, 2.0356865, -4.8029432, 5.2199960, -5.8080554, 6.8386297
3: -0.8728676, 2.3451891, -2.0835414, 8.6005402, -9.4734077, 4.4287305
4: -1.0882773, 2.3351970, -6.4269009, 5.9843197, -7.0725965, 8.7620983

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.45 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0471739, upper bound: 12.1540380
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_B1_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.5457935, 2.4314504, -4.4929471, 4.6846557, -5.2304492, 6.9243975
1: -0.5560611, 4.1216106, -3.3053031, 6.5950356, -7.1510968, 7.4269137
2: -1.0841694, 2.9876790, -4.8029432, 5.2199960, -6.3041654, 7.7906222
3: -1.2315452, 4.3375187, -2.0835414, 8.6005402, -9.8320847, 6.4210601
4: -1.9889984, 3.4472027, -6.4269009, 5.9843197, -7.9733181, 9.8741035

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.1502774, upper bound: 12.1900037
time: 0.42 seconds

## Relational analysis of IS_A1_B2_B1_A1_A2_A2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0447445, upper bound: 12.1900037
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -2.5198274, 4.0429096, -4.2572193, 4.2994070
1: -0.2778803, 2.8850155, -1.9749300, 5.7233124, -6.0011926, 4.8599453
2: -0.7044127, 2.2803013, -3.0195792, 4.5028224, -5.2072339, 5.2998805
3: -0.9286744, 2.8235273, -1.8268865, 7.4602890, -8.3889637, 4.6504140
4: -1.3212409, 2.5750978, -4.3079014, 5.1903458, -6.5115852, 6.8829994

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554049
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554049
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3161098, 2.5077143, -2.5198274, 4.0429096, -4.3590193, 5.0275416
1: -0.4254482, 4.4234543, -1.9749300, 5.7233124, -6.1487589, 6.3983841
2: -1.1336632, 3.1795123, -3.0195792, 4.5028224, -5.6364856, 6.1990914
3: -1.3724446, 4.5201435, -1.8268865, 7.4602890, -8.8327312, 6.3470302
4: -2.1406889, 3.6535268, -4.3079014, 5.1903458, -7.3310347, 7.9614282

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 1
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0296825, upper bound: 12.1554049
time: 0.41 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554049
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2143096, 1.7795796, -3.9858012, 4.3280334, -4.5423431, 5.7653809
1: -0.2778803, 2.8850155, -2.9282193, 6.0375443, -6.3154235, 5.8132348
2: -0.7044127, 2.2803013, -4.2405438, 4.8869801, -5.5913925, 6.5208454
3: -0.9286744, 2.8235273, -1.9378814, 7.7693977, -8.6980715, 4.7614088
4: -1.3212409, 2.5750978, -5.6933999, 5.6019974, -6.9232368, 8.2684975

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 1
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 3
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554049
time: 0.39 seconds

## Relational analysis of IS_A1_B2_B1_A2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -12.0238345, upper bound: 12.1554049
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3161098, 2.5077143, -3.9858012, 4.3280334, -4.6441431, 6.4935155
1: -0.4254482, 4.4234543, -2.9282193, 6.0375443, -6.4629927, 7.3516736
2: -1.1336632, 3.1795123, -4.2405438, 4.8869801, -6.0206432, 7.4200563
3: -1.3724446, 4.5201435, -1.9378814, 7.7693977, -9.1418419, 6.4580250
4: -2.1406889, 3.6535268, -5.6933999, 5.6019974, -7.7426863, 9.3469267

Time for backsubstitution: 1.50 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=13.458332061767578
rel_dist={0: [-12.270288718386343, 12.270288718386343]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1151.81 seconds
