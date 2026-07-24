## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_2.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 2.7638016924


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331)
1: (-0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803)
2: (-1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662)
3: (-1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606)
4: (-1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884)

## BASE Result
execution time: IAR + LP analysis = 1.47 + 1.04 = 2.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846


# Binary Search by BASE starts (time budget: 1197.49 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=3.285133123397827
rel_dist={0: [-2.7803829052250393, 2.7803829052250393]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=3.285133123397827
rel_dist={0: [-2.780286344644136, 2.7802863446441357]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=3.285133123397827
rel_dist={0: [-2.7801994131262022, 2.7801994131262022]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=3.285133123397827
rel_dist={0: [-2.78013659937945, 2.78013659937945]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=3.285133123397827
rel_dist={0: [-2.7790340042848785, 2.7790340042848776]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=3.285133123397827
rel_dist={0: [-2.7780414768623047, 2.778041476862305]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=3.285133123397827
rel_dist={0: [-2.777472476768272, 2.7774724767682724]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=3.285133123397827
rel_dist={0: [-2.7771819898984202, 2.7771819898984216]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=3.285133123397827
rel_dist={0: [-2.7770315308237956, 2.7770315308237947]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=3.285133123397827
rel_dist={0: [-2.77695630129431, 2.7769563012943106]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=3.285133123397827
rel_dist={0: [-2.776918686545108, 2.776918686545107]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=3.285133123397827
rel_dist={0: [-2.7768998792011317, 2.7768998792011335]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=3.285133123397827
rel_dist={0: [-2.776890475588643, 2.776890475588642]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=3.285133123397827
rel_dist={0: [-2.776885819152771, 2.7768857738947865]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=3.285133123397827
rel_dist={0: [-2.7768834397858853, 2.7768834397858857]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=3.285133123397827
rel_dist={0: [-2.776882250874421, 2.776882259450736]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=3.285133123397827
rel_dist={0: [-2.7768821313393013, 2.776881713387178]}

## Binary Search Result
Binary search time: 45.42 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1152.06 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735217, upper bound: 2.7756735
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7804673
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -2.7735217, upper bound: 2.7756735
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7804673

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.3937300, 2.2607875, -0.4974192, 2.7340121, -3.1277421, 2.7582066
1: -0.4659128, 3.1165721, -0.5533602, 3.7669601, -4.2328730, 3.6699324
2: -1.1388535, 2.1871104, -1.3512241, 2.6573057, -3.7961593, 3.5383344
3: -0.9433906, 2.6073360, -1.1182656, 3.2933886, -4.2367792, 3.7256017
4: -1.3197460, 2.8582294, -1.7044268, 3.3695371, -4.6892834, 4.5626564

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667264, upper bound: 2.7700536
time: 0.35 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7718051, upper bound: 2.7738750
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.5007467, 2.7479298, -0.5095432, 2.7755899, -3.2763367, 3.2574730
1: -0.5553178, 3.7793705, -0.5611423, 3.8165379, -4.3718557, 4.3405128
2: -1.3528485, 2.6755908, -1.3674926, 2.7016737, -4.0545225, 4.0430832
3: -1.1222062, 3.3138933, -1.1338987, 3.3577619, -4.4799681, 4.4477921
4: -1.7101582, 3.3786907, -1.7360522, 3.4053361, -5.1154943, 5.1147428

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7704332
time: 0.39 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7804438
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.24 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -2.7667264, upper bound: 2.7700536
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -2.7718051, upper bound: 2.7738750
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7704332
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.24
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7804438

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.2694138, 1.5103993, -0.4560462, 2.6190078, -2.8884215, 1.9664454
1: -0.3349953, 2.1516724, -0.5302461, 3.6145725, -3.9495678, 2.6819186
2: -0.8057680, 1.4441006, -1.2986951, 2.5268755, -3.3326435, 2.7427957
3: -0.7113398, 1.6636263, -1.0709016, 3.0826454, -3.7939854, 2.7345281
4: -0.8483682, 2.0235796, -1.5974089, 3.2710128, -4.1193810, 3.6209884

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521981, upper bound: 2.7537987
time: 0.43 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7494703, upper bound: 2.7541922
time: 0.35 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.3660870, 2.1625853, -0.4974039, 2.7339635, -3.1000504, 2.6599891
1: -0.4462218, 2.9898553, -0.5533503, 3.7668998, -4.2131214, 3.5432055
2: -1.0902050, 2.0946541, -1.3512030, 2.6572459, -3.7474508, 3.4458570
3: -0.9038171, 2.4661355, -1.1182456, 3.2933099, -4.1971269, 3.5843811
4: -1.2328745, 2.7640190, -1.7043824, 3.3694868, -4.6023612, 4.4684014

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7618593
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7738750
time: 0.35 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.2679194, 1.6677933, -0.4672572, 2.6555886, -2.9235079, 2.1350505
1: -0.3506812, 2.3546071, -0.5377486, 3.6623266, -4.0130076, 2.8923557
2: -0.8583503, 1.6077589, -1.3149562, 2.5707297, -3.4290800, 2.9227152
3: -0.7267100, 1.8529313, -1.0861664, 3.1423059, -3.8690157, 2.9390976
4: -0.9143050, 2.2346778, -1.6280193, 3.3058619, -4.2201672, 3.8626971

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
time: 0.37 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7702911
time: 0.36 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.4747180, 2.6634238, -0.5095432, 2.7755899, -3.2503080, 3.1729670
1: -0.5381627, 3.6703260, -0.5611423, 3.8165379, -4.3547006, 4.2314682
2: -1.3116175, 2.5915504, -1.3674926, 2.7016737, -4.0132914, 3.9590430
3: -1.0878518, 3.1816847, -1.1338987, 3.3577619, -4.4456139, 4.3155832
4: -1.6326261, 3.2974410, -1.7360522, 3.4053361, -5.0379620, 5.0334930

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7704332, upper bound: 2.7675101
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7704332, upper bound: 2.7804438
time: 0.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.31 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7521981, upper bound: 2.7537987
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7494703, upper bound: 2.7541922
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7618593
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7738750
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7702911
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7704332, upper bound: 2.7675101
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -2.7704332, upper bound: 2.7804438

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3660870, 2.1625853, -0.4715723, 2.6529689, -3.0190558, 2.6341577
1: -0.4462218, 2.9898553, -0.5363634, 3.6588981, -4.1051197, 3.5262187
2: -1.0902050, 2.0946541, -1.3103219, 2.5737772, -3.6639822, 3.4049759
3: -0.9038171, 2.4661355, -1.0842015, 3.1621242, -4.0659413, 3.5503368
4: -1.2328745, 2.7640190, -1.6274512, 3.2890692, -4.5219436, 4.3914700

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7530163, upper bound: 2.7597601
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7682446
time: 0.33 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.1930007, 1.3092765, -0.4607249, 2.6275926, -2.8205934, 1.7700014
1: -0.2793456, 1.8499331, -0.5320190, 3.6234608, -3.9028063, 2.3819523
2: -0.6552293, 1.2896700, -1.2994483, 2.5455186, -3.2007480, 2.5891182
3: -0.5973468, 1.4119213, -1.0750122, 3.1051345, -3.7024813, 2.4869335
4: -0.6719720, 1.8049903, -1.6038489, 3.2737558, -3.9457278, 3.4088392

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
time: 0.37 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
time: 0.34 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2027035, 1.4248056, -0.4672572, 2.6555886, -2.8582921, 1.8920629
1: -0.2976533, 2.0236523, -0.5377486, 3.6623266, -3.9599800, 2.5614009
2: -0.7181041, 1.3894733, -1.3149562, 2.5707297, -3.2888339, 2.7044296
3: -0.6241174, 1.5397948, -1.0861664, 3.1423059, -3.7664232, 2.6259613
4: -0.7333497, 1.9524517, -1.6280193, 3.3058619, -4.0392118, 3.5804710

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7620793, upper bound: 2.7629830
time: 0.41 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7702911
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4747180, 2.6634238, -0.2728533, 1.6860912, -2.1608090, 2.9362769
1: -0.5381627, 3.6703260, -0.3547124, 2.3795998, -2.9177625, 4.0250382
2: -1.3116175, 2.5915504, -0.8683753, 1.6238707, -2.9354882, 3.4599257
3: -1.0878518, 3.1816847, -0.7345715, 1.8792391, -2.9670908, 3.9162562
4: -1.6326261, 3.2974410, -0.9312305, 2.2542911, -3.8869171, 4.2286716

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7669801, upper bound: 2.7663679
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7702910, upper bound: 2.7675101
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4747180, 2.6634238, -0.4836221, 2.6905560, -3.1652741, 3.1470459
1: -0.5381627, 3.6703260, -0.5440643, 3.7076769, -4.2458396, 4.2143903
2: -1.3116175, 2.5915504, -1.3264129, 2.6181285, -3.9297462, 3.9179633
3: -1.0878518, 3.1816847, -1.0996997, 3.2257874, -4.3136392, 4.2813845
4: -1.6326261, 3.2974410, -1.6587552, 3.3243561, -4.9569821, 4.9561963

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7804438
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7700858, upper bound: 2.7804438
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.30 seconds
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7530163, upper bound: 2.7597601
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7682446
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
IS_A2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7620793, upper bound: 2.7629830
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7702911
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7669801, upper bound: 2.7663679
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7702910, upper bound: 2.7675101
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7804438
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.30
Output dim: 0, lower bound: -2.7700858, upper bound: 2.7804438

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3660870, 2.1625853, -0.4056464, 2.4017873, -2.7678742, 2.5682316
1: -0.4462218, 2.9898553, -0.4837095, 3.3162961, -3.7625179, 3.4735646
2: -1.0902050, 2.0946541, -1.1725172, 2.3317146, -3.4219196, 3.2671714
3: -0.9038171, 2.4661355, -0.9814662, 2.8128746, -3.7166915, 3.4476018
4: -1.2328745, 2.7640190, -1.4113345, 3.0007818, -4.2336564, 4.1753535

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7472162, upper bound: 2.7607231
time: 0.38 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7665200, upper bound: 2.7682225
time: 0.37 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1930007, 1.3092765, -0.3556839, 2.1820269, -2.3750277, 1.6649604
1: -0.2793456, 1.8499331, -0.4434152, 2.9940157, -3.2733612, 2.2933483
2: -0.6552293, 1.2896700, -1.0510135, 2.1491966, -2.8044260, 2.3406835
3: -0.5973468, 1.4119213, -0.9022499, 2.5187602, -3.1161070, 2.3141713
4: -0.6719720, 1.8049903, -1.2357124, 2.7616458, -3.4336178, 3.0407028

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7623451, upper bound: 2.7603385
time: 0.33 seconds

## Relational analysis of IS_A2_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582675, upper bound: 2.7607458
time: 0.35 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1930007, 1.3092765, -0.4039841, 2.4129725, -2.6059732, 1.7132607
1: -0.2793456, 1.8499331, -0.4862583, 3.3308933, -3.6102388, 2.3361914
2: -0.6552293, 1.2896700, -1.1799524, 2.3410616, -2.9962909, 2.4696224
3: -0.5973468, 1.4119213, -0.9859166, 2.8082776, -3.4056244, 2.3978379
4: -0.6719720, 1.8049903, -1.4193721, 3.0263507, -3.6983228, 3.2243624

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7623451, upper bound: 2.7603385
time: 0.38 seconds

## Relational analysis of IS_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_A1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582675, upper bound: 2.7607458
time: 0.40 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2027035, 1.4248056, -0.4549679, 2.6173713, -2.8200748, 1.8797735
1: -0.2976533, 2.0236523, -0.5297191, 3.6113737, -3.9090271, 2.5533714
2: -0.7181041, 1.3894733, -1.2964458, 2.5286741, -3.2467782, 2.6859193
3: -0.6241174, 1.5397948, -1.0696439, 3.0765905, -3.7007079, 2.6094387
4: -0.7333497, 1.9524517, -1.5925647, 3.2696197, -4.0029693, 3.5450163

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7585397, upper bound: 2.7585397
time: 0.36 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7585397, upper bound: 2.7585397
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.4678571, 2.6346989, -0.1985434, 1.3264405, -1.7942975, 2.8332424
1: -0.5322849, 3.6304960, -0.2836072, 1.8726876, -2.4049726, 3.9141033
2: -1.2958422, 2.5656059, -0.6658392, 1.3040066, -2.5998487, 3.2314451
3: -1.0763659, 3.1429250, -0.6059550, 1.4355454, -2.5119114, 3.7488799
4: -1.6080190, 3.2646074, -0.6855133, 1.8234990, -3.4315181, 3.9501207

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7603385, upper bound: 2.7623451
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7607458, upper bound: 2.7582674
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.4747180, 2.6634238, -0.2077491, 1.4434946, -1.9182125, 2.8711729
1: -0.5381627, 3.6703260, -0.3018184, 2.0486817, -2.5868444, 3.9721444
2: -1.3116175, 2.5915504, -0.7286727, 1.4051616, -2.7167792, 3.3202231
3: -1.0878518, 3.1816847, -0.6323433, 1.5639632, -2.6518149, 3.8140280
4: -1.6326261, 3.2974410, -0.7463270, 1.9729867, -3.6056128, 4.0437679

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7676421, upper bound: 2.7670926
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582861, upper bound: 2.7575392
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.4693758, 2.6483397, -0.4250804, 2.5852566, -3.0546324, 3.0734200
1: -0.5349588, 3.6501329, -0.5211089, 3.5761673, -4.1111259, 4.1712418
2: -1.3044500, 2.5751491, -1.2865007, 2.4982405, -3.8026905, 3.8616498
3: -1.0812970, 3.1544392, -1.0481679, 3.0023003, -4.0835972, 4.2026072
4: -1.6183817, 3.2837849, -1.5452410, 3.2716055, -4.8899870, 4.8290257

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801148, upper bound: 2.7803173
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801148, upper bound: 2.7804438
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.4747180, 2.6634238, -0.4047078, 2.4600189, -2.9347367, 3.0681317
1: -0.5381627, 3.6703260, -0.4967405, 3.4003437, -3.9385064, 4.1670666
2: -1.3116175, 2.5915504, -1.2186558, 2.3767066, -3.6883240, 3.8102062
3: -1.0878518, 3.1816847, -1.0018195, 2.8326006, -3.9204524, 4.1835041
4: -1.6326261, 3.2974410, -1.4532027, 3.1177409, -4.7503672, 4.7506437

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7804438
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.31 seconds
IS_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7472162, upper bound: 2.7607231
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7665200, upper bound: 2.7682225
IS_A2_A1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7623451, upper bound: 2.7603385
IS_A2_A1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7582675, upper bound: 2.7607458
IS_A2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7623451, upper bound: 2.7603385
IS_A2_A1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7582675, upper bound: 2.7607458
IS_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7585397, upper bound: 2.7585397
IS_A2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7585397, upper bound: 2.7585397
IS_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7603385, upper bound: 2.7623451
IS_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7607458, upper bound: 2.7582674
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7676421, upper bound: 2.7670926
IS_A2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7582861, upper bound: 2.7575392
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7801148, upper bound: 2.7803173
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7801148, upper bound: 2.7804438
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.31
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7804438

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3660870, 2.1625853, -0.3294576, 2.1785727, -2.5446596, 2.4920430
1: -0.4462218, 2.9898553, -0.4364458, 3.0170102, -3.4632320, 3.4263010
2: -1.0902050, 2.0946541, -1.0653431, 2.1111798, -3.2013848, 3.1599972
3: -0.9038171, 2.4661355, -0.8834851, 2.4592609, -3.3630781, 3.3496206
4: -1.2328745, 2.7640190, -1.2130371, 2.8017516, -4.0346260, 3.9770560

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7601399, upper bound: 2.7605830
time: 0.39 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535570, upper bound: 2.7594556
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4560012, 2.5703564, -0.2077491, 1.4434946, -1.8994957, 2.7781055
1: -0.5192766, 3.5445309, -0.3018184, 2.0486817, -2.5679584, 3.8463492
2: -1.2632630, 2.5034189, -0.7286727, 1.4051616, -2.6684246, 3.2320917
3: -1.0513798, 3.0557258, -0.6323433, 1.5639632, -2.6153431, 3.6880691
4: -1.5489410, 3.1905828, -0.7463270, 1.9729867, -3.5219278, 3.9369097

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7669801, upper bound: 2.7670926
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7669801, upper bound: 2.7664752
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4169641, 2.5576580, -0.4250804, 2.5852566, -3.0022206, 2.9827385
1: -0.5152790, 3.5381331, -0.5211089, 3.5761673, -4.0914464, 4.0592422
2: -1.2717404, 2.4722149, -1.2865007, 2.4982405, -3.7699809, 3.7587156
3: -1.0365596, 2.9613614, -1.0481679, 3.0023003, -4.0388598, 4.0095291
4: -1.5201240, 3.2440412, -1.5452410, 3.2716055, -4.7917295, 4.7892823

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7766139
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3969426, 2.4339795, -0.4250804, 2.5852566, -2.9821992, 2.8590598
1: -0.4911667, 3.3645411, -0.5211089, 3.5761673, -4.0673342, 3.8856499
2: -1.2047796, 2.3536205, -1.2865007, 2.4982405, -3.7030201, 3.6401212
3: -0.9906334, 2.7955449, -1.0481679, 3.0023003, -3.9929338, 3.8437128
4: -1.4297944, 3.0918128, -1.5452410, 3.2716055, -4.7013998, 4.6370540

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7801972
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4169641, 2.5576580, -0.4047078, 2.4600189, -2.8769829, 2.9623659
1: -0.5152790, 3.5381331, -0.4967405, 3.4003437, -3.9156227, 4.0348735
2: -1.2717404, 2.4722149, -1.2186558, 2.3767066, -3.6484470, 3.6908708
3: -1.0365596, 2.9613614, -1.0018195, 2.8326006, -3.8691602, 3.9631810
4: -1.5201240, 3.2440412, -1.4532027, 3.1177409, -4.6378651, 4.6972437

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7799880
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3969426, 2.4339795, -0.4047078, 2.4600189, -2.8569615, 2.8386874
1: -0.4911667, 3.3645411, -0.4967405, 3.4003437, -3.8915102, 3.8612814
2: -1.2047796, 2.3536205, -1.2186558, 2.3767066, -3.5814862, 3.5722764
3: -0.9906334, 2.7955449, -1.0018195, 2.8326006, -3.8232341, 3.7973642
4: -1.4297944, 3.0918128, -1.4532027, 3.1177409, -4.5475354, 4.5450153

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7801145
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145
time: 0.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.31 seconds
IS_A1_A2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7601399, upper bound: 2.7605830
IS_A1_A2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7535570, upper bound: 2.7594556
IS_A2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7669801, upper bound: 2.7670926
IS_A2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7669801, upper bound: 2.7664752
IS_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7766139
IS_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
IS_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7801972
IS_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
IS_A2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7799880
IS_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
IS_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7801145
IS_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145

## BFS IS instance: IS_A2_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3363741, 2.0819378, -0.2077491, 1.4434946, -1.7798687, 2.2896869
1: -0.4204853, 2.8493781, -0.3018184, 2.0486817, -2.4691670, 3.1511965
2: -0.9899735, 2.0512018, -0.7286727, 1.4051616, -2.3951352, 2.7798746
3: -0.8594006, 2.3874710, -0.6323433, 1.5639632, -2.4233637, 3.0198143
4: -1.1354136, 2.6239755, -0.7463270, 1.9729867, -3.1084003, 3.3703027

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7642129, upper bound: 2.7575121
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7670926
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3919958, 2.3270078, -0.2077491, 1.4434946, -1.8354905, 2.5347569
1: -0.4682860, 3.2111483, -0.3018184, 2.0486817, -2.5169678, 3.5129666
2: -1.1296155, 2.2633841, -0.7286727, 1.4051616, -2.5347772, 2.9920568
3: -0.9520610, 2.7149534, -0.6323433, 1.5639632, -2.5160241, 3.3472967
4: -1.3399754, 2.9089878, -0.7463270, 1.9729867, -3.3129621, 3.6553149

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7607424, upper bound: 2.7664752
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7607424, upper bound: 2.7664752
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.3031371, 2.0717506, -0.4180982, 2.5564816, -2.8596187, 2.4898489
1: -0.4158132, 2.8628929, -0.5150529, 3.5361929, -3.9520061, 3.3779459
2: -1.0028709, 2.0236433, -1.2703979, 2.4713652, -3.4742360, 3.2940412
3: -0.8422414, 2.3150473, -1.0363258, 2.9618826, -3.8041239, 3.3513732
4: -1.1176394, 2.6857204, -1.5202909, 3.2385457, -4.3561850, 4.2060113

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.4250804, 2.5852566, -2.9518538, 2.7507248
1: -0.4696044, 3.2193704, -0.5211089, 3.5761673, -4.0457716, 3.7404792
2: -1.1476570, 2.2576227, -1.2865007, 2.4982405, -3.6458974, 3.5441234
3: -0.9491717, 2.6523957, -1.0481679, 3.0023003, -3.9514718, 3.7005637
4: -1.3314362, 2.9754171, -1.5452410, 3.2716055, -4.6030416, 4.5206580

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3905433, 2.4056513, -0.3076048, 2.0884678, -2.4790111, 2.7132561
1: -0.4853692, 3.3252378, -0.4196330, 2.8851950, -3.3705642, 3.7448707
2: -1.1892383, 2.3280849, -1.0115439, 2.0395269, -3.2287652, 3.3396287
3: -0.9793400, 2.7577019, -0.8498347, 2.3376288, -3.3169689, 3.6075366
4: -1.4058888, 3.0592592, -1.1316509, 2.7040546, -4.1099434, 4.1909103

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797972, upper bound: 2.7766884
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782096, upper bound: 2.7785847
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3969426, 2.4339795, -0.3753327, 2.3569636, -2.7539062, 2.8093123
1: -0.4911667, 3.3645411, -0.4759032, 3.2600689, -3.7512355, 3.8404441
2: -1.2047796, 2.3536205, -1.1634135, 2.2862463, -3.4910259, 3.5170341
3: -0.9906334, 2.7955449, -0.9616026, 2.6963146, -3.6869478, 3.7571473
4: -1.4297944, 3.0918128, -1.3583037, 3.0054598, -4.4352541, 4.4501166

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697371, upper bound: 2.7767273
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697371, upper bound: 2.7767273
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3031371, 2.0717506, -0.3983519, 2.4315677, -2.7347047, 2.4701025
1: -0.4158132, 2.8628929, -0.4909672, 3.3609734, -3.7767866, 3.3538601
2: -1.0028709, 2.0236433, -1.2031627, 2.3510561, -3.3539271, 3.2268059
3: -0.8422414, 2.3150473, -0.9905775, 2.7944198, -3.6366611, 3.3056247
4: -1.1176394, 2.6857204, -1.4293740, 3.0850222, -4.2026615, 4.1150942

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796460, upper bound: 2.7754017
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791716, upper bound: 2.7779715
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.4047078, 2.4600189, -2.8266160, 2.7303524
1: -0.4696044, 3.2193704, -0.4967405, 3.4003437, -3.8699479, 3.7161107
2: -1.1476570, 2.2576227, -1.2186558, 2.3767066, -3.5243635, 3.4762785
3: -0.9491717, 2.6523957, -1.0018195, 2.8326006, -3.7817721, 3.6542153
4: -1.3314362, 2.9754171, -1.4532027, 3.1177409, -4.4491768, 4.4286199

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724738, upper bound: 2.7664013
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724738, upper bound: 2.7664013
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.2960344, 1.9689381, -0.3983519, 2.4315677, -2.7276020, 2.3672900
1: -0.4008421, 2.7146063, -0.4909672, 3.3609734, -3.7618155, 3.2055736
2: -0.9505634, 1.9361818, -1.2031627, 2.3510561, -3.3016195, 3.1393445
3: -0.8154755, 2.2027259, -0.9905775, 2.7944198, -3.6098952, 3.1933033
4: -1.0562243, 2.5568314, -1.4293740, 3.0850222, -4.1412468, 3.9862053

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7753306
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7783336
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.4047078, 2.4600189, -2.7913632, 2.5927243
1: -0.4382458, 3.0278041, -0.4967405, 3.4003437, -3.8385894, 3.5245447
2: -1.0667892, 2.1236391, -1.2186558, 2.3767066, -3.4434958, 3.3422949
3: -0.8872414, 2.4708958, -1.0018195, 2.8326006, -3.7198420, 3.4727154
4: -1.2173010, 2.8092000, -1.4532027, 3.1177409, -4.3350420, 4.2624025

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7801145
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7801145
time: 0.39 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.32 seconds
IS_A2_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7642129, upper bound: 2.7575121
IS_A2_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7670926
IS_A2_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7607424, upper bound: 2.7664752
IS_A2_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7607424, upper bound: 2.7664752
IS_A2_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
IS_A2_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
IS_A2_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
IS_A2_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
IS_A2_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7797972, upper bound: 2.7766884
IS_A2_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7782096, upper bound: 2.7785847
IS_A2_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7697371, upper bound: 2.7767273
IS_A2_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7697371, upper bound: 2.7767273
IS_A2_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7796460, upper bound: 2.7754017
IS_A2_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7791716, upper bound: 2.7779715
IS_A2_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7724738, upper bound: 2.7664013
IS_A2_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7724738, upper bound: 2.7664013
IS_A2_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7753306
IS_A2_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7783336
IS_A2_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7801145
IS_A2_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7801145

## BFS IS instance: IS_A2_A2_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2888264, 1.9945775, -0.2058513, 1.4355171, -1.7243435, 2.2004290
1: -0.4000432, 2.7575843, -0.3001972, 2.0390713, -2.4391146, 3.0577817
2: -0.9618342, 1.9491395, -0.7249167, 1.3978140, -2.3596482, 2.6740561
3: -0.8125226, 2.2124579, -0.6291612, 1.5546881, -2.3672109, 2.8416190
4: -1.0490496, 2.5925679, -0.7415119, 1.9646916, -3.0137413, 3.3340797

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7596451, upper bound: 2.7629310
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7613484, upper bound: 2.7607492
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2796247, 1.8826946, -0.2077491, 1.4434946, -1.7231193, 2.0904436
1: -0.3829975, 2.5972061, -0.3018184, 2.0486817, -2.4316792, 2.8990245
2: -0.9045649, 1.8533102, -0.7286727, 1.4051616, -2.3097265, 2.5819831
3: -0.7816354, 2.0899224, -0.6323433, 1.5639632, -2.3455987, 2.7222657
4: -0.9799299, 2.4523220, -0.7463270, 1.9729867, -2.9529166, 3.1986489

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7596456, upper bound: 2.7629424
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7603707, upper bound: 2.7593848
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3919958, 2.3270078, -0.1948414, 1.3757203, -1.7677162, 2.5218492
1: -0.4682860, 3.2111483, -0.2873552, 1.9522393, -2.4205253, 3.4985034
2: -1.1296155, 2.2633841, -0.6900675, 1.3430049, -2.4726205, 2.9534516
3: -0.9520610, 2.7149534, -0.6058655, 1.4762542, -2.4283152, 3.3208189
4: -1.3399754, 2.9089878, -0.6963954, 1.8879433, -3.2279186, 3.6053832

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7592310, upper bound: 2.7444432
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7666993, upper bound: 2.7664752
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3919958, 2.3270078, -0.2394581, 1.5498766, -1.9418724, 2.5664659
1: -0.4682860, 3.2111483, -0.3208745, 2.1582851, -2.6265712, 3.5320227
2: -1.1296155, 2.2633841, -0.7619050, 1.5200684, -2.6496840, 3.0252891
3: -0.9520610, 2.7149534, -0.6721208, 1.6818315, -2.6338925, 3.3870742
4: -1.3399754, 2.9089878, -0.7915561, 2.0563183, -3.3962936, 3.7005439

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7605116, upper bound: 2.7622441
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7608100, upper bound: 2.7582420
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3031371, 2.0717506, -0.3076048, 2.0884678, -2.3916049, 2.3793554
1: -0.4158132, 2.8628929, -0.4196330, 2.8851950, -3.3010082, 3.2825260
2: -1.0028709, 2.0236433, -1.0115439, 2.0395269, -3.0423980, 3.0351872
3: -0.8422414, 2.3150473, -0.8498347, 2.3376288, -3.1798701, 3.1648819
4: -1.1176394, 2.6857204, -1.1316509, 2.7040546, -3.8216939, 3.8173714

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7766139
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7694991, upper bound: 2.7755713
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3031371, 2.0717506, -0.3753327, 2.3569636, -2.6601007, 2.4470835
1: -0.4158132, 2.8628929, -0.4759032, 3.2600689, -3.6758821, 3.3387961
2: -1.0028709, 2.0236433, -1.1634135, 2.2862463, -3.2891173, 3.1870568
3: -0.8422414, 2.3150473, -0.9616026, 2.6963146, -3.5385561, 3.2766500
4: -1.1176394, 2.6857204, -1.3583037, 3.0054598, -4.1230993, 4.0440240

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7762832
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748881
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3076048, 2.0884678, -2.4550650, 2.6332493
1: -0.4696044, 3.2193704, -0.4196330, 2.8851950, -3.3547993, 3.6390033
2: -1.1476570, 2.2576227, -1.0115439, 2.0395269, -3.1871839, 3.2691665
3: -0.9491717, 2.6523957, -0.8498347, 2.3376288, -3.2868004, 3.5022304
4: -1.3314362, 2.9754171, -1.1316509, 2.7040546, -4.0354910, 4.1070681

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7662900
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3753327, 2.3569636, -2.7235608, 2.7009773
1: -0.4696044, 3.2193704, -0.4759032, 3.2600689, -3.7296734, 3.6952734
2: -1.1476570, 2.2576227, -1.1634135, 2.2862463, -3.4339032, 3.4210362
3: -0.9491717, 2.6523957, -0.9616026, 2.6963146, -3.6454864, 3.6139984
4: -1.3314362, 2.9754171, -1.3583037, 3.0054598, -4.3368959, 4.3337207

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7662900
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.3905433, 2.4056513, -0.2973586, 2.0464444, -2.4369876, 2.7030098
1: -0.4853692, 3.3252378, -0.4104910, 2.8293266, -3.3146958, 3.7357287
2: -1.1892383, 2.3280849, -0.9908106, 1.9982010, -3.1874394, 3.3188956
3: -0.9793400, 2.7577019, -0.8320768, 2.2798955, -3.2592354, 3.5897787
4: -1.4058888, 3.0592592, -1.0959523, 2.6567075, -4.0625963, 4.1552114

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785469, upper bound: 2.7741990
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.3905433, 2.4056513, -0.3086333, 2.0736451, -2.4641883, 2.7142847
1: -0.4853692, 3.3252378, -0.4168339, 2.8735497, -3.3589189, 3.7420716
2: -1.1892383, 2.3280849, -1.0126864, 2.0177553, -3.2069936, 3.3407712
3: -0.9793400, 2.7577019, -0.8451283, 2.3225791, -3.3019190, 3.6028302
4: -1.4058888, 3.0592592, -1.1221292, 2.6829336, -4.0888224, 4.1813884

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7754919
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2960344, 1.9689381, -0.3753327, 2.3569636, -2.6529980, 2.3442707
1: -0.4008421, 2.7146063, -0.4759032, 3.2600689, -3.6609111, 3.1905093
2: -0.9505634, 1.9361818, -1.1634135, 2.2862463, -3.2368097, 3.0995953
3: -0.8154755, 2.2027259, -0.9616026, 2.6963146, -3.5117900, 3.1643286
4: -1.0562243, 2.5568314, -1.3583037, 3.0054598, -4.0616841, 3.9151349

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7696262, upper bound: 2.7746064
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3753327, 2.3569636, -2.6883078, 2.5633492
1: -0.4382458, 3.0278041, -0.4759032, 3.2600689, -3.6983147, 3.5037074
2: -1.0667892, 2.1236391, -1.1634135, 2.2862463, -3.3530354, 3.2870526
3: -0.8872414, 2.4708958, -0.9616026, 2.6963146, -3.5835559, 3.4324985
4: -1.2173010, 2.8092000, -1.3583037, 3.0054598, -4.2227607, 4.1675038

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697371, upper bound: 2.7749430
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7696262, upper bound: 2.7746064
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2985941, 2.0480504, -0.3472360, 2.1678371, -2.4664311, 2.3952863
1: -0.4106064, 2.8317814, -0.4398950, 3.0290620, -3.4396684, 3.2716763
2: -0.9917817, 1.9997627, -1.1072288, 2.0700316, -3.0618134, 3.1069913
3: -0.8321357, 2.2847333, -0.8910263, 2.4721384, -3.3042741, 3.1757596
4: -1.0988803, 2.6568732, -1.2669461, 2.7950141, -3.8938944, 3.9238193

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791874, upper bound: 2.7749050
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7754015
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7751592
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3031371, 2.0717506, -0.3673147, 2.3345752, -2.6377122, 2.4390655
1: -0.4158132, 2.8628929, -0.4687061, 3.2309842, -3.6467974, 3.3315990
2: -1.0028709, 2.0236433, -1.1555011, 2.2572024, -3.2600732, 3.1791444
3: -0.8422414, 2.3150473, -0.9444780, 2.6583376, -3.5005789, 3.2595253
4: -1.1176394, 2.6857204, -1.3431969, 2.9884486, -4.1060882, 4.0289173

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7779715
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758047, upper bound: 2.7754159
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3022718, 1.9925733, -2.3591704, 2.6279163
1: -0.4696044, 3.2193704, -0.4058715, 2.7465084, -3.2161126, 3.6252418
2: -1.1476570, 2.2576227, -0.9626346, 1.9593803, -3.1070373, 3.2202573
3: -0.9491717, 2.6523957, -0.8254330, 2.2351263, -3.1842980, 3.4778287
4: -1.3314362, 2.9754171, -1.0756483, 2.5819404, -3.9133766, 4.0510654

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7721467, upper bound: 2.7662918
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3389214, 2.2148643, -2.5814614, 2.6645660
1: -0.4696044, 3.2193704, -0.4439855, 3.0646944, -3.5342989, 3.6633558
2: -1.1476570, 2.2576227, -1.0810840, 2.1477108, -3.2953677, 3.3387067
3: -0.9491717, 2.6523957, -0.8988513, 2.5088573, -3.4580288, 3.5512471
4: -1.3314362, 2.9754171, -1.2414910, 2.8358724, -4.1673088, 4.2169080

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7721467, upper bound: 2.7662918
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2914144, 1.9445134, -0.3472360, 2.1678371, -2.4592514, 2.2917495
1: -0.3954931, 2.6824191, -0.4398950, 3.0290620, -3.4245553, 3.1223140
2: -0.9390244, 1.9117186, -1.1072288, 2.0700316, -3.0090561, 3.0189474
3: -0.8052138, 2.1717138, -0.8910263, 2.4721384, -3.2773523, 3.0627401
4: -1.0368246, 2.5272179, -1.2669461, 2.7950141, -3.8318386, 3.7941639

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7751748
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2960344, 1.9689381, -0.3673147, 2.3345752, -2.6306095, 2.3362527
1: -0.4008421, 2.7146063, -0.4687061, 3.2309842, -3.6318264, 3.1833124
2: -0.9505634, 1.9361818, -1.1555011, 2.2572024, -3.2077658, 3.0916829
3: -0.8154755, 2.2027259, -0.9444780, 2.6583376, -3.4738131, 3.1472039
4: -1.0562243, 2.5568314, -1.3431969, 2.9884486, -4.0446730, 3.9000282

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7754416
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3022718, 1.9925733, -2.3239174, 2.4902883
1: -0.4382458, 3.0278041, -0.4058715, 2.7465084, -3.1847541, 3.4336758
2: -1.0667892, 2.1236391, -0.9626346, 1.9593803, -3.0261693, 3.0862737
3: -0.8872414, 2.4708958, -0.8254330, 2.2351263, -3.1223676, 3.2963288
4: -1.2173010, 2.8092000, -1.0756483, 2.5819404, -3.7992415, 3.8848484

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7767387
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781803, upper bound: 2.7785020
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3389214, 2.2148643, -2.5462084, 2.5269380
1: -0.4382458, 3.0278041, -0.4439855, 3.0646944, -3.5029402, 3.4717896
2: -1.0667892, 2.1236391, -1.0810840, 2.1477108, -3.2145000, 3.2047231
3: -0.8872414, 2.4708958, -0.8988513, 2.5088573, -3.3960986, 3.3697472
4: -1.2173010, 2.8092000, -1.2414910, 2.8358724, -4.0531735, 4.0506911

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7767387
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781803, upper bound: 2.7785020
time: 0.41 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.76 seconds
IS_A2_A2_B1_B2_A1_A1_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7596451, upper bound: 2.7629310
IS_A2_A2_B1_B2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7613484, upper bound: 2.7607492
IS_A2_A2_B1_B2_A1_A1_A2_A1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7596456, upper bound: 2.7629424
IS_A2_A2_B1_B2_A1_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7603707, upper bound: 2.7593848
IS_A2_A2_B1_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7592310, upper bound: 2.7444432
IS_A2_A2_B1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7666993, upper bound: 2.7664752
IS_A2_A2_B1_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7605116, upper bound: 2.7622441
IS_A2_A2_B1_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7608100, upper bound: 2.7582420
IS_A2_A2_B2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7698198, upper bound: 2.7766139
IS_A2_A2_B2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7694991, upper bound: 2.7755713
IS_A2_A2_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7762832
IS_A2_A2_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748881
IS_A2_A2_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7662900
IS_A2_A2_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
IS_A2_A2_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7662900
IS_A2_A2_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
IS_A2_A2_B2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
IS_A2_A2_B2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7785469, upper bound: 2.7741990
IS_A2_A2_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
IS_A2_A2_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7754919
IS_A2_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
IS_A2_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7696262, upper bound: 2.7746064
IS_A2_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7697371, upper bound: 2.7749430
IS_A2_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7696262, upper bound: 2.7746064
IS_A2_A2_B2_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7754015
IS_A2_A2_B2_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7751592
IS_A2_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7779715
IS_A2_A2_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7758047, upper bound: 2.7754159
IS_A2_A2_B2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7721467, upper bound: 2.7662918
IS_A2_A2_B2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
IS_A2_A2_B2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7721467, upper bound: 2.7662918
IS_A2_A2_B2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
IS_A2_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
IS_A2_A2_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7751748
IS_A2_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
IS_A2_A2_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7754416
IS_A2_A2_B2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7767387
IS_A2_A2_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7781803, upper bound: 2.7785020
IS_A2_A2_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7767387
IS_A2_A2_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.76
Output dim: 0, lower bound: -2.7781803, upper bound: 2.7785020

## BFS IS instance: IS_A2_A2_B1_B2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3154575, 2.1055789, -0.1948414, 1.3757203, -1.6911778, 2.3004203
1: -0.4214576, 2.9155774, -0.2873552, 1.9522393, -2.3736968, 3.2029326
2: -1.0235951, 2.0444477, -0.6900675, 1.3430049, -2.3666000, 2.7345152
3: -0.8554188, 2.3600154, -0.6058655, 1.4762542, -2.3316731, 2.9658809
4: -1.1441643, 2.7125952, -0.6963954, 1.8879433, -3.0321076, 3.4089906

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7627615, upper bound: 2.7620504
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7685367, upper bound: 2.7664585
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2806621, 1.8875258, -0.3029913, 2.0643520, -2.3450141, 2.1905169
1: -0.3806360, 2.6402259, -0.4143727, 2.8535423, -3.2341783, 3.0545986
2: -0.9465249, 1.8131994, -1.0003470, 2.0152473, -2.9617722, 2.8135464
3: -0.7766682, 2.0958865, -0.8396406, 2.3067970, -3.0834651, 2.9355271
4: -1.0165755, 2.4664505, -1.1127429, 2.6749473, -3.6915226, 3.5791934

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789902, upper bound: 2.7770080
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760796, upper bound: 2.7785684
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2856687, 2.0045700, -0.3076048, 2.0884678, -2.3741364, 2.3121748
1: -0.4007463, 2.7735984, -0.4196330, 2.8851950, -3.2859414, 3.1932316
2: -0.9694705, 1.9582894, -1.0115439, 2.0395269, -3.0089974, 2.9698334
3: -0.8125409, 2.2225275, -0.8498347, 2.3376288, -3.1501698, 3.0723622
4: -1.0605392, 2.6134980, -1.1316509, 2.7040546, -3.7645938, 3.7451489

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786695, upper bound: 2.7741536
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757589, upper bound: 2.7754662
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2909694, 2.0227432, -0.3753327, 2.3569636, -2.6479330, 2.3980761
1: -0.4052326, 2.7972746, -0.4759032, 3.2600689, -3.6653016, 3.2731776
2: -0.9782555, 1.9761380, -1.1634135, 2.2862463, -3.2645018, 3.1395516
3: -0.8216132, 2.2480628, -0.9616026, 2.6963146, -3.5179276, 3.2096653
4: -1.0757260, 2.6311049, -1.3583037, 3.0054598, -4.0811858, 3.9894085

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7762832
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7752422
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3042246, 2.0575032, -0.3753327, 2.3569636, -2.6611881, 2.4328361
1: -0.4130379, 2.8520617, -0.4759032, 3.2600689, -3.6731067, 3.3279648
2: -1.0041143, 2.0024467, -1.1634135, 2.2862463, -3.2903605, 3.1658602
3: -0.8375547, 2.3009794, -0.9616026, 2.6963146, -3.5338693, 3.2625818
4: -1.1082883, 2.6651578, -1.3583037, 3.0054598, -4.1137481, 4.0234613

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748881
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693882, upper bound: 2.7723316
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.2973586, 2.0464444, -2.4130416, 2.6230030
1: -0.4696044, 3.2193704, -0.4104910, 2.8293266, -3.2989311, 3.6298614
2: -1.1476570, 2.2576227, -0.9908106, 1.9982010, -3.1458580, 3.2484334
3: -0.9491717, 2.6523957, -0.8320768, 2.2798955, -3.2290673, 3.4844725
4: -1.3314362, 2.9754171, -1.0959523, 2.6567075, -3.9881437, 4.0713692

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7544898, upper bound: 2.7533442
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3086333, 2.0736451, -2.4402423, 2.6342778
1: -0.4696044, 3.2193704, -0.4168339, 2.8735497, -3.3431540, 3.6362042
2: -1.1476570, 2.2576227, -1.0126864, 2.0177553, -3.1654124, 3.2703090
3: -0.9491717, 2.6523957, -0.8451283, 2.3225791, -3.2717509, 3.4975240
4: -1.3314362, 2.9754171, -1.1221292, 2.6829336, -4.0143700, 4.0975466

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515212, upper bound: 2.7513427
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3629161, 2.3064585, -2.6730556, 2.6885605
1: -0.4696044, 3.2193704, -0.4652137, 3.1939573, -3.6635618, 3.6845841
2: -1.1476570, 2.2576227, -1.1392360, 2.2357109, -3.3833680, 3.3968587
3: -0.9491717, 2.6523957, -0.9405786, 2.6256416, -3.5748134, 3.5929744
4: -1.3314362, 2.9754171, -1.3152579, 2.9512329, -4.2826691, 4.2906752

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7533442
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7532613
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3708772, 2.3199959, -2.6865931, 2.6965218
1: -0.4696044, 3.2193704, -0.4697720, 3.2195220, -3.6891265, 3.6891422
2: -1.1476570, 2.2576227, -1.1537372, 2.2423575, -3.3900146, 3.4113598
3: -0.9491717, 2.6523957, -0.9502540, 2.6516135, -3.6007853, 3.6026497
4: -1.3314362, 2.9754171, -1.3322339, 2.9660292, -4.2974653, 4.3076510

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515082, upper bound: 2.7513427
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7513427, upper bound: 2.7513427
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3418609, 2.1482449, -0.2923267, 2.0209284, -2.3627894, 2.4405715
1: -0.4356224, 3.0019400, -0.4048963, 2.7956667, -3.2312891, 3.4068363
2: -1.0962515, 2.0526242, -0.9786599, 1.9727707, -3.0690222, 3.0312841
3: -0.8824854, 2.4447157, -0.8212299, 2.2473242, -3.1298096, 3.2659457
4: -1.2493535, 2.7739680, -1.0755645, 2.6259067, -3.8752604, 3.8495326

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3604465, 2.3102453, -0.2973586, 2.0464444, -2.4068909, 2.6076038
1: -0.4636661, 3.1975977, -0.4104910, 2.8293266, -3.2929926, 3.6080887
2: -1.1427677, 2.2354245, -0.9908106, 1.9982010, -3.1409688, 3.2262352
3: -0.9344901, 2.6241479, -0.8320768, 2.2798955, -3.2143855, 3.4562247
4: -1.3218516, 2.9640663, -1.0959523, 2.6567075, -3.9785590, 4.0600185

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7493137, upper bound: 2.7440268
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7528209, upper bound: 2.7507118
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.3842160, 2.3764737, -0.2889379, 1.9049499, -2.2891660, 2.6654115
1: -0.4791470, 3.2862048, -0.3853487, 2.6697907, -3.1489377, 3.6715536
2: -1.1750243, 2.2996936, -0.9618609, 1.8210847, -2.9961090, 3.2615545
3: -0.9672000, 2.7192271, -0.7863698, 2.1205049, -3.0877049, 3.5055969
4: -1.3817974, 3.0258915, -1.0324315, 2.4797454, -3.8615427, 4.0583229

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752051
time: 0.46 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752051
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.3905433, 2.4056513, -0.2890179, 1.9990376, -2.3895810, 2.6946692
1: -0.4853692, 3.3252378, -0.4002408, 2.7723556, -3.2577248, 3.7254786
2: -1.1892383, 2.3280849, -0.9748017, 1.9446510, -3.1338892, 3.3028865
3: -0.9793400, 2.7577019, -0.8124224, 2.2182291, -3.1975689, 3.5701241
4: -1.4058888, 3.0592592, -1.0574362, 2.6030040, -4.0088930, 4.1166954

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747756, upper bound: 2.7754919
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747756, upper bound: 2.7747319
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2852379, 1.9261826, -0.3753327, 2.3569636, -2.6422014, 2.3015153
1: -0.3915178, 2.6576438, -0.4759032, 3.2600689, -3.6515868, 3.1335468
2: -0.9292547, 1.8941118, -1.1634135, 2.2862463, -3.2155008, 3.0575252
3: -0.7973023, 2.1429307, -0.9616026, 2.6963146, -3.4936168, 3.1045332
4: -1.0195756, 2.5085783, -1.3583037, 3.0054598, -4.0250354, 3.8668818

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7756621
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3753327, 2.3569636, -2.6501577, 2.3316183
1: -0.3966470, 2.7066140, -0.4759032, 3.2600689, -3.6567159, 3.1825171
2: -0.9499824, 1.9157126, -1.1634135, 2.2862463, -3.2362287, 3.0791261
3: -0.8073892, 2.1858242, -0.9616026, 2.6963146, -3.5037038, 3.1474266
4: -1.0437701, 2.5382841, -1.3583037, 3.0054598, -4.0492296, 3.8965878

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
time: 0.46 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3629161, 2.3064585, -2.6378026, 2.5509324
1: -0.4382458, 3.0278041, -0.4652137, 3.1939573, -3.6322031, 3.4930177
2: -1.0667892, 2.1236391, -1.1392360, 2.2357109, -3.3025000, 3.2628751
3: -0.8872414, 2.4708958, -0.9405786, 2.6256416, -3.5128829, 3.4114745
4: -1.2173010, 2.8092000, -1.3152579, 2.9512329, -4.1685338, 4.1244578

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7495325, upper bound: 2.7507908
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3708772, 2.3199959, -2.6513400, 2.5588937
1: -0.4382458, 3.0278041, -0.4697720, 3.2195220, -3.6577678, 3.4975762
2: -1.0667892, 2.1236391, -1.1537372, 2.2423575, -3.3091466, 3.2773762
3: -0.8872414, 2.4708958, -0.9502540, 2.6516135, -3.5388548, 3.4211497
4: -1.2173010, 2.8092000, -1.3322339, 2.9660292, -4.1833301, 4.1414337

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7682916, upper bound: 2.7746064
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7682916, upper bound: 2.7746064
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2859721, 1.9972415, -0.3472360, 2.1678371, -2.4538093, 2.3444774
1: -0.3996190, 2.7636065, -0.4398950, 3.0290620, -3.4286809, 3.2035015
2: -0.9661374, 1.9507186, -1.1072288, 2.0700316, -3.0361691, 3.0579474
3: -0.8107526, 2.2156317, -0.8910263, 2.4721384, -3.2828910, 3.1066580
4: -1.0554264, 2.6002364, -1.2669461, 2.7950141, -3.8504405, 3.8671825

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734401, upper bound: 2.7749052
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7746569
time: 0.46 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7751593
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2997325, 2.0341167, -0.3472360, 2.1678371, -2.4675696, 2.3813527
1: -0.4078606, 2.8215098, -0.4398950, 3.0290620, -3.4369226, 3.2614048
2: -0.9930655, 1.9790092, -1.1072288, 2.0700316, -3.0630970, 3.0862379
3: -0.8275511, 2.2711914, -0.8910263, 2.4721384, -3.2996895, 3.1622176
4: -1.0896422, 2.6366882, -1.2669461, 2.7950141, -3.8846564, 3.9036343

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7749052
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7746569
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7751593
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2909694, 2.0227432, -0.3673147, 2.3345752, -2.6255445, 2.3900580
1: -0.4052326, 2.7972746, -0.4687061, 3.2309842, -3.6362169, 3.2659807
2: -0.9782555, 1.9761380, -1.1555011, 2.2572024, -3.2354579, 3.1316390
3: -0.8216132, 2.2480628, -0.9444780, 2.6583376, -3.4799509, 3.1925409
4: -1.0757260, 2.6311049, -1.3431969, 2.9884486, -4.0641747, 3.9743018

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7779715
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7754015
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3042246, 2.0575032, -0.3673147, 2.3345752, -2.6387997, 2.4248180
1: -0.4130379, 2.8520617, -0.4687061, 3.2309842, -3.6440220, 3.3207679
2: -1.0041143, 2.0024467, -1.1555011, 2.2572024, -3.2613168, 3.1579478
3: -0.8375547, 2.3009794, -0.9444780, 2.6583376, -3.4958923, 3.2454574
4: -1.1082883, 2.6651578, -1.3431969, 2.9884486, -4.0967369, 4.0083547

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758047, upper bound: 2.7747675
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758047, upper bound: 2.7754159
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.2913533, 1.9494442, -2.3160415, 2.6169977
1: -0.4696044, 3.2193704, -0.3965143, 2.6892331, -3.1588373, 3.6158848
2: -1.1476570, 2.2576227, -0.9412345, 1.9158479, -3.0635049, 3.1988573
3: -0.9491717, 2.6523957, -0.8072340, 2.1748338, -3.1240053, 3.4596298
4: -1.3314362, 2.9754171, -1.0388029, 2.5333719, -3.8648081, 4.0142202

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577854, upper bound: 2.7520659
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575793, upper bound: 2.7518331
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.2971243, 1.9722085, -2.3388057, 2.6227689
1: -0.4696044, 3.2193704, -0.4001294, 2.7280126, -3.1976171, 3.6194997
2: -1.1476570, 2.2576227, -0.9578750, 1.9310058, -3.0786629, 3.2154977
3: -0.9491717, 2.6523957, -0.8142713, 2.2070019, -3.1561737, 3.4666672
4: -1.3314362, 2.9754171, -1.0563898, 2.5555344, -3.8869705, 4.0318069

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3256288, 2.1637757, -2.5303729, 2.6512733
1: -0.4696044, 3.2193704, -0.4324989, 2.9961452, -3.4657497, 3.6518693
2: -1.1476570, 2.2576227, -1.0553246, 2.0967674, -3.2444243, 3.3129473
3: -0.9491717, 2.6523957, -0.8757868, 2.4374104, -3.3865819, 3.5281825
4: -1.3314362, 2.9754171, -1.1962250, 2.7796919, -4.1111279, 4.1716423

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577854, upper bound: 2.7520659
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7572478, upper bound: 2.7518331
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.3351435, 2.1843152, -2.5509124, 2.6607881
1: -0.4696044, 3.2193704, -0.4383101, 3.0287063, -3.4983106, 3.6576805
2: -1.1476570, 2.2576227, -1.0727464, 2.1102271, -3.2578840, 3.3303690
3: -0.9491717, 2.6523957, -0.8880902, 2.4713705, -3.4205422, 3.5404859
4: -1.3314362, 2.9754171, -1.2180777, 2.8008497, -4.1322861, 4.1934948

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7487747, upper bound: 2.7475630
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7479288, upper bound: 2.7475631
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2801493, 1.9003778, -0.3472360, 2.1678371, -2.4479864, 2.2476137
1: -0.3858240, 2.6233618, -0.4398950, 3.0290620, -3.4148860, 3.0632567
2: -0.9167272, 1.8683568, -1.1072288, 2.0700316, -2.9867587, 2.9755855
3: -0.7863454, 2.1098049, -0.8910263, 2.4721384, -3.2584839, 3.0008311
4: -0.9985807, 2.4773648, -1.2669461, 2.7950141, -3.7935948, 3.7443109

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741408
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7751748
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2886297, 1.9324335, -0.3472360, 2.1678371, -2.4564667, 2.2796695
1: -0.3913419, 2.6751680, -0.4398950, 3.0290620, -3.4204040, 3.1150630
2: -0.9385793, 1.8917056, -1.1072288, 2.0700316, -3.0086110, 2.9989343
3: -0.7972261, 2.1550553, -0.8910263, 2.4721384, -3.2693646, 3.0460815
4: -1.0246933, 2.5091178, -1.2669461, 2.7950141, -3.8197074, 3.7760639

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741407
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7751748
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2852379, 1.9261826, -0.3673147, 2.3345752, -2.6198130, 2.2934973
1: -0.3915178, 2.6576438, -0.4687061, 3.2309842, -3.6225021, 3.1263499
2: -0.9292547, 1.8941118, -1.1555011, 2.2572024, -3.1864572, 3.0496130
3: -0.7973023, 2.1429307, -0.9444780, 2.6583376, -3.4556398, 3.0874088
4: -1.0195756, 2.5085783, -1.3431969, 2.9884486, -4.0080242, 3.8517752

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3673147, 2.3345752, -2.6277692, 2.3236003
1: -0.3966470, 2.7066140, -0.4687061, 3.2309842, -3.6276312, 3.1753201
2: -0.9499824, 1.9157126, -1.1555011, 2.2572024, -3.2071848, 3.0712137
3: -0.8073892, 2.1858242, -0.9444780, 2.6583376, -3.4657269, 3.1303022
4: -1.0437701, 2.5382841, -1.3431969, 2.9884486, -4.0322189, 3.8814809

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7742810
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7754416
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.2913533, 1.9494442, -2.2807884, 2.4793696
1: -0.4382458, 3.0278041, -0.3965143, 2.6892331, -3.1274788, 3.4243183
2: -1.0667892, 2.1236391, -0.9412345, 1.9158479, -2.9826369, 3.0648737
3: -0.8872414, 2.4708958, -0.8072340, 2.1748338, -3.0620751, 3.2781298
4: -1.2173010, 2.8092000, -1.0388029, 2.5333719, -3.7506728, 3.8480029

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792407, upper bound: 2.7767387
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788115, upper bound: 2.7742429
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.2971243, 1.9722085, -2.3035526, 2.4851408
1: -0.4382458, 3.0278041, -0.4001294, 2.7280126, -3.1662583, 3.4279335
2: -1.0667892, 2.1236391, -0.9578750, 1.9310058, -2.9977951, 3.0815141
3: -0.8872414, 2.4708958, -0.8142713, 2.2070019, -3.0942433, 3.2851672
4: -1.2173010, 2.8092000, -1.0563898, 2.5555344, -3.7728353, 3.8655899

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760285, upper bound: 2.7782044
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756577, upper bound: 2.7754919
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3256288, 2.1637757, -2.4951200, 2.5136452
1: -0.4382458, 3.0278041, -0.4324989, 2.9961452, -3.4343910, 3.4603031
2: -1.0667892, 2.1236391, -1.0553246, 2.0967674, -3.1635566, 3.1789637
3: -0.8872414, 2.4708958, -0.8757868, 2.4374104, -3.3246517, 3.3466825
4: -1.2173010, 2.8092000, -1.1962250, 2.7796919, -3.9969931, 4.0054250

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532360, upper bound: 2.7581697
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7767387
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3351435, 2.1843152, -2.5156593, 2.5231600
1: -0.4382458, 3.0278041, -0.4383101, 3.0287063, -3.4669521, 3.4661143
2: -1.0667892, 2.1236391, -1.0727464, 2.1102271, -3.1770163, 3.1963854
3: -0.8872414, 2.4708958, -0.8880902, 2.4713705, -3.3586118, 3.3589859
4: -1.2173010, 2.8092000, -1.2180777, 2.8008497, -4.0181508, 4.0272779

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7519489, upper bound: 2.7519506
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781803, upper bound: 2.7785020
time: 0.43 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.64 seconds
IS_A2_A2_B1_B2_A1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7627615, upper bound: 2.7620504
IS_A2_A2_B1_B2_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7685367, upper bound: 2.7664585
IS_A2_A2_B2_B1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7789902, upper bound: 2.7770080
IS_A2_A2_B2_B1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7760796, upper bound: 2.7785684
IS_A2_A2_B2_B1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7786695, upper bound: 2.7741536
IS_A2_A2_B2_B1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7757589, upper bound: 2.7754662
IS_A2_A2_B2_B1_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7762832
IS_A2_A2_B2_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7752422
IS_A2_A2_B2_B1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748881
IS_A2_A2_B2_B1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7693882, upper bound: 2.7723316
IS_A2_A2_B2_B1_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7544898, upper bound: 2.7533442
IS_A2_A2_B2_B1_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
IS_A2_A2_B2_B1_A1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7515212, upper bound: 2.7513427
IS_A2_A2_B2_B1_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
IS_A2_A2_B2_B1_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7533442
IS_A2_A2_B2_B1_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7532613
IS_A2_A2_B2_B1_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7515082, upper bound: 2.7513427
IS_A2_A2_B2_B1_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7513427, upper bound: 2.7513427
IS_A2_A2_B2_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
IS_A2_A2_B2_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
IS_A2_A2_B2_B1_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7493137, upper bound: 2.7440268
IS_A2_A2_B2_B1_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7528209, upper bound: 2.7507118
IS_A2_A2_B2_B1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752051
IS_A2_A2_B2_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752051
IS_A2_A2_B2_B1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7747756, upper bound: 2.7754919
IS_A2_A2_B2_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7747756, upper bound: 2.7747319
IS_A2_A2_B2_B1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
IS_A2_A2_B2_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7756621
IS_A2_A2_B2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
IS_A2_A2_B2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
IS_A2_A2_B2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
IS_A2_A2_B2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7495325, upper bound: 2.7507908
IS_A2_A2_B2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7682916, upper bound: 2.7746064
IS_A2_A2_B2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7682916, upper bound: 2.7746064
IS_A2_A2_B2_B2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7746569
IS_A2_A2_B2_B2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7751593
IS_A2_A2_B2_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7746569
IS_A2_A2_B2_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7751593
IS_A2_A2_B2_B2_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7779715
IS_A2_A2_B2_B2_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7754015
IS_A2_A2_B2_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7758047, upper bound: 2.7747675
IS_A2_A2_B2_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7758047, upper bound: 2.7754159
IS_A2_A2_B2_B2_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7577854, upper bound: 2.7520659
IS_A2_A2_B2_B2_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7575793, upper bound: 2.7518331
IS_A2_A2_B2_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
IS_A2_A2_B2_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
IS_A2_A2_B2_B2_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7577854, upper bound: 2.7520659
IS_A2_A2_B2_B2_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7572478, upper bound: 2.7518331
IS_A2_A2_B2_B2_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7487747, upper bound: 2.7475630
IS_A2_A2_B2_B2_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7479288, upper bound: 2.7475631
IS_A2_A2_B2_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741408
IS_A2_A2_B2_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7751748
IS_A2_A2_B2_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741407
IS_A2_A2_B2_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7751748
IS_A2_A2_B2_B2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
IS_A2_A2_B2_B2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
IS_A2_A2_B2_B2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7742810
IS_A2_A2_B2_B2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7754416
IS_A2_A2_B2_B2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7792407, upper bound: 2.7767387
IS_A2_A2_B2_B2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7788115, upper bound: 2.7742429
IS_A2_A2_B2_B2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7760285, upper bound: 2.7782044
IS_A2_A2_B2_B2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7756577, upper bound: 2.7754919
IS_A2_A2_B2_B2_A2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7532360, upper bound: 2.7581697
IS_A2_A2_B2_B2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7767387
IS_A2_A2_B2_B2_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7519489, upper bound: 2.7519506
IS_A2_A2_B2_B2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.64
Output dim: 0, lower bound: -2.7781803, upper bound: 2.7785020

## BFS IS instance: IS_A2_A2_B1_B2_A1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.3071057, 2.0731227, -0.1948414, 1.3757203, -1.6828260, 2.2679641
1: -0.4145108, 2.8722401, -0.2873552, 1.9522393, -2.3667502, 3.1595953
2: -1.0085254, 2.0125806, -0.6900675, 1.3430049, -2.3515303, 2.7026482
3: -0.8413924, 2.3148570, -0.6058655, 1.4762542, -2.3176465, 2.9207225
4: -1.1167248, 2.6804931, -0.6963954, 1.8879433, -3.0046682, 3.3768885

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7597453, upper bound: 2.7611288
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B1_B2_A1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_A2_B1_B2_A1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7597453, upper bound: 2.7664585
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2806621, 1.8875258, -0.2923267, 2.0209284, -2.3015904, 2.1798525
1: -0.3806360, 2.6402259, -0.4048963, 2.7956667, -3.1763027, 3.0451221
2: -0.9465249, 1.8131994, -0.9786599, 1.9727707, -2.9192955, 2.7918591
3: -0.7766682, 2.0958865, -0.8212299, 2.2473242, -3.0239925, 2.9171164
4: -1.0165755, 2.4664505, -1.0755645, 2.6259067, -3.6424823, 3.5420151

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768307, upper bound: 2.7740156
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752891, upper bound: 2.7726597
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2806621, 1.8875258, -0.3041376, 2.0502391, -2.3309011, 2.1916633
1: -0.3806360, 2.6402259, -0.4116657, 2.8429527, -3.2235887, 3.0518916
2: -0.9465249, 1.8131994, -1.0016354, 1.9943073, -2.9408321, 2.8148348
3: -0.7766682, 2.0958865, -0.8351074, 2.2927485, -3.0694165, 2.9309940
4: -1.0165755, 2.4664505, -1.1034777, 2.6544933, -3.6710687, 3.5699282

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739644, upper bound: 2.7755613
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737101, upper bound: 2.7753213
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2856687, 2.0045700, -0.2973586, 2.0464444, -2.3321133, 2.3019285
1: -0.4007463, 2.7735984, -0.4104910, 2.8293266, -3.2300730, 3.1840894
2: -0.9694705, 1.9582894, -0.9908106, 1.9982010, -2.9676714, 2.9491000
3: -0.8125409, 2.2225275, -0.8320768, 2.2798955, -3.0924363, 3.0546043
4: -1.0605392, 2.6134980, -1.0959523, 2.6567075, -3.7172468, 3.7094502

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786695, upper bound: 2.7740592
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786695, upper bound: 2.7740592
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2856687, 2.0045700, -0.3086333, 2.0736451, -2.3593140, 2.3132033
1: -0.4007463, 2.7735984, -0.4168339, 2.8735497, -3.2742960, 3.1904323
2: -0.9694705, 1.9582894, -1.0126864, 2.0177553, -2.9872258, 2.9709759
3: -0.8125409, 2.2225275, -0.8451283, 2.3225791, -3.1351199, 3.0676558
4: -1.0605392, 2.6134980, -1.1221292, 2.6829336, -3.7434728, 3.7356272

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757589, upper bound: 2.7751895
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757589, upper bound: 2.7751895
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2677826, 1.8381294, -0.3693268, 2.3271992, -2.5949817, 2.2074561
1: -0.3697071, 2.5732079, -0.4698166, 3.2208519, -3.5905590, 3.0430245
2: -0.9205275, 1.7657945, -1.1496696, 2.2569191, -3.1774466, 2.9154642
3: -0.7550522, 2.0262730, -0.9498016, 2.6556087, -3.4106607, 2.9760747
4: -0.9717644, 2.4110608, -1.3350083, 2.9717481, -3.9435124, 3.7460690

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7762832
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7762832
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2756731, 1.9642451, -0.3753327, 2.3569636, -2.6326368, 2.3395777
1: -0.3918094, 2.7198572, -0.4759032, 3.2600689, -3.6518784, 3.1957603
2: -0.9491056, 1.9184160, -1.1634135, 2.2862463, -3.2353520, 3.0818295
3: -0.7951761, 2.1666198, -0.9616026, 2.6963146, -3.4914906, 3.1282225
4: -1.0252471, 2.5676005, -1.3583037, 3.0054598, -4.0307069, 3.9259043

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7752422
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678125, upper bound: 2.7752422
time: 0.47 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.2853531, 1.8911947, -0.3693268, 2.3271992, -2.6125522, 2.2605214
1: -0.3823905, 2.6506677, -0.4698166, 3.2208519, -3.6032424, 3.1204844
2: -0.9541993, 1.8089762, -1.1496696, 2.2569191, -3.2111185, 2.9586458
3: -0.7804253, 2.1019111, -0.9498016, 2.6556087, -3.4360340, 3.0517125
4: -1.0201142, 2.4645104, -1.3350083, 2.9717481, -3.9918623, 3.7995186

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748881
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748881
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.2847048, 1.9835527, -0.3753327, 2.3569636, -2.6416683, 2.3588853
1: -0.3965250, 2.7515988, -0.4759032, 3.2600689, -3.6565938, 3.2275019
2: -0.9664359, 1.9298962, -1.1634135, 2.2862463, -3.2526822, 3.0933099
3: -0.8049830, 2.1974092, -0.9616026, 2.6963146, -3.5012975, 3.1590118
4: -1.0439100, 2.5858889, -1.3583037, 3.0054598, -4.0493698, 3.9441924

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693882, upper bound: 2.7723316
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693882, upper bound: 2.7723316
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2591349, 1.7701197, -0.2923267, 2.0209284, -2.2800632, 2.0624464
1: -0.3568611, 2.4715261, -0.4048963, 2.7956667, -3.1525278, 2.8764224
2: -0.8803689, 1.7118559, -0.9786599, 1.9727707, -2.8531396, 2.6905158
3: -0.7305660, 1.9627409, -0.8212299, 2.2473242, -2.9778903, 2.7839708
4: -0.9410986, 2.3190031, -1.0755645, 2.6259067, -3.5670052, 3.3945675

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2975173, 1.9552003, -0.2923267, 2.0209284, -2.3184457, 2.2475271
1: -0.3960572, 2.7384527, -0.4048963, 2.7956667, -3.1917238, 3.1433489
2: -0.9879949, 1.8708196, -0.9786599, 1.9727707, -2.9607656, 2.8494794
3: -0.8069820, 2.1896915, -0.8212299, 2.2473242, -3.0543063, 3.0109215
4: -1.0846331, 2.5465209, -1.0755645, 2.6259067, -3.7105398, 3.6220856

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789075, upper bound: 2.7766884
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2914144, 1.9445134, -0.2889379, 1.9049499, -2.1963644, 2.2334514
1: -0.3954931, 2.6824191, -0.3853487, 2.6697907, -3.0652838, 3.0677676
2: -0.9390244, 1.9117186, -0.9618609, 1.8210847, -2.7601092, 2.8735795
3: -0.8052138, 2.1717138, -0.7863698, 2.1205049, -2.9257188, 2.9580836
4: -1.0368246, 2.5272179, -1.0324315, 2.4797454, -3.5165701, 3.5596495

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7752053
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7746791
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3253865, 2.1591175, -0.2889379, 1.9049499, -2.2303364, 2.4480553
1: -0.4321831, 2.9894443, -0.3853487, 2.6697907, -3.1019740, 3.3747931
2: -1.0530849, 2.0953832, -0.9618609, 1.8210847, -2.8741696, 3.0572441
3: -0.8754766, 2.4329910, -0.7863698, 2.1205049, -2.9959815, 3.2193608
4: -1.1940833, 2.7761521, -1.0324315, 2.4797454, -3.6738286, 3.8085837

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754452, upper bound: 2.7735084
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7752053
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7746792
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3768151, 2.3552387, -0.2890179, 1.9990376, -2.3758528, 2.6442566
1: -0.4741576, 3.2577703, -0.4002408, 2.7723556, -3.2465131, 3.6580110
2: -1.1642399, 2.2777932, -0.9748017, 1.9446510, -3.1088910, 3.2525949
3: -0.9569162, 2.6878850, -0.8124224, 2.2182291, -3.1751451, 3.5003076
4: -1.3621224, 3.0037341, -1.0574362, 2.6030040, -3.9651265, 4.0611706

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747530, upper bound: 2.7754919
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747530, upper bound: 2.7752052
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3803409, 2.3737307, -0.2890179, 1.9990376, -2.3793786, 2.6627486
1: -0.4775637, 3.2876668, -0.4002408, 2.7723556, -3.2499194, 3.6879077
2: -1.1773490, 2.2890823, -0.9748017, 1.9446510, -3.1220000, 3.2638841
3: -0.9636787, 2.7165272, -0.8124224, 2.2182291, -3.1819077, 3.5289497
4: -1.3771966, 3.0225368, -1.0574362, 2.6030040, -3.9802005, 4.0799732

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747530, upper bound: 2.7747319
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747530, upper bound: 2.7746792
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2458000, 1.7208488, -0.3693268, 2.3271992, -2.5729992, 2.0901756
1: -0.3454927, 2.4054158, -0.4698166, 3.2208519, -3.5663445, 2.8752322
2: -0.8536351, 1.6646116, -1.1496696, 2.2569191, -3.1105542, 2.8142812
3: -0.7079096, 1.8960061, -0.9498016, 2.6556087, -3.3635182, 2.8458076
4: -0.9026028, 2.2619066, -1.3350083, 2.9717481, -3.8743510, 3.5969148

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2705797, 1.8707315, -0.3753327, 2.3569636, -2.6275434, 2.2460642
1: -0.3789070, 2.5838137, -0.4759032, 3.2600689, -3.6389759, 3.0597167
2: -0.9023267, 1.8385874, -1.1634135, 2.2862463, -3.1885729, 3.0020008
3: -0.7723326, 2.0637321, -0.9616026, 2.6963146, -3.4686472, 3.0253348
4: -0.9726133, 2.4480281, -1.3583037, 3.0054598, -3.9780731, 3.8063316

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7756620
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7756621
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3629161, 2.3064585, -2.5996525, 2.3192017
1: -0.3966470, 2.7066140, -0.4652137, 3.1939573, -3.5906043, 3.1718278
2: -0.9499824, 1.9157126, -1.1392360, 2.2357109, -3.1856933, 3.0549486
3: -0.8073892, 2.1858242, -0.9405786, 2.6256416, -3.4330308, 3.1264029
4: -1.0437701, 2.5382841, -1.3152579, 2.9512329, -3.9950030, 3.8535419

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628420, upper bound: 2.7645788
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693286, upper bound: 2.7723775
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3708772, 2.3199959, -2.6131899, 2.3271627
1: -0.3966470, 2.7066140, -0.4697720, 3.2195220, -3.6161690, 3.1763859
2: -0.9499824, 1.9157126, -1.1537372, 2.2423575, -3.1923399, 3.0694499
3: -0.8073892, 2.1858242, -0.9502540, 2.6516135, -3.4590027, 3.1360781
4: -1.0437701, 2.5382841, -1.3322339, 2.9660292, -4.0097990, 3.8705180

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628420, upper bound: 2.7645788
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693286, upper bound: 2.7723775
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3154575, 2.1055789, -0.3629161, 2.3064585, -2.6219161, 2.4684949
1: -0.4214576, 2.9155774, -0.4652137, 3.1939573, -3.6154151, 3.3807912
2: -1.0235951, 2.0444477, -1.1392360, 2.2357109, -3.2593060, 3.1836836
3: -0.8554188, 2.3600154, -0.9405786, 2.6256416, -3.4810605, 3.3005941
4: -1.1441643, 2.7125952, -1.3152579, 2.9512329, -4.0953970, 4.0278530

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7493327, upper bound: 2.7499283
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7493327, upper bound: 2.7507908
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3182679, 2.1371770, -0.3708772, 2.3199959, -2.6382637, 2.5080543
1: -0.4268867, 2.9595909, -0.4697720, 3.2195220, -3.6464088, 3.4293628
2: -1.0412936, 2.0729208, -1.1537372, 2.2423575, -3.2836511, 3.2266579
3: -0.8644805, 2.3999858, -0.9502540, 2.6516135, -3.5160940, 3.3502398
4: -1.1726522, 2.7533128, -1.3322339, 2.9660292, -4.1386814, 4.0855465

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7617107, upper bound: 2.7635633
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7478429
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3276167, 2.1572509, -0.3708772, 2.3199959, -2.6476126, 2.5281281
1: -0.4325817, 2.9916034, -0.4697720, 3.2195220, -3.6521037, 3.4613752
2: -1.0583760, 2.0858326, -1.1537372, 2.2423575, -3.3007336, 3.2395697
3: -0.8765814, 2.4330173, -0.9502540, 2.6516135, -3.5281949, 3.3832712
4: -1.1936884, 2.7738743, -1.3322339, 2.9660292, -4.1597176, 4.1061082

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7617107, upper bound: 2.7635633
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7478429
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2859721, 1.9972415, -0.3366727, 2.1246564, -2.4106286, 2.3339143
1: -0.3996190, 2.7636065, -0.4304063, 2.9713082, -3.3709273, 3.1940129
2: -0.9661374, 1.9507186, -1.0854695, 2.0262046, -2.9923420, 3.0361881
3: -0.8107526, 2.2156317, -0.8722178, 2.4124658, -3.2232184, 3.0878496
4: -1.0554264, 2.6002364, -1.2293684, 2.7454641, -3.8008904, 3.8296049

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734401, upper bound: 2.7742517
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7752528
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7752528
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2859721, 1.9972415, -0.3442136, 2.1465864, -2.4325585, 2.3414550
1: -0.3996190, 2.7636065, -0.4363874, 3.0061598, -3.4057789, 3.1999941
2: -0.9661374, 1.9507186, -1.1004975, 2.0408866, -3.0070240, 3.0512161
3: -0.8107526, 2.2156317, -0.8845118, 2.4401507, -3.2509034, 3.1001434
4: -1.0554264, 2.6002364, -1.2462673, 2.7676198, -3.8230462, 3.8465037

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734401, upper bound: 2.7749052
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7754017
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748275, upper bound: 2.7754017
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2997325, 2.0341167, -0.3366727, 2.1246564, -2.4243889, 2.3707895
1: -0.4078606, 2.8215098, -0.4304063, 2.9713082, -3.3791690, 3.2519162
2: -0.9930655, 1.9790092, -1.0854695, 2.0262046, -3.0192699, 3.0644786
3: -0.8275511, 2.2711914, -0.8722178, 2.4124658, -3.2400169, 3.1434093
4: -1.0896422, 2.6366882, -1.2293684, 2.7454641, -3.8351064, 3.8660567

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761990, upper bound: 2.7738905
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7746569
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770037, upper bound: 2.7746569
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2997325, 2.0341167, -0.3442136, 2.1465864, -2.4463189, 2.3783302
1: -0.4078606, 2.8215098, -0.4363874, 3.0061598, -3.4140205, 3.2578974
2: -0.9930655, 1.9790092, -1.1004975, 2.0408866, -3.0339522, 3.0795066
3: -0.8275511, 2.2711914, -0.8845118, 2.4401507, -3.2677019, 3.1557031
4: -1.0896422, 2.6366882, -1.2462673, 2.7676198, -3.8572621, 3.8829556

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7743644
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770036, upper bound: 2.7747722
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770036, upper bound: 2.7747722
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2677826, 1.8381294, -0.3673147, 2.3345752, -2.6023579, 2.2054441
1: -0.3697071, 2.5732079, -0.4687061, 3.2309842, -3.6006913, 3.0419140
2: -0.9205275, 1.7657945, -1.1555011, 2.2572024, -3.1777298, 2.9212956
3: -0.7550522, 2.0262730, -0.9444780, 2.6583376, -3.4133897, 2.9707510
4: -0.9717644, 2.4110608, -1.3431969, 2.9884486, -3.9602132, 3.7542577

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7779715
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7779715
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2756731, 1.9642451, -0.3673147, 2.3345752, -2.6102483, 2.3315597
1: -0.3918094, 2.7198572, -0.4687061, 3.2309842, -3.6227937, 3.1885633
2: -0.9491056, 1.9184160, -1.1555011, 2.2572024, -3.2063079, 3.0739172
3: -0.7951761, 2.1666198, -0.9444780, 2.6583376, -3.4535136, 3.1110978
4: -1.0252471, 2.5676005, -1.3431969, 2.9884486, -4.0136957, 3.9107974

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7752525
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7754015
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3042246, 2.0575032, -0.3541468, 2.2870154, -2.5912399, 2.4116499
1: -0.4130379, 2.8520617, -0.4580086, 3.1671107, -3.5801487, 3.3100705
2: -1.0041143, 2.0024467, -1.1315486, 2.2098856, -3.2139997, 3.1339953
3: -0.8375547, 2.3009794, -0.9232545, 2.5914588, -3.4290135, 3.2242339
4: -1.1082883, 2.6651578, -1.3013200, 2.9357991, -4.0440874, 3.9664779

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754901, upper bound: 2.7741444
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7608470, upper bound: 2.7603791
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3042246, 2.0575032, -0.3578824, 2.3051138, -2.6093383, 2.4153857
1: -0.4130379, 2.8520617, -0.4614864, 3.1964364, -3.6094742, 3.3135481
2: -1.0041143, 2.0024467, -1.1438384, 2.2210517, -3.2251658, 3.1462851
3: -0.8375547, 2.3009794, -0.9302158, 2.6164603, -3.4540150, 3.2311952
4: -1.1082883, 2.6651578, -1.3141679, 2.9544411, -4.0627294, 3.9793258

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754901, upper bound: 2.7744818
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7608470, upper bound: 2.7603791
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3536609, 2.2733560, -0.2971243, 1.9722085, -2.3258696, 2.5704803
1: -0.4585382, 3.1511228, -0.4001294, 2.7280126, -3.1865509, 3.5512521
2: -1.1225939, 2.2052293, -0.9578750, 1.9310058, -3.0535998, 3.1631043
3: -0.9274352, 2.5793722, -0.8142713, 2.2070019, -3.1344371, 3.3936434
4: -1.2869700, 2.9193895, -1.0563898, 2.5555344, -3.8425045, 3.9757793

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 18

Time for candidate selection: 3.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7576301, upper bound: 2.7604848
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7637152, upper bound: 2.7630007
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3630877, 2.2926280, -0.2971243, 1.9722085, -2.3352962, 2.5897522
1: -0.4641411, 3.1831021, -0.4001294, 2.7280126, -3.1921537, 3.5832314
2: -1.1396843, 2.2169552, -0.9578750, 1.9310058, -3.0706902, 3.1748302
3: -0.9391701, 2.6123335, -0.8142713, 2.2070019, -3.1461720, 3.4266047
4: -1.3082904, 2.9394026, -1.0563898, 2.5555344, -3.8638248, 3.9957924

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 18

Time for candidate selection: 3.26 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7576301, upper bound: 2.7624915
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A1_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7637152, upper bound: 2.7630008
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2801493, 1.9003778, -0.3366727, 2.1246564, -2.4048057, 2.2370505
1: -0.3858240, 2.6233618, -0.4304063, 2.9713082, -3.3571322, 3.0537682
2: -0.9167272, 1.8683568, -1.0854695, 2.0262046, -2.9429317, 2.9538264
3: -0.7863454, 2.1098049, -0.8722178, 2.4124658, -3.1988113, 2.9820228
4: -0.9985807, 2.4773648, -1.2293684, 2.7454641, -3.7440448, 3.7067332

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746769
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746772
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2801493, 1.9003778, -0.3442136, 2.1465864, -2.4267356, 2.2445912
1: -0.3858240, 2.6233618, -0.4363874, 3.0061598, -3.3919837, 3.0597491
2: -0.9167272, 1.8683568, -1.1004975, 2.0408866, -2.9576139, 2.9688544
3: -0.7863454, 2.1098049, -0.8845118, 2.4401507, -3.2264962, 2.9943166
4: -0.9985807, 2.4773648, -1.2462673, 2.7676198, -3.7662005, 3.7236321

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2886297, 1.9324335, -0.3366727, 2.1246564, -2.4132862, 2.2691061
1: -0.3913419, 2.6751680, -0.4304063, 2.9713082, -3.3626502, 3.1055744
2: -0.9385793, 1.8917056, -1.0854695, 2.0262046, -2.9647839, 2.9771752
3: -0.7972261, 2.1550553, -0.8722178, 2.4124658, -3.2096920, 3.0272732
4: -1.0246933, 2.5091178, -1.2293684, 2.7454641, -3.7701573, 3.7384863

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741407
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741407
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2886297, 1.9324335, -0.3442136, 2.1465864, -2.4352160, 2.2766471
1: -0.3913419, 2.6751680, -0.4363874, 3.0061598, -3.3975017, 3.1115556
2: -0.9385793, 1.8917056, -1.1004975, 2.0408866, -2.9794660, 2.9922032
3: -0.7972261, 2.1550553, -0.8845118, 2.4401507, -3.2373769, 3.0395670
4: -1.0246933, 2.5091178, -1.2462673, 2.7676198, -3.7923131, 3.7553852

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746774
time: 0.47 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746774
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2852379, 1.9261826, -0.2869928, 1.9336224, -2.2188601, 2.2131753
1: -0.3915178, 2.6576438, -0.3927403, 2.6686208, -3.0601387, 3.0503840
2: -0.9292547, 1.8941118, -0.9346623, 1.8997009, -2.8289557, 2.8287740
3: -0.7973023, 2.1429307, -0.7993886, 2.1493618, -2.9466641, 2.9423194
4: -1.0195756, 2.5085783, -1.0269583, 2.5181310, -3.5377066, 3.5355368

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7774857
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2852379, 1.9261826, -0.3140926, 2.1232934, -2.4085312, 2.2402751
1: -0.3915178, 2.6576438, -0.4233531, 2.9424019, -3.3339198, 3.0809970
2: -0.9292547, 1.8941118, -1.0367260, 2.0574467, -2.9867015, 2.9308376
3: -0.7973023, 2.1429307, -0.8564682, 2.3797078, -3.1770101, 2.9993989
4: -1.0195756, 2.5085783, -1.1613970, 2.7425013, -3.7620769, 3.6699753

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7776243
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783336
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3541468, 2.2870154, -2.5802095, 2.3104324
1: -0.3966470, 2.7066140, -0.4580086, 3.1671107, -3.5637577, 3.1646228
2: -0.9499824, 1.9157126, -1.1315486, 2.2098856, -3.1598680, 3.0472612
3: -0.8073892, 2.1858242, -0.9232545, 2.5914588, -3.3988481, 3.1090786
4: -1.0437701, 2.5382841, -1.3013200, 2.9357991, -3.9795692, 3.8396039

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7742429
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7742810
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3578824, 2.3051138, -2.5983078, 2.3141680
1: -0.3966470, 2.7066140, -0.4614864, 3.1964364, -3.5930834, 3.1681004
2: -0.9499824, 1.9157126, -1.1438384, 2.2210517, -3.1710341, 3.0595510
3: -0.8073892, 2.1858242, -0.9302158, 2.6164603, -3.4238496, 3.1160400
4: -1.0437701, 2.5382841, -1.3141679, 2.9544411, -3.9982111, 3.8524518

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7747277
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7747346
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2975173, 1.9552003, -0.2861977, 1.9233072, -2.2208245, 2.2413979
1: -0.3960572, 2.7384527, -0.3907812, 2.6545448, -3.0506020, 3.1292338
2: -0.9879949, 1.8708196, -0.9286579, 1.8898368, -2.8778317, 2.7994776
3: -0.8069820, 2.1896915, -0.7961895, 2.1413364, -2.9483185, 2.9858811
4: -1.0846331, 2.5465209, -1.0177705, 2.5019808, -3.5866139, 3.5642915

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7681997, upper bound: 2.7639853
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788115, upper bound: 2.7741393
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788115, upper bound: 2.7742429
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3084403, 2.1027575, -0.2913533, 1.9494442, -2.2578845, 2.3941107
1: -0.4188761, 2.9142489, -0.3965143, 2.6892331, -3.1081090, 3.3107634
2: -1.0257192, 2.0389187, -0.9412345, 1.9158479, -2.9415669, 2.9801531
3: -0.8475755, 2.3515916, -0.8072340, 2.1748338, -3.0224094, 3.1588256
4: -1.1437144, 2.7211394, -1.0388029, 2.5333719, -3.6770864, 3.7599423

Time for backsubstitution: 1.68 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728438, upper bound: 2.7734686
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803612, upper bound: 2.7803613
time: 0.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -2.7728438, upper bound: 2.7734686
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -2.7803612, upper bound: 2.7803613

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.3937300, 2.2607875, -0.4715014, 2.6545389, -3.0482688, 2.7322888
1: -0.4659128, 3.1165721, -0.5367672, 3.6636696, -4.1295824, 3.6533394
2: -1.1388535, 2.1871104, -1.3161142, 2.5634563, -3.7023098, 3.5032246
3: -0.9433906, 2.6073360, -1.0849349, 3.1588147, -4.1022053, 3.6922710
4: -1.3197460, 2.8582294, -1.6355078, 3.2938135, -4.6135597, 4.4937372

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7642997, upper bound: 2.7669333
time: 0.30 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7701175, upper bound: 2.7713935
time: 0.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.5007467, 2.7479298, -0.5095432, 2.7755899, -3.2763367, 3.2574730
1: -0.5553178, 3.7793705, -0.5611423, 3.8165379, -4.3718557, 4.3405128
2: -1.3528485, 2.6755908, -1.3674926, 2.7016737, -4.0545225, 4.0430832
3: -1.1222062, 3.3138933, -1.1338987, 3.3577619, -4.4799681, 4.4477921
4: -1.7101582, 3.3786907, -1.7360522, 3.4053361, -5.1154943, 5.1147428

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7670271, upper bound: 2.7645061
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803377, upper bound: 2.7803377
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.17 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7642997, upper bound: 2.7669333
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7701175, upper bound: 2.7713935
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7670271, upper bound: 2.7645061
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.17
Output dim: 0, lower bound: -2.7803377, upper bound: 2.7803377

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.2694138, 1.5103993, -0.3663085, 2.3382554, -2.6076694, 1.8767078
1: -0.3349953, 2.1516724, -0.4723832, 3.2475135, -3.5825088, 2.6240556
2: -0.8057680, 1.4441006, -1.1713521, 2.2537899, -3.0595579, 2.6154528
3: -0.7113398, 1.6636263, -0.9524465, 2.6886563, -3.3999963, 2.6160727
4: -0.8483682, 2.0235796, -1.3667691, 3.0175381, -3.8659062, 3.3903487

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7485000, upper bound: 2.7499630
time: 0.37 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7422068, upper bound: 2.7483538
time: 0.36 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.3660870, 2.1625853, -0.4714772, 2.6544621, -3.0205491, 2.6340623
1: -0.4462218, 2.9898553, -0.5367516, 3.6635723, -4.1097941, 3.5266070
2: -1.0902050, 2.0946541, -1.3160801, 2.5633631, -3.6535680, 3.4107342
3: -0.9038171, 2.4661355, -1.0849037, 3.1586890, -4.0625062, 3.5510392
4: -1.2328745, 2.7640190, -1.6354345, 3.2937355, -4.5266099, 4.3994536

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7612840, upper bound: 2.7607675
time: 0.33 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7612840, upper bound: 2.7713935
time: 0.33 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.3846325, 2.4077125, -0.2728533, 1.6860912, -2.0707238, 2.6805658
1: -0.4871436, 3.3304827, -0.3547124, 2.3795998, -2.8667436, 3.6851950
2: -1.1991270, 2.3286099, -0.8683753, 1.6238707, -2.8229976, 3.1969852
3: -0.9822053, 2.7720141, -0.7345715, 1.8792391, -2.8614445, 3.5065856
4: -1.4174317, 3.0858316, -0.9312305, 2.2542911, -3.6717229, 4.0170622

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7634520, upper bound: 2.7621594
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7670271, upper bound: 2.7645061
time: 0.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.5007467, 2.7479298, -0.4836221, 2.6905560, -3.1913028, 3.2315519
1: -0.5553178, 3.7793705, -0.5440643, 3.7076769, -4.2629948, 4.3234348
2: -1.3528485, 2.6755908, -1.3264129, 2.6181285, -3.9709771, 4.0020037
3: -1.1222062, 3.3138933, -1.0996997, 3.2257874, -4.3479939, 4.4135933
4: -1.7101582, 3.3786907, -1.6587552, 3.3243561, -5.0345144, 5.0374460

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7645061, upper bound: 2.7670271
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7645061, upper bound: 2.7803377
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.22 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7485000, upper bound: 2.7499630
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7422068, upper bound: 2.7483538
IS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7612840, upper bound: 2.7607675
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7612840, upper bound: 2.7713935
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7634520, upper bound: 2.7621594
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7670271, upper bound: 2.7645061
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7645061, upper bound: 2.7670271
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.22
Output dim: 0, lower bound: -2.7645061, upper bound: 2.7803377

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3660870, 2.1625853, -0.4464076, 2.5728307, -2.9389176, 2.6089928
1: -0.4462218, 2.9898553, -0.5202224, 3.5553193, -4.0015411, 3.5100777
2: -1.0902050, 2.0946541, -1.2758083, 2.4786229, -3.5688279, 3.3704624
3: -0.9038171, 2.4661355, -1.0518147, 3.0295630, -3.9333801, 3.5179501
4: -1.2328745, 2.7640190, -1.5592734, 3.2125986, -4.4454732, 4.3232923

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534537, upper bound: 2.7600887
time: 0.45 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7348733, upper bound: 2.7563866
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.3682767, 2.3456261, -0.2077491, 1.4434946, -1.8117712, 2.5533752
1: -0.4736199, 3.2456224, -0.3018184, 2.0486817, -2.5223017, 3.5474408
2: -1.1639061, 2.2702942, -0.7286727, 1.4051616, -2.5690677, 2.9989669
3: -0.9558219, 2.6877842, -0.6323433, 1.5639632, -2.5197849, 3.3201275
4: -1.3640976, 3.0129039, -0.7463270, 1.9729867, -3.3370843, 3.7592311

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7600849, upper bound: 2.7573945
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7581348, upper bound: 2.7573945
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2679194, 1.6677933, -0.4836221, 2.6905560, -2.9584754, 2.1514153
1: -0.3506812, 2.3546071, -0.5440643, 3.7076769, -4.0583582, 2.8986714
2: -0.8583503, 1.6077589, -1.3264129, 2.6181285, -3.4764788, 2.9341717
3: -0.7267100, 1.8529313, -1.0996997, 3.2257874, -3.9524975, 2.9526310
4: -0.9143050, 2.2346778, -1.6587552, 3.3243561, -4.2386608, 3.8934331

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575250, upper bound: 2.7634519
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7581911, upper bound: 2.7670270
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4747180, 2.6634238, -0.4836221, 2.6905560, -3.1652741, 3.1470459
1: -0.5381627, 3.6703260, -0.5440643, 3.7076769, -4.2458396, 4.2143903
2: -1.3116175, 2.5915504, -1.3264129, 2.6181285, -3.9297462, 3.9179633
3: -1.0878518, 3.1816847, -1.0996997, 3.2257874, -4.3136392, 4.2813845
4: -1.6326261, 3.2974410, -1.6587552, 3.3243561, -4.9569821, 4.9561963

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7542386, upper bound: 2.7803377
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7580277, upper bound: 2.7580277
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.25 seconds
IS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7534537, upper bound: 2.7600887
IS_A1_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7348733, upper bound: 2.7563866
IS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7600849, upper bound: 2.7573945
IS_A2_B1_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7581348, upper bound: 2.7573945
IS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7575250, upper bound: 2.7634519
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7581911, upper bound: 2.7670270
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7542386, upper bound: 2.7803377
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.25
Output dim: 0, lower bound: -2.7580277, upper bound: 2.7580277

## BFS IS instance: IS_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2027035, 1.4248056, -0.4652545, 2.6211228, -2.8238263, 1.8900602
1: -0.2976533, 2.0236523, -0.5295218, 3.6135693, -3.9112225, 2.5531740
2: -0.7181041, 1.3894733, -1.2885467, 2.5499463, -3.2680504, 2.6780200
3: -0.6241174, 1.5397948, -1.0711973, 3.1288435, -3.7529609, 2.6109920
4: -0.7333497, 1.9524517, -1.5991359, 3.2449117, -3.9782615, 3.5515876

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7581802, upper bound: 2.7586815
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7531035, upper bound: 2.7584240
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4454918, 2.5787899, -0.4250804, 2.5852566, -3.0307484, 3.0038705
1: -0.5205739, 3.5574338, -0.5211089, 3.5761673, -4.0967412, 4.0785427
2: -1.2714649, 2.4991918, -1.2865007, 2.4982405, -3.7697053, 3.7856925
3: -1.0519731, 3.0320582, -1.0481679, 3.0023003, -4.0542736, 4.0802259
4: -1.5526068, 3.2206376, -1.5452410, 3.2716055, -4.8242121, 4.7658787

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797972, upper bound: 2.7801972
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689750, upper bound: 2.7762072
time: 0.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.29 seconds
IS_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -2.7581802, upper bound: 2.7586815
IS_A2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.29
Output dim: 0, lower bound: -2.7531035, upper bound: 2.7584240
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 0, lower bound: -2.7797972, upper bound: 2.7801972
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.29
Output dim: 0, lower bound: -2.7689750, upper bound: 2.7762072

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.4025486, 2.3927507, -0.3076048, 2.0884678, -2.4910164, 2.7003555
1: -0.4825866, 3.2977738, -0.4196330, 2.8851950, -3.3677816, 3.7174067
2: -1.1682276, 2.3320112, -1.0115439, 2.0395269, -3.2077546, 3.3435550
3: -0.9781964, 2.7868094, -0.8498347, 2.3376288, -3.3158252, 3.6366441
4: -1.3950206, 3.0056338, -1.1316509, 2.7040546, -4.0990753, 4.1372848

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797972, upper bound: 2.7764849
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782096, upper bound: 2.7785847
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.4276377, 2.5095294, -0.3753327, 2.3569636, -2.7846012, 2.8848619
1: -0.5060825, 3.4632559, -0.4759032, 3.2600689, -3.7661514, 3.9391589
2: -1.2336650, 2.4304042, -1.1634135, 2.2862463, -3.5199113, 3.5938177
3: -1.0236963, 2.9366641, -0.9616026, 2.6963146, -3.7200108, 3.8982668
4: -1.4934676, 3.1411386, -1.3583037, 3.0054598, -4.4989271, 4.4994421

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689750, upper bound: 2.7762072
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689750, upper bound: 2.7762072
time: 0.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.31 seconds
IS_A2_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7797972, upper bound: 2.7764849
IS_A2_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7782096, upper bound: 2.7785847
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7689750, upper bound: 2.7762072
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.31
Output dim: 0, lower bound: -2.7689750, upper bound: 2.7762072

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.4025486, 2.3927507, -0.2973586, 2.0464444, -2.4489930, 2.6901093
1: -0.4825866, 3.2977738, -0.4104910, 2.8293266, -3.3119133, 3.7082648
2: -1.1682276, 2.3320112, -0.9908106, 1.9982010, -3.1664286, 3.3228219
3: -0.9781964, 2.7868094, -0.8320768, 2.2798955, -3.2580919, 3.6188862
4: -1.3950206, 3.0056338, -1.0959523, 2.6567075, -4.0517282, 4.1015863

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740856
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785469, upper bound: 2.7741990
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.4025486, 2.3927507, -0.3086333, 2.0736451, -2.4761937, 2.7013841
1: -0.4825866, 3.2977738, -0.4168339, 2.8735497, -3.3561363, 3.7146077
2: -1.1682276, 2.3320112, -1.0126864, 2.0177553, -3.1859827, 3.3446975
3: -0.9781964, 2.7868094, -0.8451283, 2.3225791, -3.3007755, 3.6319377
4: -1.3950206, 3.0056338, -1.1221292, 2.6829336, -4.0779543, 4.1277628

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764715, upper bound: 2.7752051
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7754919
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3317343, 2.0929952, -0.3753327, 2.3569636, -2.6886978, 2.4683280
1: -0.4236188, 2.8752916, -0.4759032, 3.2600689, -3.6836877, 3.3511949
2: -1.0036168, 2.0634935, -1.1634135, 2.2862463, -3.2898631, 3.2269070
3: -0.8625162, 2.3927832, -0.9616026, 2.6963146, -3.5588307, 3.3543859
4: -1.1514467, 2.6629903, -1.3583037, 3.0054598, -4.1569066, 4.0212941

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7762072
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3810499, 2.3267608, -0.3753327, 2.3569636, -2.7380135, 2.7020936
1: -0.4680253, 3.2144878, -0.4759032, 3.2600689, -3.7280941, 3.6903911
2: -1.1341581, 2.2601120, -1.1634135, 2.2862463, -3.4204044, 3.4235256
3: -0.9495662, 2.6868331, -0.9616026, 2.6963146, -3.6458807, 3.6484356
4: -1.3403040, 2.9307985, -1.3583037, 3.0054598, -4.3457637, 4.2891021

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7762072
time: 0.37 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.34 seconds
IS_A2_B2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740856
IS_A2_B2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7785469, upper bound: 2.7741990
IS_A2_B2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7764715, upper bound: 2.7752051
IS_A2_B2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7754919
IS_A2_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
IS_A2_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7762072
IS_A2_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
IS_A2_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.34
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7762072

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.3812420, 2.2891955, -0.2717244, 1.8525931, -2.2338352, 2.5609200
1: -0.4610502, 3.1591778, -0.3729628, 2.5932012, -3.0542512, 3.5321407
2: -1.1185905, 2.2309954, -0.9286113, 1.7790446, -2.8976350, 3.1596067
3: -0.9362877, 2.6493425, -0.7617036, 2.0464137, -2.9827013, 3.4110460
4: -1.3106452, 2.8866336, -0.9849690, 2.4273558, -3.7380009, 3.8716025

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740857
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740857
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.4025486, 2.3927507, -0.2820300, 1.9873157, -2.3898642, 2.6747808
1: -0.4825866, 3.2977738, -0.3970020, 2.7511249, -3.2337115, 3.6947758
2: -1.1682276, 2.3320112, -0.9615395, 1.9399827, -3.1082101, 3.2935507
3: -0.9781964, 2.7868094, -0.8055222, 2.1982713, -3.1764677, 3.5923316
4: -1.3950206, 3.0056338, -1.0453598, 2.5924871, -3.9875078, 4.0509939

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7741990
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.3812420, 2.2891955, -0.2889379, 1.9049499, -2.2861919, 2.5781333
1: -0.4610502, 3.1591778, -0.3853487, 2.6697907, -3.1308408, 3.5445266
2: -1.1185905, 2.2309954, -0.9618609, 1.8210847, -2.9396753, 3.1928563
3: -0.9362877, 2.6493425, -0.7863698, 2.1205049, -3.0567925, 3.4357123
4: -1.3106452, 2.8866336, -1.0324315, 2.4797454, -3.7903905, 3.9190650

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752051
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.4025486, 2.3927507, -0.2890179, 1.9990376, -2.4015863, 2.6817687
1: -0.4825866, 3.2977738, -0.4002408, 2.7723556, -3.2549422, 3.6980147
2: -1.1682276, 2.3320112, -0.9748017, 1.9446510, -3.1128786, 3.3068128
3: -0.9781964, 2.7868094, -0.8124224, 2.2182291, -3.1964254, 3.5992317
4: -1.3950206, 3.0056338, -1.0574362, 2.6030040, -3.9980245, 4.0630703

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7754919
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3031371, 2.0717158, -0.3753327, 2.3569636, -2.6601007, 2.4470487
1: -0.4158132, 2.8628645, -0.4759032, 3.2600689, -3.6758821, 3.3387675
2: -1.0028709, 2.0236099, -1.1634135, 2.2862463, -3.2891173, 3.1870234
3: -0.8422414, 2.3150051, -0.9616026, 2.6963146, -3.5385561, 3.2766075
4: -1.1176394, 2.6856921, -1.3583037, 3.0054598, -4.1230993, 4.0439959

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678183, upper bound: 2.7762297
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7691768, upper bound: 2.7755699
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2960344, 1.9689381, -0.3753327, 2.3569636, -2.6529980, 2.3442707
1: -0.4008421, 2.7146063, -0.4759032, 3.2600689, -3.6609111, 3.1905093
2: -0.9505634, 1.9361818, -1.1634135, 2.2862463, -3.2368097, 3.0995953
3: -0.8154755, 2.2027259, -0.9616026, 2.6963146, -3.5117900, 3.1643286
4: -1.0562243, 2.5568314, -1.3583037, 3.0054598, -4.0616841, 3.9151349

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7758956
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7733116
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3252814, -0.3753327, 2.3569636, -2.7235608, 2.7006140
1: -0.4696044, 3.2191162, -0.4759032, 3.2600689, -3.7296734, 3.6950192
2: -1.1476570, 2.2572851, -1.1634135, 2.2862463, -3.4339032, 3.4206986
3: -0.9491717, 2.6519766, -0.9616026, 2.6963146, -3.6454864, 3.6135793
4: -1.3314362, 2.9751580, -1.3583037, 3.0054598, -4.3368959, 4.3334618

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7662900
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3753327, 2.3569636, -2.6883078, 2.5633492
1: -0.4382458, 3.0278041, -0.4759032, 3.2600689, -3.6983147, 3.5037074
2: -1.0667892, 2.1236391, -1.1634135, 2.2862463, -3.3530354, 3.2870526
3: -0.8872414, 2.4708958, -0.9616026, 2.6963146, -3.5835559, 3.4324985
4: -1.2173010, 2.8092000, -1.3583037, 3.0054598, -4.2227607, 4.1675038

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7726802
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7723025
time: 0.50 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.52 seconds
IS_A2_B2_A2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
IS_A2_B2_A2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7741990
IS_A2_B2_A2_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752051
IS_A2_B2_A2_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
IS_A2_B2_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
IS_A2_B2_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7754919
IS_A2_B2_A2_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7678183, upper bound: 2.7762297
IS_A2_B2_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7691768, upper bound: 2.7755699
IS_A2_B2_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7758956
IS_A2_B2_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7733116
IS_A2_B2_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7662900
IS_A2_B2_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
IS_A2_B2_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7660718, upper bound: 2.7726802
IS_A2_B2_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.52
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7723025

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3105701, 1.9908373, -0.2717244, 1.8525931, -2.1631632, 2.2625618
1: -0.4016057, 2.7402065, -0.3729628, 2.5932012, -2.9948068, 3.1131692
2: -0.9552937, 1.9617456, -0.9286113, 1.7790446, -2.7343383, 2.8903570
3: -0.8198207, 2.2574906, -0.7617036, 2.0464137, -2.8662343, 3.0191941
4: -1.0688707, 2.5450819, -0.9849690, 2.4273558, -3.4962263, 3.5300508

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776571, upper bound: 2.7740857
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776571, upper bound: 2.7740857
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3596670, 2.2237973, -0.2717244, 1.8525931, -2.2122600, 2.4955218
1: -0.4464675, 3.0763843, -0.3729628, 2.5932012, -3.0396686, 3.4493470
2: -1.0844438, 2.1603224, -0.9286113, 1.7790446, -2.8634884, 3.0889337
3: -0.9075895, 2.5500841, -0.7617036, 2.0464137, -2.9540031, 3.3117876
4: -1.2560790, 2.8128216, -0.9849690, 2.4273558, -3.6834347, 3.7977905

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740856
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740857
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3704951, 2.3637142, -0.2820300, 1.9873157, -2.3578107, 2.6457443
1: -0.4745487, 3.2686713, -0.3970020, 2.7511249, -3.2256737, 3.6656733
2: -1.1631755, 2.2910173, -0.9615395, 1.9399827, -3.1031580, 3.2525568
3: -0.9569656, 2.6909490, -0.8055222, 2.1982713, -3.1552367, 3.4964712
4: -1.3534044, 3.0198841, -1.0453598, 2.5924871, -3.9458914, 4.0652437

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3545275, 2.2452011, -0.2820300, 1.9873157, -2.3418431, 2.5272312
1: -0.4528053, 3.1015203, -0.3970020, 2.7511249, -3.2039301, 3.4985223
2: -1.1005495, 2.1841080, -0.9615395, 1.9399827, -3.0405321, 3.1456475
3: -0.9158913, 2.5481915, -0.8055222, 2.1982713, -3.1141624, 3.3537138
4: -1.2706814, 2.8744109, -1.0453598, 2.5924871, -3.8631685, 3.9197707

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7709401, upper bound: 2.7690101
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7528209, upper bound: 2.7507118
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3105701, 1.9908373, -0.2889379, 1.9049499, -2.2155199, 2.2797751
1: -0.4016057, 2.7402065, -0.3853487, 2.6697907, -3.0713964, 3.1255550
2: -0.9552937, 1.9617456, -0.9618609, 1.8210847, -2.7763784, 2.9236064
3: -0.8198207, 2.2574906, -0.7863698, 2.1205049, -2.9403255, 3.0438604
4: -1.0688707, 2.5450819, -1.0324315, 2.4797454, -3.5486159, 3.5775132

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7752053
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7746792
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3596670, 2.2237973, -0.2889379, 1.9049499, -2.2646170, 2.5127351
1: -0.4464675, 3.0763843, -0.3853487, 2.6697907, -3.1162581, 3.4617329
2: -1.0844438, 2.1603224, -0.9618609, 1.8210847, -2.9055285, 3.1221833
3: -0.9075895, 2.5500841, -0.7863698, 2.1205049, -3.0280943, 3.3364539
4: -1.2560790, 2.8128216, -1.0324315, 2.4797454, -3.7358243, 3.8452530

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3704951, 2.3637142, -0.2890179, 1.9990376, -2.3695326, 2.6527321
1: -0.4745487, 3.2686713, -0.4002408, 2.7723556, -3.2469044, 3.6689119
2: -1.1631755, 2.2910173, -0.9748017, 1.9446510, -3.1078265, 3.2658191
3: -0.9569656, 2.6909490, -0.8124224, 2.2182291, -3.1751947, 3.5033712
4: -1.3534044, 3.0198841, -1.0574362, 2.6030040, -3.9564085, 4.0773201

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3545275, 2.2452011, -0.2890179, 1.9990376, -2.3535652, 2.5342190
1: -0.4528053, 3.1015203, -0.4002408, 2.7723556, -3.2251608, 3.5017610
2: -1.1005495, 2.1841080, -0.9748017, 1.9446510, -3.0452003, 3.1589098
3: -0.9158913, 2.5481915, -0.8124224, 2.2182291, -3.1341205, 3.3606138
4: -1.2706814, 2.8744109, -1.0574362, 2.6030040, -3.8736854, 3.9318471

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7754919
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7747319
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2806621, 1.8875258, -0.3536721, 2.2504725, -2.5311346, 2.2411978
1: -0.3806360, 2.6402259, -0.4539548, 3.1193335, -3.4999695, 3.0941806
2: -0.9465249, 1.8131994, -1.1137347, 2.1815369, -3.1280618, 2.9269342
3: -0.7766682, 2.0958865, -0.9190763, 2.5514789, -3.3281469, 3.0149627
4: -1.0165755, 2.4664505, -1.2742250, 2.8849490, -3.9015245, 3.7406754

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7696285, upper bound: 2.7740357
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7695084, upper bound: 2.7736951
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2856687, 2.0045700, -0.3753327, 2.3569636, -2.6426325, 2.3799028
1: -0.4007463, 2.7735984, -0.4759032, 3.2600689, -3.6608152, 3.2495017
2: -0.9694705, 1.9582894, -1.1634135, 2.2862463, -3.2557168, 3.1217029
3: -0.8125409, 2.2225275, -0.9616026, 2.6963146, -3.5088553, 3.1841302
4: -1.0605392, 2.6134980, -1.3583037, 3.0054598, -4.0659990, 3.9718018

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7752404
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693506, upper bound: 2.7723261
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2852379, 1.9261826, -0.3753327, 2.3569636, -2.6422014, 2.3015153
1: -0.3915178, 2.6576438, -0.4759032, 3.2600689, -3.6515868, 3.1335468
2: -0.9292547, 1.8941118, -1.1634135, 2.2862463, -3.2155008, 3.0575252
3: -0.7973023, 2.1429307, -0.9616026, 2.6963146, -3.4936168, 3.1045332
4: -1.0195756, 2.5085783, -1.3583037, 3.0054598, -4.0250354, 3.8668818

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7674585, upper bound: 2.7759766
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678142, upper bound: 2.7756463
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3753327, 2.3569636, -2.6501577, 2.3316183
1: -0.3966470, 2.7066140, -0.4759032, 3.2600689, -3.6567159, 3.1825171
2: -0.9499824, 1.9157126, -1.1634135, 2.2862463, -3.2362287, 3.0791261
3: -0.8073892, 2.1858242, -0.9616026, 2.6963146, -3.5037038, 3.1474266
4: -1.0437701, 2.5382841, -1.3583037, 3.0054598, -4.0492296, 3.8965878

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7733116
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7733116
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3252814, -0.3629161, 2.3064585, -2.6730556, 2.6881974
1: -0.4696044, 3.2191162, -0.4652137, 3.1939573, -3.6635618, 3.6843300
2: -1.1476570, 2.2572851, -1.1392360, 2.2357109, -3.3833680, 3.3965211
3: -0.9491717, 2.6519766, -0.9405786, 2.6256416, -3.5748134, 3.5925553
4: -1.3314362, 2.9751580, -1.3152579, 2.9512329, -4.2826691, 4.2904158

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7533442
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7532613
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3252814, -0.3708772, 2.3199959, -2.6865931, 2.6961586
1: -0.4696044, 3.2191162, -0.4697720, 3.2195220, -3.6891265, 3.6888881
2: -1.1476570, 2.2572851, -1.1537372, 2.2423575, -3.3900146, 3.4110222
3: -0.9491717, 2.6519766, -0.9502540, 2.6516135, -3.6007853, 3.6022305
4: -1.3314362, 2.9751580, -1.3322339, 2.9660292, -4.2974653, 4.3073921

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515082, upper bound: 2.7513427
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7513427, upper bound: 2.7513427
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3629161, 2.3064585, -2.6378026, 2.5509324
1: -0.4382458, 3.0278041, -0.4652137, 3.1939573, -3.6322031, 3.4930177
2: -1.0667892, 2.1236391, -1.1392360, 2.2357109, -3.3025000, 3.2628751
3: -0.8872414, 2.4708958, -0.9405786, 2.6256416, -3.5128829, 3.4114745
4: -1.2173010, 2.8092000, -1.3152579, 2.9512329, -4.1685338, 4.1244578

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7493327, upper bound: 2.7499283
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7495325, upper bound: 2.7507908
time: 0.53 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3313442, 2.1880164, -0.3708772, 2.3199959, -2.6513400, 2.5588937
1: -0.4382458, 3.0278041, -0.4697720, 3.2195220, -3.6577678, 3.4975762
2: -1.0667892, 2.1236391, -1.1537372, 2.2423575, -3.3091466, 3.2773762
3: -0.8872414, 2.4708958, -0.9502540, 2.6516135, -3.5388548, 3.4211497
4: -1.2173010, 2.8092000, -1.3322339, 2.9660292, -4.1833301, 4.1414337

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678477, upper bound: 2.7723025
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678477, upper bound: 2.7723025
time: 0.41 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.80 seconds
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7776571, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7776571, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740856
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7791046, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
IS_A2_B2_A2_B1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7709401, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7528209, upper bound: 2.7507118
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7752053
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7748114, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7764716, upper bound: 2.7752053
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7754919
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7747319
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7696285, upper bound: 2.7740357
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7695084, upper bound: 2.7736951
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7752404
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7693506, upper bound: 2.7723261
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7674585, upper bound: 2.7759766
IS_A2_B2_A2_B1_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7678142, upper bound: 2.7756463
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7733116
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7733116
IS_A2_B2_A2_B1_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7533442
IS_A2_B2_A2_B1_B2_A2_A1_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7530892, upper bound: 2.7532613
IS_A2_B2_A2_B1_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7515082, upper bound: 2.7513427
IS_A2_B2_A2_B1_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7513427, upper bound: 2.7513427
IS_A2_B2_A2_B1_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7493327, upper bound: 2.7499283
IS_A2_B2_A2_B1_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7495325, upper bound: 2.7507908
IS_A2_B2_A2_B1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7678477, upper bound: 2.7723025
IS_A2_B2_A2_B1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.80
Output dim: 0, lower bound: -2.7678477, upper bound: 2.7723025

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2933785, 1.9254615, -0.2717244, 1.8525931, -2.1459715, 2.1971860
1: -0.3881600, 2.6552501, -0.3729628, 2.5932012, -2.9813612, 3.0282130
2: -0.9234380, 1.8988812, -0.9286113, 1.7790446, -2.7024827, 2.8274925
3: -0.7931215, 2.1667268, -0.7617036, 2.0464137, -2.8395352, 2.9284306
4: -1.0140216, 2.4781418, -0.9849690, 2.4273558, -3.4413774, 3.4631109

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7654552, upper bound: 2.7611498
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740592
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740592
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3074463, 1.9780474, -0.2717244, 1.8525931, -2.1600394, 2.2497718
1: -0.3971092, 2.7358575, -0.3729628, 2.5932012, -2.9903104, 3.1088204
2: -0.9541693, 1.9347870, -0.9286113, 1.7790446, -2.7332139, 2.8633983
3: -0.8110955, 2.2350283, -0.7617036, 2.0464137, -2.8575091, 2.9967318
4: -1.0563571, 2.5269520, -0.9849690, 2.4273558, -3.4837129, 3.5119209

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7654552, upper bound: 2.7611498
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740592
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740857
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.3231174, 2.0654058, -0.2717244, 1.8525931, -2.1757104, 2.3371303
1: -0.4157538, 2.8843560, -0.3729628, 2.5932012, -3.0089550, 3.2573190
2: -1.0334489, 1.9777371, -0.9286113, 1.7790446, -2.8124936, 2.9063482
3: -0.8464797, 2.3422604, -0.7617036, 2.0464137, -2.8928933, 3.1039639
4: -1.1675417, 2.6453815, -0.9849690, 2.4273558, -3.5948975, 3.6303506

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740857
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740857
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.3465745, 2.2301984, -0.2717244, 1.8525931, -2.1991677, 2.5019228
1: -0.4452873, 3.0852342, -0.3729628, 2.5932012, -3.0384884, 3.4581971
2: -1.0840776, 2.1610651, -0.9286113, 1.7790446, -2.8631222, 3.0896764
3: -0.9021518, 2.5303700, -0.7617036, 2.0464137, -2.9485655, 3.2920737
4: -1.2483793, 2.8331316, -0.9849690, 2.4273558, -3.6757350, 3.8181005

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740857
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740856
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3031371, 2.0717506, -0.2820300, 1.9873157, -2.2904527, 2.3537807
1: -0.4158132, 2.8628929, -0.3970020, 2.7511249, -3.1669381, 3.2598948
2: -1.0028709, 2.0236433, -0.9615395, 1.9399827, -2.9428535, 2.9851828
3: -0.8422414, 2.3150473, -0.8055222, 2.1982713, -3.0405126, 3.1205695
4: -1.1176394, 2.6857204, -1.0453598, 2.5924871, -3.7101264, 3.7310803

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7709401, upper bound: 2.7658043
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3665972, 2.3256445, -0.2820300, 1.9873157, -2.3539128, 2.6076746
1: -0.4696044, 3.2193704, -0.3970020, 2.7511249, -3.2207294, 3.6163723
2: -1.1476570, 2.2576227, -0.9615395, 1.9399827, -3.0876398, 3.2191622
3: -0.9491717, 2.6523957, -0.8055222, 2.1982713, -3.1474428, 3.4579179
4: -1.3314362, 2.9754171, -1.0453598, 2.5924871, -3.9239233, 4.0207767

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7544898, upper bound: 2.7533442
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3352187, 2.1514726, -0.2820300, 1.9873157, -2.3225343, 2.4335027
1: -0.4334549, 2.9734864, -0.3970020, 2.7511249, -3.1845798, 3.3704884
2: -1.0506161, 2.0943696, -0.9615395, 1.9399827, -2.9905987, 3.0559092
3: -0.8786925, 2.4222009, -0.8055222, 2.1982713, -3.0769639, 3.2277231
4: -1.1862975, 2.7643881, -1.0453598, 2.5924871, -3.7787848, 3.8097479

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2933785, 1.9254615, -0.2889379, 1.9049499, -2.1983285, 2.2143993
1: -0.3881600, 2.6552501, -0.3853487, 2.6697907, -3.0579507, 3.0405989
2: -0.9234380, 1.8988812, -0.9618609, 1.8210847, -2.7445226, 2.8607421
3: -0.7931215, 2.1667268, -0.7863698, 2.1205049, -2.9136262, 2.9530966
4: -1.0140216, 2.4781418, -1.0324315, 2.4797454, -3.4937670, 3.5105734

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7610972, upper bound: 2.7580870
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7751869
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7752053
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3074463, 1.9780474, -0.2889379, 1.9049499, -2.2123961, 2.2669852
1: -0.3971092, 2.7358575, -0.3853487, 2.6697907, -3.0669000, 3.1212063
2: -0.9541693, 1.9347870, -0.9618609, 1.8210847, -2.7752540, 2.8966479
3: -0.8110955, 2.2350283, -0.7863698, 2.1205049, -2.9316003, 3.0213981
4: -1.0563571, 2.5269520, -1.0324315, 2.4797454, -3.5361025, 3.5593834

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7610964, upper bound: 2.7580856
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7746611
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7746792
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.3231174, 2.0654058, -0.2889379, 1.9049499, -2.2280674, 2.3543437
1: -0.4157538, 2.8843560, -0.3853487, 2.6697907, -3.0855446, 3.2697048
2: -1.0334489, 1.9777371, -0.9618609, 1.8210847, -2.8545337, 2.9395981
3: -0.8464797, 2.3422604, -0.7863698, 2.1205049, -2.9669845, 3.1286302
4: -1.1675417, 2.6453815, -1.0324315, 2.4797454, -3.6472871, 3.6778131

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7752053
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7746792
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.3465745, 2.2301984, -0.2889379, 1.9049499, -2.2515244, 2.5191362
1: -0.4452873, 3.0852342, -0.3853487, 2.6697907, -3.1150780, 3.4705830
2: -1.0840776, 2.1610651, -0.9618609, 1.8210847, -2.9051623, 3.1229260
3: -0.9021518, 2.5303700, -0.7863698, 2.1205049, -3.0226567, 3.3167398
4: -1.2483793, 2.8331316, -1.0324315, 2.4797454, -3.7281246, 3.8655629

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7752051
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7746792
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3578414, 2.3138692, -0.2890179, 1.9990376, -2.3568790, 2.6028872
1: -0.4637145, 3.2022610, -0.4002408, 2.7723556, -3.2360702, 3.6025019
2: -1.1388912, 2.2404590, -0.9748017, 1.9446510, -3.0835423, 3.2152605
3: -0.9356593, 2.6202211, -0.8124224, 2.2182291, -3.1538882, 3.4326434
4: -1.3105438, 2.9649637, -1.0574362, 2.6030040, -3.9135478, 4.0223999

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3649357, 2.3264618, -0.2890179, 1.9990376, -2.3639734, 2.6154797
1: -0.4678703, 3.2243400, -0.4002408, 2.7723556, -3.2402258, 3.6245809
2: -1.1521792, 2.2473352, -0.9748017, 1.9446510, -3.0968304, 3.2221370
3: -0.9445353, 2.6441708, -0.8124224, 2.2182291, -3.1627643, 3.4565930
4: -1.3248823, 2.9783530, -1.0574362, 2.6030040, -3.9278862, 4.0357895

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3409981, 2.1949234, -0.2890179, 1.9990376, -2.3400357, 2.4839413
1: -0.4416623, 3.0343451, -0.4002408, 2.7723556, -3.2140179, 3.4345860
2: -1.0756550, 2.1337790, -0.9748017, 1.9446510, -3.0203061, 3.1085806
3: -0.8936343, 2.4786105, -0.8124224, 2.2182291, -3.1118634, 3.2910328
4: -1.2270881, 2.8188548, -1.0574362, 2.6030040, -3.8300920, 3.8762910

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7754919
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7752053
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3452685, 2.2133813, -0.2890179, 1.9990376, -2.3443060, 2.5023992
1: -0.4452651, 3.0645375, -0.4002408, 2.7723556, -3.2176206, 3.4647784
2: -1.0893548, 2.1448741, -0.9748017, 1.9446510, -3.0340056, 3.1196756
3: -0.9010228, 2.5083795, -0.8124224, 2.2182291, -3.1192517, 3.3208017
4: -1.2427499, 2.8374698, -1.0574362, 2.6030040, -3.8457539, 3.8949060

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7747319
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7746791
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2806621, 1.8875258, -0.3411655, 2.1992579, -2.4799199, 2.2286913
1: -0.3806360, 2.6402259, -0.4431228, 3.0523765, -3.4330125, 3.0833488
2: -0.9465249, 1.8131994, -1.0892384, 2.1301794, -3.0767043, 2.9024377
3: -0.7766682, 2.0958865, -0.8978004, 2.4804626, -3.2571306, 2.9936869
4: -1.0165755, 2.4664505, -1.2308609, 2.8298907, -3.8464661, 3.6973114

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2806621, 1.8875258, -0.3501772, 2.2155571, -2.4962192, 2.2377031
1: -0.3806360, 2.6402259, -0.4485214, 3.0818214, -3.4624574, 3.0887473
2: -0.9465249, 1.8131994, -1.1055143, 2.1401927, -3.0867176, 2.9187136
3: -0.7766682, 2.0958865, -0.9091074, 2.5106003, -3.2872686, 3.0049939
4: -1.0165755, 2.4664505, -1.2508855, 2.8473842, -3.8639598, 3.7173359

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2756731, 1.9642451, -0.3753327, 2.3569636, -2.6326368, 2.3395777
1: -0.3918094, 2.7198572, -0.4759032, 3.2600689, -3.6518784, 3.1957603
2: -0.9491056, 1.9184160, -1.1634135, 2.2862463, -3.2353520, 3.0818295
3: -0.7951761, 2.1666198, -0.9616026, 2.6963146, -3.4914906, 3.1282225
4: -1.0252471, 2.5676005, -1.3583037, 3.0054598, -4.0307069, 3.9259043

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7723261
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7723261
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2847048, 1.9835527, -0.3753327, 2.3569636, -2.6416683, 2.3588853
1: -0.3965250, 2.7515988, -0.4759032, 3.2600689, -3.6565938, 3.2275019
2: -0.9664359, 1.9298962, -1.1634135, 2.2862463, -3.2526822, 3.0933099
3: -0.8049830, 2.1974092, -0.9616026, 2.6963146, -3.5012975, 3.1590118
4: -1.0439100, 2.5858889, -1.3583037, 3.0054598, -4.0493698, 3.9441924

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693506, upper bound: 2.7723261
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693506, upper bound: 2.7723261
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2458000, 1.7208488, -0.3536721, 2.2504725, -2.4962726, 2.0745208
1: -0.3454927, 2.4054158, -0.4539548, 3.1193335, -3.4648261, 2.8593705
2: -0.8536351, 1.6646116, -1.1137347, 2.1815369, -3.0351720, 2.7783463
3: -0.7079096, 1.8960061, -0.9190763, 2.5514789, -3.2593884, 2.8150823
4: -0.9026028, 2.2619066, -1.2742250, 2.8849490, -3.7875519, 3.5361316

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7674585, upper bound: 2.7759766
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7674585, upper bound: 2.7759766
time: 0.55 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2705797, 1.8707315, -0.3753327, 2.3569636, -2.6275434, 2.2460642
1: -0.3789070, 2.5838137, -0.4759032, 3.2600689, -3.6389759, 3.0597167
2: -0.9023267, 1.8385874, -1.1634135, 2.2862463, -3.1885729, 3.0020008
3: -0.7723326, 2.0637321, -0.9616026, 2.6963146, -3.4686472, 3.0253348
4: -0.9726133, 2.4480281, -1.3583037, 3.0054598, -3.9780731, 3.8063316

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678142, upper bound: 2.7756463
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678142, upper bound: 2.7756463
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3629161, 2.3064585, -2.5996525, 2.3192017
1: -0.3966470, 2.7066140, -0.4652137, 3.1939573, -3.5906043, 3.1718278
2: -0.9499824, 1.9157126, -1.1392360, 2.2357109, -3.1856933, 3.0549486
3: -0.8073892, 2.1858242, -0.9405786, 2.6256416, -3.4330308, 3.1264029
4: -1.0437701, 2.5382841, -1.3152579, 2.9512329, -3.9950030, 3.8535419

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675807, upper bound: 2.7733116
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7723650
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2931940, 1.9562856, -0.3708772, 2.3199959, -2.6131899, 2.3271627
1: -0.3966470, 2.7066140, -0.4697720, 3.2195220, -3.6161690, 3.1763859
2: -0.9499824, 1.9157126, -1.1537372, 2.2423575, -3.1923399, 3.0694499
3: -0.8073892, 2.1858242, -0.9502540, 2.6516135, -3.4590027, 3.1360781
4: -1.0437701, 2.5382841, -1.3322339, 2.9660292, -4.0097990, 3.8705180

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675807, upper bound: 2.7733116
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7723650
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3182679, 2.1371770, -0.3708772, 2.3199959, -2.6382637, 2.5080543
1: -0.4268867, 2.9595909, -0.4697720, 3.2195220, -3.6464088, 3.4293628
2: -1.0412936, 2.0729208, -1.1537372, 2.2423575, -3.2836511, 3.2266579
3: -0.8644805, 2.3999858, -0.9502540, 2.6516135, -3.5160940, 3.3502398
4: -1.1726522, 2.7533128, -1.3322339, 2.9660292, -4.1386814, 4.0855465

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7478249, upper bound: 2.7475631
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7478429
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3276167, 2.1572509, -0.3708772, 2.3199959, -2.6476126, 2.5281281
1: -0.4325817, 2.9916034, -0.4697720, 3.2195220, -3.6521037, 3.4613752
2: -1.0583760, 2.0858326, -1.1537372, 2.2423575, -3.3007336, 3.2395697
3: -0.8765814, 2.4330173, -0.9502540, 2.6516135, -3.5281949, 3.3832712
4: -1.1936884, 2.7738743, -1.3322339, 2.9660292, -4.1597176, 4.1061082

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7478249, upper bound: 2.7475631
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7478429
time: 0.45 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 2.57 seconds
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740592
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740592
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740592
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7779449, upper bound: 2.7740856
IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7709401, upper bound: 2.7658043
IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7544898, upper bound: 2.7533442
IS_A2_B2_A2_B1_B1_B1_B2_A1_A2_B2, status: Status.VERIFIED, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7751869
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7752053
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7746611
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7752053
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7752051
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7751236, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7660595
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7754919
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7752053
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7747319
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7745400, upper bound: 2.7746791
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7677857, upper bound: 2.7736951
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7723261
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7678042, upper bound: 2.7723261
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7693506, upper bound: 2.7723261
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7693506, upper bound: 2.7723261
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7674585, upper bound: 2.7759766
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7674585, upper bound: 2.7759766
IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7678142, upper bound: 2.7756463
IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7678142, upper bound: 2.7756463
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7675807, upper bound: 2.7733116
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7723650
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7675807, upper bound: 2.7733116
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7690471, upper bound: 2.7723650
IS_A2_B2_A2_B1_B2_A2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7478249, upper bound: 2.7475631
IS_A2_B2_A2_B1_B2_A2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7478429
IS_A2_B2_A2_B1_B2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7478249, upper bound: 2.7475631
IS_A2_B2_A2_B1_B2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 2.57
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7478429

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2742207, 1.9354399, -0.2717244, 1.8525931, -2.1268139, 2.2071643
1: -0.3859153, 2.6822429, -0.3729628, 2.5932012, -2.9791164, 3.0552058
2: -0.9369248, 1.8887759, -0.9286113, 1.7790446, -2.7159696, 2.8173871
3: -0.7844094, 2.1373806, -0.7617036, 2.0464137, -2.8308229, 2.8990841
4: -1.0066166, 2.5245895, -0.9849690, 2.4273558, -3.4339724, 3.5095587

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7661277, upper bound: 2.7685349
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745409
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745411
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2667916, 1.8328767, -0.2717244, 1.8525931, -2.1193848, 2.1046011
1: -0.3707577, 2.5342102, -0.3729628, 2.5932012, -2.9639587, 2.9071732
2: -0.8837141, 1.8015516, -0.9286113, 1.7790446, -2.6627588, 2.7301629
3: -0.7573927, 2.0254762, -0.7617036, 2.0464137, -2.8038063, 2.7871799
4: -0.9488536, 2.3947587, -0.9849690, 2.4273558, -3.3762093, 3.3797278

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7661277, upper bound: 2.7685341
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745435
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745435
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2881563, 1.9737076, -0.2717244, 1.8525931, -2.1407495, 2.2454319
1: -0.3943375, 2.7422757, -0.3729628, 2.5932012, -2.9875388, 3.1152387
2: -0.9641743, 1.9185419, -0.9286113, 1.7790446, -2.7432189, 2.8471532
3: -0.8016165, 2.1940377, -0.7617036, 2.0464137, -2.8480301, 2.9557414
4: -1.0413297, 2.5623701, -0.9849690, 2.4273558, -3.4686856, 3.5473390

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7654552, upper bound: 2.7611498
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740592
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740592
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2770597, 1.8715515, -0.2717244, 1.8525931, -2.1296527, 2.1432760
1: -0.3775736, 2.5947576, -0.3729628, 2.5932012, -2.9707747, 2.9677205
2: -0.9088897, 1.8306731, -0.9286113, 1.7790446, -2.6879344, 2.7592845
3: -0.7709047, 2.0766945, -0.7617036, 2.0464137, -2.8173184, 2.8383980
4: -0.9763392, 2.4333420, -0.9849690, 2.4273558, -3.4036951, 3.4183111

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7654552, upper bound: 2.7611498
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740857
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740857
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3109539, 2.0162413, -0.2717244, 1.8525931, -2.1635470, 2.2879658
1: -0.4051618, 2.8182204, -0.3729628, 2.5932012, -2.9983630, 3.1911831
2: -1.0086639, 1.9288844, -0.9286113, 1.7790446, -2.7877085, 2.8574958
3: -0.8253425, 2.2734647, -0.7617036, 2.0464137, -2.8717561, 3.0351682
4: -1.1241037, 2.5908298, -0.9849690, 2.4273558, -3.5514593, 3.5757990

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18

Time for candidate selection: 2.48 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731158, upper bound: 2.7736986
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7741935
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3202214, 2.0352769, -0.2717244, 1.8525931, -2.1728144, 2.3070014
1: -0.4117236, 2.8488946, -0.3729628, 2.5932012, -3.0049248, 3.2218575
2: -1.0251302, 1.9413912, -0.9286113, 1.7790446, -2.8041749, 2.8700025
3: -0.8389386, 2.3040559, -0.7617036, 2.0464137, -2.8853521, 3.0657597
4: -1.1437246, 2.6106558, -0.9849690, 2.4273558, -3.5710802, 3.5956249

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 18

Time for candidate selection: 2.51 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731158, upper bound: 2.7736986
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7741935
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3320477, 2.1800489, -0.2717244, 1.8525931, -2.1846409, 2.4517734
1: -0.4339747, 3.0176001, -0.3729628, 2.5932012, -3.0271759, 3.3905630
2: -1.0587735, 2.1115923, -0.9286113, 1.7790446, -2.8378181, 3.0402036
3: -0.8793849, 2.4603066, -0.7617036, 2.0464137, -2.9257984, 3.2220101
4: -1.2040153, 2.7777414, -0.9849690, 2.4273558, -3.6313710, 3.7627106

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 41

Time for candidate selection: 2.61 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727574, upper bound: 2.7711751
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7717660
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3379650, 2.1988478, -0.2717244, 1.8525931, -2.1905580, 2.4705722
1: -0.4380774, 3.0492973, -0.3729628, 2.5932012, -3.0312786, 3.4222603
2: -1.0738218, 2.1241112, -0.9286113, 1.7790446, -2.8528664, 3.0527225
3: -0.8880580, 2.4930398, -0.7617036, 2.0464137, -2.9344716, 3.2547436
4: -1.2224107, 2.7969630, -0.9849690, 2.4273558, -3.6497664, 3.7819319

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 18

Time for candidate selection: 2.65 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727574, upper bound: 2.7711751
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7717660
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2888264, 1.9945775, -0.2820300, 1.9873157, -2.2761421, 2.2766075
1: -0.4000432, 2.7575843, -0.3970020, 2.7511249, -3.1511681, 3.1545863
2: -0.9618342, 1.9491395, -0.9615395, 1.9399827, -2.9018168, 2.9106789
3: -0.8125226, 2.2124579, -0.8055222, 2.1982713, -3.0107939, 3.0179801
4: -1.0490496, 2.5925679, -1.0453598, 2.5924871, -3.6415367, 3.6379278

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769998, upper bound: 2.7727325
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769998, upper bound: 2.7724724
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2796247, 1.8826946, -0.2820300, 1.9873157, -2.2669404, 2.1647246
1: -0.3829975, 2.5972061, -0.3970020, 2.7511249, -3.1341224, 2.9942081
2: -0.9045649, 1.8533102, -0.9615395, 1.9399827, -2.8445475, 2.8148499
3: -0.7816354, 2.0899224, -0.8055222, 2.1982713, -2.9799066, 2.8954446
4: -0.9799299, 2.4523220, -1.0453598, 2.5924871, -3.5724170, 3.4976819

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3154575, 2.1055789, -0.2820300, 1.9873157, -2.3027732, 2.3876090
1: -0.4214576, 2.9155774, -0.3970020, 2.7511249, -3.1725826, 3.3125794
2: -1.0235951, 2.0444477, -0.9615395, 1.9399827, -2.9635777, 3.0059872
3: -0.8554188, 2.3600154, -0.8055222, 2.1982713, -3.0536900, 3.1655376
4: -1.1441643, 2.7125952, -1.0453598, 2.5924871, -3.7366514, 3.7579551

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2742207, 1.9354399, -0.2889379, 1.9049499, -2.1791706, 2.2243779
1: -0.3859153, 2.6822429, -0.3853487, 2.6697907, -3.0557060, 3.0675917
2: -0.9369248, 1.8887759, -0.9618609, 1.8210847, -2.7580094, 2.8506370
3: -0.7844094, 2.1373806, -0.7863698, 2.1205049, -2.9049144, 2.9237504
4: -1.0066166, 2.5245895, -1.0324315, 2.4797454, -3.4863620, 3.5570211

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7623966, upper bound: 2.7672984
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753251
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753251
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2667916, 1.8328767, -0.2889379, 1.9049499, -2.1717415, 2.1218145
1: -0.3707577, 2.5342102, -0.3853487, 2.6697907, -3.0405483, 2.9195590
2: -0.8837141, 1.8015516, -0.9618609, 1.8210847, -2.7047987, 2.7634125
3: -0.7573927, 2.0254762, -0.7863698, 2.1205049, -2.8778975, 2.8118460
4: -0.9488536, 2.3947587, -1.0324315, 2.4797454, -3.4285989, 3.4271903

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7623966, upper bound: 2.7673002
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753271
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753271
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2881563, 1.9737076, -0.2889379, 1.9049499, -2.1931062, 2.2626455
1: -0.3943375, 2.7422757, -0.3853487, 2.6697907, -3.0641284, 3.1276245
2: -0.9641743, 1.9185419, -0.9618609, 1.8210847, -2.7852590, 2.8804028
3: -0.8016165, 2.1940377, -0.7863698, 2.1205049, -2.9221213, 2.9804075
4: -1.0413297, 2.5623701, -1.0324315, 2.4797454, -3.5210752, 3.5948014

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7610972, upper bound: 2.7580870
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746611
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746611
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2770597, 1.8715515, -0.2889379, 1.9049499, -2.1820097, 2.1604893
1: -0.3775736, 2.5947576, -0.3853487, 2.6697907, -3.0473642, 2.9801064
2: -0.9088897, 1.8306731, -0.9618609, 1.8210847, -2.7299743, 2.7925339
3: -0.7709047, 2.0766945, -0.7863698, 2.1205049, -2.8914094, 2.8630643
4: -0.9763392, 2.4333420, -1.0324315, 2.4797454, -3.4560847, 3.4657736

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7610964, upper bound: 2.7580856
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746792
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746791
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3109539, 2.0162413, -0.2889379, 1.9049499, -2.2159038, 2.3051791
1: -0.4051618, 2.8182204, -0.3853487, 2.6697907, -3.0749526, 3.2035689
2: -1.0086639, 1.9288844, -0.9618609, 1.8210847, -2.8297486, 2.8907452
3: -0.8253425, 2.2734647, -0.7863698, 2.1205049, -2.9458475, 3.0598345
4: -1.1241037, 2.5908298, -1.0324315, 2.4797454, -3.6038489, 3.6232615

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18

Time for candidate selection: 2.57 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710236, upper bound: 2.7760116
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7736057, upper bound: 2.7765065
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3202214, 2.0352769, -0.2889379, 1.9049499, -2.2251713, 2.3242147
1: -0.4117236, 2.8488946, -0.3853487, 2.6697907, -3.0815144, 3.2342434
2: -1.0251302, 1.9413912, -0.9618609, 1.8210847, -2.8462148, 2.9032521
3: -0.8389386, 2.3040559, -0.7863698, 2.1205049, -2.9594436, 3.0904257
4: -1.1437246, 2.6106558, -1.0324315, 2.4797454, -3.6234698, 3.6430874

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18

Time for candidate selection: 2.56 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710236, upper bound: 2.7748529
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7736057, upper bound: 2.7753646
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3320477, 2.1800489, -0.2889379, 1.9049499, -2.2369976, 2.4689867
1: -0.4339747, 3.0176001, -0.3853487, 2.6697907, -3.1037655, 3.4029489
2: -1.0587735, 2.1115923, -0.9618609, 1.8210847, -2.8798583, 3.0734532
3: -0.8793849, 2.4603066, -0.7863698, 2.1205049, -2.9998899, 3.2466764
4: -1.2040153, 2.7777414, -1.0324315, 2.4797454, -3.6837606, 3.8101730

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 41

Time for candidate selection: 2.65 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7721732, upper bound: 2.7694934
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727245, upper bound: 2.7725973
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3379650, 2.1988478, -0.2889379, 1.9049499, -2.2429149, 2.4877856
1: -0.4380774, 3.0492973, -0.3853487, 2.6697907, -3.1078682, 3.4346461
2: -1.0738218, 2.1241112, -0.9618609, 1.8210847, -2.8949065, 3.0859721
3: -0.8880580, 2.4930398, -0.7863698, 2.1205049, -3.0085628, 3.2794096
4: -1.2224107, 2.7969630, -1.0324315, 2.4797454, -3.7021561, 3.8293943

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 41

Time for candidate selection: 2.67 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698548, upper bound: 2.7718295
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727245, upper bound: 2.7723777
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2909694, 2.0227432, -0.2890179, 1.9990376, -2.2900071, 2.3117611
1: -0.4052326, 2.7972746, -0.4002408, 2.7723556, -3.1775882, 3.1975155
2: -0.9782555, 1.9761380, -0.9748017, 1.9446510, -2.9229064, 2.9509397
3: -0.8216132, 2.2480628, -0.8124224, 2.2182291, -3.0398421, 3.0604854
4: -1.0757260, 2.6311049, -1.0574362, 2.6030040, -3.6787300, 3.6885412

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7670011, upper bound: 2.7659914
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533576, upper bound: 2.7533477
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.3536609, 2.2733560, -0.2890179, 1.9990376, -2.3526986, 2.5623739
1: -0.4585382, 3.1511228, -0.4002408, 2.7723556, -3.2308939, 3.5513635
2: -1.1225939, 2.2052293, -0.9748017, 1.9446510, -3.0672450, 3.1800308
3: -0.9274352, 2.5793722, -0.8124224, 2.2182291, -3.1456642, 3.3917947
4: -1.2869700, 2.9193895, -1.0574362, 2.6030040, -3.8899741, 3.9768257

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533547, upper bound: 2.7530929
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533576, upper bound: 2.7533477
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.3042246, 2.0575032, -0.2890179, 1.9990376, -2.3032622, 2.3465211
1: -0.4130379, 2.8520617, -0.4002408, 2.7723556, -3.1853933, 3.2523026
2: -1.0041143, 2.0024467, -0.9748017, 1.9446510, -2.9487653, 2.9772482
3: -0.8375547, 2.3009794, -0.8124224, 2.2182291, -3.0557837, 3.1134019
4: -1.1082883, 2.6651578, -1.0574362, 2.6030040, -3.7112923, 3.7225940

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7666747, upper bound: 2.7658292
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.3630877, 2.2926280, -0.2890179, 1.9990376, -2.3621254, 2.5816460
1: -0.4641411, 3.1831021, -0.4002408, 2.7723556, -3.2364967, 3.5833430
2: -1.1396843, 2.2169552, -0.9748017, 1.9446510, -3.0843353, 3.1917567
3: -0.9391701, 2.6123335, -0.8124224, 2.2182291, -3.1573992, 3.4247561
4: -1.3082904, 2.9394026, -1.0574362, 2.6030040, -3.9112945, 3.9968388

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515212, upper bound: 2.7513427
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2997308, 1.9636698, -0.2890179, 1.9990376, -2.2987685, 2.2526877
1: -0.3961045, 2.7462509, -0.4002408, 2.7723556, -3.1684601, 3.1464915
2: -0.9927866, 1.8816230, -0.9748017, 1.9446510, -2.9374375, 2.8564248
3: -0.8057804, 2.2014689, -0.8124224, 2.2182291, -3.0240095, 3.0138912
4: -1.0913504, 2.5538836, -1.0574362, 2.6030040, -3.6943545, 3.6113198

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7783693
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7783966
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3191950, 2.1188316, -0.2890179, 1.9990376, -2.3182325, 2.4078496
1: -0.4243059, 2.9335325, -0.4002408, 2.7723556, -3.1966615, 3.3337731
2: -1.0395632, 2.0570145, -0.9748017, 1.9446510, -2.9842143, 3.0318160
3: -0.8582977, 2.3725855, -0.8124224, 2.2182291, -3.0765266, 3.1850080
4: -1.1624240, 2.7384832, -1.0574362, 2.6030040, -3.7654281, 3.7959194

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7753268
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7753268
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3077968, 1.9852371, -0.2890179, 1.9990376, -2.3068345, 2.2742550
1: -0.4024566, 2.7807822, -0.4002408, 2.7723556, -3.1748123, 3.1810231
2: -1.0082695, 1.8957026, -0.9748017, 1.9446510, -2.9529204, 2.8705044
3: -0.8189222, 2.2298522, -0.8124224, 2.2182291, -3.0371513, 3.0422745
4: -1.1089811, 2.5756531, -1.0574362, 2.6030040, -3.7119851, 3.6330893

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7747319
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7747319
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3229951, 2.1372235, -0.2890179, 1.9990376, -2.3220327, 2.4262414
1: -0.4279416, 2.9636190, -0.4002408, 2.7723556, -3.2002971, 3.3638597
2: -1.0527887, 2.0682073, -0.9748017, 1.9446510, -2.9974399, 3.0430088
3: -0.8656837, 2.3995986, -0.8124224, 2.2182291, -3.0839128, 3.2120209
4: -1.1768974, 2.7573173, -1.0574362, 2.6030040, -3.7799015, 3.8147535

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7746792
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7746792
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2677826, 1.8381294, -0.3411655, 2.1992579, -2.4670405, 2.1792948
1: -0.3697071, 2.5732079, -0.4431228, 3.0523765, -3.4220836, 3.0163307
2: -0.9205275, 1.7657945, -1.0892384, 2.1301794, -3.0507069, 2.8550329
3: -0.7550522, 2.0262730, -0.8978004, 2.4804626, -3.2355146, 2.9240735
4: -0.9717644, 2.4110608, -1.2308609, 2.8298907, -3.8016553, 3.6419218

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 41

Time for candidate selection: 2.72 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7634963, upper bound: 2.7647070
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7656358, upper bound: 2.7712183
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2853531, 1.8911947, -0.3411655, 2.1992579, -2.4846110, 2.2323601
1: -0.3823905, 2.6506677, -0.4431228, 3.0523765, -3.4347670, 3.0937905
2: -0.9541993, 1.8089762, -1.0892384, 2.1301794, -3.0843787, 2.8982146
3: -0.7804253, 2.1019111, -0.8978004, 2.4804626, -3.2608879, 2.9997115
4: -1.0201142, 2.4645104, -1.2308609, 2.8298907, -3.8500049, 3.6953714

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 41

Time for candidate selection: 2.72 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7627036, upper bound: 2.7707965
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7656358, upper bound: 2.7712184
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2677826, 1.8381294, -0.3501772, 2.2155571, -2.4833398, 2.1883066
1: -0.3697071, 2.5732079, -0.4485214, 3.0818214, -3.4515285, 3.0217292
2: -0.9205275, 1.7657945, -1.1055143, 2.1401927, -3.0607202, 2.8713088
3: -0.7550522, 2.0262730, -0.9091074, 2.5106003, -3.2656527, 2.9353805
4: -0.9717644, 2.4110608, -1.2508855, 2.8473842, -3.8191485, 3.6619463

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 18

Time for candidate selection: 2.62 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7634963, upper bound: 2.7661962
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7654782, upper bound: 2.7706501
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2853531, 1.8911947, -0.3501772, 2.2155571, -2.5009103, 2.2413719
1: -0.3823905, 2.6506677, -0.4485214, 3.0818214, -3.4642119, 3.0991890
2: -0.9541993, 1.8089762, -1.1055143, 2.1401927, -3.0943921, 2.9144905
3: -0.7804253, 2.1019111, -0.9091074, 2.5106003, -3.2910256, 3.0110185
4: -1.0201142, 2.4645104, -1.2508855, 2.8473842, -3.8674984, 3.7153959

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 41

Time for candidate selection: 2.71 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7625419, upper bound: 2.7702287
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7654782, upper bound: 2.7706501
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2756731, 1.9642451, -0.3629161, 2.3064585, -2.5821316, 2.3271611
1: -0.3918094, 2.7198572, -0.4652137, 3.1939573, -3.5857668, 3.1850710
2: -0.9491056, 1.9184160, -1.1392360, 2.2357109, -3.1848164, 3.0576520
3: -0.7951761, 2.1666198, -0.9405786, 2.6256416, -3.4208176, 3.1071985
4: -1.0252471, 2.5676005, -1.3152579, 2.9512329, -3.9764800, 3.8828583

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663368, upper bound: 2.7730405
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7595657, upper bound: 2.7626000
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2756731, 1.9642451, -0.3708772, 2.3199959, -2.5956690, 2.3351223
1: -0.3918094, 2.7198572, -0.4697720, 3.2195220, -3.6113315, 3.1896291
2: -0.9491056, 1.9184160, -1.1537372, 2.2423575, -3.1914630, 3.0721531
3: -0.7951761, 2.1666198, -0.9502540, 2.6516135, -3.4467895, 3.1168737
4: -1.0252471, 2.5676005, -1.3322339, 2.9660292, -3.9912763, 3.8998344

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663368, upper bound: 2.7730405
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7595657, upper bound: 2.7626000
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2847048, 1.9835527, -0.3629161, 2.3064585, -2.5911632, 2.3464687
1: -0.3965250, 2.7515988, -0.4652137, 3.1939573, -3.5904822, 3.2168126
2: -0.9664359, 1.9298962, -1.1392360, 2.2357109, -3.2021468, 3.0691323
3: -0.8049830, 2.1974092, -0.9405786, 2.6256416, -3.4306245, 3.1379879
4: -1.0439100, 2.5858889, -1.3152579, 2.9512329, -3.9951429, 3.9011469

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675707, upper bound: 2.7697524
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7596147, upper bound: 2.7583180
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2847048, 1.9835527, -0.3708772, 2.3199959, -2.6047006, 2.3544300
1: -0.3965250, 2.7515988, -0.4697720, 3.2195220, -3.6160469, 3.2213707
2: -0.9664359, 1.9298962, -1.1537372, 2.2423575, -3.2087934, 3.0836334
3: -0.8049830, 2.1974092, -0.9502540, 2.6516135, -3.4565964, 3.1476631
4: -1.0439100, 2.5858889, -1.3322339, 2.9660292, -4.0099392, 3.9181228

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675707, upper bound: 2.7697524
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7596146, upper bound: 2.7583180
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2458000, 1.7208488, -0.3411655, 2.1992579, -2.4450579, 2.0620143
1: -0.3454927, 2.4054158, -0.4431228, 3.0523765, -3.3978691, 2.8485386
2: -0.8536351, 1.6646116, -1.0892384, 2.1301794, -2.9838145, 2.7538500
3: -0.7079096, 1.8960061, -0.8978004, 2.4804626, -3.1883721, 2.7938066
4: -0.9026028, 2.2619066, -1.2308609, 2.8298907, -3.7324934, 3.4927676

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 18

Time for candidate selection: 2.53 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7625419, upper bound: 2.7718604
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7651227, upper bound: 2.7731087
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2458000, 1.7208488, -0.3501772, 2.2155571, -2.4613571, 2.0710261
1: -0.3454927, 2.4054158, -0.4485214, 3.0818214, -3.4273140, 2.8539371
2: -0.8536351, 1.6646116, -1.1055143, 2.1401927, -2.9938278, 2.7701259
3: -0.7079096, 1.8960061, -0.9091074, 2.5106003, -3.2185099, 2.8051136
4: -0.9026028, 2.2619066, -1.2508855, 2.8473842, -3.7499871, 3.5127921

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 18

Time for candidate selection: 2.67 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7609458, upper bound: 2.7686439
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7651227, upper bound: 2.7731087
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2705797, 1.8707315, -0.3629161, 2.3064585, -2.5770383, 2.2336476
1: -0.3789070, 2.5838137, -0.4652137, 3.1939573, -3.5728643, 3.0490274
2: -0.9023267, 1.8385874, -1.1392360, 2.2357109, -3.1380377, 2.9778233
3: -0.7723326, 2.0637321, -0.9405786, 2.6256416, -3.3979743, 3.0043108
4: -0.9726133, 2.4480281, -1.3152579, 2.9512329, -3.9238462, 3.7632861

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7518036, upper bound: 2.7622824
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7517690, upper bound: 2.7591097
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2705797, 1.8707315, -0.3708772, 2.3199959, -2.5905757, 2.2416086
1: -0.3789070, 2.5838137, -0.4697720, 3.2195220, -3.5984290, 3.0535855
2: -0.9023267, 1.8385874, -1.1537372, 2.2423575, -3.1446843, 2.9923246
3: -0.7723326, 2.0637321, -0.9502540, 2.6516135, -3.4239461, 3.0139861
4: -0.9726133, 2.4480281, -1.3322339, 2.9660292, -3.9386425, 3.7802620

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7518036, upper bound: 2.7622824
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7517690, upper bound: 2.7591097
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2648096, 1.7750605, -0.3411655, 2.1992579, -2.4640675, 2.1162260
1: -0.3593262, 2.4844408, -0.4431228, 3.0523765, -3.4117026, 2.9275637
2: -0.8891013, 1.7091408, -1.0892384, 2.1301794, -3.0192807, 2.7983792
3: -0.7361317, 1.9713337, -0.8978004, 2.4804626, -3.2165942, 2.8691342
4: -0.9439448, 2.3187432, -1.2308609, 2.8298907, -3.7738357, 3.5496042

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 41

Time for candidate selection: 2.56 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7634499, upper bound: 2.7705947
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7653844, upper bound: 2.7708325
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.2747284, 1.8879770, -0.3629161, 2.3064585, -2.5811868, 2.2508931
1: -0.3815731, 2.6145062, -0.4652137, 3.1939573, -3.5755305, 3.0797200
2: -0.9161618, 1.8483543, -1.1392360, 2.2357109, -3.1518726, 2.9875903
3: -0.7776511, 2.0904698, -0.9405786, 2.6256416, -3.4032927, 3.0310485
4: -0.9852120, 2.4648352, -1.3152579, 2.9512329, -3.9364448, 3.7800932

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7615205, upper bound: 2.7607130
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 18

Time for candidate selection: 3.27 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7645503, upper bound: 2.7635281
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7669663, upper bound: 2.7698212
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.2648096, 1.7750605, -0.3501772, 2.2155571, -2.4803667, 2.1252377
1: -0.3593262, 2.4844408, -0.4485214, 3.0818214, -3.4411478, 2.9329622
2: -0.8891013, 1.7091408, -1.1055143, 2.1401927, -3.0292940, 2.8146551
3: -0.7361317, 1.9713337, -0.9091074, 2.5106003, -3.2467320, 2.8804412
4: -0.9439448, 2.3187432, -1.2508855, 2.8473842, -3.7913289, 3.5696287

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 18

Time for candidate selection: 2.62 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7631810, upper bound: 2.7699912
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7650956, upper bound: 2.7702321
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.2747284, 1.8879770, -0.3708772, 2.3199959, -2.5947242, 2.2588542
1: -0.3815731, 2.6145062, -0.4697720, 3.2195220, -3.6010952, 3.0842781
2: -0.9161618, 1.8483543, -1.1537372, 2.2423575, -3.1585193, 3.0020914
3: -0.7776511, 2.0904698, -0.9502540, 2.6516135, -3.4292645, 3.0407238
4: -0.9852120, 2.4648352, -1.3322339, 2.9660292, -3.9512410, 3.7970691

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7594909, upper bound: 2.7584911
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 18

Time for candidate selection: 3.16 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7654793, upper bound: 2.7644474
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667986, upper bound: 2.7692660
time: 0.50 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 7.01 seconds
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745409
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745411
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745435
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7776079, upper bound: 2.7745435
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740592
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740592
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7731158, upper bound: 2.7736986
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7741935
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7731158, upper bound: 2.7736986
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7741935
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7727574, upper bound: 2.7711751
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7717660
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7727574, upper bound: 2.7711751
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7717660
IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7769998, upper bound: 2.7727325
IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7769998, upper bound: 2.7724724
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753251
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753251
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753271
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753271
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746611
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746611
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746791
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7710236, upper bound: 2.7760116
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7736057, upper bound: 2.7765065
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7710236, upper bound: 2.7748529
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7736057, upper bound: 2.7753646
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7721732, upper bound: 2.7694934
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7727245, upper bound: 2.7725973
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7698548, upper bound: 2.7718295
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7727245, upper bound: 2.7723777
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7670011, upper bound: 2.7659914
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7533576, upper bound: 2.7533477
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7533547, upper bound: 2.7530929
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7533576, upper bound: 2.7533477
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7666747, upper bound: 2.7658292
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_B1, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7515212, upper bound: 2.7513427
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A2_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7783693
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7783966
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7753268
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7753268
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7747319
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7747319
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7746792
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7634963, upper bound: 2.7647070
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7656358, upper bound: 2.7712183
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7627036, upper bound: 2.7707965
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7656358, upper bound: 2.7712184
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7634963, upper bound: 2.7661962
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7654782, upper bound: 2.7706501
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7625419, upper bound: 2.7702287
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7654782, upper bound: 2.7706501
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7663368, upper bound: 2.7730405
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7595657, upper bound: 2.7626000
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7663368, upper bound: 2.7730405
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7595657, upper bound: 2.7626000
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7675707, upper bound: 2.7697524
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7596147, upper bound: 2.7583180
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7675707, upper bound: 2.7697524
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7596146, upper bound: 2.7583180
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7625419, upper bound: 2.7718604
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7651227, upper bound: 2.7731087
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7609458, upper bound: 2.7686439
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7651227, upper bound: 2.7731087
IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7518036, upper bound: 2.7622824
IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B1_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7517690, upper bound: 2.7591097
IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7518036, upper bound: 2.7622824
IS_A2_B2_A2_B1_B2_A1_A2_A1_A2_B2_B2, status: Status.VERIFIED, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7517690, upper bound: 2.7591097
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7634499, upper bound: 2.7705947
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7653844, upper bound: 2.7708325
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7645503, upper bound: 2.7635281
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7669663, upper bound: 2.7698212
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7631810, upper bound: 2.7699912
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7650956, upper bound: 2.7702321
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7654793, upper bound: 2.7644474
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 7.01
Output dim: 0, lower bound: -2.7667986, upper bound: 2.7692660

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2677826, 1.8381294, -0.2717244, 1.8525931, -2.1203756, 2.1098537
1: -0.3697071, 2.5732079, -0.3729628, 2.5932012, -2.9629083, 2.9461708
2: -0.9205275, 1.7657945, -0.9286113, 1.7790446, -2.6995721, 2.6944058
3: -0.7550522, 2.0262730, -0.7617036, 2.0464137, -2.8014660, 2.7879767
4: -0.9717644, 2.4110608, -0.9849690, 2.4273558, -3.3991203, 3.3960299

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18

Time for candidate selection: 2.20 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741227, upper bound: 2.7682413
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754644, upper bound: 2.7719431
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2756731, 1.9642451, -0.2717244, 1.8525931, -2.1282661, 2.2359695
1: -0.3918094, 2.7198572, -0.3729628, 2.5932012, -2.9850106, 3.0928202
2: -0.9491056, 1.9184160, -0.9286113, 1.7790446, -2.7281504, 2.8470273
3: -0.7951761, 2.1666198, -0.7617036, 2.0464137, -2.8415897, 2.9283233
4: -1.0252471, 2.5676005, -0.9849690, 2.4273558, -3.4526029, 3.5525694

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 41

Time for candidate selection: 2.23 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741227, upper bound: 2.7682413
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754644, upper bound: 2.7719431
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2458000, 1.7208488, -0.2717244, 1.8525931, -2.0983930, 1.9925733
1: -0.3454927, 2.4054158, -0.3729628, 2.5932012, -2.9386938, 2.7783785
2: -0.8536351, 1.6646116, -0.9286113, 1.7790446, -2.6326797, 2.5932229
3: -0.7079096, 1.8960061, -0.7617036, 2.0464137, -2.7543232, 2.6577096
4: -0.9026028, 2.2619066, -0.9849690, 2.4273558, -3.3299584, 3.2468758

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7655572, upper bound: 2.7684281
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 18

Time for candidate selection: 2.63 seconds

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727154, upper bound: 2.7720611
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754833, upper bound: 2.7718245
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2705797, 1.8702737, -0.2717244, 1.8525931, -2.1231728, 2.1419981
1: -0.3789070, 2.5829499, -0.3729628, 2.5932012, -2.9721081, 2.9559126
2: -0.9023267, 1.8380101, -0.9286113, 1.7790446, -2.6813712, 2.7666214
3: -0.7723326, 2.0621796, -0.7617036, 2.0464137, -2.8187463, 2.8238831
4: -0.9695092, 2.4480281, -0.9849690, 2.4273558, -3.3968649, 3.4329972

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7655572, upper bound: 2.7685349
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 41

Time for candidate selection: 2.74 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741411, upper bound: 2.7680991
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754833, upper bound: 2.7718243
time: 0.41 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 6.45 seconds
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7741227, upper bound: 2.7682413
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7754644, upper bound: 2.7719431
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7741227, upper bound: 2.7682413
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7754644, upper bound: 2.7719431
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7727154, upper bound: 2.7720611
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7754833, upper bound: 2.7718245
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7741411, upper bound: 2.7680991
IS_A2_B2_A2_B1_B1_B1_B1_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 6.45
Output dim: 0, lower bound: -2.7754833, upper bound: 2.7718243
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740592
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740592
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7791713, upper bound: 2.7740857
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7731158, upper bound: 2.7736986
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7741935
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7731158, upper bound: 2.7736986
IS_A2_B2_A2_B1_B1_B1_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7741935
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7727574, upper bound: 2.7711751
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7717660
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7727574, upper bound: 2.7711751
IS_A2_B2_A2_B1_B1_B1_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7758176, upper bound: 2.7717660
IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7769998, upper bound: 2.7727325
IS_A2_B2_A2_B1_B1_B1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7769998, upper bound: 2.7724724
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7734514, upper bound: 2.7690101
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753251
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753251
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753271
IS_A2_B2_A2_B1_B1_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7747674, upper bound: 2.7753271
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746611
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746611
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7765559, upper bound: 2.7746791
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7710236, upper bound: 2.7760116
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7736057, upper bound: 2.7765065
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7710236, upper bound: 2.7748529
IS_A2_B2_A2_B1_B1_B2_B1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7736057, upper bound: 2.7753646
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7721732, upper bound: 2.7694934
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7727245, upper bound: 2.7725973
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7698548, upper bound: 2.7718295
IS_A2_B2_A2_B1_B1_B2_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7727245, upper bound: 2.7723777
IS_A2_B2_A2_B1_B1_B2_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7670011, upper bound: 2.7659914
IS_A2_B2_A2_B1_B1_B2_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7666747, upper bound: 2.7658292
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7783693
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7783966
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7753268
IS_A2_B2_A2_B1_B1_B2_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7742335, upper bound: 2.7753268
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7747319
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7747319
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7746792
IS_A2_B2_A2_B1_B1_B2_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7756363, upper bound: 2.7746792
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7634963, upper bound: 2.7647070
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7656358, upper bound: 2.7712183
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7627036, upper bound: 2.7707965
IS_A2_B2_A2_B1_B2_A1_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7656358, upper bound: 2.7712184
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7634963, upper bound: 2.7661962
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7654782, upper bound: 2.7706501
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7625419, upper bound: 2.7702287
IS_A2_B2_A2_B1_B2_A1_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7654782, upper bound: 2.7706501
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7663368, upper bound: 2.7730405
IS_A2_B2_A2_B1_B2_A1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7663368, upper bound: 2.7730405
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7675707, upper bound: 2.7697524
IS_A2_B2_A2_B1_B2_A1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7675707, upper bound: 2.7697524
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7625419, upper bound: 2.7718604
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7651227, upper bound: 2.7731087
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7609458, upper bound: 2.7686439
IS_A2_B2_A2_B1_B2_A1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7651227, upper bound: 2.7731087
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7634499, upper bound: 2.7705947
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7653844, upper bound: 2.7708325
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7645503, upper bound: 2.7635281
IS_A2_B2_A2_B1_B2_A1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7669663, upper bound: 2.7698212
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7631810, upper bound: 2.7699912
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7650956, upper bound: 2.7702321
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7654793, upper bound: 2.7644474
IS_A2_B2_A2_B1_B2_A1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.45
Output dim: 0, lower bound: -2.7667986, upper bound: 2.7692660
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=3.285133123397827
rel_dist={0: [-2.7803829052250393, 2.7803829052250393]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720536, upper bound: 2.7716346
time: 0.38 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802701, upper bound: 2.7802701
time: 0.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.90 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -2.7720536, upper bound: 2.7716346
IS_B2, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -2.7802701, upper bound: 2.7802701

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.4416689, 2.5614049, -0.3937300, 2.2607875, -2.7024565, 2.9551349
1: -0.5173086, 3.5417147, -0.4659128, 3.1165721, -3.6338806, 4.0076275
2: -1.2744279, 2.4602361, -1.1388535, 2.1871104, -3.4615383, 3.5990896
3: -1.0457175, 3.0060875, -0.9433906, 2.6073360, -3.6530535, 3.9494781
4: -1.5548723, 3.2051432, -1.3197460, 2.8582294, -4.4131017, 4.5248890

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7626402, upper bound: 2.7615758
time: 0.34 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7695544, upper bound: 2.7684479
time: 0.40 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.5095432, 2.7755899, -0.5007467, 2.7479298, -3.2574730, 3.2763367
1: -0.5611423, 3.8165379, -0.5553178, 3.7793705, -4.3405128, 4.3718557
2: -1.3674926, 2.7016737, -1.3528485, 2.6755908, -4.0430832, 4.0545225
3: -1.1338987, 3.3577619, -1.1222062, 3.3138933, -4.4477921, 4.4799681
4: -1.7360522, 3.4053361, -1.7101582, 3.3786907, -5.1147428, 5.1154943

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7616859, upper bound: 2.7627959
time: 0.37 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802493, upper bound: 2.7802493
time: 0.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.23 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 2.23
Output dim: 0, lower bound: -2.7626402, upper bound: 2.7615758
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -2.7695544, upper bound: 2.7684479
IS_B2_A1, status: Status.VERIFIED, split count: 2, time: 2.23
Output dim: 0, lower bound: -2.7616859, upper bound: 2.7627959
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -2.7802493, upper bound: 2.7802493

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -0.4346480, 2.5385976, -0.3660870, 2.1625853, -2.5972333, 2.9046845
1: -0.5126103, 3.5111556, -0.4462218, 2.9898553, -3.5024655, 3.9573774
2: -1.2628827, 2.4379914, -1.0902050, 2.0946541, -3.3575368, 3.5281963
3: -1.0362089, 2.9705076, -0.9038171, 2.4661355, -3.5023444, 3.8743248
4: -1.5330046, 3.1822793, -1.2328745, 2.7640190, -4.2970238, 4.4151540

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B2_A1

### Relational analysis result of IS_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7595307, upper bound: 2.7593811
time: 0.40 seconds

## Relational analysis of IS_B1_B2_A2

### Relational analysis result of IS_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7595307, upper bound: 2.7684479
time: 0.35 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.4836221, 2.6905560, -0.4932828, 2.7232521, -3.2068741, 3.1838388
1: -0.5440643, 3.7076769, -0.5504035, 3.7479644, -4.2920289, 4.2580805
2: -1.3264129, 2.6181285, -1.3410298, 2.6514711, -3.9778841, 3.9591584
3: -1.0996997, 3.2257874, -1.1123638, 3.2757754, -4.3754749, 4.3381510
4: -1.6587552, 3.3243561, -1.6879110, 3.3553219, -5.0140772, 5.0122671

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802493, upper bound: 2.7801530
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802493, upper bound: 2.7802493
time: 0.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.30 seconds
IS_B1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 0, lower bound: -2.7595307, upper bound: 2.7593811
IS_B1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -2.7595307, upper bound: 2.7684479
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -2.7802493, upper bound: 2.7801530
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -2.7802493, upper bound: 2.7802493

## BFS IS instance: IS_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4174378, 2.4826860, -0.3660870, 2.1625853, -2.5800231, 2.8487730
1: -0.5010115, 3.4363956, -0.4462218, 2.9898553, -3.4908667, 3.8826175
2: -1.2345645, 2.3842325, -1.0902050, 2.0946541, -3.3292186, 3.4744375
3: -1.0127002, 2.8843901, -0.9038171, 2.4661355, -3.4788356, 3.7882071
4: -1.4810382, 3.1259696, -1.2328745, 2.7640190, -4.2450571, 4.3588443

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B1_B2_A2_A1

### Relational analysis result of IS_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7419771, upper bound: 2.7612175
time: 0.37 seconds

## Relational analysis of IS_B1_B2_A2_A2

### Relational analysis result of IS_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7324292, upper bound: 2.7544693
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.4250804, 2.5852566, -0.4451678, 2.5806375, -3.0057178, 3.0304244
1: -0.5211089, 3.5761673, -0.5209967, 3.5599334, -4.0810423, 4.0971642
2: -1.2865007, 2.4982405, -1.2741983, 2.4979095, -3.7844102, 3.7724388
3: -1.0481679, 3.0023003, -1.0522817, 3.0286231, -4.0767908, 4.0545821
4: -1.5452410, 3.2716055, -1.5573623, 3.2266369, -4.7718778, 4.8289680

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801775, upper bound: 2.7799282
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743155, upper bound: 2.7678531
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.4047078, 2.4600189, -0.4550146, 2.6126692, -3.0173771, 2.9150333
1: -0.4967405, 3.4003437, -0.5280871, 3.6034970, -4.1002374, 3.9284308
2: -1.2186558, 2.3767066, -1.2909219, 2.5311210, -3.7497768, 3.6676285
3: -1.0018195, 2.8326006, -1.0666820, 3.0787945, -4.0806141, 3.8992825
4: -1.4532027, 3.1177409, -1.5862188, 3.2589617, -4.7121644, 4.7039595

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7624818, upper bound: 2.7616579
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7624819, upper bound: 2.7802493
time: 0.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.29 seconds
IS_B1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7419771, upper bound: 2.7612175
IS_B1_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7324292, upper bound: 2.7544693
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7801775, upper bound: 2.7799282
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7743155, upper bound: 2.7678531
IS_B2_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7624818, upper bound: 2.7616579
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7624819, upper bound: 2.7802493

## BFS IS instance: IS_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.3076048, 2.0884678, -0.3769161, 2.2807512, -2.5883560, 2.4653840
1: -0.4196330, 2.8851950, -0.4596784, 3.1412697, -3.5609026, 3.3448734
2: -1.0115439, 2.0395269, -1.1072766, 2.2329235, -3.2444673, 3.1468034
3: -0.8498347, 2.3376288, -0.9333512, 2.6402533, -3.4900880, 3.2709801
4: -1.1316509, 2.7040546, -1.3050870, 2.8765621, -4.0082130, 4.0091414

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796784, upper bound: 2.7774580
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790801, upper bound: 2.7782493
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.3753327, 2.3569636, -0.4104998, 2.4450758, -2.8204083, 2.7674634
1: -0.4759032, 3.2600689, -0.4927739, 3.3758583, -3.8517613, 3.7528429
2: -1.1634135, 2.2862463, -1.2001539, 2.3684855, -3.5318990, 3.4864001
3: -0.9616026, 2.6963146, -0.9973001, 2.8427961, -3.8043985, 3.6936147
4: -1.3583037, 3.0054598, -1.4429557, 3.0700436, -4.4283471, 4.4484158

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A2_B1

### Relational analysis result of IS_B2_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7605301, upper bound: 2.7554607
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2

### Relational analysis result of IS_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7605302, upper bound: 2.7664014
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4047078, 2.4600189, -0.4375663, 2.5565093, -2.9612172, 2.8975852
1: -0.4967405, 3.4003437, -0.5163251, 3.5281458, -4.0248861, 3.9166689
2: -1.2186558, 2.3767066, -1.2617882, 2.4721022, -3.6907580, 3.6384950
3: -1.0018195, 2.8326006, -1.0429542, 2.9900739, -3.9918933, 3.8755548
4: -1.4532027, 3.1177409, -1.5331733, 3.2024903, -4.6556931, 4.6509142

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7597917, upper bound: 2.7801145
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7624819, upper bound: 2.7801145
time: 0.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.30 seconds
IS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -2.7796784, upper bound: 2.7774580
IS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -2.7790801, upper bound: 2.7782493
IS_B2_A2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.30
Output dim: 0, lower bound: -2.7605301, upper bound: 2.7554607
IS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -2.7605302, upper bound: 2.7664014
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -2.7597917, upper bound: 2.7801145
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.30
Output dim: 0, lower bound: -2.7624819, upper bound: 2.7801145

## BFS IS instance: IS_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2818971, 1.9536383, -0.3172453, 2.0206575, -2.3025546, 2.2708836
1: -0.3898954, 2.7076778, -0.4059046, 2.8135741, -3.2034695, 3.1135824
2: -0.9478806, 1.9045155, -1.0074635, 1.9452149, -2.8930955, 2.9119790
3: -0.7926321, 2.1656933, -0.8269950, 2.2903981, -3.0830302, 2.9926882
4: -1.0248098, 2.5395565, -1.1349168, 2.5894983, -3.6143081, 3.6744733

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790801, upper bound: 2.7774580
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790801, upper bound: 2.7774580
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3034027, 2.0719838, -0.3468301, 2.1838102, -2.4872129, 2.4188139
1: -0.4159359, 2.8633971, -0.4381542, 3.0114746, -3.4274106, 3.3015513
2: -1.0035447, 2.0233877, -1.0590985, 2.1322331, -3.1357780, 3.0824862
3: -0.8425676, 2.3149433, -0.8887890, 2.4853041, -3.3278718, 3.2037323
4: -1.1178741, 2.6864338, -1.2168373, 2.7797287, -3.8976028, 3.9032712

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7782493
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757661, upper bound: 2.7756363
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3753327, 2.3569636, -0.3919042, 2.3856845, -2.7610173, 2.7488678
1: -0.4759032, 3.2600689, -0.4802060, 3.2960272, -3.7719302, 3.7402749
2: -1.1634135, 2.2862463, -1.1695201, 2.3103971, -3.4738107, 3.4557664
3: -0.9616026, 2.6963146, -0.9720069, 2.7496693, -3.7112718, 3.6683216
4: -1.3583037, 3.0054598, -1.3875504, 3.0101581, -4.3684616, 4.3930101

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7605154, upper bound: 2.7565502
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577663, upper bound: 2.7541046
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3022718, 1.9925733, -0.3642406, 2.2400019, -2.5422738, 2.3568139
1: -0.4058715, 2.7465084, -0.4519633, 3.0871091, -3.4929805, 3.1984715
2: -0.9626346, 1.9593803, -1.0869578, 2.1925445, -3.1551790, 3.0463381
3: -0.8254330, 2.2351263, -0.9176500, 2.5782814, -3.4037144, 3.1527762
4: -1.0756483, 2.5819404, -1.2679894, 2.8364604, -3.9121087, 3.8499298

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757669, upper bound: 2.7801145
time: 0.45 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783787, upper bound: 2.7787060
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3389214, 2.2148643, -0.4012665, 2.4172280, -2.7561493, 2.6161308
1: -0.4439855, 3.0646944, -0.4869935, 3.3384731, -3.7824585, 3.5516880
2: -1.0810840, 2.1477108, -1.1852508, 2.3409665, -3.4220505, 3.3329616
3: -0.8988513, 2.5088573, -0.9857627, 2.7983422, -3.6971936, 3.4946198
4: -1.2414910, 2.8358724, -1.4144943, 3.0423744, -4.2838655, 4.2503667

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792429, upper bound: 2.7796443
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7783620
time: 0.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.28 seconds
IS_B2_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7790801, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7790801, upper bound: 2.7774580
IS_B2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7742291, upper bound: 2.7782493
IS_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7757661, upper bound: 2.7756363
IS_B2_A2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7605154, upper bound: 2.7565502
IS_B2_A2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7577663, upper bound: 2.7541046
IS_B2_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7757669, upper bound: 2.7801145
IS_B2_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7783787, upper bound: 2.7787060
IS_B2_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7792429, upper bound: 2.7796443
IS_B2_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.28
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7783620

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2844167, 1.9012821, -0.3172453, 2.0206575, -2.3050742, 2.2185273
1: -0.3836576, 2.6593406, -0.4059046, 2.8135741, -3.1972318, 3.0652452
2: -0.9543507, 1.8255115, -1.0074635, 1.9452149, -2.8995657, 2.8329749
3: -0.7827784, 2.1150155, -0.8269950, 2.2903981, -3.0731764, 2.9420104
4: -1.0292071, 2.4816728, -1.1349168, 2.5894983, -3.6187053, 3.6165895

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796784, upper bound: 2.7774580
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796784, upper bound: 2.7774580
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2919451, 2.0275006, -0.3172453, 2.0206575, -2.3126025, 2.3447459
1: -0.4058816, 2.8045843, -0.4059046, 2.8135741, -3.2194557, 3.2104888
2: -0.9817809, 1.9796841, -1.0074635, 1.9452149, -2.9269958, 2.9871476
3: -0.8227718, 2.2539344, -0.8269950, 2.2903981, -3.1131699, 3.0809293
4: -1.0805242, 2.6382742, -1.1349168, 2.5894983, -3.6700225, 3.7731910

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7757802
time: 0.44 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2933387, 2.0308990, -0.3468301, 2.1838102, -2.4771490, 2.3777292
1: -0.4069531, 2.8087568, -0.4381542, 3.0114746, -3.4184277, 3.2469110
2: -0.9831082, 1.9829079, -1.0590985, 2.1322331, -3.1153412, 3.0420065
3: -0.8251112, 2.2583125, -0.8887890, 2.4853041, -3.3104153, 3.1471014
4: -1.0825255, 2.6399064, -1.2168373, 2.7797287, -3.8622541, 3.8567438

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535870, upper bound: 2.7552049
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7535870, upper bound: 2.7779715
time: 0.37 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3035588, 2.0542922, -0.3467528, 2.1835070, -2.4870658, 2.4010451
1: -0.4125428, 2.8472562, -0.4380866, 3.0110672, -3.4236100, 3.2853427
2: -1.0028613, 1.9987938, -1.0589467, 2.1319237, -3.1347849, 3.0577407
3: -0.8366684, 2.2953863, -0.8886551, 2.4848809, -3.3215494, 3.1840415
4: -1.1052647, 2.6622438, -1.2165774, 2.7793901, -3.8846548, 3.8788214

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526668, upper bound: 2.7503910
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526669, upper bound: 2.7503911
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.2913533, 1.9494442, -0.3642406, 2.2400019, -2.5313554, 2.3136847
1: -0.3965143, 2.6892331, -0.4519633, 3.0871091, -3.4836235, 3.1411963
2: -0.9412345, 1.9158479, -1.0869578, 2.1925445, -3.1337790, 3.0028057
3: -0.8072340, 2.1748338, -0.9176500, 2.5782814, -3.3855155, 3.0924838
4: -1.0388029, 2.5333719, -1.2679894, 2.8364604, -3.8752632, 3.8013613

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7783335
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.2971243, 1.9722085, -0.3641655, 2.2397013, -2.5368257, 2.3363740
1: -0.4001294, 2.7280126, -0.4518974, 3.0867066, -3.4868360, 3.1799099
2: -0.9578750, 1.9310058, -1.0868068, 2.1922441, -3.1501191, 3.0178127
3: -0.8142713, 2.2070019, -0.9175199, 2.5778639, -3.3921351, 3.1245217
4: -1.0563898, 2.5555344, -1.2677337, 2.8361254, -3.8925152, 3.8232679

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760656, upper bound: 2.7778191
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7754416
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.3029714, 1.9758003, -0.3645653, 2.2454779, -2.5484493, 2.3403656
1: -0.4003568, 2.7667799, -0.4507895, 3.1085761, -3.5089328, 3.2175694
2: -0.9991634, 1.8891554, -1.1022553, 2.1741815, -3.1733449, 2.9914107
3: -0.8155400, 2.2180204, -0.9154611, 2.5681973, -3.3837371, 3.1334815
4: -1.1026624, 2.5684884, -1.2737134, 2.8464110, -3.9490733, 3.8422017

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7753514
time: 0.46 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7783620
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3140926, 2.1232934, -0.3927014, 2.3930073, -2.7070999, 2.5159948
1: -0.4233531, 2.9424019, -0.4813277, 3.3061571, -3.7295103, 3.4237294
2: -1.0367260, 2.0574467, -1.1728952, 2.3161459, -3.3528719, 3.2303419
3: -0.8564682, 2.3797078, -0.9739293, 2.7560031, -3.6124713, 3.3536372
4: -1.1613970, 2.7425013, -1.3917834, 3.0180120, -4.1794090, 4.1342845

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667024, upper bound: 2.7680316
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7783620
time: 0.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.41 seconds
IS_B2_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7796784, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7796784, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7757802
IS_B2_A2_A1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7535870, upper bound: 2.7552049
IS_B2_A2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7535870, upper bound: 2.7779715
IS_B2_A2_A1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7526668, upper bound: 2.7503910
IS_B2_A2_A1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7526669, upper bound: 2.7503911
IS_B2_A2_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
IS_B2_A2_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7783335
IS_B2_A2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7760656, upper bound: 2.7778191
IS_B2_A2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7754416
IS_B2_A2_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7753514
IS_B2_A2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7783620
IS_B2_A2_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7667024, upper bound: 2.7680316
IS_B2_A2_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.41
Output dim: 0, lower bound: -2.7790223, upper bound: 2.7783620

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2844167, 1.9012821, -0.2782299, 1.8474497, -2.1318665, 2.1795120
1: -0.3836576, 2.6593406, -0.3716577, 2.5725513, -2.9562089, 3.0309982
2: -0.9543507, 1.8255115, -0.9150798, 1.7869606, -2.7413113, 2.7405913
3: -0.7827784, 2.1150155, -0.7603004, 2.0667460, -2.8495245, 2.8753159
4: -1.0292071, 2.4816728, -0.9986765, 2.3961380, -3.4253449, 3.4803493

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7799282
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7786571
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2844167, 1.9012821, -0.3287072, 2.0794437, -2.3638604, 2.2299893
1: -0.3836576, 2.6593406, -0.4193263, 2.9038768, -3.2875345, 3.0786669
2: -0.9543507, 1.8255115, -1.0449104, 1.9904075, -2.9447582, 2.8704219
3: -0.7827784, 2.1150155, -0.8534839, 2.3650570, -3.1478353, 2.9684994
4: -1.0292071, 2.4816728, -1.1882963, 2.6632438, -3.6924510, 3.6699691

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7799282
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7786571
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2820300, 1.9873157, -0.3172453, 2.0206575, -2.3026876, 2.3045609
1: -0.3970020, 2.7511249, -0.4059046, 2.8135741, -3.2105761, 3.1570294
2: -0.9615395, 1.9399827, -1.0074635, 1.9452149, -2.9067545, 2.9474461
3: -0.8055222, 2.1982713, -0.8269950, 2.2903981, -3.0959203, 3.0252662
4: -1.0453598, 2.5924871, -1.1349168, 2.5894983, -3.6348581, 3.7274039

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2890179, 1.9990376, -0.3171669, 2.0203631, -2.3093810, 2.3162045
1: -0.4002408, 2.7723556, -0.4058405, 2.8131759, -3.2134166, 3.1781960
2: -0.9748017, 1.9446510, -1.0073085, 1.9449313, -2.9197330, 2.9519596
3: -0.8124224, 2.2182291, -0.8268667, 2.2899933, -3.1024156, 3.0450959
4: -1.0574362, 2.6030040, -1.1346540, 2.5891700, -3.6466062, 3.7376580

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7757802
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7757802
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2933387, 2.0308990, -0.3286344, 2.1222587, -2.4155974, 2.3595333
1: -0.4069531, 2.8087568, -0.4252181, 2.9291036, -3.3360567, 3.2339749
2: -0.9831082, 1.9829079, -1.0281134, 2.0698578, -3.0529661, 3.0110211
3: -0.8251112, 2.2583125, -0.8625802, 2.3921413, -3.2172525, 3.1208925
4: -1.0825255, 2.6399064, -1.1610584, 2.7175934, -3.8001189, 3.8009648

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7534056, upper bound: 2.7745106
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7534056, upper bound: 2.7779715
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.2498522, 1.7357445, -0.3282668, 2.0704947, -2.3203468, 2.0640113
1: -0.3489222, 2.4258590, -0.4162362, 2.8597138, -3.2086360, 2.8420951
2: -0.8619993, 1.6778553, -1.0047234, 2.0267153, -2.8887146, 2.6825786
3: -0.7148793, 1.9152986, -0.8481596, 2.3521070, -3.0669863, 2.7634583
4: -0.9132509, 2.2788439, -1.1295152, 2.6423745, -3.5556254, 3.4083591

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.2764732, 1.8929052, -0.3563612, 2.2154276, -2.4919009, 2.2492664
1: -0.3837109, 2.6139765, -0.4462438, 3.0544045, -3.4381154, 3.0602202
2: -0.9139111, 1.8594116, -1.0744777, 2.1667128, -3.0806239, 2.9338894
3: -0.7817896, 2.0936637, -0.9058086, 2.5385308, -3.3203204, 2.9994721
4: -0.9912694, 2.4718633, -1.2449846, 2.8116820, -3.8029513, 3.7168479

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7517243, upper bound: 2.7622349
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7505308, upper bound: 2.7566708
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -0.2686107, 1.7899325, -0.3281909, 2.0701923, -2.3388031, 2.1181235
1: -0.3625808, 2.5046179, -0.4161705, 2.8593082, -3.2218890, 2.9207883
2: -0.8971289, 1.7222278, -1.0045698, 2.0264115, -2.9235406, 2.7267976
3: -0.7427063, 1.9909970, -0.8480300, 2.3516846, -3.0943909, 2.8390269
4: -0.9565146, 2.3351691, -1.1292567, 2.6420372, -3.5985518, 3.4644258

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7751748
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7754416
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -0.2799054, 1.9078822, -0.3562855, 2.2151256, -2.4950309, 2.2641678
1: -0.3859175, 2.6413178, -0.4461774, 3.0539994, -3.4399168, 3.0874953
2: -0.9264836, 1.8670630, -1.0743264, 2.1664083, -3.0928919, 2.9413896
3: -0.7861645, 2.1172347, -0.9056774, 2.5381083, -3.3242729, 3.0229120
4: -1.0017887, 2.4863176, -1.2447263, 2.8113456, -3.8131342, 3.7310438

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7744970
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7747662
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3029714, 1.9758003, -0.3399627, 2.1412797, -2.4442511, 2.3157630
1: -0.4003568, 2.7667799, -0.4318531, 2.9897790, -3.3901358, 3.1986330
2: -0.9991634, 1.8891554, -1.0796771, 2.0471630, -3.0463264, 2.9688325
3: -0.8155400, 2.2180204, -0.8770121, 2.4389844, -3.2545242, 3.0950327
4: -1.1026624, 2.5684884, -1.2337385, 2.7423425, -3.8450050, 3.8022270

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747307, upper bound: 2.7796443
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759834, upper bound: 2.7777564
time: 0.46 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3029714, 1.9758003, -0.3684069, 2.3242297, -2.6272011, 2.3442073
1: -0.4003568, 2.7667799, -0.4651923, 3.2138858, -3.6142426, 3.2319722
2: -0.9991634, 1.8891554, -1.1377780, 2.2485032, -3.2476666, 3.0269334
3: -0.8155400, 2.2180204, -0.9401004, 2.6491013, -3.4646411, 3.1581209
4: -1.1026624, 2.5684884, -1.3273412, 2.9488654, -4.0515280, 3.8958297

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792429, upper bound: 2.7765702
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759834, upper bound: 2.7777564
time: 0.47 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.4461385, 2.7662539, -0.3363041, 2.2255838, -2.6717224, 3.1025581
1: -0.5598933, 3.8898566, -0.4438351, 3.0810294, -3.6409225, 4.3336916
2: -1.3933828, 2.6360474, -1.0872071, 2.1544590, -3.5478418, 3.7232544
3: -1.1361436, 3.2483170, -0.8972466, 2.5058210, -3.6419644, 4.1455636
4: -1.6816521, 3.5539007, -1.2343891, 2.8551092, -4.5367613, 4.7882900

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7609055, upper bound: 2.7605008
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7531744, upper bound: 2.7548267
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.3097464, 2.1038358, -0.3927014, 2.3930073, -2.7027538, 2.4965372
1: -0.4192142, 2.9166768, -0.4813277, 3.3061571, -3.7253714, 3.3980045
2: -1.0282929, 2.0379205, -1.1728952, 2.3161459, -3.3444388, 3.2108157
3: -0.8484409, 2.3540249, -0.9739293, 2.7560031, -3.6044440, 3.3279543
4: -1.1468025, 2.7217529, -1.3917834, 3.0180120, -4.1648145, 4.1135364

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7743100
time: 0.36 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7783620
time: 0.40 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 2.46 seconds
IS_B2_A2_A1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7799282
IS_B2_A2_A1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7786571
IS_B2_A2_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7799282
IS_B2_A2_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7786571
IS_B2_A2_A1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7757802
IS_B2_A2_A1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7757802
IS_B2_A2_A1_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7534056, upper bound: 2.7745106
IS_B2_A2_A1_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7534056, upper bound: 2.7779715
IS_B2_A2_A2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
IS_B2_A2_A2_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
IS_B2_A2_A2_B2_A1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7517243, upper bound: 2.7622349
IS_B2_A2_A2_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7505308, upper bound: 2.7566708
IS_B2_A2_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7751748
IS_B2_A2_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7754416
IS_B2_A2_A2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7744970
IS_B2_A2_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7747662
IS_B2_A2_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7747307, upper bound: 2.7796443
IS_B2_A2_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7759834, upper bound: 2.7777564
IS_B2_A2_A2_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7792429, upper bound: 2.7765702
IS_B2_A2_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7759834, upper bound: 2.7777564
IS_B2_A2_A2_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7609055, upper bound: 2.7605008
IS_B2_A2_A2_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7531744, upper bound: 2.7548267
IS_B2_A2_A2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7743100
IS_B2_A2_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.46
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7783620

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2717244, 1.8525931, -0.2782299, 1.8474497, -2.1191740, 2.1308229
1: -0.3729628, 2.5932012, -0.3716577, 2.5725513, -2.9455142, 2.9648590
2: -0.9286113, 1.7790446, -0.9150798, 1.7869606, -2.7155719, 2.6941245
3: -0.7617036, 2.0464137, -0.7603004, 2.0667460, -2.8284497, 2.8067141
4: -0.9849690, 2.4273558, -0.9986765, 2.3961380, -3.3811069, 3.4260323

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7800357
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7800357
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.2889379, 1.9049499, -0.2781777, 1.8472334, -2.1361713, 2.1831276
1: -0.3853487, 2.6697907, -0.3716098, 2.5722616, -2.9576101, 3.0414004
2: -0.9618609, 1.8210847, -0.9149680, 1.7867472, -2.7486081, 2.7360528
3: -0.7863698, 2.1205049, -0.7602062, 2.0664544, -2.8528242, 2.8807111
4: -1.0324315, 2.4797454, -0.9984884, 2.3958895, -3.4283209, 3.4782338

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7764222
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7788416
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2717244, 1.8525931, -0.3287072, 2.0794437, -2.3511682, 2.1813002
1: -0.3729628, 2.5932012, -0.4193263, 2.9038768, -3.2768397, 3.0125275
2: -0.9286113, 1.7790446, -1.0449104, 1.9904075, -2.9190187, 2.8239551
3: -0.7617036, 2.0464137, -0.8534839, 2.3650570, -3.1267605, 2.8998976
4: -0.9849690, 2.4273558, -1.1882963, 2.6632438, -3.6482129, 3.6156521

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7763369
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7786571
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.2889379, 1.9049499, -0.3286379, 2.0791621, -2.3680999, 2.2335877
1: -0.3853487, 2.6697907, -0.4192663, 2.9034960, -3.2888446, 3.0890570
2: -0.9618609, 1.8210847, -1.0447648, 1.9901369, -2.9519978, 2.8658495
3: -0.7863698, 2.1205049, -0.8533641, 2.3646705, -3.1510403, 2.9738688
4: -1.0324315, 2.4797454, -1.1880488, 2.6629286, -3.6953602, 3.6677942

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7763369
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7786571
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2820300, 1.9873157, -0.2782299, 1.8474497, -2.1294796, 2.2655456
1: -0.3970020, 2.7511249, -0.3716577, 2.5725513, -2.9695532, 3.1227827
2: -0.9615395, 1.9399827, -0.9150798, 1.7869606, -2.7485001, 2.8550625
3: -0.8055222, 2.1982713, -0.7603004, 2.0667460, -2.8722682, 2.9585717
4: -1.0453598, 2.5924871, -0.9986765, 2.3961380, -3.4414978, 3.5911636

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7764103
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2820300, 1.9873157, -0.3287072, 2.0794437, -2.3614736, 2.3160229
1: -0.3970020, 2.7511249, -0.4193263, 2.9038768, -3.3008788, 3.1704512
2: -0.9615395, 1.9399827, -1.0449104, 1.9904075, -2.9519470, 2.9848931
3: -0.8055222, 2.1982713, -0.8534839, 2.3650570, -3.1705792, 3.0517552
4: -1.0453598, 2.5924871, -1.1882963, 2.6632438, -3.7086036, 3.7807834

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7764103
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2890179, 1.9990376, -0.2781777, 1.8472334, -2.1362514, 2.2772155
1: -0.4002408, 2.7723556, -0.3716098, 2.5722616, -2.9725022, 3.1439652
2: -0.9748017, 1.9446510, -0.9149680, 1.7867472, -2.7615490, 2.8596191
3: -0.8124224, 2.2182291, -0.7602062, 2.0664544, -2.8788767, 2.9784353
4: -1.0574362, 2.6030040, -0.9984884, 2.3958895, -3.4533257, 3.6014924

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7742335
time: 0.46 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7748342
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2890179, 1.9990376, -0.3286379, 2.0791621, -2.3681800, 2.3276756
1: -0.4002408, 2.7723556, -0.4192663, 2.9034960, -3.3037367, 3.1916218
2: -0.9748017, 1.9446510, -1.0447648, 1.9901369, -2.9649386, 2.9894156
3: -0.8124224, 2.2182291, -0.8533641, 2.3646705, -3.1770930, 3.0715933
4: -1.0574362, 2.6030040, -1.1880488, 2.6629286, -3.7203648, 3.7910528

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7743062
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7748397
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2933387, 2.0308990, -0.3191969, 2.1560540, -2.4493928, 2.3500960
1: -0.4069531, 2.8087568, -0.4299223, 2.9855862, -3.3925393, 3.2386792
2: -0.9831082, 1.9829079, -1.0537493, 2.0934625, -3.0765705, 3.0366573
3: -0.8251112, 2.2583125, -0.8691810, 2.4104123, -3.2355235, 3.1274934
4: -1.0825255, 2.6399064, -1.1781275, 2.7858243, -3.8683498, 3.8180339

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7553012, upper bound: 2.7577949
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541295, upper bound: 2.7571481
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2933387, 2.0308990, -0.3080829, 2.0485001, -2.3418388, 2.3389819
1: -0.4069531, 2.8087568, -0.4115947, 2.8330650, -3.2400181, 3.2203517
2: -0.9831082, 1.9829079, -0.9985578, 1.9972117, -2.9803200, 2.9814658
3: -0.8251112, 2.2583125, -0.8345631, 2.2880187, -3.1131299, 3.0928755
4: -1.0825255, 2.6399064, -1.1085553, 2.6515625, -3.7340879, 3.7484617

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7553012, upper bound: 2.7577949
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541295, upper bound: 2.7571481
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2498522, 1.7357445, -0.2922257, 1.9101553, -2.1600075, 2.0279703
1: -0.3489222, 2.4258590, -0.3858527, 2.6366305, -2.9855528, 2.8117118
2: -0.8619993, 1.6778553, -0.9193681, 1.8825984, -2.7445977, 2.5972233
3: -0.7148793, 1.9152986, -0.7886305, 2.1467710, -2.8616502, 2.7039289
4: -0.9132509, 2.2788439, -1.0067019, 2.4620130, -3.3752639, 3.2855458

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7782248
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2498522, 1.7357445, -0.3368993, 2.1370778, -2.3869300, 2.0726438
1: -0.3489222, 2.4258590, -0.4281969, 2.9606328, -3.3095551, 2.8540559
2: -0.8619993, 1.6778553, -1.0432029, 2.0746379, -2.9366372, 2.7210581
3: -0.7148793, 1.9152986, -0.8713563, 2.4255347, -3.1404140, 2.7866549
4: -0.9132509, 2.2788439, -1.1845312, 2.7216258, -3.6348767, 3.4633751

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2686107, 1.7899325, -0.3013673, 1.9662584, -2.2348690, 2.0912998
1: -0.3625808, 2.5046179, -0.3952734, 2.7411184, -3.1036992, 2.8998914
2: -0.8971289, 1.7222278, -0.9791040, 1.8948953, -2.7920241, 2.7013319
3: -0.7427063, 1.9909970, -0.8054570, 2.2121744, -2.9548807, 2.7964540
4: -0.9565146, 2.3351691, -1.0846816, 2.5357726, -3.4922872, 3.4198508

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760656, upper bound: 2.7765358
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760656, upper bound: 2.7772001
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2686107, 1.7899325, -0.3352407, 2.1452107, -2.4138215, 2.1251731
1: -0.3625808, 2.5046179, -0.4305971, 2.9606981, -3.3232789, 2.9352150
2: -0.8971289, 1.7222278, -1.0398868, 2.0922604, -2.9893894, 2.7621145
3: -0.7427063, 1.9909970, -0.8733870, 2.4261951, -3.1689014, 2.8643839
4: -0.9565146, 2.3351691, -1.1808650, 2.7409091, -3.6974237, 3.5160341

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7778191
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7778191
time: 0.46 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2799054, 1.9078822, -0.3430017, 2.1650214, -2.4449267, 2.2508838
1: -0.3859175, 2.6413178, -0.4350375, 2.9869099, -3.3728273, 3.0763555
2: -0.9264836, 1.8670630, -1.0495744, 2.1141596, -3.0406432, 2.9166374
3: -0.7861645, 2.1172347, -0.8833915, 2.4662464, -3.2524109, 3.0006261
4: -1.0017887, 2.4863176, -1.2015169, 2.7557297, -3.7575183, 3.6878345

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7507392, upper bound: 2.7520929
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678435, upper bound: 2.7704501
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678435, upper bound: 2.7742810
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2799054, 1.9078822, -0.3461347, 2.1798525, -2.4597578, 2.2540169
1: -0.3859175, 2.6413178, -0.4375314, 3.0144095, -3.4003270, 3.0788493
2: -0.9264836, 1.8670630, -1.0611887, 2.1206467, -3.0471303, 2.9282517
3: -0.7861645, 2.1172347, -0.8885728, 2.4825411, -3.2687056, 3.0058074
4: -1.0017887, 2.4863176, -1.2127098, 2.7714510, -3.7732396, 3.6990275

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7507392, upper bound: 2.7520929
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7747312
time: 0.58 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7747313
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2925015, 1.9341680, -0.3399627, 2.1412797, -2.4337811, 2.2741306
1: -0.3910138, 2.7106080, -0.4318531, 2.9897790, -3.3807929, 3.1424611
2: -0.9777339, 1.8478855, -1.0796771, 2.0471630, -3.0248969, 2.9275627
3: -0.7969239, 2.1597757, -0.8770121, 2.4389844, -3.2359083, 3.0367880
4: -1.0653149, 2.5205028, -1.2337385, 2.7423425, -3.8076572, 3.7542415

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760407, upper bound: 2.7766300
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760407, upper bound: 2.7785020
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3005981, 1.9561236, -0.3398917, 2.1409926, -2.4415908, 2.2960153
1: -0.3972500, 2.7450514, -0.4317920, 2.9893932, -3.3866432, 3.1768434
2: -0.9933329, 1.8629159, -1.0795288, 2.0468860, -3.0402188, 2.9424448
3: -0.8096988, 2.1886795, -0.8768905, 2.4385912, -3.2482901, 3.0655699
4: -1.0831859, 2.5429926, -1.2334855, 2.7420232, -3.8252091, 3.7764781

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782712, upper bound: 2.7766300
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782712, upper bound: 2.7785020
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.3029714, 1.9758003, -0.3544002, 2.2744720, -2.5774434, 2.3302004
1: -0.4003568, 2.7667799, -0.4540269, 3.1468425, -3.5471992, 3.2208068
2: -0.9991634, 1.8891554, -1.1126282, 2.1994314, -3.1985948, 3.0017836
3: -0.8155400, 2.2180204, -0.9176872, 2.5795469, -3.3950868, 3.1357076
4: -1.1026624, 2.5684884, -1.2832001, 2.8939590, -3.9966216, 3.8516884

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791502, upper bound: 2.7761443
time: 0.54 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791502, upper bound: 2.7762445
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.3029111, 1.9755520, -0.3582784, 2.2925892, -2.5955002, 2.3338304
1: -0.4003025, 2.7664475, -0.4572957, 3.1761456, -3.5764480, 3.2237432
2: -0.9990335, 1.8889129, -1.1261404, 2.2105923, -3.2096257, 3.0150533
3: -0.8154335, 2.2176867, -0.9243510, 2.6107152, -3.4261487, 3.1420376
4: -1.1024466, 2.5682011, -1.2992344, 2.9124341, -4.0148807, 3.8674355

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747307, upper bound: 2.7777564
time: 0.39 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747307, upper bound: 2.7777564
time: 0.44 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3097464, 2.1038358, -0.3825232, 2.4036002, -2.7133467, 2.4863591
1: -0.4192142, 2.9166768, -0.4850632, 3.3296161, -3.7488303, 3.4017401
2: -1.0282929, 2.0379205, -1.1921978, 2.3280528, -3.3563457, 3.2301183
3: -0.8484409, 2.3540249, -0.9782482, 2.7541327, -3.6025736, 3.3322730
4: -1.1468025, 2.7217529, -1.3948834, 3.0714338, -4.2182364, 4.1166363

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7699838
time: 0.38 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7677170, upper bound: 2.7696068
time: 0.46 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3097464, 2.1038358, -0.3517131, 2.2722392, -2.5819857, 2.4555488
1: -0.4192142, 2.9166768, -0.4558137, 3.1438556, -3.5630698, 3.3724904
2: -1.0282929, 2.0379205, -1.1155756, 2.2015338, -3.2298267, 3.1534960
3: -0.8484409, 2.3540249, -0.9207952, 2.5787268, -3.4271677, 3.2748201
4: -1.1468025, 2.7217529, -1.2884810, 2.9099033, -4.0567060, 4.0102339

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7745775
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2_A2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7677170, upper bound: 2.7754416
time: 0.44 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 2.56 seconds
IS_B2_A2_A1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7800357
IS_B2_A2_A1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7800357
IS_B2_A2_A1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7764222
IS_B2_A2_A1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7788416
IS_B2_A2_A1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7763369
IS_B2_A2_A1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760594, upper bound: 2.7786571
IS_B2_A2_A1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7763369
IS_B2_A2_A1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7785349, upper bound: 2.7786571
IS_B2_A2_A1_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7764103
IS_B2_A2_A1_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7764103
IS_B2_A2_A1_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7759137, upper bound: 2.7774580
IS_B2_A2_A1_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7742335
IS_B2_A2_A1_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7748342
IS_B2_A2_A1_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7743062
IS_B2_A2_A1_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7775646, upper bound: 2.7748397
IS_B2_A2_A1_A1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7553012, upper bound: 2.7577949
IS_B2_A2_A1_A1_B2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7541295, upper bound: 2.7571481
IS_B2_A2_A1_A1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7553012, upper bound: 2.7577949
IS_B2_A2_A1_A1_B2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7541295, upper bound: 2.7571481
IS_B2_A2_A2_B2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7782248
IS_B2_A2_A2_B2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
IS_B2_A2_A2_B2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
IS_B2_A2_A2_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
IS_B2_A2_A2_B2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760656, upper bound: 2.7765358
IS_B2_A2_A2_B2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760656, upper bound: 2.7772001
IS_B2_A2_A2_B2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7778191
IS_B2_A2_A2_B2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7778191
IS_B2_A2_A2_B2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7678435, upper bound: 2.7704501
IS_B2_A2_A2_B2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7678435, upper bound: 2.7742810
IS_B2_A2_A2_B2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7747312
IS_B2_A2_A2_B2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7747313
IS_B2_A2_A2_B2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760407, upper bound: 2.7766300
IS_B2_A2_A2_B2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7760407, upper bound: 2.7785020
IS_B2_A2_A2_B2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7782712, upper bound: 2.7766300
IS_B2_A2_A2_B2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7782712, upper bound: 2.7785020
IS_B2_A2_A2_B2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7791502, upper bound: 2.7761443
IS_B2_A2_A2_B2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7791502, upper bound: 2.7762445
IS_B2_A2_A2_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7747307, upper bound: 2.7777564
IS_B2_A2_A2_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7747307, upper bound: 2.7777564
IS_B2_A2_A2_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7699838
IS_B2_A2_A2_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7677170, upper bound: 2.7696068
IS_B2_A2_A2_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7678523, upper bound: 2.7745775
IS_B2_A2_A2_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.56
Output dim: 0, lower bound: -2.7677170, upper bound: 2.7754416

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2717244, 1.8525931, -0.2922369, 1.9329662, -2.2046907, 2.1448300
1: -0.3729628, 2.5932012, -0.3895455, 2.7013831, -3.0743461, 2.9827466
2: -0.9286113, 1.7790446, -0.9691267, 1.8554987, -2.7841101, 2.7481713
3: -0.7617036, 2.0464137, -0.7945237, 2.1567588, -2.9184623, 2.8409374
4: -0.9849690, 2.4273558, -1.0531660, 2.5133586, -3.4983277, 3.4805217

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7780055
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7800357
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.2717244, 1.8525931, -0.2693810, 1.8078651, -2.0795896, 2.1219740
1: -0.3729628, 2.5932012, -0.3648047, 2.5216575, -2.8946204, 2.9580059
2: -0.9286113, 1.7790446, -0.9005327, 1.7479432, -2.6765544, 2.6795774
3: -0.7617036, 2.0464137, -0.7463789, 2.0156932, -2.7773967, 2.7927926
4: -0.9849690, 2.4273558, -0.9752934, 2.3596635, -3.3446326, 3.4026492

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7780063
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763053, upper bound: 2.7800357
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.2889379, 1.9049499, -0.2638234, 1.7935154, -2.0824533, 2.1687734
1: -0.3853487, 2.6697907, -0.3596354, 2.4994867, -2.8848352, 3.0294261
2: -0.9618609, 1.8210847, -0.8863502, 1.7350838, -2.6969447, 2.7074349
3: -0.7863698, 2.1205049, -0.7363384, 1.9913274, -2.7776971, 2.8568432
4: -1.0324315, 2.4797454, -0.9508004, 2.3367088, -3.3691401, 3.4305458

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7763179
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7764222
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.2889379, 1.9049499, -0.2824586, 1.8528644, -2.1418023, 2.1874084
1: -0.3853487, 2.6697907, -0.3732200, 2.5871220, -2.9724708, 3.0430107
2: -0.9618609, 1.8210847, -0.9223183, 1.7844286, -2.7462895, 2.7434030
3: -0.7863698, 2.1205049, -0.7637820, 2.0766785, -2.8630483, 2.8842869
4: -1.0324315, 2.4797454, -1.0017049, 2.3958540, -3.4282856, 3.4814503

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7775811
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788425, upper bound: 2.7776511
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.2717244, 1.8525931, -0.3166355, 2.0308909, -2.3026154, 2.1692286
1: -0.3729628, 2.5932012, -0.4088413, 2.8385818, -3.2115445, 3.0020423
2: -0.9286113, 1.7790446, -1.0204396, 1.9420912, -2.8707025, 2.7994843
3: -0.7617036, 2.0464137, -0.8325315, 2.2972150, -3.0589185, 2.8789451
4: -0.9849690, 2.4273558, -1.1453919, 2.6093647, -3.5943336, 3.5727477

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18

Time for candidate selection: 2.84 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732059, upper bound: 2.7731633
time: 0.48 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7736283, upper bound: 2.7757481
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.2717244, 1.8525931, -0.3249844, 2.0504947, -2.3222191, 2.1775775
1: -0.3729628, 2.5932012, -0.4150970, 2.8698602, -3.2428231, 3.0082982
2: -0.9286113, 1.7790446, -1.0357146, 1.9549512, -2.8835626, 2.8147593
3: -0.7617036, 2.0464137, -0.8454655, 2.3253436, -3.0870471, 2.8918791
4: -0.9849690, 2.4273558, -1.1625791, 2.6297333, -3.6147022, 3.5899348

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 18

Time for candidate selection: 2.87 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732059, upper bound: 2.7759101
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7736283, upper bound: 2.7778759
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.2889379, 1.9049499, -0.3166355, 2.0308909, -2.3198287, 2.2215853
1: -0.3853487, 2.6697907, -0.4088413, 2.8385818, -3.2239304, 3.0786319
2: -0.9618609, 1.8210847, -1.0204396, 1.9420912, -2.9039521, 2.8415244
3: -0.7863698, 2.1205049, -0.8325315, 2.2972150, -3.0835848, 2.9530363
4: -1.0324315, 2.4797454, -1.1453919, 2.6093647, -3.6417961, 3.6251373

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18

Time for candidate selection: 2.83 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759032, upper bound: 2.7714995
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763982, upper bound: 2.7739443
time: 0.46 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.2889379, 1.9049499, -0.3249844, 2.0504947, -2.3394325, 2.2299342
1: -0.3853487, 2.6697907, -0.4150970, 2.8698602, -3.2552090, 3.0848877
2: -0.9618609, 1.8210847, -1.0357146, 1.9549512, -2.9168119, 2.8567994
3: -0.7863698, 2.1205049, -0.8454655, 2.3253436, -3.1117134, 2.9659705
4: -1.0324315, 2.4797454, -1.1625791, 2.6297333, -3.6621647, 3.6423244

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18

Time for candidate selection: 2.84 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759032, upper bound: 2.7730854
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763982, upper bound: 2.7754296
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2820300, 1.9873157, -0.2638234, 1.7935154, -2.0755453, 2.2511392
1: -0.3970020, 2.7511249, -0.3596354, 2.4994867, -2.8964887, 3.1107602
2: -0.9615395, 1.9399827, -0.8863502, 1.7350838, -2.6966233, 2.8263328
3: -0.8055222, 2.1982713, -0.7363384, 1.9913274, -2.7968497, 2.9346097
4: -1.0453598, 2.5924871, -0.9508004, 2.3367088, -3.3820686, 3.5432875

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761766, upper bound: 2.7770938
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761766, upper bound: 2.7771068
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2820300, 1.9873157, -0.2824586, 1.8528644, -2.1348944, 2.2697742
1: -0.3970020, 2.7511249, -0.3732200, 2.5871220, -2.9841239, 3.1243448
2: -0.9615395, 1.9399827, -0.9223183, 1.7844286, -2.7459681, 2.8623009
3: -0.8055222, 2.1982713, -0.7637820, 2.0766785, -2.8822007, 2.9620533
4: -1.0453598, 2.5924871, -1.0017049, 2.3958540, -3.4412138, 3.5941920

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761766, upper bound: 2.7781308
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761766, upper bound: 2.7781308
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2820300, 1.9873157, -0.3166355, 2.0308909, -2.3129210, 2.3039513
1: -0.3970020, 2.7511249, -0.4088413, 2.8385818, -3.2355838, 3.1599660
2: -0.9615395, 1.9399827, -1.0204396, 1.9420912, -2.9036307, 2.9604223
3: -0.8055222, 2.1982713, -0.8325315, 2.2972150, -3.1027372, 3.0308027
4: -1.0453598, 2.5924871, -1.1453919, 2.6093647, -3.6547246, 3.7378790

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 41

Time for candidate selection: 2.91 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731411, upper bound: 2.7714401
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735244, upper bound: 2.7741634
time: 0.44 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2820300, 1.9873157, -0.3249844, 2.0504947, -2.3325248, 2.3123000
1: -0.3970020, 2.7511249, -0.4150970, 2.8698602, -3.2668622, 3.1662219
2: -0.9615395, 1.9399827, -1.0357146, 1.9549512, -2.9164906, 2.9756973
3: -0.8055222, 2.1982713, -0.8454655, 2.3253436, -3.1308658, 3.0437369
4: -1.0453598, 2.5924871, -1.1625791, 2.6297333, -3.6750932, 3.7550662

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 41

Time for candidate selection: 2.86 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731411, upper bound: 2.7728743
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735244, upper bound: 2.7752672
time: 0.45 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2890179, 1.9990376, -0.2638234, 1.7935154, -2.0825334, 2.2628610
1: -0.4002408, 2.7723556, -0.3596354, 2.4994867, -2.8997273, 3.1319909
2: -0.9748017, 1.9446510, -0.8863502, 1.7350838, -2.7098856, 2.8310013
3: -0.8124224, 2.2182291, -0.7363384, 1.9913274, -2.8037498, 2.9545674
4: -1.0574362, 2.6030040, -0.9508004, 2.3367088, -3.3941450, 3.5538044

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778275, upper bound: 2.7741832
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778275, upper bound: 2.7742335
time: 0.40 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2890179, 1.9990376, -0.2824586, 1.8528644, -2.1418824, 2.2814963
1: -0.4002408, 2.7723556, -0.3732200, 2.5871220, -2.9873629, 3.1455755
2: -0.9748017, 1.9446510, -0.9223183, 1.7844286, -2.7592301, 2.8669693
3: -0.8124224, 2.2182291, -0.7637820, 2.0766785, -2.8891010, 2.9820111
4: -1.0574362, 2.6030040, -1.0017049, 2.3958540, -3.4532902, 3.6047089

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778275, upper bound: 2.7748406
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778275, upper bound: 2.7748406
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2890179, 1.9990376, -0.3166355, 2.0308909, -2.3199089, 2.3156731
1: -0.4002408, 2.7723556, -0.4088413, 2.8385818, -3.2388225, 3.1811967
2: -0.9748017, 1.9446510, -1.0204396, 1.9420912, -2.9168930, 2.9650908
3: -0.8124224, 2.2182291, -0.8325315, 2.2972150, -3.1096373, 3.0507605
4: -1.0574362, 2.6030040, -1.1453919, 2.6093647, -3.6668010, 3.7483959

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 41

Time for candidate selection: 2.94 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748387, upper bound: 2.7691539
time: 0.47 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752400, upper bound: 2.7719731
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2890179, 1.9990376, -0.3249844, 2.0504947, -2.3395126, 2.3240221
1: -0.4002408, 2.7723556, -0.4150970, 2.8698602, -3.2701011, 3.1874526
2: -0.9748017, 1.9446510, -1.0357146, 1.9549512, -2.9297528, 2.9803658
3: -0.8124224, 2.2182291, -0.8454655, 2.3253436, -3.1377659, 3.0636945
4: -1.0574362, 2.6030040, -1.1625791, 2.6297333, -3.6871696, 3.7655830

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 41

Time for candidate selection: 2.87 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748387, upper bound: 2.7699065
time: 0.47 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752400, upper bound: 2.7724327
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2498522, 1.7357445, -0.2750586, 1.8485012, -2.0983534, 2.0108030
1: -0.3489222, 2.4258590, -0.3723598, 2.5539541, -2.9028764, 2.7982187
2: -0.8619993, 1.6778553, -0.8874640, 1.8207785, -2.6827779, 2.5653193
3: -0.7148793, 1.9152986, -0.7618650, 2.0604444, -2.7753236, 2.6771636
4: -0.9132509, 2.2788439, -0.9597875, 2.3949852, -3.3082361, 3.2386312

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7782248
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7782248
time: 0.46 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2498522, 1.7357445, -0.2893965, 1.9016511, -2.1515033, 2.0251410
1: -0.3489222, 2.4258590, -0.3816713, 2.6341915, -2.9831138, 2.8075304
2: -0.8619993, 1.6778553, -0.9185903, 1.8592347, -2.7212338, 2.5964456
3: -0.7148793, 1.9152986, -0.7806731, 2.1272354, -2.8421147, 2.6959717
4: -0.9132509, 2.2788439, -0.9975452, 2.4454205, -3.3586714, 3.2763891

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796784
time: 0.37 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796784
time: 0.41 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2498522, 1.7357445, -0.3169692, 2.0393405, -2.2891927, 2.0527136
1: -0.3489222, 2.4258590, -0.4115121, 2.8502831, -3.1992054, 2.8373711
2: -0.8619993, 1.6778553, -1.0234690, 1.9525166, -2.8145158, 2.7013242
3: -0.7148793, 1.9152986, -0.8378575, 2.3051753, -3.0200546, 2.7531562
4: -0.9132509, 2.2788439, -1.1483159, 2.6240706, -3.5373216, 3.4271598

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7784891
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.49 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2498522, 1.7357445, -0.3381015, 2.2106109, -2.4604630, 2.0738459
1: -0.3489222, 2.4258590, -0.4408157, 3.0583348, -3.4072571, 2.8666747
2: -0.8619993, 1.6778553, -1.0741737, 2.1422040, -3.0042033, 2.7520289
3: -0.7148793, 1.9152986, -0.8925020, 2.4995527, -3.2144320, 2.8078005
4: -0.9132509, 2.2788439, -1.2294081, 2.8181005, -3.7313514, 3.5082521

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7784891
time: 0.41 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742794, upper bound: 2.7796443
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.2686107, 1.7899325, -0.2894652, 1.9184904, -2.1871011, 2.0793977
1: -0.3625808, 2.5046179, -0.3847837, 2.6769810, -3.0395617, 2.8894017
2: -0.8971289, 1.7222278, -0.9548855, 1.8469553, -2.7440844, 2.6771133
3: -0.7427063, 1.9909970, -0.7846484, 2.1455696, -2.8882759, 2.7756453
4: -0.9565146, 2.3351691, -1.0426308, 2.4824779, -3.4389925, 3.3778000

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 18

Time for candidate selection: 2.91 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757182, upper bound: 2.7717542
time: 0.40 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7762352, upper bound: 2.7742755
time: 0.38 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.2686107, 1.7899325, -0.3026787, 1.9537526, -2.2223635, 2.0926113
1: -0.3625808, 2.5046179, -0.3943729, 2.7299430, -3.0925238, 2.8989909
2: -0.8971289, 1.7222278, -0.9790335, 1.8745556, -2.7716846, 2.7012613
3: -0.7427063, 1.9909970, -0.8044777, 2.1972787, -2.9399850, 2.7954745
4: -0.9565146, 2.3351691, -1.0747997, 2.5185428, -3.4750574, 3.4099689

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 13
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 5
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 25
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18

Time for candidate selection: 2.92 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757182, upper bound: 2.7730854
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7762352, upper bound: 2.7754516
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.2686107, 1.7899325, -0.2980190, 1.9792039, -2.2478147, 2.0879514
1: -0.3625808, 2.5046179, -0.4009111, 2.7268662, -3.0894470, 2.9055290
2: -0.8971289, 1.7222278, -0.9501152, 1.9495702, -2.8466992, 2.6723430
3: -0.7427063, 1.9909970, -0.8158891, 2.2144580, -2.9571643, 2.8068862
4: -0.9565146, 2.3351691, -1.0531934, 2.5587103, -3.5152249, 3.3883624

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7763405
time: 0.49 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7771398
time: 0.44 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.2686107, 1.7899325, -0.3380236, 2.2103078, -2.4789186, 2.1279562
1: -0.3625808, 2.5046179, -0.4407489, 3.0579267, -3.4205074, 2.9453669
2: -0.8971289, 1.7222278, -1.0740206, 2.1419098, -3.0390387, 2.7962484
3: -0.7427063, 1.9909970, -0.8923699, 2.4991393, -3.2418456, 2.8833668
4: -0.9565146, 2.3351691, -1.2291472, 2.8177657, -3.7742803, 3.5643163

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7763405
time: 0.45 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760461, upper bound: 2.7771398
time: 0.42 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.2799054, 1.9078822, -0.3217086, 2.1637869, -2.4436922, 2.2295909
1: -0.3859175, 2.6413178, -0.4319691, 2.9951830, -3.3811004, 3.0732870
2: -0.9264836, 1.8670630, -1.0566359, 2.1016927, -3.0281763, 2.9236989
3: -0.7861645, 2.1172347, -0.8735865, 2.4206874, -3.2068520, 2.9908214
4: -1.0017887, 2.4863176, -1.1834191, 2.7921784, -3.7939672, 3.6697369

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562058, upper bound: 2.7563335
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7679812, upper bound: 2.7708386
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7679812, upper bound: 2.7708386
time: 0.39 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.2799054, 1.9078822, -0.3095315, 2.0523829, -2.3322883, 2.2174137
1: -0.3859175, 2.6413178, -0.4128166, 2.8373108, -3.2232282, 3.0541344
2: -0.9264836, 1.8670630, -0.9991217, 2.0023096, -2.9287932, 2.8661847
3: -0.7861645, 2.1172347, -0.8374678, 2.2932167, -3.0793812, 2.9547024
4: -1.0017887, 2.4863176, -1.1103311, 2.6539612, -3.6557498, 3.5966487

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562058, upper bound: 2.7563335
time: 0.44 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7679812, upper bound: 2.7742429
time: 0.43 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7679812, upper bound: 2.7742810
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.2799054, 1.9078822, -0.3026787, 1.9537526, -2.2336581, 2.2105608
1: -0.3859175, 2.6413178, -0.3943729, 2.7299430, -3.1158605, 3.0356908
2: -0.9264836, 1.8670630, -0.9790335, 1.8745556, -2.8010392, 2.8460965
3: -0.7861645, 2.1172347, -0.8044777, 2.1972787, -2.9834433, 2.9217124
4: -1.0017887, 2.4863176, -1.0747997, 2.5185428, -3.5203314, 3.5611172

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 6
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 10
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 25
type: A, layer: 3, pos: 5
type: B, layer: 3, pos: 25
type: A, layer: 3, pos: 10
type: B, layer: 3, pos: 18
type: A, layer: 3, pos: 18
type: B, layer: 3, pos: 41

Time for candidate selection: 3.02 seconds

### Candidate
type: B, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 6

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7729383, upper bound: 2.7689741
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7733326, upper bound: 2.7724550
time: 0.45 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.2799054, 1.9078822, -0.3252559, 2.1111951, -2.3911004, 2.2331381
1: -0.3859175, 2.6413178, -0.4223713, 2.9217782, -3.3076956, 3.0636892
2: -0.9264836, 1.8670630, -1.0273952, 2.0531788, -2.9796624, 2.8944583
3: -0.7861645, 2.1172347, -0.8570579, 2.3760862, -3.1622508, 2.9742928
4: -1.0017887, 2.4863176, -1.1500295, 2.7024202, -3.7042089, 3.6363473

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_B1

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7746770
time: 0.42 seconds

## Relational analysis of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_B2

### Relational analysis result of IS_B2_A2_A2_B2_A1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757452, upper bound: 2.7747313
time: 0.43 seconds

## BFS IS instance: IS_B2_A2_A2_B2_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.2925015, 1.9341680, -0.3278462, 2.0922771, -2.3847785, 2.2620142
1: -0.3910138, 2.7106080, -0.4213070, 2.9239202, -3.3149340, 3.1319151
2: -0.9777339, 1.8478855, -1.0550969, 1.9983690, -2.9761028, 2.9029822
3: -0.7969239, 2.1597757, -0.8559048, 2.3705678, -3.1674917, 3.0156806
4: -1.0653149, 2.5205028, -1.1906550, 2.6879668, -3.7532816, 3.7111578

Time for backsubstitution: 1.68 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=3.285133123397827
rel_dist={0: [-2.780286344644136, 2.7802863446441357]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1153.16 seconds
