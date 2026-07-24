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
execution time: IAR + LP analysis = 1.47 + 1.05 = 2.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846


# Binary Search by BASE starts (time budget: 1197.48 seconds, max iter: 100)

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
Binary search time: 46.66 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1150.81 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
time: 0.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.35 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.85
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.4334733, 2.4970913, -0.5033236, 2.7462137, -3.1796870, 3.0004148
1: -0.5019389, 3.4698267, -0.5549647, 3.7774627, -4.2794018, 4.0247912
2: -1.2600799, 2.3818066, -1.3533173, 2.6732659, -3.9333458, 3.7351239
3: -1.0162834, 2.9394152, -1.1219059, 3.3177133, -4.3339968, 4.0613213
4: -1.5453744, 3.1090739, -1.7115899, 3.3719947, -4.9173689, 4.8206639

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.39 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.5095432, 2.7755899, -3.2509441, 3.1871667
1: -0.5395899, 3.6900327, -0.5611423, 3.8165379, -4.3561277, 4.2511749
2: -1.3205540, 2.6000621, -1.3674926, 2.7016737, -4.0222278, 3.9675546
3: -1.0897965, 3.1838365, -1.1338987, 3.3577619, -4.4475584, 4.3177352
4: -1.6415637, 3.3140776, -1.7360522, 3.4053361, -5.0468998, 5.0501299

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.32 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.32
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.4334733, 2.4970913, -0.4334733, 2.4970913, -2.9305646, 2.9305646
1: -0.5019389, 3.4698267, -0.5019389, 3.4698267, -3.9717655, 3.9717655
2: -1.2600799, 2.3818066, -1.2600799, 2.3818066, -3.6418865, 3.6418865
3: -1.0162834, 2.9394152, -1.0162834, 2.9394152, -3.9556985, 3.9556985
4: -1.5453744, 3.1090739, -1.5453744, 3.1090739, -4.6544485, 4.6544485

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.4334733, 2.4970913, -0.4753543, 2.6776235, -3.1110969, 2.9724455
1: -0.5019389, 3.4698267, -0.5395899, 3.6900327, -4.1919718, 4.0094166
2: -1.2600799, 2.3818066, -1.3205540, 2.6000621, -3.8601420, 3.7023606
3: -1.0162834, 2.9394152, -1.0897965, 3.1838365, -4.2001200, 4.0292120
4: -1.5453744, 3.1090739, -1.6415637, 3.3140776, -4.8594522, 4.7506375

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
time: 0.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.4334733, 2.4970913, -2.9724455, 3.1110969
1: -0.5395899, 3.6900327, -0.5019389, 3.4698267, -4.0094166, 4.1919718
2: -1.3205540, 2.6000621, -1.2600799, 2.3818066, -3.7023606, 3.8601420
3: -1.0897965, 3.1838365, -1.0162834, 2.9394152, -4.0292120, 4.2001200
4: -1.6415637, 3.3140776, -1.5453744, 3.1090739, -4.7506375, 4.8594522

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.37 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.4753543, 2.6776235, -3.1529779, 3.1529779
1: -0.5395899, 3.6900327, -0.5395899, 3.6900327, -4.2296228, 4.2296228
2: -1.3205540, 2.6000621, -1.3205540, 2.6000621, -3.9206161, 3.9206161
3: -1.0897965, 3.1838365, -1.0897965, 3.1838365, -4.2736330, 4.2736330
4: -1.6415637, 3.3140776, -1.6415637, 3.3140776, -4.9556413, 4.9556413

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.29 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.29
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4292223, 2.4830091, -2.8848131, 2.8493788
1: -0.4905685, 3.3785930, -0.4990866, 3.4508967, -3.9414654, 3.8776796
2: -1.2406874, 2.2999053, -1.2536042, 2.3682904, -3.6089778, 3.5535095
3: -0.9905678, 2.8093987, -1.0104907, 2.9184167, -3.9089847, 3.8198893
4: -1.4657205, 3.0825684, -1.5326307, 3.0961943, -4.5619149, 4.6151991

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4334733, 2.4970913, -2.8721917, 2.7071152
1: -0.4616931, 3.1732640, -0.5019389, 3.4698267, -3.9315197, 3.6752028
2: -1.1646090, 2.1681395, -1.2600799, 2.3818066, -3.5464156, 3.4282193
3: -0.9341383, 2.6221490, -1.0162834, 2.9394152, -3.8735535, 3.6384325
4: -1.3658996, 2.9074883, -1.5453744, 3.1090739, -4.4749737, 4.4528627

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4702159, 2.6624470, -3.0642509, 2.8903723
1: -0.4905685, 3.3785930, -0.5364853, 3.6699429, -4.1605115, 3.9150782
2: -1.2406874, 2.2999053, -1.3134742, 2.5835495, -3.8242369, 3.6133795
3: -0.9905678, 2.8093987, -1.0834836, 3.1569023, -4.1474700, 3.8928823
4: -1.4657205, 3.0825684, -1.6277233, 3.3002880, -4.7660084, 4.7102919

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7778969
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772093, upper bound: 2.7788218
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4753543, 2.6776235, -3.0527239, 2.7489963
1: -0.4616931, 3.1732640, -0.5395899, 3.6900327, -4.1517258, 3.7128539
2: -1.1646090, 2.1681395, -1.3205540, 2.6000621, -3.7646711, 3.4886935
3: -0.9341383, 2.6221490, -1.0897965, 3.1838365, -4.1179748, 3.7119455
4: -1.3658996, 2.9074883, -1.6415637, 3.3140776, -4.6799774, 4.5490522

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4292223, 2.4830091, -2.9004710, 2.9891875
1: -0.5153302, 3.5426629, -0.4990866, 3.4508967, -3.9662271, 4.0417495
2: -1.2751217, 2.4697607, -1.2536042, 2.3682904, -3.6434121, 3.7233648
3: -1.0362854, 2.9608207, -1.0104907, 2.9184167, -3.9547021, 3.9713113
4: -1.5216012, 3.2471507, -1.5326307, 3.0961943, -4.6177955, 4.7797813

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4334733, 2.4970913, -2.8958321, 2.8817961
1: -0.4923947, 3.3845534, -0.5019389, 3.4698267, -3.9622214, 3.8864923
2: -1.2141361, 2.3641450, -1.2600799, 2.3818066, -3.5959427, 3.6242249
3: -0.9919410, 2.8182275, -1.0162834, 2.9394152, -3.9313562, 3.8345108
4: -1.4456424, 3.1076982, -1.5453744, 3.1090739, -4.5547161, 4.6530724

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4702159, 2.6624470, -3.0799088, 3.0301809
1: -0.5153302, 3.5426629, -0.5364853, 3.6699429, -4.1852732, 4.0791483
2: -1.2751217, 2.4697607, -1.3134742, 2.5835495, -3.8586712, 3.7832348
3: -1.0362854, 2.9608207, -1.0834836, 3.1569023, -4.1931877, 4.0443044
4: -1.5216012, 3.2471507, -1.6277233, 3.3002880, -4.8218889, 4.8748741

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7758395
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7768017
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4753543, 2.6776235, -3.0763640, 2.9236770
1: -0.4923947, 3.3845534, -0.5395899, 3.6900327, -4.1824274, 3.9241433
2: -1.2141361, 2.3641450, -1.3205540, 2.6000621, -3.8141983, 3.6846991
3: -0.9919410, 2.8182275, -1.0897965, 3.1838365, -4.1757774, 3.9080241
4: -1.4456424, 3.1076982, -1.6415637, 3.3140776, -4.7597198, 4.7492619

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239
time: 0.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.35 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7778969
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7772093, upper bound: 2.7788218
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7758395
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7768017
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.35
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4018040, 2.4201565, -2.8219604, 2.8219604
1: -0.4905685, 3.3785930, -0.4905685, 3.3785930, -3.8691616, 3.8691616
2: -1.2406874, 2.2999053, -1.2406874, 2.2999053, -3.5405927, 3.5405927
3: -0.9905678, 2.8093987, -0.9905678, 2.8093987, -3.7999663, 3.7999663
4: -1.4657205, 3.0825684, -1.4657205, 3.0825684, -4.5482888, 4.5482888

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.3751003, 2.2736418, -2.6754458, 2.7952569
1: -0.4905685, 3.3785930, -0.4616931, 3.1732640, -3.6638327, 3.8402860
2: -1.2406874, 2.2999053, -1.1646090, 2.1681395, -3.4088268, 3.4645143
3: -0.9905678, 2.8093987, -0.9341383, 2.6221490, -3.6127167, 3.7435369
4: -1.4657205, 3.0825684, -1.3658996, 2.9074883, -4.3732090, 4.4484682

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4018040, 2.4201565, -2.7952569, 2.6754458
1: -0.4616931, 3.1732640, -0.4905685, 3.3785930, -3.8402860, 3.6638327
2: -1.1646090, 2.1681395, -1.2406874, 2.2999053, -3.4645143, 3.4088268
3: -0.9341383, 2.6221490, -0.9905678, 2.8093987, -3.7435369, 3.6127167
4: -1.3658996, 2.9074883, -1.4657205, 3.0825684, -4.4484682, 4.3732090

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.3751003, 2.2736418, -2.6487422, 2.6487422
1: -0.4616931, 3.1732640, -0.4616931, 3.1732640, -3.6349571, 3.6349571
2: -1.1646090, 2.1681395, -1.1646090, 2.1681395, -3.3327484, 3.3327484
3: -0.9341383, 2.6221490, -0.9341383, 2.6221490, -3.5562873, 3.5562873
4: -1.3658996, 2.9074883, -1.3658996, 2.9074883, -4.2733879, 4.2733879

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4561949, 2.6133635, -3.0151675, 2.8763514
1: -0.4905685, 3.3785930, -0.5258589, 3.6044338, -4.0950022, 3.9044518
2: -1.2406874, 2.2999053, -1.2888968, 2.5323260, -3.7730134, 3.5888021
3: -0.9905678, 2.8093987, -1.0625300, 3.0840850, -4.0746527, 3.8719287
4: -1.4657205, 3.0825684, -1.5837104, 3.2471111, -4.7128315, 4.6662788

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7778969
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7778969
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4588744, 2.6281669, -3.0299709, 2.8790309
1: -0.4905685, 3.3785930, -0.5282684, 3.6314750, -4.1220436, 3.9068613
2: -1.2406874, 2.2999053, -1.3008018, 2.5315521, -3.7722394, 3.6007071
3: -0.9905678, 2.8093987, -1.0670657, 3.0996389, -4.0902066, 3.8764644
4: -1.4657205, 3.0825684, -1.5958318, 3.2616765, -4.7273970, 4.6784000

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7788218
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7788218
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4174619, 2.5599651, -2.9350655, 2.6911037
1: -0.4616931, 3.1732640, -0.5153302, 3.5426629, -4.0043559, 3.6885943
2: -1.1646090, 2.1681395, -1.2751217, 2.4697607, -3.6343696, 3.4432611
3: -0.9341383, 2.6221490, -1.0362854, 2.9608207, -3.8949590, 3.6584344
4: -1.3658996, 2.9074883, -1.5216012, 3.2471507, -4.6130505, 4.4290895

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.3987406, 2.4483228, -2.8234231, 2.6723824
1: -0.4616931, 3.1732640, -0.4923947, 3.3845534, -3.8462465, 3.6656587
2: -1.1646090, 2.1681395, -1.2141361, 2.3641450, -3.5287540, 3.3822756
3: -0.9341383, 2.6221490, -0.9919410, 2.8182275, -3.7523658, 3.6140900
4: -1.3658996, 2.9074883, -1.4456424, 3.1076982, -4.4735975, 4.3531308

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4018040, 2.4201565, -2.8376184, 2.9617691
1: -0.5153302, 3.5426629, -0.4905685, 3.3785930, -3.8939233, 4.0332313
2: -1.2751217, 2.4697607, -1.2406874, 2.2999053, -3.5750270, 3.7104480
3: -1.0362854, 2.9608207, -0.9905678, 2.8093987, -3.8456841, 3.9513884
4: -1.5216012, 3.2471507, -1.4657205, 3.0825684, -4.6041698, 4.7128711

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.3751003, 2.2736418, -2.6911037, 2.9350655
1: -0.5153302, 3.5426629, -0.4616931, 3.1732640, -3.6885943, 4.0043559
2: -1.2751217, 2.4697607, -1.1646090, 2.1681395, -3.4432611, 3.6343696
3: -1.0362854, 2.9608207, -0.9341383, 2.6221490, -3.6584344, 3.8949590
4: -1.5216012, 3.2471507, -1.3658996, 2.9074883, -4.4290895, 4.6130505

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4018040, 2.4201565, -2.8188972, 2.8501267
1: -0.4923947, 3.3845534, -0.4905685, 3.3785930, -3.8709877, 3.8751221
2: -1.2141361, 2.3641450, -1.2406874, 2.2999053, -3.5140414, 3.6048324
3: -0.9919410, 2.8182275, -0.9905678, 2.8093987, -3.8013396, 3.8087955
4: -1.4456424, 3.1076982, -1.4657205, 3.0825684, -4.5282106, 4.5734186

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.3751003, 2.2736418, -2.6723824, 2.8234231
1: -0.4923947, 3.3845534, -0.4616931, 3.1732640, -3.6656587, 3.8462465
2: -1.2141361, 2.3641450, -1.1646090, 2.1681395, -3.3822756, 3.5287540
3: -0.9919410, 2.8182275, -0.9341383, 2.6221490, -3.6140900, 3.7523658
4: -1.4456424, 3.1076982, -1.3658996, 2.9074883, -4.3531308, 4.4735975

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7772093
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4561949, 2.6133635, -3.0308254, 3.0161600
1: -0.5153302, 3.5426629, -0.5258589, 3.6044338, -4.1197639, 4.0685215
2: -1.2751217, 2.4697607, -1.2888968, 2.5323260, -3.8074477, 3.7586575
3: -1.0362854, 2.9608207, -1.0625300, 3.0840850, -4.1203704, 4.0233507
4: -1.5216012, 3.2471507, -1.5837104, 3.2471111, -4.7687120, 4.8308611

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7758395
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7758395
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4588744, 2.6281669, -3.0456288, 3.0188396
1: -0.5153302, 3.5426629, -0.5282684, 3.6314750, -4.1468053, 4.0709314
2: -1.2751217, 2.4697607, -1.3008018, 2.5315521, -3.8066738, 3.7705624
3: -1.0362854, 2.9608207, -1.0670657, 3.0996389, -4.1359243, 4.0278864
4: -1.5216012, 3.2471507, -1.5958318, 3.2616765, -4.7832775, 4.8429823

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7768017
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7768017
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4174619, 2.5599651, -2.9587059, 2.8657846
1: -0.4923947, 3.3845534, -0.5153302, 3.5426629, -4.0350575, 3.8998837
2: -1.2141361, 2.3641450, -1.2751217, 2.4697607, -3.6838968, 3.6392667
3: -0.9919410, 2.8182275, -1.0362854, 2.9608207, -3.9527617, 3.8545129
4: -1.4456424, 3.1076982, -1.5216012, 3.2471507, -4.6927929, 4.6292992

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.3987406, 2.4483228, -2.8470635, 2.8470635
1: -0.4923947, 3.3845534, -0.4923947, 3.3845534, -3.8769481, 3.8769481
2: -1.2141361, 2.3641450, -1.2141361, 2.3641450, -3.5782812, 3.5782812
3: -0.9919410, 2.8182275, -0.9919410, 2.8182275, -3.8101685, 3.8101685
4: -1.4456424, 3.1076982, -1.4456424, 3.1076982, -4.5533404, 4.5533404

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629
time: 0.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.40 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7778969
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7778969
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7788218
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7788218
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7772093
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7758395
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7758395
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7768017
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7768017
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4018040, 2.4201565, -2.8097630, 2.7723832
1: -0.4801092, 3.3123057, -0.4905685, 3.3785930, -3.8587022, 3.8028741
2: -1.2162256, 2.2505226, -1.2406874, 2.2999053, -3.5161309, 3.4912100
3: -0.9699728, 2.7406662, -0.9905678, 2.8093987, -3.7793715, 3.7312341
4: -1.4223459, 3.0282202, -1.4657205, 3.0825684, -4.5049143, 4.4939408

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7773000
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7791682
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4018040, 2.4201565, -2.8175697, 2.7951903
1: -0.4861144, 3.3476477, -0.4905685, 3.3785930, -3.8647075, 3.8382163
2: -1.2313797, 2.2667561, -1.2406874, 2.2999053, -3.5312850, 3.5074434
3: -0.9823158, 2.7700262, -0.9905678, 2.8093987, -3.7917144, 3.7605939
4: -1.4410114, 3.0509758, -1.4657205, 3.0825684, -4.5235796, 4.5166965

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7773000
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791682
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3751003, 2.2736418, -2.6632483, 2.7456796
1: -0.4801092, 3.3123057, -0.4616931, 3.1732640, -3.6533732, 3.7739987
2: -1.2162256, 2.2505226, -1.1646090, 2.1681395, -3.3843651, 3.4151316
3: -0.9699728, 2.7406662, -0.9341383, 2.6221490, -3.5921218, 3.6748044
4: -1.4223459, 3.0282202, -1.3658996, 2.9074883, -4.3298340, 4.3941197

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7778597
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3751003, 2.2736418, -2.6710553, 2.7684867
1: -0.4861144, 3.3476477, -0.4616931, 3.1732640, -3.6593785, 3.8093407
2: -1.2313797, 2.2667561, -1.1646090, 2.1681395, -3.3995192, 3.4313650
3: -0.9823158, 2.7700262, -0.9341383, 2.6221490, -3.6044648, 3.7041645
4: -1.4410114, 3.0509758, -1.3658996, 2.9074883, -4.3484998, 4.4168754

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7778597
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4018040, 2.4201565, -2.7832375, 2.6270947
1: -0.4512913, 3.1082866, -0.4905685, 3.3785930, -3.8298843, 3.5988550
2: -1.1403358, 2.1198447, -1.2406874, 2.2999053, -3.4402411, 3.3605320
3: -0.9133595, 2.5548997, -0.9905678, 2.8093987, -3.7227583, 3.5454674
4: -1.3233099, 2.8539722, -1.4657205, 3.0825684, -4.4058781, 4.3196926

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7772892
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7791573
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4018040, 2.4201565, -2.7911177, 2.6521373
1: -0.4575120, 3.1468184, -0.4905685, 3.3785930, -3.8361049, 3.6373868
2: -1.1562662, 2.1376271, -1.2406874, 2.2999053, -3.4561715, 3.3783145
3: -0.9258730, 2.5864434, -0.9905678, 2.8093987, -3.7352717, 3.5770111
4: -1.3418519, 2.8791902, -1.4657205, 3.0825684, -4.4244204, 4.3449106

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7772892
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791573
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3751003, 2.2736418, -2.6367228, 2.6003911
1: -0.4512913, 3.1082866, -0.4616931, 3.1732640, -3.6245553, 3.5699797
2: -1.1403358, 2.1198447, -1.1646090, 2.1681395, -3.3084753, 3.2844536
3: -0.9133595, 2.5548997, -0.9341383, 2.6221490, -3.5355086, 3.4890380
4: -1.3233099, 2.8539722, -1.3658996, 2.9074883, -4.2307982, 4.2198715

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7772892
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7791573
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3751003, 2.2736418, -2.6446030, 2.6254337
1: -0.4575120, 3.1468184, -0.4616931, 3.1732640, -3.6307759, 3.6085114
2: -1.1562662, 2.1376271, -1.1646090, 2.1681395, -3.3244057, 3.3022361
3: -0.9258730, 2.5864434, -0.9341383, 2.6221490, -3.5480220, 3.5205817
4: -1.3418519, 2.8791902, -1.3658996, 2.9074883, -4.2493401, 4.2450895

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7772892
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4561949, 2.6133635, -3.0029700, 2.8267741
1: -0.4801092, 3.3123057, -0.5258589, 3.6044338, -4.0845432, 3.8381646
2: -1.2162256, 2.2505226, -1.2888968, 2.5323260, -3.7485516, 3.5394194
3: -0.9699728, 2.7406662, -1.0625300, 3.0840850, -4.0540581, 3.8031962
4: -1.4223459, 3.0282202, -1.5837104, 3.2471111, -4.6694570, 4.6119308

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7773018
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7778969
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4561949, 2.6133635, -3.0107770, 2.8495812
1: -0.4861144, 3.3476477, -0.5258589, 3.6044338, -4.0905480, 3.8735065
2: -1.2313797, 2.2667561, -1.2888968, 2.5323260, -3.7637057, 3.5556529
3: -0.9823158, 2.7700262, -1.0625300, 3.0840850, -4.0664005, 3.8325562
4: -1.4410114, 3.0509758, -1.5837104, 3.2471111, -4.6881227, 4.6346865

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7773018
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7778969
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4588744, 2.6281669, -3.0177734, 2.8294537
1: -0.4801092, 3.3123057, -0.5282684, 3.6314750, -4.1115842, 3.8405740
2: -1.2162256, 2.2505226, -1.3008018, 2.5315521, -3.7477777, 3.5513244
3: -0.9699728, 2.7406662, -1.0670657, 3.0996389, -4.0696115, 3.8077319
4: -1.4223459, 3.0282202, -1.5958318, 3.2616765, -4.6840224, 4.6240520

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4588744, 2.6281669, -3.0255804, 2.8522608
1: -0.4861144, 3.3476477, -0.5282684, 3.6314750, -4.1175895, 3.8759160
2: -1.2313797, 2.2667561, -1.3008018, 2.5315521, -3.7629318, 3.5675578
3: -0.9823158, 2.7700262, -1.0670657, 3.0996389, -4.0819550, 3.8370919
4: -1.4410114, 3.0509758, -1.5958318, 3.2616765, -4.7026882, 4.6468077

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7780966
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7780966
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4174619, 2.5599651, -2.9230461, 2.6427526
1: -0.4512913, 3.1082866, -0.5153302, 3.5426629, -3.9939542, 3.6236167
2: -1.1403358, 2.1198447, -1.2751217, 2.4697607, -3.6100965, 3.3949664
3: -0.9133595, 2.5548997, -1.0362854, 2.9608207, -3.8741803, 3.5911851
4: -1.3233099, 2.8539722, -1.5216012, 3.2471507, -4.5704603, 4.3755732

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7772910
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7788109
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4174619, 2.5599651, -2.9309263, 2.6677952
1: -0.4575120, 3.1468184, -0.5153302, 3.5426629, -4.0001750, 3.6621485
2: -1.1562662, 2.1376271, -1.2751217, 2.4697607, -3.6260269, 3.4127488
3: -0.9258730, 2.5864434, -1.0362854, 2.9608207, -3.8866937, 3.6227288
4: -1.3418519, 2.8791902, -1.5216012, 3.2471507, -4.5890026, 4.4007912

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3987406, 2.4483228, -2.8114038, 2.6240315
1: -0.4512913, 3.1082866, -0.4923947, 3.3845534, -3.8358448, 3.6006813
2: -1.1403358, 2.1198447, -1.2141361, 2.3641450, -3.5044808, 3.3339808
3: -0.9133595, 2.5548997, -0.9919410, 2.8182275, -3.7315869, 3.5468407
4: -1.3233099, 2.8539722, -1.4456424, 3.1076982, -4.4310083, 4.2996144

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7772910
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788109
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3987406, 2.4483228, -2.8192840, 2.6490741
1: -0.4575120, 3.1468184, -0.4923947, 3.3845534, -3.8420653, 3.6392131
2: -1.1562662, 2.1376271, -1.2141361, 2.3641450, -3.5204113, 3.3517632
3: -0.9258730, 2.5864434, -0.9919410, 2.8182275, -3.7441006, 3.5783844
4: -1.3418519, 2.8791902, -1.4456424, 3.1076982, -4.4495502, 4.3248324

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4018040, 2.4201565, -2.8257761, 2.9158249
1: -0.5055367, 3.4809954, -0.4905685, 3.3785930, -3.8841295, 3.9715638
2: -1.2521493, 2.4233136, -1.2406874, 2.2999053, -3.5520546, 3.6640010
3: -1.0172695, 2.8950694, -0.9905678, 2.8093987, -3.8266683, 3.8856373
4: -1.4811087, 3.1972914, -1.4657205, 3.0825684, -4.5636768, 4.6630120

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7752799
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7771481
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4018040, 2.4201565, -2.8321061, 2.9269733
1: -0.5089593, 3.5015557, -0.4905685, 3.3785930, -3.8875523, 3.9921241
2: -1.2644246, 2.4245837, -1.2406874, 2.2999053, -3.5643299, 3.6652710
3: -1.0243788, 2.9105000, -0.9905678, 2.8093987, -3.8337774, 3.9010677
4: -1.4927979, 3.2066495, -1.4657205, 3.0825684, -4.5753660, 4.6723700

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7752799
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7771481
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3751003, 2.2736418, -2.6792614, 2.8891213
1: -0.5055367, 3.4809954, -0.4616931, 3.1732640, -3.6788006, 3.9426885
2: -1.2521493, 2.4233136, -1.1646090, 2.1681395, -3.4202888, 3.5879226
3: -1.0172695, 2.8950694, -0.9341383, 2.6221490, -3.6394186, 3.8292077
4: -1.4811087, 3.1972914, -1.3658996, 2.9074883, -4.3885970, 4.5631909

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7758395
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7771481
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3751003, 2.2736418, -2.6855912, 2.9002697
1: -0.5089593, 3.5015557, -0.4616931, 3.1732640, -3.6822233, 3.9632487
2: -1.2644246, 2.4245837, -1.1646090, 2.1681395, -3.4325640, 3.5891926
3: -1.0243788, 2.9105000, -0.9341383, 2.6221490, -3.6465278, 3.8446383
4: -1.4927979, 3.2066495, -1.3658996, 2.9074883, -4.4002862, 4.5725489

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7758395
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4018040, 2.4201565, -2.8059001, 2.8036869
1: -0.4820945, 3.3220072, -0.4905685, 3.3785930, -3.8606875, 3.8125758
2: -1.1904538, 2.3187947, -1.2406874, 2.2999053, -3.4903591, 3.5594821
3: -0.9715070, 2.7528343, -0.9905678, 2.8093987, -3.7809057, 3.7434020
4: -1.4044091, 3.0569649, -1.4657205, 3.0825684, -4.4869776, 4.5226855

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7753411
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7772093
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4018040, 2.4201565, -2.8081899, 2.8200445
1: -0.4848027, 3.3492410, -0.4905685, 3.3785930, -3.8633957, 3.8398094
2: -1.2020583, 2.3271937, -1.2406874, 2.2999053, -3.5019636, 3.5678811
3: -0.9766790, 2.7750435, -0.9905678, 2.8093987, -3.7860775, 3.7656112
4: -1.4156674, 3.0728226, -1.4657205, 3.0825684, -4.4982357, 4.5385432

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7753411
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7772093
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3751003, 2.2736418, -2.6593854, 2.7769833
1: -0.4820945, 3.3220072, -0.4616931, 3.1732640, -3.6553586, 3.7837002
2: -1.1904538, 2.3187947, -1.1646090, 2.1681395, -3.3585932, 3.4834037
3: -0.9715070, 2.7528343, -0.9341383, 2.6221490, -3.5936561, 3.6869726
4: -1.4044091, 3.0569649, -1.3658996, 2.9074883, -4.3118973, 4.4228644

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7757081
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7772093
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3751003, 2.2736418, -2.6616752, 2.7933409
1: -0.4848027, 3.3492410, -0.4616931, 3.1732640, -3.6580667, 3.8109341
2: -1.2020583, 2.3271937, -1.1646090, 2.1681395, -3.3701978, 3.4918027
3: -0.9766790, 2.7750435, -0.9341383, 2.6221490, -3.5988278, 3.7091818
4: -1.4156674, 3.0728226, -1.3658996, 2.9074883, -4.3231559, 4.4387221

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4561949, 2.6133635, -3.0189831, 2.9702158
1: -0.5055367, 3.4809954, -0.5258589, 3.6044338, -4.1099706, 4.0068541
2: -1.2521493, 2.4233136, -1.2888968, 2.5323260, -3.7844753, 3.7122104
3: -1.0172695, 2.8950694, -1.0625300, 3.0840850, -4.1013546, 3.9575994
4: -1.4811087, 3.1972914, -1.5837104, 3.2471111, -4.7282200, 4.7810020

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7752799
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7758395
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4561949, 2.6133635, -3.0253129, 2.9813643
1: -0.5089593, 3.5015557, -0.5258589, 3.6044338, -4.1133928, 4.0274143
2: -1.2644246, 2.4245837, -1.2888968, 2.5323260, -3.7967505, 3.7134805
3: -1.0243788, 2.9105000, -1.0625300, 3.0840850, -4.1084638, 3.9730301
4: -1.4927979, 3.2066495, -1.5837104, 3.2471111, -4.7399092, 4.7903600

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7752799
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7758395
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4588744, 2.6281669, -3.0337865, 2.9728954
1: -0.5055367, 3.4809954, -0.5282684, 3.6314750, -4.1370115, 4.0092640
2: -1.2521493, 2.4233136, -1.3008018, 2.5315521, -3.7837014, 3.7241154
3: -1.0172695, 2.8950694, -1.0670657, 3.0996389, -4.1169086, 3.9621351
4: -1.4811087, 3.1972914, -1.5958318, 3.2616765, -4.7427855, 4.7931232

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4588744, 2.6281669, -3.0401163, 2.9840438
1: -0.5089593, 3.5015557, -0.5282684, 3.6314750, -4.1404343, 4.0298243
2: -1.2644246, 2.4245837, -1.3008018, 2.5315521, -3.7959766, 3.7253854
3: -1.0243788, 2.9105000, -1.0670657, 3.0996389, -4.1240177, 3.9775658
4: -1.4927979, 3.2066495, -1.5958318, 3.2616765, -4.7544746, 4.8024812

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7759360
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7759810
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4174619, 2.5599651, -2.9457088, 2.8193448
1: -0.4820945, 3.3220072, -0.5153302, 3.5426629, -4.0247574, 3.8373375
2: -1.1904538, 2.3187947, -1.2751217, 2.4697607, -3.6602144, 3.5939164
3: -0.9715070, 2.7528343, -1.0362854, 2.9608207, -3.9323277, 3.7891197
4: -1.4044091, 3.0569649, -1.5216012, 3.2471507, -4.6515598, 4.5785661

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7753411
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4174619, 2.5599651, -2.9479985, 2.8357024
1: -0.4848027, 3.3492410, -0.5153302, 3.5426629, -4.0274653, 3.8645711
2: -1.2020583, 2.3271937, -1.2751217, 2.4697607, -3.6718190, 3.6023154
3: -0.9766790, 2.7750435, -1.0362854, 2.9608207, -3.9374995, 3.8113289
4: -1.4156674, 3.0728226, -1.5216012, 3.2471507, -4.6628180, 4.5944238

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7753411
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3987406, 2.4483228, -2.8340664, 2.8006234
1: -0.4820945, 3.3220072, -0.4923947, 3.3845534, -3.8666480, 3.8144019
2: -1.1904538, 2.3187947, -1.2141361, 2.3641450, -3.5545988, 3.5329309
3: -0.9715070, 2.7528343, -0.9919410, 2.8182275, -3.7897344, 3.7447753
4: -1.4044091, 3.0569649, -1.4456424, 3.1076982, -4.5121074, 4.5026073

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759339, upper bound: 2.7757081
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759339, upper bound: 2.7768629
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3987406, 2.4483228, -2.8363562, 2.8169813
1: -0.4848027, 3.3492410, -0.4923947, 3.3845534, -3.8693562, 3.8416357
2: -1.2020583, 2.3271937, -1.2141361, 2.3641450, -3.5662034, 3.5413299
3: -0.9766790, 2.7750435, -0.9919410, 2.8182275, -3.7949066, 3.7669845
4: -1.4156674, 3.0728226, -1.4456424, 3.1076982, -4.5233655, 4.5184650

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7757081
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
time: 0.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.47 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7773000
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7791682
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7773000
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791682
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7778597
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7778597
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7772892
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7791573
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7772892
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791573
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7772892
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7791573
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7772892
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7773018
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7778969
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7773018
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777795, upper bound: 2.7778969
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7780966
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7780966
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7772910
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7788109
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7772910
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788109
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7752799
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7771481
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7752799
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7771481
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7758395
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7771481
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7758395
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7753411
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7772093
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7753411
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7772093
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7757081
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7772093
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7752799
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7758395
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7752799
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7777813, upper bound: 2.7758395
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7759360
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7759810
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7753411
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7753411
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7759339, upper bound: 2.7757081
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7759339, upper bound: 2.7768629
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7757081
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3896064, 2.3705792, -2.7601857, 2.7601857
1: -0.4801092, 3.3123057, -0.4801092, 3.3123057, -3.7924149, 3.7924149
2: -1.2162256, 2.2505226, -1.2162256, 2.2505226, -3.4667482, 3.4667482
3: -0.9699728, 2.7406662, -0.9699728, 2.7406662, -3.7106390, 3.7106390
4: -1.4223459, 3.0282202, -1.4223459, 3.0282202, -4.4505663, 4.4505663

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7765046
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7742238
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3974134, 2.3933864, -2.7829928, 2.7679925
1: -0.4801092, 3.3123057, -0.4861144, 3.3476477, -3.8277569, 3.7984202
2: -1.2162256, 2.2505226, -1.2313797, 2.2667561, -3.4829817, 3.4819024
3: -0.9699728, 2.7406662, -0.9823158, 2.7700262, -3.7399991, 3.7229819
4: -1.4223459, 3.0282202, -1.4410114, 3.0509758, -4.4733219, 4.4692316

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7777688
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3896064, 2.3705792, -2.7679925, 2.7829928
1: -0.4861144, 3.3476477, -0.4801092, 3.3123057, -3.7984202, 3.8277569
2: -1.2313797, 2.2667561, -1.2162256, 2.2505226, -3.4819024, 3.4829817
3: -0.9823158, 2.7700262, -0.9699728, 2.7406662, -3.7229819, 3.7399991
4: -1.4410114, 3.0509758, -1.4223459, 3.0282202, -4.4692316, 4.4733219

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7747670
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7726478
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3974134, 2.3933864, -2.7907996, 2.7907996
1: -0.4861144, 3.3476477, -0.4861144, 3.3476477, -3.8337622, 3.8337622
2: -1.2313797, 2.2667561, -1.2313797, 2.2667561, -3.4981358, 3.4981358
3: -0.9823158, 2.7700262, -0.9823158, 2.7700262, -3.7523420, 3.7523420
4: -1.4410114, 3.0509758, -1.4410114, 3.0509758, -4.4919872, 4.4919872

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7750911
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3630810, 2.2252908, -2.6148973, 2.7336602
1: -0.4801092, 3.3123057, -0.4512913, 3.1082866, -3.5883958, 3.7635970
2: -1.2162256, 2.2505226, -1.1403358, 2.1198447, -3.3360703, 3.3908584
3: -0.9699728, 2.7406662, -0.9133595, 2.5548997, -3.5248725, 3.6540256
4: -1.4223459, 3.0282202, -1.3233099, 2.8539722, -4.2763181, 4.3515301

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7789359
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.42 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7719448, upper bound: 2.7762100
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754712, upper bound: 2.7773894
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3709612, 2.2503333, -2.6399398, 2.7415404
1: -0.4801092, 3.3123057, -0.4575120, 3.1468184, -3.6269276, 3.7698176
2: -1.2162256, 2.2505226, -1.1562662, 2.1376271, -3.3538527, 3.4067888
3: -0.9699728, 2.7406662, -0.9258730, 2.5864434, -3.5564163, 3.6665392
4: -1.4223459, 3.0282202, -1.3418519, 2.8791902, -4.3015361, 4.3700724

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7800277
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.46 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7719448, upper bound: 2.7774819
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754712, upper bound: 2.7786613
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3630810, 2.2252908, -2.6227040, 2.7564673
1: -0.4861144, 3.3476477, -0.4512913, 3.1082866, -3.5944011, 3.7989390
2: -1.2313797, 2.2667561, -1.1403358, 2.1198447, -3.3512244, 3.4070919
3: -0.9823158, 2.7700262, -0.9133595, 2.5548997, -3.5372155, 3.6833858
4: -1.4410114, 3.0509758, -1.3233099, 2.8539722, -4.2949839, 4.3742857

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786091, upper bound: 2.7777588
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.43 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746233, upper bound: 2.7755073
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772569, upper bound: 2.7760229
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3709612, 2.2503333, -2.6477466, 2.7643476
1: -0.4861144, 3.3476477, -0.4575120, 3.1468184, -3.6329329, 3.8051596
2: -1.2313797, 2.2667561, -1.1562662, 2.1376271, -3.3690069, 3.4230223
3: -0.9823158, 2.7700262, -0.9258730, 2.5864434, -3.5687592, 3.6958992
4: -1.4410114, 3.0509758, -1.3418519, 2.8791902, -4.3202019, 4.3928280

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786091, upper bound: 2.7779473
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.46 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746233, upper bound: 2.7758399
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772569, upper bound: 2.7763455
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3896064, 2.3705792, -2.7336602, 2.6148973
1: -0.4512913, 3.1082866, -0.4801092, 3.3123057, -3.7635970, 3.5883958
2: -1.1403358, 2.1198447, -1.2162256, 2.2505226, -3.3908584, 3.3360703
3: -0.9133595, 2.5548997, -0.9699728, 2.7406662, -3.6540256, 3.5248725
4: -1.3233099, 2.8539722, -1.4223459, 3.0282202, -4.3515301, 4.2763181

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745329, upper bound: 2.7766235
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.40 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723755, upper bound: 2.7758359
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760229, upper bound: 2.7768918
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3974134, 2.3933864, -2.7564673, 2.6227040
1: -0.4512913, 3.1082866, -0.4861144, 3.3476477, -3.7989390, 3.5944011
2: -1.1403358, 2.1198447, -1.2313797, 2.2667561, -3.4070919, 3.3512244
3: -0.9133595, 2.5548997, -0.9823158, 2.7700262, -3.6833858, 3.5372155
4: -1.3233099, 2.8539722, -1.4410114, 3.0509758, -4.3742857, 4.2949839

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745329, upper bound: 2.7778879
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.43 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723755, upper bound: 2.7776216
time: 0.46 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760229, upper bound: 2.7786775
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3896064, 2.3705792, -2.7415404, 2.6399398
1: -0.4575120, 3.1468184, -0.4801092, 3.3123057, -3.7698176, 3.6269276
2: -1.1562662, 2.1376271, -1.2162256, 2.2505226, -3.4067888, 3.3538527
3: -0.9258730, 2.5864434, -0.9699728, 2.7406662, -3.6665392, 3.5564163
4: -1.3418519, 2.8791902, -1.4223459, 3.0282202, -4.3700724, 4.3015361

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7747084
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.50 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746612, upper bound: 2.7749925
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772948, upper bound: 2.7754712
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3974134, 2.3933864, -2.7643476, 2.6477466
1: -0.4575120, 3.1468184, -0.4861144, 3.3476477, -3.8051596, 3.6329329
2: -1.1562662, 2.1376271, -1.2313797, 2.2667561, -3.4230223, 3.3690069
3: -0.9258730, 2.5864434, -0.9823158, 2.7700262, -3.6958992, 3.5687592
4: -1.3418519, 2.8791902, -1.4410114, 3.0509758, -4.3928280, 4.3202019

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7750266
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.44 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746612, upper bound: 2.7758361
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772948, upper bound: 2.7762811
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3630810, 2.2252908, -2.5883718, 2.5883718
1: -0.4512913, 3.1082866, -0.4512913, 3.1082866, -3.5595779, 3.5595779
2: -1.1403358, 2.1198447, -1.1403358, 2.1198447, -3.2601805, 3.2601805
3: -0.9133595, 2.5548997, -0.9133595, 2.5548997, -3.4682593, 3.4682593
4: -1.3233099, 2.8539722, -1.3233099, 2.8539722, -4.1772823, 4.1772823

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7786973
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7787504
time: 0.48 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3709612, 2.2503333, -2.6134143, 2.5962520
1: -0.4512913, 3.1082866, -0.4575120, 3.1468184, -3.5981097, 3.5657985
2: -1.1403358, 2.1198447, -1.1562662, 2.1376271, -3.2779629, 3.2761109
3: -0.9133595, 2.5548997, -0.9258730, 2.5864434, -3.4998031, 3.4807727
4: -1.3233099, 2.8539722, -1.3418519, 2.8791902, -4.2025003, 4.1958241

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7801671
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7801671
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3630810, 2.2252908, -2.5962520, 2.6134143
1: -0.4575120, 3.1468184, -0.4512913, 3.1082866, -3.5657985, 3.5981097
2: -1.1562662, 2.1376271, -1.1403358, 2.1198447, -3.2761109, 3.2779629
3: -0.9258730, 2.5864434, -0.9133595, 2.5548997, -3.4807727, 3.4998031
4: -1.3418519, 2.8791902, -1.3233099, 2.8539722, -4.1958241, 4.2025003

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7770552
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7769808
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3709612, 2.2503333, -2.6212945, 2.6212945
1: -0.4575120, 3.1468184, -0.4575120, 3.1468184, -3.6043303, 3.6043303
2: -1.1562662, 2.1376271, -1.1562662, 2.1376271, -3.2938933, 3.2938933
3: -0.9258730, 2.5864434, -0.9258730, 2.5864434, -3.5123165, 3.5123165
4: -1.3418519, 2.8791902, -1.3418519, 2.8791902, -4.2210422, 4.2210422

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7778307
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7777063
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4056197, 2.5140209, -2.9036274, 2.7761989
1: -0.4801092, 3.3123057, -0.5055367, 3.4809954, -3.9611046, 3.8178425
2: -1.2162256, 2.2505226, -1.2521493, 2.4233136, -3.6395392, 3.5026720
3: -0.9699728, 2.7406662, -1.0172695, 2.8950694, -3.8650422, 3.7579355
4: -1.4223459, 3.0282202, -1.4811087, 3.1972914, -4.6196375, 4.5093288

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7714556, upper bound: 2.7706598
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3857437, 2.4018829, -2.7914894, 2.7563229
1: -0.4801092, 3.3123057, -0.4820945, 3.3220072, -3.8021164, 3.7944002
2: -1.2162256, 2.2505226, -1.1904538, 2.3187947, -3.5350204, 3.4409764
3: -0.9699728, 2.7406662, -0.9715070, 2.7528343, -3.7228072, 3.7121730
4: -1.4223459, 3.0282202, -1.4044091, 3.0569649, -4.4793110, 4.4326291

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7714556, upper bound: 2.7706598
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4056197, 2.5140209, -2.9114342, 2.7990060
1: -0.4861144, 3.3476477, -0.5055367, 3.4809954, -3.9671099, 3.8531842
2: -1.2313797, 2.2667561, -1.2521493, 2.4233136, -3.6546934, 3.5189054
3: -0.9823158, 2.7700262, -1.0172695, 2.8950694, -3.8773851, 3.7872958
4: -1.4410114, 3.0509758, -1.4811087, 3.1972914, -4.6383028, 4.5320845

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731543, upper bound: 2.7685290
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710662, upper bound: 2.7663166
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3857437, 2.4018829, -2.7992964, 2.7791300
1: -0.4861144, 3.3476477, -0.4820945, 3.3220072, -3.8081217, 3.8297422
2: -1.2313797, 2.2667561, -1.1904538, 2.3187947, -3.5501745, 3.4572098
3: -0.9823158, 2.7700262, -0.9715070, 2.7528343, -3.7351501, 3.7415333
4: -1.4410114, 3.0509758, -1.4044091, 3.0569649, -4.4979763, 4.4553847

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731543, upper bound: 2.7685290
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710662, upper bound: 2.7663166
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4119495, 2.5251694, -2.9147758, 2.7825289
1: -0.4801092, 3.3123057, -0.5089593, 3.5015557, -3.9816649, 3.8212650
2: -1.2162256, 2.2505226, -1.2644246, 2.4245837, -3.6408093, 3.5149472
3: -0.9699728, 2.7406662, -1.0243788, 2.9105000, -3.8804729, 3.7650449
4: -1.4223459, 3.0282202, -1.4927979, 3.2066495, -4.6289954, 4.5210180

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3880334, 2.4182405, -2.8078470, 2.7586126
1: -0.4801092, 3.3123057, -0.4848027, 3.3492410, -3.8293502, 3.7971084
2: -1.2162256, 2.2505226, -1.2020583, 2.3271937, -3.5434194, 3.4525809
3: -0.9699728, 2.7406662, -0.9766790, 2.7750435, -3.7450163, 3.7173452
4: -1.4223459, 3.0282202, -1.4156674, 3.0728226, -4.4951687, 4.4438877

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4119495, 2.5251694, -2.9225826, 2.8053360
1: -0.4861144, 3.3476477, -0.5089593, 3.5015557, -3.9876702, 3.8566070
2: -1.2313797, 2.2667561, -1.2644246, 2.4245837, -3.6559634, 3.5311806
3: -0.9823158, 2.7700262, -1.0243788, 2.9105000, -3.8928158, 3.7944050
4: -1.4410114, 3.0509758, -1.4927979, 3.2066495, -4.6476612, 4.5437737

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7647012, upper bound: 2.7647692
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754901
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 3.00 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7758398
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751152, upper bound: 2.7763454
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3880334, 2.4182405, -2.8156538, 2.7814198
1: -0.4861144, 3.3476477, -0.4848027, 3.3492410, -3.8353555, 3.8324504
2: -1.2313797, 2.2667561, -1.2020583, 2.3271937, -3.5585735, 3.4688144
3: -0.9823158, 2.7700262, -0.9766790, 2.7750435, -3.7573593, 3.7467051
4: -1.4410114, 3.0509758, -1.4156674, 3.0728226, -4.5138340, 4.4666433

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7647012, upper bound: 2.7647692
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7779244
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.89 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7758398
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751152, upper bound: 2.7763454
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4056197, 2.5140209, -2.8771019, 2.6309104
1: -0.4512913, 3.1082866, -0.5055367, 3.4809954, -3.9322867, 3.6138234
2: -1.1403358, 2.1198447, -1.2521493, 2.4233136, -3.5636494, 3.3719940
3: -0.9133595, 2.5548997, -1.0172695, 2.8950694, -3.8084288, 3.5721693
4: -1.3233099, 2.8539722, -1.4811087, 3.1972914, -4.5206013, 4.3350811

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743753, upper bound: 2.7767873
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4119495, 2.5251694, -2.8882504, 2.6372404
1: -0.4512913, 3.1082866, -0.5089593, 3.5015557, -3.9528470, 3.6172459
2: -1.1403358, 2.1198447, -1.2644246, 2.4245837, -3.5649195, 3.3842692
3: -0.9133595, 2.5548997, -1.0243788, 2.9105000, -3.8238597, 3.5792785
4: -1.3233099, 2.8539722, -1.4927979, 3.2066495, -4.5299597, 4.3467703

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743753, upper bound: 2.7794065
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4056197, 2.5140209, -2.8849821, 2.6559529
1: -0.4575120, 3.1468184, -0.5055367, 3.4809954, -3.9385073, 3.6523552
2: -1.1562662, 2.1376271, -1.2521493, 2.4233136, -3.5795798, 3.3897765
3: -0.9258730, 2.5864434, -1.0172695, 2.8950694, -3.8209424, 3.6037130
4: -1.3418519, 2.8791902, -1.4811087, 3.1972914, -4.5391436, 4.3602991

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759765, upper bound: 2.7748129
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4119495, 2.5251694, -2.8961306, 2.6622829
1: -0.4575120, 3.1468184, -0.5089593, 3.5015557, -3.9590676, 3.6557777
2: -1.1562662, 2.1376271, -1.2644246, 2.4245837, -3.5808499, 3.4020517
3: -0.9258730, 2.5864434, -1.0243788, 2.9105000, -3.8363731, 3.6108222
4: -1.3418519, 2.8791902, -1.4927979, 3.2066495, -4.5485015, 4.3719883

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759765, upper bound: 2.7760454
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3857437, 2.4018829, -2.7649639, 2.6110344
1: -0.4512913, 3.1082866, -0.4820945, 3.3220072, -3.7732985, 3.5903811
2: -1.1403358, 2.1198447, -1.1904538, 2.3187947, -3.4591305, 3.3102984
3: -0.9133595, 2.5548997, -0.9715070, 2.7528343, -3.6661940, 3.5264068
4: -1.3233099, 2.8539722, -1.4044091, 3.0569649, -4.3802748, 4.2583814

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608479, upper bound: 2.7670756
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7786973
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7787504
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3880334, 2.4182405, -2.7813215, 2.6133242
1: -0.4512913, 3.1082866, -0.4848027, 3.3492410, -3.8005323, 3.5930893
2: -1.1403358, 2.1198447, -1.2020583, 2.3271937, -3.4675295, 3.3219030
3: -0.9133595, 2.5548997, -0.9766790, 2.7750435, -3.6884031, 3.5315785
4: -1.3233099, 2.8539722, -1.4156674, 3.0728226, -4.3961325, 4.2696395

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608479, upper bound: 2.7670756
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7799106
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7799106
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3857437, 2.4018829, -2.7728441, 2.6360769
1: -0.4575120, 3.1468184, -0.4820945, 3.3220072, -3.7795191, 3.6289129
2: -1.1562662, 2.1376271, -1.1904538, 2.3187947, -3.4750609, 3.3280809
3: -0.9258730, 2.5864434, -0.9715070, 2.7528343, -3.6787074, 3.5579505
4: -1.3418519, 2.8791902, -1.4044091, 3.0569649, -4.3988171, 4.2835994

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608755, upper bound: 2.7644214
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7770552
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7769808
time: 0.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3880334, 2.4182405, -2.7892017, 2.6383667
1: -0.4575120, 3.1468184, -0.4848027, 3.3492410, -3.8067529, 3.6316211
2: -1.1562662, 2.1376271, -1.2020583, 2.3271937, -3.4834599, 3.3396854
3: -0.9258730, 2.5864434, -0.9766790, 2.7750435, -3.7009165, 3.5631223
4: -1.3418519, 2.8791902, -1.4156674, 3.0728226, -4.4146748, 4.2948575

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608755, upper bound: 2.7644214
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7778236
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7776769
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3896064, 2.3705792, -2.7761989, 2.9036274
1: -0.5055367, 3.4809954, -0.4801092, 3.3123057, -3.8178425, 3.9611046
2: -1.2521493, 2.4233136, -1.2162256, 2.2505226, -3.5026720, 3.6395392
3: -1.0172695, 2.8950694, -0.9699728, 2.7406662, -3.7579355, 3.8650422
4: -1.4811087, 3.1972914, -1.4223459, 3.0282202, -4.5093288, 4.6196375

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7760001
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7742530
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3974134, 2.3933864, -2.7990060, 2.9114342
1: -0.5055367, 3.4809954, -0.4861144, 3.3476477, -3.8531842, 3.9671099
2: -1.2521493, 2.4233136, -1.2313797, 2.2667561, -3.5189054, 3.6546934
3: -1.0172695, 2.8950694, -0.9823158, 2.7700262, -3.7872958, 3.8773851
4: -1.4811087, 3.1972914, -1.4410114, 3.0509758, -4.5320845, 4.6383028

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7772661
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7755190
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3896064, 2.3705792, -2.7825289, 2.9147758
1: -0.5089593, 3.5015557, -0.4801092, 3.3123057, -3.8212650, 3.9816649
2: -1.2644246, 2.4245837, -1.2162256, 2.2505226, -3.5149472, 3.6408093
3: -1.0243788, 2.9105000, -0.9699728, 2.7406662, -3.7650449, 3.8804729
4: -1.4927979, 3.2066495, -1.4223459, 3.0282202, -4.5210180, 4.6289954

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7735495
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7725126
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3974134, 2.3933864, -2.8053360, 2.9225826
1: -0.5089593, 3.5015557, -0.4861144, 3.3476477, -3.8566070, 3.9876702
2: -1.2644246, 2.4245837, -1.2313797, 2.2667561, -3.5311806, 3.6559634
3: -1.0243788, 2.9105000, -0.9823158, 2.7700262, -3.7944050, 3.8928158
4: -1.4927979, 3.2066495, -1.4410114, 3.0509758, -4.5437737, 4.6476612

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7738449
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7728425
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3630810, 2.2252908, -2.6309104, 2.8771019
1: -0.5055367, 3.4809954, -0.4512913, 3.1082866, -3.6138234, 3.9322867
2: -1.2521493, 2.4233136, -1.1403358, 2.1198447, -3.3719940, 3.5636494
3: -1.0172695, 2.8950694, -0.9133595, 2.5548997, -3.5721693, 3.8084288
4: -1.4811087, 3.1972914, -1.3233099, 2.8539722, -4.3350811, 4.5206013

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7782382
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7699404
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3709612, 2.2503333, -2.6559529, 2.8849821
1: -0.5055367, 3.4809954, -0.4575120, 3.1468184, -3.6523552, 3.9385073
2: -1.2521493, 2.4233136, -1.1562662, 2.1376271, -3.3897765, 3.5795798
3: -1.0172695, 2.8950694, -0.9258730, 2.5864434, -3.6037130, 3.8209424
4: -1.4811087, 3.1972914, -1.3418519, 2.8791902, -4.3602991, 4.5391436

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7793301
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7711735
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3630810, 2.2252908, -2.6372404, 2.8882504
1: -0.5089593, 3.5015557, -0.4512913, 3.1082866, -3.6172459, 3.9528470
2: -1.2644246, 2.4245837, -1.1403358, 2.1198447, -3.3842692, 3.5649195
3: -1.0243788, 2.9105000, -0.9133595, 2.5548997, -3.5792785, 3.8238597
4: -1.4927979, 3.2066495, -1.3233099, 2.8539722, -4.3467703, 4.5299597

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783385, upper bound: 2.7757387
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752083, upper bound: 2.7698186
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3709612, 2.2503333, -2.6622829, 2.8961306
1: -0.5089593, 3.5015557, -0.4575120, 3.1468184, -3.6557777, 3.9590676
2: -1.2644246, 2.4245837, -1.1562662, 2.1376271, -3.4020517, 3.5808499
3: -1.0243788, 2.9105000, -0.9258730, 2.5864434, -3.6108222, 3.8363731
4: -1.4927979, 3.2066495, -1.3418519, 2.8791902, -4.3719883, 4.5485015

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7592106, upper bound: 2.7533886
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787334, upper bound: 2.7751537
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3896064, 2.3705792, -2.7563229, 2.7914894
1: -0.4820945, 3.3220072, -0.4801092, 3.3123057, -3.7944002, 3.8021164
2: -1.1904538, 2.3187947, -1.2162256, 2.2505226, -3.4409764, 3.5350204
3: -0.9715070, 2.7528343, -0.9699728, 2.7406662, -3.7121730, 3.7228072
4: -1.4044091, 3.0569649, -1.4223459, 3.0282202, -4.4326291, 4.4793110

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7763857
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7698002
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3974134, 2.3933864, -2.7791300, 2.7992964
1: -0.4820945, 3.3220072, -0.4861144, 3.3476477, -3.8297422, 3.8081217
2: -1.1904538, 2.3187947, -1.2313797, 2.2667561, -3.4572098, 3.5501745
3: -0.9715070, 2.7528343, -0.9823158, 2.7700262, -3.7415333, 3.7351501
4: -1.4044091, 3.0569649, -1.4410114, 3.0509758, -4.4553847, 4.4979763

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7776517
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7710662
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3896064, 2.3705792, -2.7586126, 2.8078470
1: -0.4848027, 3.3492410, -0.4801092, 3.3123057, -3.7971084, 3.8293502
2: -1.2020583, 2.3271937, -1.2162256, 2.2505226, -3.4525809, 3.5434194
3: -0.9766790, 2.7750435, -0.9699728, 2.7406662, -3.7173452, 3.7450163
4: -1.4156674, 3.0728226, -1.4223459, 3.0282202, -4.4438877, 4.4951687

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7736132
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7745035
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3974134, 2.3933864, -2.7814198, 2.8156538
1: -0.4848027, 3.3492410, -0.4861144, 3.3476477, -3.8324504, 3.8353555
2: -1.2020583, 2.3271937, -1.2313797, 2.2667561, -3.4688144, 3.5585735
3: -0.9766790, 2.7750435, -0.9823158, 2.7700262, -3.7467051, 3.7573593
4: -1.4156674, 3.0728226, -1.4410114, 3.0509758, -4.4666433, 4.5138340

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7738952
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7751405
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3630810, 2.2252908, -2.6110344, 2.7649639
1: -0.4820945, 3.3220072, -0.4512913, 3.1082866, -3.5903811, 3.7732985
2: -1.1904538, 2.3187947, -1.1403358, 2.1198447, -3.3102984, 3.4591305
3: -0.9715070, 2.7528343, -0.9133595, 2.5548997, -3.5264068, 3.6661940
4: -1.4044091, 3.0569649, -1.3233099, 2.8539722, -4.2583814, 4.3802748

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7757595
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7786459
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3709612, 2.2503333, -2.6360769, 2.7728441
1: -0.4820945, 3.3220072, -0.4575120, 3.1468184, -3.6289129, 3.7795191
2: -1.1904538, 2.3187947, -1.1562662, 2.1376271, -3.3280809, 3.4750609
3: -0.9715070, 2.7528343, -0.9258730, 2.5864434, -3.5579505, 3.6787074
4: -1.4044091, 3.0569649, -1.3418519, 2.8791902, -4.2835994, 4.3988171

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7762219
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3630810, 2.2252908, -2.6133242, 2.7813215
1: -0.4848027, 3.3492410, -0.4512913, 3.1082866, -3.5930893, 3.8005323
2: -1.2020583, 2.3271937, -1.1403358, 2.1198447, -3.3219030, 3.4675295
3: -0.9766790, 2.7750435, -0.9133595, 2.5548997, -3.5315785, 3.6884031
4: -1.4156674, 3.0728226, -1.3233099, 2.8539722, -4.2696395, 4.3961325

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3709612, 2.2503333, -2.6383667, 2.7892017
1: -0.4848027, 3.3492410, -0.4575120, 3.1468184, -3.6316211, 3.8067529
2: -1.2020583, 2.3271937, -1.1562662, 2.1376271, -3.3396854, 3.4834599
3: -0.9766790, 2.7750435, -0.9258730, 2.5864434, -3.5631223, 3.7009165
4: -1.4156674, 3.0728226, -1.3418519, 2.8791902, -4.2948575, 4.4146748

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7760039
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4056197, 2.5140209, -2.9196405, 2.9196405
1: -0.5055367, 3.4809954, -0.5055367, 3.4809954, -3.9865322, 3.9865322
2: -1.2521493, 2.4233136, -1.2521493, 2.4233136, -3.6754630, 3.6754630
3: -1.0172695, 2.8950694, -1.0172695, 2.8950694, -3.9123387, 3.9123387
4: -1.4811087, 3.1972914, -1.4811087, 3.1972914, -4.6784000, 4.6784000

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3857437, 2.4018829, -2.8075025, 2.8997645
1: -0.5055367, 3.4809954, -0.4820945, 3.3220072, -3.8275437, 3.9630899
2: -1.2521493, 2.4233136, -1.1904538, 2.3187947, -3.5709441, 3.6137674
3: -1.0172695, 2.8950694, -0.9715070, 2.7528343, -3.7701039, 3.8665762
4: -1.4811087, 3.1972914, -1.4044091, 3.0569649, -4.5380735, 4.6017003

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4056197, 2.5140209, -2.9259706, 2.9307890
1: -0.5089593, 3.5015557, -0.5055367, 3.4809954, -3.9899547, 4.0070925
2: -1.2644246, 2.4245837, -1.2521493, 2.4233136, -3.6877382, 3.6767330
3: -1.0243788, 2.9105000, -1.0172695, 2.8950694, -3.9194481, 3.9277697
4: -1.4927979, 3.2066495, -1.4811087, 3.1972914, -4.6900892, 4.6877584

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689538, upper bound: 2.7570467
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689836, upper bound: 2.7570490
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3857437, 2.4018829, -2.8138323, 2.9109130
1: -0.5089593, 3.5015557, -0.4820945, 3.3220072, -3.8309665, 3.9836502
2: -1.2644246, 2.4245837, -1.1904538, 2.3187947, -3.5832193, 3.6150374
3: -1.0243788, 2.9105000, -0.9715070, 2.7528343, -3.7772131, 3.8820071
4: -1.4927979, 3.2066495, -1.4044091, 3.0569649, -4.5497627, 4.6110587

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689538, upper bound: 2.7570467
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689836, upper bound: 2.7570490
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4119495, 2.5251694, -2.9307890, 2.9259706
1: -0.5055367, 3.4809954, -0.5089593, 3.5015557, -4.0070925, 3.9899547
2: -1.2521493, 2.4233136, -1.2644246, 2.4245837, -3.6767330, 3.6877382
3: -1.0172695, 2.8950694, -1.0243788, 2.9105000, -3.9277697, 3.9194481
4: -1.4811087, 3.1972914, -1.4927979, 3.2066495, -4.6877584, 4.6900892

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
time: 0.50 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3880334, 2.4182405, -2.8238602, 2.9020543
1: -0.5055367, 3.4809954, -0.4848027, 3.3492410, -3.8547778, 3.9657981
2: -1.2521493, 2.4233136, -1.2020583, 2.3271937, -3.5793431, 3.6253719
3: -1.0172695, 2.8950694, -0.9766790, 2.7750435, -3.7923131, 3.8717484
4: -1.4811087, 3.1972914, -1.4156674, 3.0728226, -4.5539312, 4.6129589

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4119495, 2.5251694, -2.9371190, 2.9371190
1: -0.5089593, 3.5015557, -0.5089593, 3.5015557, -4.0105152, 4.0105152
2: -1.2644246, 2.4245837, -1.2644246, 2.4245837, -3.6890082, 3.6890082
3: -1.0243788, 2.9105000, -1.0243788, 2.9105000, -3.9348788, 3.9348788
4: -1.4927979, 3.2066495, -1.4927979, 3.2066495, -4.6994476, 4.6994476

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3880334, 2.4182405, -2.8301902, 2.9132028
1: -0.5089593, 3.5015557, -0.4848027, 3.3492410, -3.8582003, 3.9863584
2: -1.2644246, 2.4245837, -1.2020583, 2.3271937, -3.5916183, 3.6266420
3: -1.0243788, 2.9105000, -0.9766790, 2.7750435, -3.7994223, 3.8871789
4: -1.4927979, 3.2066495, -1.4156674, 3.0728226, -4.5656204, 4.6223168

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4056197, 2.5140209, -2.8997645, 2.8075025
1: -0.4820945, 3.3220072, -0.5055367, 3.4809954, -3.9630899, 3.8275437
2: -1.1904538, 2.3187947, -1.2521493, 2.4233136, -3.6137674, 3.5709441
3: -0.9715070, 2.7528343, -1.0172695, 2.8950694, -3.8665762, 3.7701039
4: -1.4044091, 3.0569649, -1.4811087, 3.1972914, -4.6017003, 4.5380735

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7763899
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7700042
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4119495, 2.5251694, -2.9109130, 2.8138323
1: -0.4820945, 3.3220072, -0.5089593, 3.5015557, -3.9836502, 3.8309665
2: -1.1904538, 2.3187947, -1.2644246, 2.4245837, -3.6150374, 3.5832193
3: -0.9715070, 2.7528343, -1.0243788, 2.9105000, -3.8820071, 3.7772131
4: -1.4044091, 3.0569649, -1.4927979, 3.2066495, -4.6110587, 4.5497627

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7790485
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7725818
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4056197, 2.5140209, -2.9020543, 2.8238602
1: -0.4848027, 3.3492410, -0.5055367, 3.4809954, -3.9657981, 3.8547778
2: -1.2020583, 2.3271937, -1.2521493, 2.4233136, -3.6253719, 3.5793431
3: -0.9766790, 2.7750435, -1.0172695, 2.8950694, -3.8717484, 3.7923131
4: -1.4156674, 3.0728226, -1.4811087, 3.1972914, -4.6129589, 4.5539312

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7736132
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4119495, 2.5251694, -2.9132028, 2.8301902
1: -0.4848027, 3.3492410, -0.5089593, 3.5015557, -3.9863584, 3.8582003
2: -1.2020583, 2.3271937, -1.2644246, 2.4245837, -3.6266420, 3.5916183
3: -0.9766790, 2.7750435, -1.0243788, 2.9105000, -3.8871789, 3.7994223
4: -1.4156674, 3.0728226, -1.4927979, 3.2066495, -4.6223168, 4.5656204

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7741817
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3857437, 2.4018829, -2.7876265, 2.7876265
1: -0.4820945, 3.3220072, -0.4820945, 3.3220072, -3.8041017, 3.8041017
2: -1.1904538, 2.3187947, -1.1904538, 2.3187947, -3.5092485, 3.5092485
3: -0.9715070, 2.7528343, -0.9715070, 2.7528343, -3.7243414, 3.7243414
4: -1.4044091, 3.0569649, -1.4044091, 3.0569649, -4.4613738, 4.4613738

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3880334, 2.4182405, -2.8039842, 2.7899163
1: -0.4820945, 3.3220072, -0.4848027, 3.3492410, -3.8313355, 3.8068099
2: -1.1904538, 2.3187947, -1.2020583, 2.3271937, -3.5176475, 3.5208530
3: -0.9715070, 2.7528343, -0.9766790, 2.7750435, -3.7465506, 3.7295132
4: -1.4044091, 3.0569649, -1.4156674, 3.0728226, -4.4772315, 4.4726324

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3857437, 2.4018829, -2.7899163, 2.8039842
1: -0.4848027, 3.3492410, -0.4820945, 3.3220072, -3.8068099, 3.8313355
2: -1.2020583, 2.3271937, -1.1904538, 2.3187947, -3.5208530, 3.5176475
3: -0.9766790, 2.7750435, -0.9715070, 2.7528343, -3.7295132, 3.7465506
4: -1.4156674, 3.0728226, -1.4044091, 3.0569649, -4.4726324, 4.4772315

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3880334, 2.4182405, -2.8062739, 2.8062739
1: -0.4848027, 3.3492410, -0.4848027, 3.3492410, -3.8340437, 3.8340437
2: -1.2020583, 2.3271937, -1.2020583, 2.3271937, -3.5292521, 3.5292521
3: -0.9766790, 2.7750435, -0.9766790, 2.7750435, -3.7517223, 3.7517223
4: -1.4156674, 3.0728226, -1.4156674, 3.0728226, -4.4884901, 4.4884901

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.43 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7765046
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7742238
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7777688
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7747670
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7726478
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7750911
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7719448, upper bound: 2.7762100
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7754712, upper bound: 2.7773894
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7719448, upper bound: 2.7774819
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7754712, upper bound: 2.7786613
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7746233, upper bound: 2.7755073
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7772569, upper bound: 2.7760229
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7746233, upper bound: 2.7758399
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7772569, upper bound: 2.7763455
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7723755, upper bound: 2.7758359
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7760229, upper bound: 2.7768918
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7723755, upper bound: 2.7776216
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7760229, upper bound: 2.7786775
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7746612, upper bound: 2.7749925
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7772948, upper bound: 2.7754712
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7746612, upper bound: 2.7758361
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7772948, upper bound: 2.7762811
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7786973
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7787504
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7801671
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7801671
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7770552
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7769808
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7778307
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7777063
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7714556, upper bound: 2.7706598
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7714556, upper bound: 2.7706598
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7731543, upper bound: 2.7685290
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7710662, upper bound: 2.7663166
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7731543, upper bound: 2.7685290
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7710662, upper bound: 2.7663166
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7758398
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7751152, upper bound: 2.7763454
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7758398
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7751152, upper bound: 2.7763454
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7786973
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7787504
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7799106
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7799106
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7770552
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7769808
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7778236
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7776769
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7760001
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7742530
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7772661
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7755190
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7735495
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7725126
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7738449
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7728425
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7782382
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7699404
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7793301
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7711735
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7783385, upper bound: 2.7757387
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7752083, upper bound: 2.7698186
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7592106, upper bound: 2.7533886
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7787334, upper bound: 2.7751537
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7763857
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7698002
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7776517
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7710662
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7745035
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7751405
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7757595
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7786459
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7762219
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7760039
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7689538, upper bound: 2.7570467
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7689836, upper bound: 2.7570490
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7689538, upper bound: 2.7570467
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7689836, upper bound: 2.7570490
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7763899
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7700042
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7790485
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7725818
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7736132
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7741817
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.73
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3739614, 2.2898805, -0.3896064, 2.3705792, -2.7445407, 2.6794870
1: -0.4637825, 3.2020445, -0.4801092, 3.3123057, -3.7760882, 3.6821537
2: -1.1733108, 2.1740575, -1.2162256, 2.2505226, -3.4238334, 3.3902831
3: -0.9386251, 2.6226339, -0.9699728, 2.7406662, -3.6792912, 3.5926068
4: -1.3490784, 2.9333854, -1.4223459, 3.0282202, -4.3772984, 4.3557310

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4101669, 2.4480877, -0.3855862, 2.3537929, -2.7639599, 2.8336740
1: -0.4886297, 3.3980122, -0.4765965, 3.2889633, -3.7775931, 3.8746085
2: -1.2252474, 2.3572233, -1.2068617, 2.2354035, -3.4606509, 3.5640850
3: -0.9874058, 2.8292980, -0.9630527, 2.7163746, -3.7037804, 3.7923508
4: -1.4586418, 3.0615449, -1.4068248, 3.0093212, -4.4679632, 4.4683695

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3739614, 2.2898805, -0.3974134, 2.3933864, -2.7673478, 2.6872940
1: -0.4637825, 3.2020445, -0.4861144, 3.3476477, -3.8114302, 3.6881590
2: -1.1733108, 2.1740575, -1.2313797, 2.2667561, -3.4400668, 3.4054372
3: -0.9386251, 2.6226339, -0.9823158, 2.7700262, -3.7086513, 3.6049497
4: -1.3490784, 2.9333854, -1.4410114, 3.0509758, -4.4000540, 4.3743968

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4101669, 2.4480877, -0.3937279, 2.3775983, -2.7877650, 2.8418155
1: -0.4886297, 3.3980122, -0.4828582, 3.3257217, -3.8143516, 3.8808703
2: -1.2252474, 2.3572233, -1.2226243, 2.2524691, -3.4777164, 3.5798476
3: -0.9874058, 2.8292980, -0.9758934, 2.7471776, -3.7345834, 3.8051915
4: -1.4586418, 3.0615449, -1.4264481, 3.0330350, -4.4916768, 4.4879932

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3809908, 2.3117228, -0.3896064, 2.3705792, -2.7515700, 2.7013292
1: -0.4694015, 3.2361622, -0.4801092, 3.3123057, -3.7817073, 3.7162714
2: -1.1875644, 2.1893144, -1.2162256, 2.2505226, -3.4380870, 3.4055400
3: -0.9499736, 2.6506827, -0.9699728, 2.7406662, -3.6906397, 3.6206555
4: -1.3663981, 2.9551375, -1.4223459, 3.0282202, -4.3946180, 4.3774834

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.4191682, 2.4729218, -0.3855862, 2.3537929, -2.7729611, 2.8585081
1: -0.4953873, 3.4356165, -0.4765965, 3.2889633, -3.7843506, 3.9122128
2: -1.2422593, 2.3750792, -1.2068617, 2.2354035, -3.4776628, 3.5819409
3: -1.0012507, 2.8649900, -0.9630527, 2.7163746, -3.7176254, 3.8280427
4: -1.4800587, 3.0869579, -1.4068248, 3.0093212, -4.4893799, 4.4937830

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3809908, 2.3117228, -0.3974134, 2.3933864, -2.7743771, 2.7091360
1: -0.4694015, 3.2361622, -0.4861144, 3.3476477, -3.8170493, 3.7222767
2: -1.1875644, 2.1893144, -1.2313797, 2.2667561, -3.4543204, 3.4206941
3: -0.9499736, 2.6506827, -0.9823158, 2.7700262, -3.7199998, 3.6329985
4: -1.3663981, 2.9551375, -1.4410114, 3.0509758, -4.4173737, 4.3961487

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.4191682, 2.4729218, -0.3937279, 2.3775983, -2.7967665, 2.8666496
1: -0.4953873, 3.4356165, -0.4828582, 3.3257217, -3.8211091, 3.9184747
2: -1.2422593, 2.3750792, -1.2226243, 2.2524691, -3.4947283, 3.5977035
3: -1.0012507, 2.8649900, -0.9758934, 2.7471776, -3.7484283, 3.8408833
4: -1.4800587, 3.0869579, -1.4264481, 3.0330350, -4.5130939, 4.5134058

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.2636126, 1.9244066, -0.3630810, 2.2252908, -2.4889035, 2.2874875
1: -0.3881378, 2.7189453, -0.4512913, 3.1082866, -3.4964244, 3.1702366
2: -0.9683192, 1.8626842, -1.1403358, 2.1198447, -3.0881639, 3.0030200
3: -0.7888877, 2.1286426, -0.9133595, 2.5548997, -3.3437874, 3.0420022
4: -1.0292611, 2.5780277, -1.3233099, 2.8539722, -3.8832333, 3.9013376

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7695340, upper bound: 2.7725720
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 25

Time for candidate selection: 2.63 seconds

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723095, upper bound: 2.7725626
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723095, upper bound: 2.7725626
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3810645, 2.3678570, -0.3630810, 2.2252908, -2.6063552, 2.7309380
1: -0.4877526, 3.3072553, -0.4512913, 3.1082866, -3.5960393, 3.7585466
2: -1.2041907, 2.2659509, -1.1403358, 2.1198447, -3.3240354, 3.4062867
3: -0.9885994, 2.7563348, -0.9133595, 2.5548997, -3.5434990, 3.6696944
4: -1.4238694, 3.0649676, -1.3233099, 2.8539722, -4.2778416, 4.3882775

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747315, upper bound: 2.7745404
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 25

Time for candidate selection: 2.58 seconds

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758359, upper bound: 2.7737420
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758359, upper bound: 2.7773894
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.2636126, 1.9244066, -0.3709612, 2.2503333, -2.5139461, 2.2953677
1: -0.3881378, 2.7189453, -0.4575120, 3.1468184, -3.5349562, 3.1764572
2: -0.9683192, 1.8626842, -1.1562662, 2.1376271, -3.1059463, 3.0189505
3: -0.7888877, 2.1286426, -0.9258730, 2.5864434, -3.3753312, 3.0545156
4: -1.0292611, 2.5780277, -1.3418519, 2.8791902, -3.9084513, 3.9198797

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7672857, upper bound: 2.7741393
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 25

Time for candidate selection: 2.56 seconds

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7714661, upper bound: 2.7748483
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7714661, upper bound: 2.7748483
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3810645, 2.3678570, -0.3709612, 2.2503333, -2.6313977, 2.7388182
1: -0.4877526, 3.3072553, -0.4575120, 3.1468184, -3.6345711, 3.7647672
2: -1.2041907, 2.2659509, -1.1562662, 2.1376271, -3.3418179, 3.4222171
3: -0.9885994, 2.7563348, -0.9258730, 2.5864434, -3.5750427, 3.6822078
4: -1.4238694, 3.0649676, -1.3418519, 2.8791902, -4.3030596, 4.4068193

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726219, upper bound: 2.7761142
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 25

Time for candidate selection: 2.57 seconds

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749925, upper bound: 2.7760277
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749925, upper bound: 2.7786613
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.2794446, 1.9706439, -0.3630810, 2.2252908, -2.5047355, 2.3337250
1: -0.3996434, 2.7860620, -0.4512913, 3.1082866, -3.5079300, 3.2373533
2: -0.9987776, 1.8992174, -1.1403358, 2.1198447, -3.1186223, 3.0395532
3: -0.8124233, 2.1936581, -0.9133595, 2.5548997, -3.3673229, 3.1070175
4: -1.0716901, 2.6247184, -1.3233099, 2.8539722, -3.9256623, 3.9480283

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7709326, upper bound: 2.7715338
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 25

Time for candidate selection: 2.60 seconds

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749880, upper bound: 2.7718599
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749880, upper bound: 2.7718599
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3916628, 2.3988516, -0.3630810, 2.2252908, -2.6169536, 2.7619326
1: -0.4951562, 3.3508790, -0.4512913, 3.1082866, -3.6034427, 3.8021703
2: -1.2243512, 2.2904301, -1.1403358, 2.1198447, -3.3441958, 3.4307659
3: -1.0028315, 2.7935500, -0.9133595, 2.5548997, -3.5577312, 3.7069097
4: -1.4461796, 3.0963032, -1.3233099, 2.8539722, -4.3001518, 4.4196129

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756610, upper bound: 2.7725883
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 5
type: B, layer: 3, pos: 10
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 18
type: B, layer: 3, pos: 6
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 13
type: B, layer: 3, pos: 25

Time for candidate selection: 2.68 seconds

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776216, upper bound: 2.7723755
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776216, upper bound: 2.7723755
time: 0.47 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.34 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7723095, upper bound: 2.7725626
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7723095, upper bound: 2.7725626
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7758359, upper bound: 2.7737420
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7758359, upper bound: 2.7773894
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7714661, upper bound: 2.7748483
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7714661, upper bound: 2.7748483
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7749925, upper bound: 2.7760277
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7749925, upper bound: 2.7786613
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7749880, upper bound: 2.7718599
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7749880, upper bound: 2.7718599
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7776216, upper bound: 2.7723755
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.34
Output dim: 0, lower bound: -2.7776216, upper bound: 2.7723755
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7746233, upper bound: 2.7758399
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7772569, upper bound: 2.7763455
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7723755, upper bound: 2.7758359
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7760229, upper bound: 2.7768918
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7723755, upper bound: 2.7776216
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7760229, upper bound: 2.7786775
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7746612, upper bound: 2.7749925
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7772948, upper bound: 2.7754712
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7746612, upper bound: 2.7758361
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7772948, upper bound: 2.7762811
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7786973
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7787504
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7801671
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7801671
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7770552
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7769808
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7778307
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7777063
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7714556, upper bound: 2.7706598
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7714556, upper bound: 2.7706598
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7731543, upper bound: 2.7685290
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7710662, upper bound: 2.7663166
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7731543, upper bound: 2.7685290
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7710662, upper bound: 2.7663166
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7758398
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7751152, upper bound: 2.7763454
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7758398
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7751152, upper bound: 2.7763454
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7786973
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7787504
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7799106
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7799106
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7770552
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7769808
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7778236
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7776769
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7760001
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7742530
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7772661
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7755190
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7735495
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7725126
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7738449
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7728425
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7782382
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7699404
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7793301
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7711735
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7783385, upper bound: 2.7757387
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7752083, upper bound: 2.7698186
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7787334, upper bound: 2.7751537
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7763857
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7698002
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7776517
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7710662
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7745035
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7751405
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7757595
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7786459
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7762219
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7760039
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7689538, upper bound: 2.7570467
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7689836, upper bound: 2.7570490
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7689538, upper bound: 2.7570467
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7689836, upper bound: 2.7570490
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7763899
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7700042
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7790485
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7725818
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7736132
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.34
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7741817
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=3.285133123397827
rel_dist={0: [-2.7804845941221, 2.7804845941221004]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7800292
time: 0.37 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.88 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7800292
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.88
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.4334733, 2.4970913, -0.4869425, 2.6686676, -3.1021409, 2.9840338
1: -0.5019389, 3.4698267, -0.5386804, 3.6743641, -4.1763029, 4.0085073
2: -1.2600799, 2.3818066, -1.3158708, 2.5983098, -3.8583896, 3.6976774
3: -1.0162834, 2.9394152, -1.0903339, 3.2125225, -4.2288060, 4.0297489
4: -1.5453744, 3.1090739, -1.6470305, 3.2839675, -4.8293419, 4.7561045

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772986
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
time: 0.36 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.5095432, 2.7755899, -3.2509441, 3.1871667
1: -0.5395899, 3.6900327, -0.5611423, 3.8165379, -4.3561277, 4.2511749
2: -1.3205540, 2.6000621, -1.3674926, 2.7016737, -4.0222278, 3.9675546
3: -1.0897965, 3.1838365, -1.1338987, 3.3577619, -4.4475584, 4.3177352
4: -1.6415637, 3.3140776, -1.7360522, 3.4053361, -5.0468998, 5.0501299

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522
time: 0.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.26 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772986
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.4334733, 2.4970913, -0.4721184, 2.6139584, -3.0474317, 2.9692097
1: -0.5019389, 3.4698267, -0.5271653, 3.6039305, -4.1058693, 3.9969921
2: -1.2600799, 2.3818066, -1.2901279, 2.5430479, -3.8031278, 3.6719346
3: -1.0162834, 2.9394152, -1.0674360, 3.1326091, -4.1488924, 4.0068512
4: -1.5453744, 3.1090739, -1.6003956, 3.2267430, -4.7721176, 4.7094693

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772708
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772986
time: 0.37 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.4334733, 2.4970913, -0.4760821, 2.6324074, -3.0658808, 2.9731734
1: -0.5019389, 3.4698267, -0.5306048, 3.6395037, -4.1414428, 4.0004315
2: -1.2600799, 2.3818066, -1.3034103, 2.5473106, -3.8073905, 3.6852169
3: -1.0162834, 2.9394152, -1.0741812, 3.1553285, -4.1716118, 4.0135965
4: -1.5453744, 3.1090739, -1.6171616, 3.2466135, -4.7919879, 4.7262354

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
time: 0.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.4334733, 2.4970913, -2.9724455, 3.1110969
1: -0.5395899, 3.6900327, -0.5019389, 3.4698267, -4.0094166, 4.1919718
2: -1.3205540, 2.6000621, -1.2600799, 2.3818066, -3.7023606, 3.8601420
3: -1.0897965, 3.1838365, -1.0162834, 2.9394152, -4.0292120, 4.2001200
4: -1.6415637, 3.3140776, -1.5453744, 3.1090739, -4.7506375, 4.8594522

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7792479
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.4753543, 2.6776235, -3.1529779, 3.1529779
1: -0.5395899, 3.6900327, -0.5395899, 3.6900327, -4.2296228, 4.2296228
2: -1.3205540, 2.6000621, -1.3205540, 2.6000621, -3.9206161, 3.9206161
3: -1.0897965, 3.1838365, -1.0897965, 3.1838365, -4.2736330, 4.2736330
4: -1.6415637, 3.3140776, -1.6415637, 3.3140776, -4.9556413, 4.9556413

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7792479
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522
time: 0.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.25 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772708
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772986
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7792479
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7792479
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.25
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7796522

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4424449, 2.5282881, -2.9300921, 2.8626013
1: -0.4905685, 3.3785930, -0.5094109, 3.4904478, -3.9810162, 3.8880038
2: -1.2406874, 2.2999053, -1.2503626, 2.4504304, -3.6911178, 3.5502679
3: -0.9905678, 2.8093987, -1.0311543, 2.9814494, -3.9720173, 3.8405528
4: -1.4657205, 3.0825684, -1.5212888, 3.1499283, -4.6156487, 4.6038570

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772708
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772708
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4552263, 2.5661674, -2.9412677, 2.7288680
1: -0.4616931, 3.1732640, -0.5173952, 3.5414946, -4.0031877, 3.6906593
2: -1.1646090, 2.1681395, -1.2686813, 2.4914229, -3.6560318, 3.4368207
3: -0.9341383, 2.6221490, -1.0474337, 3.0464823, -3.9806206, 3.6695828
4: -1.3658996, 2.9074883, -1.5565467, 3.1856947, -4.5515943, 4.4640350

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795803, upper bound: 2.7769815
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795803, upper bound: 2.7772986
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4465230, 2.5474181, -2.9492221, 2.8666794
1: -0.4905685, 3.3785930, -0.5129495, 3.5237522, -4.0143209, 3.8915424
2: -1.2406874, 2.2999053, -1.2639635, 2.4545760, -3.6952634, 3.5638688
3: -0.9905678, 2.8093987, -1.0380840, 3.0041614, -3.9947290, 3.8474827
4: -1.4657205, 3.0825684, -1.5380543, 3.1694603, -4.6351810, 4.6206226

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4591975, 2.5851049, -2.9602053, 2.7328393
1: -0.4616931, 3.1732640, -0.5208060, 3.5755916, -4.0372849, 3.6940701
2: -1.1646090, 2.1681395, -1.2820011, 2.4948659, -3.6594748, 3.4501405
3: -0.9341383, 2.6221490, -1.0541071, 3.0689542, -4.0030928, 3.6762562
4: -1.3658996, 2.9074883, -1.5733650, 3.2050641, -4.5709639, 4.4808531

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770663, upper bound: 2.7786200
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770663, upper bound: 2.7786728
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4107991, 2.4194169, -2.8368788, 2.9707642
1: -0.5153302, 3.5426629, -0.4863987, 3.3667943, -3.8821244, 4.0290613
2: -1.2751217, 2.4697607, -1.2244968, 2.3069098, -3.5820315, 3.6942575
3: -1.0362854, 2.9608207, -0.9848766, 2.8255253, -3.8618107, 3.9456973
4: -1.5216012, 3.2471507, -1.4775673, 3.0376947, -4.5592957, 4.7247181

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800292, upper bound: 2.7755707
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786200, upper bound: 2.7770663
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4190712, 2.4496202, -2.8483610, 2.8673940
1: -0.4923947, 3.3845534, -0.4927001, 3.4072039, -3.8995986, 3.8772535
2: -1.2141361, 2.3641450, -1.2390170, 2.3361087, -3.5502448, 3.6031621
3: -0.9919410, 2.8182275, -0.9974864, 2.8689785, -3.8609195, 3.8157139
4: -1.4456424, 3.1076982, -1.5042461, 3.0669675, -4.5126100, 4.6119442

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800292, upper bound: 2.7758126
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786728, upper bound: 2.7771076
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4472826, 2.5931993, -3.0106611, 3.0072477
1: -0.5153302, 3.5426629, -0.5222347, 3.5779586, -4.0932889, 4.0648975
2: -1.2751217, 2.4697607, -1.2803242, 2.5071547, -3.7822764, 3.7500849
3: -1.0362854, 2.9608207, -1.0540519, 3.0400903, -4.0763760, 4.0148726
4: -1.5216012, 3.2471507, -1.5653882, 3.2363043, -4.7579055, 4.8125391

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7755707
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767840, upper bound: 2.7767365
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4592464, 2.6301217, -3.0288625, 2.9075692
1: -0.4923947, 3.3845534, -0.5301664, 3.6276648, -4.1200595, 3.9147198
2: -1.2141361, 2.3641450, -1.2989151, 2.5471928, -3.7613289, 3.6630602
3: -0.9919410, 2.8182275, -1.0703456, 3.1006763, -4.0926170, 3.8885732
4: -1.4456424, 3.1076982, -1.5985987, 3.2716894, -4.7173319, 4.7062969

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7758126
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767840, upper bound: 2.7767840
time: 0.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.32 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772708
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7799791, upper bound: 2.7772708
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7795803, upper bound: 2.7769815
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7795803, upper bound: 2.7772986
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7771076, upper bound: 2.7786728
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7770663, upper bound: 2.7786200
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7770663, upper bound: 2.7786728
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7800292, upper bound: 2.7755707
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7786200, upper bound: 2.7770663
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7800292, upper bound: 2.7758126
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7786728, upper bound: 2.7771076
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7755707
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7767840, upper bound: 2.7767365
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7796522, upper bound: 2.7758126
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.32
Output dim: 0, lower bound: -2.7767840, upper bound: 2.7767840

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.3976916, 2.3701842, -2.7719882, 2.8178482
1: -0.4905685, 3.3785930, -0.4759254, 3.3006201, -3.7911887, 3.8545184
2: -1.2406874, 2.2999053, -1.2000978, 2.2573438, -3.4980311, 3.5000031
3: -0.9905678, 2.8093987, -0.9639099, 2.7569559, -3.7475238, 3.7733085
4: -1.4657205, 3.0825684, -1.4348797, 2.9833052, -4.4490256, 4.5174479

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4339360, 2.5453248, -2.9471288, 2.8540926
1: -0.4905685, 3.3785930, -0.5117657, 3.5136716, -4.0042400, 3.8903587
2: -1.2406874, 2.2999053, -1.2562408, 2.4565895, -3.6972768, 3.5561461
3: -0.9905678, 2.8093987, -1.0333462, 2.9686604, -3.9592280, 3.8427448
4: -1.4657205, 3.0825684, -1.5234973, 3.1840131, -4.6497335, 4.6060658

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4131082, 2.5138421, -2.8889425, 2.6867499
1: -0.4616931, 3.1732640, -0.5049461, 3.4821463, -3.9438393, 3.6782103
2: -1.1646090, 2.1681395, -1.2529514, 2.4245839, -3.5891929, 3.4210908
3: -0.9341383, 2.6221490, -1.0170228, 2.9047923, -3.8389306, 3.6391718
4: -1.3658996, 2.9074883, -1.4890478, 3.1800227, -4.5459223, 4.3965359

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.3933438, 2.3846712, -2.7597716, 2.6669855
1: -0.4616931, 3.1732640, -0.4801708, 3.3002474, -3.7619405, 3.6534348
2: -1.1646090, 2.1681395, -1.1844540, 2.2999473, -3.4645562, 3.3525934
3: -0.9341383, 2.6221490, -0.9702330, 2.7382944, -3.6724327, 3.5923820
4: -1.3658996, 2.9074883, -1.3987203, 3.0223463, -4.3882456, 4.3062086

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7771095
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7771095
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4045002, 2.3892779, -2.7910819, 2.8246567
1: -0.4905685, 3.3785930, -0.4820384, 3.3310199, -3.8215885, 3.8606315
2: -1.2406874, 2.2999053, -1.2148471, 2.2700236, -3.5107110, 3.5147524
3: -0.9905678, 2.8093987, -0.9765917, 2.7838321, -3.7743998, 3.7859902
4: -1.4657205, 3.0825684, -1.4514358, 3.0033937, -4.4691143, 4.5340042

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7786728
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7780272
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4018040, 2.4201565, -0.4363415, 2.5598364, -2.9616404, 2.8564980
1: -0.4905685, 3.3785930, -0.5141369, 3.5384102, -4.0289788, 3.8927298
2: -1.2406874, 2.2999053, -1.2679358, 2.4622645, -3.7029519, 3.5678411
3: -0.9905678, 2.8093987, -1.0378445, 2.9840109, -3.9745789, 3.8472433
4: -1.4657205, 3.0825684, -1.5353248, 3.1980126, -4.6637330, 4.6178932

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7786728
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7780272
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.4214399, 2.5317283, -2.9068286, 2.6950817
1: -0.4616931, 3.1732640, -0.5101770, 3.5111363, -3.9728293, 3.6834412
2: -1.1646090, 2.1681395, -1.2683390, 2.4343724, -3.5989814, 3.4364786
3: -0.9341383, 2.6221490, -1.0279799, 2.9328279, -3.8669662, 3.6501288
4: -1.3658996, 2.9074883, -1.5075796, 3.1978621, -4.5637617, 4.4150677

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786200
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778624
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3751003, 2.2736418, -0.3966825, 2.4039001, -2.7790005, 2.6703243
1: -0.4616931, 3.1732640, -0.4837550, 3.3311136, -3.7928066, 3.6570191
2: -1.1646090, 2.1681395, -1.1977184, 2.3121138, -3.4767227, 3.3658578
3: -0.9341383, 2.6221490, -0.9773662, 2.7677941, -3.7019324, 3.5995152
4: -1.3658996, 2.9074883, -1.4139899, 3.0421400, -4.4080396, 4.3214784

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786200
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778778
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.3976916, 2.3701842, -2.7876461, 2.9576569
1: -0.5153302, 3.5426629, -0.4759254, 3.3006201, -3.8159504, 4.0185881
2: -1.2751217, 2.4697607, -1.2000978, 2.2573438, -3.5324655, 3.6698585
3: -1.0362854, 2.9608207, -0.9639099, 2.7569559, -3.7932413, 3.9247305
4: -1.5216012, 3.2471507, -1.4348797, 2.9833052, -4.5049067, 4.6820302

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7755707
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7755707
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4045002, 2.3892779, -2.8067398, 2.9644654
1: -0.5153302, 3.5426629, -0.4820384, 3.3310199, -3.8463502, 4.0247011
2: -1.2751217, 2.4697607, -1.2148471, 2.2700236, -3.5451453, 3.6846077
3: -1.0362854, 2.9608207, -0.9765917, 2.7838321, -3.8201175, 3.9374123
4: -1.5216012, 3.2471507, -1.4514358, 3.0033937, -4.5249949, 4.6985865

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
time: 0.36 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4052889, 2.3988190, -2.7975597, 2.8536117
1: -0.4923947, 3.3845534, -0.4816256, 3.3393538, -3.8317485, 3.8661790
2: -1.2141361, 2.3641450, -1.2137136, 2.2852054, -3.4993415, 3.5778587
3: -0.9919410, 2.8182275, -0.9754038, 2.7983217, -3.7902627, 3.7936313
4: -1.4456424, 3.1076982, -1.4600146, 3.0111508, -4.4567933, 4.5677128

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7758126
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7758126
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4127932, 2.4226701, -2.8214107, 2.8611159
1: -0.4923947, 3.3845534, -0.4886633, 3.3760819, -3.8684766, 3.8732166
2: -1.2141361, 2.3641450, -1.2310851, 2.3023915, -3.5165277, 3.5952301
3: -0.9919410, 2.8182275, -0.9900044, 2.8328435, -3.8247845, 3.8082318
4: -1.4456424, 3.1076982, -1.4809371, 3.0359046, -4.4815469, 4.5886354

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7771076
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7771076
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4339360, 2.5453739, -2.9628358, 2.9939013
1: -0.5153302, 3.5426629, -0.5117657, 3.5137279, -4.0290580, 4.0544286
2: -1.2751217, 2.4697607, -1.2562408, 2.4566569, -3.7317786, 3.7260015
3: -1.0362854, 2.9608207, -1.0333462, 2.9687085, -4.0049939, 3.9941669
4: -1.5216012, 3.2471507, -1.5234973, 3.1840360, -4.7056370, 4.7706480

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7755707
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7755707
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4363415, 2.5598497, -2.9773116, 2.9963067
1: -0.5153302, 3.5426629, -0.5141369, 3.5384238, -4.0537539, 4.0567999
2: -1.2751217, 2.4697607, -1.2679358, 2.4622810, -3.7374027, 3.7376964
3: -1.0362854, 2.9608207, -1.0378445, 2.9840209, -4.0203066, 3.9986653
4: -1.5216012, 3.2471507, -1.5353248, 3.1980190, -4.7196202, 4.7824755

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7767365
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7767365
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4457266, 2.5817947, -2.9805355, 2.8940494
1: -0.4923947, 3.3845534, -0.5196065, 3.5628102, -4.0552049, 3.9041600
2: -1.2141361, 2.3641450, -1.2744284, 2.4964032, -3.7105393, 3.6385734
3: -0.9919410, 2.8182275, -1.0494717, 3.0287817, -4.0207224, 3.8676991
4: -1.4456424, 3.1076982, -1.5558999, 3.2188327, -4.6644750, 4.6635981

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7758126
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7758126
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4482549, 2.5963612, -2.9951019, 2.8965778
1: -0.4923947, 3.3845534, -0.5219529, 3.5887427, -4.0811377, 3.9065063
2: -1.2141361, 2.3641450, -1.2862134, 2.4982758, -3.7124119, 3.6503584
3: -0.9919410, 2.8182275, -1.0539751, 3.0440583, -4.0359993, 3.8722026
4: -1.4456424, 3.1076982, -1.5680395, 3.2330320, -4.6786742, 4.6757379

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7767840
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7767840
time: 0.45 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.43 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7781350, upper bound: 2.7772708
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7771095
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7771095
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7786728
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7780272
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7786728
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7780272
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786200
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778624
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786200
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778778
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7755707
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7755707
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7758126
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7758126
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7771076
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7772986, upper bound: 2.7771076
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7755707
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7755707
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7767365
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7752668, upper bound: 2.7767365
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7758126
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7758126
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7767840
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7767840

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3976916, 2.3701842, -2.7597907, 2.7682710
1: -0.4801092, 3.3123057, -0.4759254, 3.3006201, -3.7807293, 3.7882311
2: -1.2162256, 2.2505226, -1.2000978, 2.2573438, -3.4735694, 3.4506204
3: -0.9699728, 2.7406662, -0.9639099, 2.7569559, -3.7269287, 3.7045760
4: -1.4223459, 3.0282202, -1.4348797, 2.9833052, -4.4056511, 4.4631000

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7770258
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7772720
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3976916, 2.3701842, -2.7675977, 2.7910781
1: -0.4861144, 3.3476477, -0.4759254, 3.3006201, -3.7867346, 3.8235731
2: -1.2313797, 2.2667561, -1.2000978, 2.2573438, -3.4887235, 3.4668539
3: -0.9823158, 2.7700262, -0.9639099, 2.7569559, -3.7392716, 3.7339361
4: -1.4410114, 3.0509758, -1.4348797, 2.9833052, -4.4243164, 4.4858556

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7770258
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7772720
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4339360, 2.5453248, -2.9349313, 2.8045154
1: -0.4801092, 3.3123057, -0.5117657, 3.5136716, -3.9937809, 3.8240714
2: -1.2162256, 2.2505226, -1.2562408, 2.4565895, -3.6728151, 3.5067635
3: -0.9699728, 2.7406662, -1.0333462, 2.9686604, -3.9386332, 3.7740123
4: -1.4223459, 3.0282202, -1.5234973, 3.1840131, -4.6063590, 4.5517178

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7770245
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7770245
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4339360, 2.5453248, -2.9427381, 2.8273225
1: -0.4861144, 3.3476477, -0.5117657, 3.5136716, -3.9997861, 3.8594134
2: -1.2313797, 2.2667561, -1.2562408, 2.4565895, -3.6879692, 3.5229969
3: -0.9823158, 2.7700262, -1.0333462, 2.9686604, -3.9509761, 3.8033724
4: -1.4410114, 3.0509758, -1.5234973, 3.1840131, -4.6250248, 4.5744734

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7770245
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7772708
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4131082, 2.5138421, -2.8769231, 2.6383991
1: -0.4512913, 3.1082866, -0.5049461, 3.4821463, -3.9334376, 3.6132326
2: -1.1403358, 2.1198447, -1.2529514, 2.4245839, -3.5649197, 3.3727961
3: -0.9133595, 2.5548997, -1.0170228, 2.9047923, -3.8181520, 3.5719225
4: -1.3233099, 2.8539722, -1.4890478, 3.1800227, -4.5033326, 4.3430200

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4131082, 2.5138421, -2.8848033, 2.6634417
1: -0.4575120, 3.1468184, -0.5049461, 3.4821463, -3.9396582, 3.6517644
2: -1.1562662, 2.1376271, -1.2529514, 2.4245839, -3.5808501, 3.3905785
3: -0.9258730, 2.5864434, -1.0170228, 2.9047923, -3.8306653, 3.6034663
4: -1.3418519, 2.8791902, -1.4890478, 3.1800227, -4.5218744, 4.3682380

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3933438, 2.3846712, -2.7477522, 2.6186347
1: -0.4512913, 3.1082866, -0.4801708, 3.3002474, -3.7515388, 3.5884576
2: -1.1403358, 2.1198447, -1.1844540, 2.2999473, -3.4402831, 3.3042986
3: -0.9133595, 2.5548997, -0.9702330, 2.7382944, -3.6516538, 3.5251327
4: -1.3233099, 2.8539722, -1.3987203, 3.0223463, -4.3456564, 4.2526922

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3933438, 2.3846712, -2.7556324, 2.6436772
1: -0.4575120, 3.1468184, -0.4801708, 3.3002474, -3.7577593, 3.6269894
2: -1.1562662, 2.1376271, -1.1844540, 2.2999473, -3.4562135, 3.3220811
3: -0.9258730, 2.5864434, -0.9702330, 2.7382944, -3.6641674, 3.5566764
4: -1.3418519, 2.8791902, -1.3987203, 3.0223463, -4.3641982, 4.2779102

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4045002, 2.3892779, -2.7788844, 2.7750795
1: -0.4801092, 3.3123057, -0.4820384, 3.3310199, -3.8111291, 3.7943442
2: -1.2162256, 2.2505226, -1.2148471, 2.2700236, -3.4862492, 3.4653697
3: -0.9699728, 2.7406662, -0.9765917, 2.7838321, -3.7538049, 3.7172580
4: -1.4223459, 3.0282202, -1.4514358, 3.0033937, -4.4257393, 4.4796562

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7791135
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7791135
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4045002, 2.3892779, -2.7866912, 2.7978866
1: -0.4861144, 3.3476477, -0.4820384, 3.3310199, -3.8171344, 3.8296862
2: -1.2313797, 2.2667561, -1.2148471, 2.2700236, -3.5014033, 3.4816031
3: -0.9823158, 2.7700262, -0.9765917, 2.7838321, -3.7661479, 3.7466178
4: -1.4410114, 3.0509758, -1.4514358, 3.0033937, -4.4444051, 4.5024118

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7780688
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7780688
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4363415, 2.5598364, -2.9494429, 2.8069208
1: -0.4801092, 3.3123057, -0.5141369, 3.5384102, -4.0185194, 3.8264425
2: -1.2162256, 2.2505226, -1.2679358, 2.4622645, -3.6784902, 3.5184584
3: -0.9699728, 2.7406662, -1.0378445, 2.9840109, -3.9539838, 3.7785106
4: -1.4223459, 3.0282202, -1.5353248, 3.1980126, -4.6203585, 4.5635452

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7786504
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7786728
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4363415, 2.5598364, -2.9572496, 2.8297279
1: -0.4861144, 3.3476477, -0.5141369, 3.5384102, -4.0245247, 3.8617845
2: -1.2313797, 2.2667561, -1.2679358, 2.4622645, -3.6936443, 3.5346918
3: -0.9823158, 2.7700262, -1.0378445, 2.9840109, -3.9663267, 3.8078709
4: -1.4410114, 3.0509758, -1.5353248, 3.1980126, -4.6390238, 4.5863008

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7779043
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7780272
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4214399, 2.5317283, -2.8948092, 2.6467307
1: -0.4512913, 3.1082866, -0.5101770, 3.5111363, -3.9624276, 3.6184635
2: -1.1403358, 2.1198447, -1.2683390, 2.4343724, -3.5747082, 3.3881836
3: -0.9133595, 2.5548997, -1.0279799, 2.9328279, -3.8461876, 3.5828795
4: -1.3233099, 2.8539722, -1.5075796, 3.1978621, -4.5211720, 4.3615518

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786199
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786200
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4214399, 2.5317283, -2.9026895, 2.6717732
1: -0.4575120, 3.1468184, -0.5101770, 3.5111363, -3.9686482, 3.6569953
2: -1.1562662, 2.1376271, -1.2683390, 2.4343724, -3.5906386, 3.4059663
3: -0.9258730, 2.5864434, -1.0279799, 2.9328279, -3.8587010, 3.6144233
4: -1.3418519, 2.8791902, -1.5075796, 3.1978621, -4.5397139, 4.3867698

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778624
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778624
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3966825, 2.4039001, -2.7669811, 2.6219733
1: -0.4512913, 3.1082866, -0.4837550, 3.3311136, -3.7824049, 3.5920415
2: -1.1403358, 2.1198447, -1.1977184, 2.3121138, -3.4524496, 3.3175631
3: -0.9133595, 2.5548997, -0.9773662, 2.7677941, -3.6811538, 3.5322659
4: -1.3233099, 2.8539722, -1.4139899, 3.0421400, -4.3654499, 4.2679620

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7786200
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7786200
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3966825, 2.4039001, -2.7748613, 2.6470158
1: -0.4575120, 3.1468184, -0.4837550, 3.3311136, -3.7886255, 3.6305733
2: -1.1562662, 2.1376271, -1.1977184, 2.3121138, -3.4683800, 3.3353455
3: -0.9258730, 2.5864434, -0.9773662, 2.7677941, -3.6936672, 3.5638096
4: -1.3418519, 2.8791902, -1.4139899, 3.0421400, -4.3839922, 4.2931800

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7778778
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7778778
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3976916, 2.3701842, -2.7758038, 2.9117126
1: -0.5055367, 3.4809954, -0.4759254, 3.3006201, -3.8061566, 3.9569209
2: -1.2521493, 2.4233136, -1.2000978, 2.2573438, -3.5094931, 3.6234114
3: -1.0172695, 2.8950694, -0.9639099, 2.7569559, -3.7742252, 3.8589792
4: -1.4811087, 3.1972914, -1.4348797, 2.9833052, -4.4644136, 4.6321712

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7752192
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7755707
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3976916, 2.3701842, -2.7821336, 2.9228611
1: -0.5089593, 3.5015557, -0.4759254, 3.3006201, -3.8095794, 3.9774811
2: -1.2644246, 2.4245837, -1.2000978, 2.2573438, -3.5217683, 3.6246815
3: -1.0243788, 2.9105000, -0.9639099, 2.7569559, -3.7813346, 3.8744099
4: -1.4927979, 3.2066495, -1.4348797, 2.9833052, -4.4761028, 4.6415291

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7752192
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7752192
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4045002, 2.3892779, -2.7948976, 2.9185212
1: -0.5055367, 3.4809954, -0.4820384, 3.3310199, -3.8365564, 3.9630339
2: -1.2521493, 2.4233136, -1.2148471, 2.2700236, -3.5221729, 3.6381607
3: -1.0172695, 2.8950694, -0.9765917, 2.7838321, -3.8011017, 3.8716612
4: -1.4811087, 3.1972914, -1.4514358, 3.0033937, -4.4845023, 4.6487274

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4045002, 2.3892779, -2.8012276, 2.9296696
1: -0.5089593, 3.5015557, -0.4820384, 3.3310199, -3.8399792, 3.9835942
2: -1.2644246, 2.4245837, -1.2148471, 2.2700236, -3.5344481, 3.6394308
3: -1.0243788, 2.9105000, -0.9765917, 2.7838321, -3.8082108, 3.8870916
4: -1.4927979, 3.2066495, -1.4514358, 3.0033937, -4.4961915, 4.6580853

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7758676
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7758937
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4052889, 2.3988190, -2.7845626, 2.8071718
1: -0.4820945, 3.3220072, -0.4816256, 3.3393538, -3.8214483, 3.8036327
2: -1.1904538, 2.3187947, -1.2137136, 2.2852054, -3.4756591, 3.5325084
3: -0.9715070, 2.7528343, -0.9754038, 2.7983217, -3.7698288, 3.7282381
4: -1.4044091, 3.0569649, -1.4600146, 3.0111508, -4.4155598, 4.5169792

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7752668
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7756912
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4052889, 2.3988190, -2.7868524, 2.8235295
1: -0.4848027, 3.3492410, -0.4816256, 3.3393538, -3.8241565, 3.8308666
2: -1.2020583, 2.3271937, -1.2137136, 2.2852054, -3.4872637, 3.5409074
3: -0.9766790, 2.7750435, -0.9754038, 2.7983217, -3.7750006, 3.7504473
4: -1.4156674, 3.0728226, -1.4600146, 3.0111508, -4.4268184, 4.5328369

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7752668
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7756912
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4127932, 2.4226701, -2.8084137, 2.8146760
1: -0.4820945, 3.3220072, -0.4886633, 3.3760819, -3.8581765, 3.8106704
2: -1.1904538, 2.3187947, -1.2310851, 2.3023915, -3.4928453, 3.5498798
3: -0.9715070, 2.7528343, -0.9900044, 2.8328435, -3.8043504, 3.7428389
4: -1.4044091, 3.0569649, -1.4809371, 3.0359046, -4.4403138, 4.5379019

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7771076
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7771074
time: 0.41 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4127932, 2.4226701, -2.8107035, 2.8310337
1: -0.4848027, 3.3492410, -0.4886633, 3.3760819, -3.8608847, 3.8379045
2: -1.2020583, 2.3271937, -1.2310851, 2.3023915, -3.5044498, 3.5582788
3: -0.9766790, 2.7750435, -0.9900044, 2.8328435, -3.8095226, 3.7650480
4: -1.4156674, 3.0728226, -1.4809371, 3.0359046, -4.4515719, 4.5537596

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7759331
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7759331
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4339360, 2.5453739, -2.9509935, 2.9479570
1: -0.5055367, 3.4809954, -0.5117657, 3.5137279, -4.0192647, 3.9927611
2: -1.2521493, 2.4233136, -1.2562408, 2.4566569, -3.7088063, 3.6795545
3: -1.0172695, 2.8950694, -1.0333462, 2.9687085, -3.9859781, 3.9284155
4: -1.4811087, 3.1972914, -1.5234973, 3.1840360, -4.6651449, 4.7207890

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4339360, 2.5453739, -2.9573236, 2.9591055
1: -0.5089593, 3.5015557, -0.5117657, 3.5137279, -4.0226870, 4.0133214
2: -1.2644246, 2.4245837, -1.2562408, 2.4566569, -3.7210815, 3.6808245
3: -1.0243788, 2.9105000, -1.0333462, 2.9687085, -3.9930873, 3.9438462
4: -1.4927979, 3.2066495, -1.5234973, 3.1840360, -4.6768341, 4.7301469

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4363415, 2.5598497, -2.9654694, 2.9503624
1: -0.5055367, 3.4809954, -0.5141369, 3.5384238, -4.0439606, 3.9951322
2: -1.2521493, 2.4233136, -1.2679358, 2.4622810, -3.7144303, 3.6912494
3: -1.0172695, 2.8950694, -1.0378445, 2.9840209, -4.0012903, 3.9329138
4: -1.4811087, 3.1972914, -1.5353248, 3.1980190, -4.6791277, 4.7326164

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7767365
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7767365
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4363415, 2.5598497, -2.9717994, 2.9615109
1: -0.5089593, 3.5015557, -0.5141369, 3.5384238, -4.0473833, 4.0156927
2: -1.2644246, 2.4245837, -1.2679358, 2.4622810, -3.7267056, 3.6925194
3: -1.0243788, 2.9105000, -1.0378445, 2.9840209, -4.0084000, 3.9483447
4: -1.4927979, 3.2066495, -1.5353248, 3.1980190, -4.6908169, 4.7419744

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7758676
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7758676
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4457266, 2.5817947, -2.9675384, 2.8476095
1: -0.4820945, 3.3220072, -0.5196065, 3.5628102, -4.0449047, 3.8416138
2: -1.1904538, 2.3187947, -1.2744284, 2.4964032, -3.6868570, 3.5932231
3: -0.9715070, 2.7528343, -1.0494717, 3.0287817, -4.0002885, 3.8023062
4: -1.4044091, 3.0569649, -1.5558999, 3.2188327, -4.6232419, 4.6128645

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7752668
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7756912
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4457266, 2.5817947, -2.9698281, 2.8639672
1: -0.4848027, 3.3492410, -0.5196065, 3.5628102, -4.0476127, 3.8688476
2: -1.2020583, 2.3271937, -1.2744284, 2.4964032, -3.6984615, 3.6016221
3: -0.9766790, 2.7750435, -1.0494717, 3.0287817, -4.0054607, 3.8245153
4: -1.4156674, 3.0728226, -1.5558999, 3.2188327, -4.6345000, 4.6287222

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7752668
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7756912
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4482549, 2.5963612, -2.9821048, 2.8501377
1: -0.4820945, 3.3220072, -0.5219529, 3.5887427, -4.0708370, 3.8439600
2: -1.1904538, 2.3187947, -1.2862134, 2.4982758, -3.6887295, 3.6050081
3: -0.9715070, 2.7528343, -1.0539751, 3.0440583, -4.0155654, 3.8068094
4: -1.4044091, 3.0569649, -1.5680395, 3.2330320, -4.6374412, 4.6250043

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7767840
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7767840
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4482549, 2.5963612, -2.9843946, 2.8664956
1: -0.4848027, 3.3492410, -0.5219529, 3.5887427, -4.0735455, 3.8711939
2: -1.2020583, 2.3271937, -1.2862134, 2.4982758, -3.7003341, 3.6134071
3: -0.9766790, 2.7750435, -1.0539751, 3.0440583, -4.0207372, 3.8290186
4: -1.4156674, 3.0728226, -1.5680395, 3.2330320, -4.6486993, 4.6408620

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7759331
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7759331
time: 0.42 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.48 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7770258
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7772720
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7770258
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7772720
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7770245
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7770245
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7770245
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7772708
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7779655, upper bound: 2.7769815
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786688, upper bound: 2.7771095
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7791135
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7791135
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7780688
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769864, upper bound: 2.7780688
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7786504
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7786728
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7779043
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7780272
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786199
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7786200
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778624
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7755707, upper bound: 2.7778624
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7786200
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7786200
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7778778
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7758126, upper bound: 2.7778778
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7752192
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7755707
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7752192
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7784081, upper bound: 2.7752192
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7770663
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7758676
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7769815, upper bound: 2.7758937
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7752668
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7756912
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7752668
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7786161, upper bound: 2.7756912
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7771076
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7771074
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7759331
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7772708, upper bound: 2.7759331
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7777306, upper bound: 2.7752192
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7767365
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7767365
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7758676
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7752192, upper bound: 2.7758676
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7752668
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7756912
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7752668
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7780532, upper bound: 2.7756912
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7767840
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7767840
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7759331
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -2.7756392, upper bound: 2.7759331

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3896064, 2.3704360, -2.7600424, 2.7601857
1: -0.4801092, 3.3123057, -0.4801092, 3.3121889, -3.7922981, 3.7924149
2: -1.2162256, 2.2505226, -1.2162256, 2.2503803, -3.4666059, 3.4667482
3: -0.9699728, 2.7406662, -0.9699728, 2.7404900, -3.7104628, 3.7106390
4: -1.4223459, 3.0282202, -1.4223459, 3.0280967, -4.4504423, 4.4505663

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7733165, upper bound: 2.7754665
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766597, upper bound: 2.7766597
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3630810, 2.2252908, -2.6148973, 2.7336602
1: -0.4801092, 3.3123057, -0.4512913, 3.1082866, -3.5883958, 3.7635970
2: -1.2162256, 2.2505226, -1.1403358, 2.1198447, -3.3360703, 3.3908584
3: -0.9699728, 2.7406662, -0.9133595, 2.5548997, -3.5248725, 3.6540256
4: -1.4223459, 3.0282202, -1.3233099, 2.8539722, -4.2763181, 4.3515301

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7733165, upper bound: 2.7755704
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766597, upper bound: 2.7767666
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3896064, 2.3704360, -2.7678494, 2.7829928
1: -0.4861144, 3.3476477, -0.4801092, 3.3121889, -3.7983034, 3.8277569
2: -1.2313797, 2.2667561, -1.2162256, 2.2503803, -3.4817600, 3.4829817
3: -0.9823158, 2.7700262, -0.9699728, 2.7404900, -3.7228057, 3.7399991
4: -1.4410114, 3.0509758, -1.4223459, 3.0280967, -4.4691081, 4.4733219

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.07 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760277, upper bound: 2.7745997
time: 0.45 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785562, upper bound: 2.7751177
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3630810, 2.2252908, -2.6227040, 2.7564673
1: -0.4861144, 3.3476477, -0.4512913, 3.1082866, -3.5944011, 3.7989390
2: -1.2313797, 2.2667561, -1.1403358, 2.1198447, -3.3512244, 3.4070919
3: -0.9823158, 2.7700262, -0.9133595, 2.5548997, -3.5372155, 3.6833858
4: -1.4410114, 3.0509758, -1.3233099, 2.8539722, -4.2949839, 4.3742857

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760277, upper bound: 2.7748090
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7785562, upper bound: 2.7753655
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4056197, 2.5140209, -2.9036274, 2.7761989
1: -0.4801092, 3.3123057, -0.5055367, 3.4809954, -3.9611046, 3.8178425
2: -1.2162256, 2.2505226, -1.2521493, 2.4233136, -3.6395392, 3.5026720
3: -0.9699728, 2.7406662, -1.0172695, 2.8950694, -3.8650422, 3.7579355
4: -1.4223459, 3.0282202, -1.4811087, 3.1972914, -4.6196375, 4.5093288

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7694452, upper bound: 2.7768467
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.68 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724698, upper bound: 2.7754665
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758426, upper bound: 2.7766597
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3857437, 2.4018829, -2.7914894, 2.7563229
1: -0.4801092, 3.3123057, -0.4820945, 3.3220072, -3.8021164, 3.7944002
2: -1.2162256, 2.2505226, -1.1904538, 2.3187947, -3.5350204, 3.4409764
3: -0.9699728, 2.7406662, -0.9715070, 2.7528343, -3.7228072, 3.7121730
4: -1.4223459, 3.0282202, -1.4044091, 3.0569649, -4.4793110, 4.4326291

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7694452, upper bound: 2.7785518
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.52 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724698, upper bound: 2.7755704
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758426, upper bound: 2.7767666
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4056197, 2.5140209, -2.9114342, 2.7990060
1: -0.4861144, 3.3476477, -0.5055367, 3.4809954, -3.9671099, 3.8531842
2: -1.2313797, 2.2667561, -1.2521493, 2.4233136, -3.6546934, 3.5189054
3: -0.9823158, 2.7700262, -1.0172695, 2.8950694, -3.8773851, 3.7872958
4: -1.4410114, 3.0509758, -1.4811087, 3.1972914, -4.6383028, 4.5320845

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711050, upper bound: 2.7746169
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.47 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751022, upper bound: 2.7745997
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776515, upper bound: 2.7751177
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3857437, 2.4018829, -2.7992964, 2.7791300
1: -0.4861144, 3.3476477, -0.4820945, 3.3220072, -3.8081217, 3.8297422
2: -1.2313797, 2.2667561, -1.1904538, 2.3187947, -3.5501745, 3.4572098
3: -0.9823158, 2.7700262, -0.9715070, 2.7528343, -3.7351501, 3.7415333
4: -1.4410114, 3.0509758, -1.4044091, 3.0569649, -4.4979763, 4.4553847

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711050, upper bound: 2.7746169
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.45 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724698, upper bound: 2.7748087
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776515, upper bound: 2.7751177
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3896064, 2.3705792, -2.7336602, 2.6148973
1: -0.4512913, 3.1082866, -0.4801092, 3.3123057, -3.7635970, 3.5883958
2: -1.1403358, 2.1198447, -1.2162256, 2.2505226, -3.3908584, 3.3360703
3: -0.9133595, 2.5548997, -0.9699728, 2.7406662, -3.6540256, 3.5248725
4: -1.3233099, 2.8539722, -1.4223459, 3.0282202, -4.3515301, 4.2763181

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7680553, upper bound: 2.7755769
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.44 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727044, upper bound: 2.7754684
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760429, upper bound: 2.7766597
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4056197, 2.5140209, -2.8771019, 2.6309104
1: -0.4512913, 3.1082866, -0.5055367, 3.4809954, -3.9322867, 3.6138234
2: -1.1403358, 2.1198447, -1.2521493, 2.4233136, -3.5636494, 3.3719940
3: -0.9133595, 2.5548997, -1.0172695, 2.8950694, -3.8084288, 3.5721693
4: -1.3233099, 2.8539722, -1.4811087, 3.1972914, -4.5206013, 4.3350811

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7680553, upper bound: 2.7759193
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.48 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727044, upper bound: 2.7754684
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760429, upper bound: 2.7766597
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3896064, 2.3705792, -2.7415404, 2.6399398
1: -0.4575120, 3.1468184, -0.4801092, 3.3123057, -3.7698176, 3.6269276
2: -1.1562662, 2.1376271, -1.2162256, 2.2505226, -3.4067888, 3.3538527
3: -0.9258730, 2.5864434, -0.9699728, 2.7406662, -3.6665392, 3.5564163
4: -1.3418519, 2.8791902, -1.4223459, 3.0282202, -4.3700724, 4.3015361

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750976, upper bound: 2.7745545
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776419, upper bound: 2.7749588
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4056197, 2.5140209, -2.8849821, 2.6559529
1: -0.4575120, 3.1468184, -0.5055367, 3.4809954, -3.9385073, 3.6523552
2: -1.1562662, 2.1376271, -1.2521493, 2.4233136, -3.5795798, 3.3897765
3: -0.9258730, 2.5864434, -1.0172695, 2.8950694, -3.8209424, 3.6037130
4: -1.3418519, 2.8791902, -1.4811087, 3.1972914, -4.5391436, 4.3602991

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.09 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727044, upper bound: 2.7745545
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776419, upper bound: 2.7749588
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3630810, 2.2252908, -2.5883718, 2.5883718
1: -0.4512913, 3.1082866, -0.4512913, 3.1082866, -3.5595779, 3.5595779
2: -1.1403358, 2.1198447, -1.1403358, 2.1198447, -3.2601805, 3.2601805
3: -0.9133595, 2.5548997, -0.9133595, 2.5548997, -3.4682593, 3.4682593
4: -1.3233099, 2.8539722, -1.3233099, 2.8539722, -4.1772823, 4.1772823

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756702, upper bound: 2.7786973
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786228, upper bound: 2.7787504
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3857437, 2.4018829, -2.7649639, 2.6110344
1: -0.4512913, 3.1082866, -0.4820945, 3.3220072, -3.7732985, 3.5903811
2: -1.1403358, 2.1198447, -1.1904538, 2.3187947, -3.4591305, 3.3102984
3: -0.9133595, 2.5548997, -0.9715070, 2.7528343, -3.6661940, 3.5264068
4: -1.3233099, 2.8539722, -1.4044091, 3.0569649, -4.3802748, 4.2583814

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756702, upper bound: 2.7786973
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786228, upper bound: 2.7787504
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3630810, 2.2252908, -2.5962520, 2.6134143
1: -0.4575120, 3.1468184, -0.4512913, 3.1082866, -3.5657985, 3.5981097
2: -1.1562662, 2.1376271, -1.1403358, 2.1198447, -3.2761109, 3.2779629
3: -0.9258730, 2.5864434, -0.9133595, 2.5548997, -3.4807727, 3.4998031
4: -1.3418519, 2.8791902, -1.3233099, 2.8539722, -4.1958241, 4.2025003

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768823, upper bound: 2.7770148
time: 0.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796031, upper bound: 2.7769808
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3857437, 2.4018829, -2.7728441, 2.6360769
1: -0.4575120, 3.1468184, -0.4820945, 3.3220072, -3.7795191, 3.6289129
2: -1.1562662, 2.1376271, -1.1904538, 2.3187947, -3.4750609, 3.3280809
3: -0.9258730, 2.5864434, -0.9715070, 2.7528343, -3.6787074, 3.5579505
4: -1.3418519, 2.8791902, -1.4044091, 3.0569649, -4.3988171, 4.2835994

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768822, upper bound: 2.7770148
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796031, upper bound: 2.7769808
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3974134, 2.3932111, -2.7828176, 2.7679925
1: -0.4801092, 3.3123057, -0.4861144, 3.3475065, -3.8276157, 3.7984202
2: -1.2162256, 2.2505226, -1.2313797, 2.2665799, -3.4828055, 3.4819024
3: -0.9699728, 2.7406662, -0.9823158, 2.7698097, -3.7397826, 3.7229819
4: -1.4223459, 3.0282202, -1.4410114, 3.0508230, -4.4731688, 4.4692316

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.08 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7718694, upper bound: 2.7773767
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749641, upper bound: 2.7785562
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3709612, 2.2503333, -2.6399398, 2.7415404
1: -0.4801092, 3.3123057, -0.4575120, 3.1468184, -3.6269276, 3.7698176
2: -1.2162256, 2.2505226, -1.1562662, 2.1376271, -3.3538527, 3.4067888
3: -0.9699728, 2.7406662, -0.9258730, 2.5864434, -3.5564163, 3.6665392
4: -1.4223459, 3.0282202, -1.3418519, 2.8791902, -4.3015361, 4.3700724

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7718694, upper bound: 2.7773767
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749641, upper bound: 2.7785562
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3974134, 2.3932111, -2.7906246, 2.7907996
1: -0.4861144, 3.3476477, -0.4861144, 3.3475065, -3.8336210, 3.8337622
2: -1.2313797, 2.2667561, -1.2313797, 2.2665799, -3.4979596, 3.4981358
3: -0.9823158, 2.7700262, -0.9823158, 2.7698097, -3.7521255, 3.7523420
4: -1.4410114, 3.0509758, -1.4410114, 3.0508230, -4.4918346, 4.4919872

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.09 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746233, upper bound: 2.7756820
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771256, upper bound: 2.7761858
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3709612, 2.2503333, -2.6477466, 2.7643476
1: -0.4861144, 3.3476477, -0.4575120, 3.1468184, -3.6329329, 3.8051596
2: -1.2313797, 2.2667561, -1.1562662, 2.1376271, -3.3690069, 3.4230223
3: -0.9823158, 2.7700262, -0.9258730, 2.5864434, -3.5687592, 3.6958992
4: -1.4410114, 3.0509758, -1.3418519, 2.8791902, -4.3202019, 4.3928280

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746233, upper bound: 2.7756841
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771256, upper bound: 2.7761873
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.4119495, 2.5251694, -2.9147758, 2.7825289
1: -0.4801092, 3.3123057, -0.5089593, 3.5015557, -3.9816649, 3.8212650
2: -1.2162256, 2.2505226, -1.2644246, 2.4245837, -3.6408093, 3.5149472
3: -0.9699728, 2.7406662, -1.0243788, 2.9105000, -3.8804729, 3.7650449
4: -1.4223459, 3.0282202, -1.4927979, 3.2066495, -4.6289954, 4.5210180

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693198, upper bound: 2.7765415
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.64 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698031, upper bound: 2.7770773
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732200, upper bound: 2.7782713
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3705792, -0.3880334, 2.4182405, -2.8078470, 2.7586126
1: -0.4801092, 3.3123057, -0.4848027, 3.3492410, -3.8293502, 3.7971084
2: -1.2162256, 2.2505226, -1.2020583, 2.3271937, -3.5434194, 3.4525809
3: -0.9699728, 2.7406662, -0.9766790, 2.7750435, -3.7450163, 3.7173452
4: -1.4223459, 3.0282202, -1.4156674, 3.0728226, -4.4951687, 4.4438877

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693198, upper bound: 2.7797651
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.63 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698031, upper bound: 2.7770773
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732200, upper bound: 2.7782713
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.4119495, 2.5251694, -2.9225826, 2.8053360
1: -0.4861144, 3.3476477, -0.5089593, 3.5015557, -3.9876702, 3.8566070
2: -1.2313797, 2.2667561, -1.2644246, 2.4245837, -3.6559634, 3.5311806
3: -0.9823158, 2.7700262, -1.0243788, 2.9105000, -3.8928158, 3.7944050
4: -1.4410114, 3.0509758, -1.4927979, 3.2066495, -4.6476612, 4.5437737

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7709895, upper bound: 2.7743048
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.58 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7755248
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750055, upper bound: 2.7760484
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3933864, -0.3880334, 2.4182405, -2.8156538, 2.7814198
1: -0.4861144, 3.3476477, -0.4848027, 3.3492410, -3.8353555, 3.8324504
2: -1.2313797, 2.2667561, -1.2020583, 2.3271937, -3.5585735, 3.4688144
3: -0.9823158, 2.7700262, -0.9766790, 2.7750435, -3.7573593, 3.7467051
4: -1.4410114, 3.0509758, -1.4156674, 3.0728226, -4.5138340, 4.4666433

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7709895, upper bound: 2.7779086
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.61 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7756524
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750055, upper bound: 2.7761310
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3974134, 2.3933864, -2.7564673, 2.6227040
1: -0.4512913, 3.1082866, -0.4861144, 3.3476477, -3.7989390, 3.5944011
2: -1.1403358, 2.1198447, -1.2313797, 2.2667561, -3.4070919, 3.3512244
3: -0.9133595, 2.5548997, -0.9823158, 2.7700262, -3.6833858, 3.5372155
4: -1.3233099, 2.8539722, -1.4410114, 3.0509758, -4.3742857, 4.2949839

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7668273, upper bound: 2.7759499
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.49 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7701243, upper bound: 2.7770792
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735719, upper bound: 2.7782713
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4119495, 2.5251694, -2.8882504, 2.6372404
1: -0.4512913, 3.1082866, -0.5089593, 3.5015557, -3.9528470, 3.6172459
2: -1.1403358, 2.1198447, -1.2644246, 2.4245837, -3.5649195, 3.3842692
3: -0.9133595, 2.5548997, -1.0243788, 2.9105000, -3.8238597, 3.5792785
4: -1.3233099, 2.8539722, -1.4927979, 3.2066495, -4.5299597, 4.3467703

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7668273, upper bound: 2.7785484
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.50 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7701243, upper bound: 2.7770792
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735719, upper bound: 2.7782713
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3974134, 2.3933864, -2.7643476, 2.6477466
1: -0.4575120, 3.1468184, -0.4861144, 3.3476477, -3.8051596, 3.6329329
2: -1.1562662, 2.1376271, -1.2313797, 2.2667561, -3.4230223, 3.3690069
3: -0.9258730, 2.5864434, -0.9823158, 2.7700262, -3.6958992, 3.5687592
4: -1.3418519, 2.8791902, -1.4410114, 3.0509758, -4.3928280, 4.3202019

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.21 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7754936
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750055, upper bound: 2.7758944
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4119495, 2.5251694, -2.8961306, 2.6622829
1: -0.4575120, 3.1468184, -0.5089593, 3.5015557, -3.9590676, 3.6557777
2: -1.1562662, 2.1376271, -1.2644246, 2.4245837, -3.5808499, 3.4020517
3: -0.9258730, 2.5864434, -1.0243788, 2.9105000, -3.8363731, 3.6108222
4: -1.3418519, 2.8791902, -1.4927979, 3.2066495, -4.5485015, 4.3719883

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.18 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724816, upper bound: 2.7754936
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750055, upper bound: 2.7758945
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3709612, 2.2503333, -2.6134143, 2.5962520
1: -0.4512913, 3.1082866, -0.4575120, 3.1468184, -3.5981097, 3.5657985
2: -1.1403358, 2.1198447, -1.1562662, 2.1376271, -3.2779629, 3.2761109
3: -0.9133595, 2.5548997, -0.9258730, 2.5864434, -3.4998031, 3.4807727
4: -1.3233099, 2.8539722, -1.3418519, 2.8791902, -4.2025003, 4.1958241

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7798990
time: 0.48 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757757, upper bound: 2.7798990
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3880334, 2.4182405, -2.7813215, 2.6133242
1: -0.4512913, 3.1082866, -0.4848027, 3.3492410, -3.8005323, 3.5930893
2: -1.1403358, 2.1198447, -1.2020583, 2.3271937, -3.4675295, 3.3219030
3: -0.9133595, 2.5548997, -0.9766790, 2.7750435, -3.6884031, 3.5315785
4: -1.3233099, 2.8539722, -1.4156674, 3.0728226, -4.3961325, 4.2696395

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7798990
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757757, upper bound: 2.7798990
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3709612, 2.2503333, -2.6212945, 2.6212945
1: -0.4575120, 3.1468184, -0.4575120, 3.1468184, -3.6043303, 3.6043303
2: -1.1562662, 2.1376271, -1.1562662, 2.1376271, -3.2938933, 3.2938933
3: -0.9258730, 2.5864434, -0.9258730, 2.5864434, -3.5123165, 3.5123165
4: -1.3418519, 2.8791902, -1.3418519, 2.8791902, -4.2210422, 4.2210422

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7777758
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7776757
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3880334, 2.4182405, -2.7892017, 2.6383667
1: -0.4575120, 3.1468184, -0.4848027, 3.3492410, -3.8067529, 3.6316211
2: -1.1562662, 2.1376271, -1.2020583, 2.3271937, -3.4834599, 3.3396854
3: -0.9258730, 2.5864434, -0.9766790, 2.7750435, -3.7009165, 3.5631223
4: -1.3418519, 2.8791902, -1.4156674, 3.0728226, -4.4146748, 4.2948575

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7777758
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7776757
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3896064, 2.3704360, -2.7760556, 2.9036274
1: -0.5055367, 3.4809954, -0.4801092, 3.3121889, -3.8177257, 3.9611046
2: -1.2521493, 2.4233136, -1.2162256, 2.2503803, -3.5025296, 3.6395392
3: -1.0172695, 2.8950694, -0.9699728, 2.7404900, -3.7577596, 3.8650422
4: -1.4811087, 3.1972914, -1.4223459, 3.0280967, -4.5092053, 4.6196375

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.15 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7730905, upper bound: 2.7744728
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766597, upper bound: 2.7758426
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3630810, 2.2252908, -2.6309104, 2.8771019
1: -0.5055367, 3.4809954, -0.4512913, 3.1082866, -3.6138234, 3.9322867
2: -1.2521493, 2.4233136, -1.1403358, 2.1198447, -3.3719940, 3.5636494
3: -1.0172695, 2.8950694, -0.9133595, 2.5548997, -3.5721693, 3.8084288
4: -1.4811087, 3.1972914, -1.3233099, 2.8539722, -4.3350811, 4.5206013

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7730905, upper bound: 2.7746794
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766597, upper bound: 2.7760429
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3896064, 2.3704360, -2.7823853, 2.9147758
1: -0.5089593, 3.5015557, -0.4801092, 3.3121889, -3.8211482, 3.9816649
2: -1.2644246, 2.4245837, -1.2162256, 2.2503803, -3.5148048, 3.6408093
3: -1.0243788, 2.9105000, -0.9699728, 2.7404900, -3.7648687, 3.8804729
4: -1.4927979, 3.2066495, -1.4223459, 3.0280967, -4.5208945, 4.6289954

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.14 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753627, upper bound: 2.7726165
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782713, upper bound: 2.7732200
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3630810, 2.2252908, -2.6372404, 2.8882504
1: -0.5089593, 3.5015557, -0.4512913, 3.1082866, -3.6172459, 3.9528470
2: -1.2644246, 2.4245837, -1.1403358, 2.1198447, -3.3842692, 3.5649195
3: -1.0243788, 2.9105000, -0.9133595, 2.5548997, -3.5792785, 3.8238597
4: -1.4927979, 3.2066495, -1.3233099, 2.8539722, -4.3467703, 4.5299597

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.13 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753627, upper bound: 2.7729046
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782713, upper bound: 2.7735719
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3974134, 2.3932111, -2.7988307, 2.9114342
1: -0.5055367, 3.4809954, -0.4861144, 3.3475065, -3.8530431, 3.9671099
2: -1.2521493, 2.4233136, -1.2313797, 2.2665799, -3.5187292, 3.6546934
3: -1.0172695, 2.8950694, -0.9823158, 2.7698097, -3.7870793, 3.8773851
4: -1.4811087, 3.1972914, -1.4410114, 3.0508230, -4.5319319, 4.6383028

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.13 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7713940, upper bound: 2.7762830
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749588, upper bound: 2.7776420
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3709612, 2.2503333, -2.6559529, 2.8849821
1: -0.5055367, 3.4809954, -0.4575120, 3.1468184, -3.6523552, 3.9385073
2: -1.2521493, 2.4233136, -1.1562662, 2.1376271, -3.3897765, 3.5795798
3: -1.0172695, 2.8950694, -0.9258730, 2.5864434, -3.6037130, 3.8209424
4: -1.4811087, 3.1972914, -1.3418519, 2.8791902, -4.3602991, 4.5391436

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.11 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7713940, upper bound: 2.7762830
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749588, upper bound: 2.7776420
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3974134, 2.3932111, -2.8051605, 2.9225826
1: -0.5089593, 3.5015557, -0.4861144, 3.3475065, -3.8564658, 3.9876702
2: -1.2644246, 2.4245837, -1.2313797, 2.2665799, -3.5310044, 3.6559634
3: -1.0243788, 2.9105000, -0.9823158, 2.7698097, -3.7941885, 3.8928158
4: -1.4927979, 3.2066495, -1.4410114, 3.0508230, -4.5436211, 4.6476612

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.16 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7738182, upper bound: 2.7733051
time: 0.56 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766655, upper bound: 2.7739303
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3709612, 2.2503333, -2.6622829, 2.8961306
1: -0.5089593, 3.5015557, -0.4575120, 3.1468184, -3.6557777, 3.9590676
2: -1.2644246, 2.4245837, -1.1562662, 2.1376271, -3.4020517, 3.5808499
3: -1.0243788, 2.9105000, -0.9258730, 2.5864434, -3.6108222, 3.8363731
4: -1.4927979, 3.2066495, -1.3418519, 2.8791902, -4.3719883, 4.5485015

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.15 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7738182, upper bound: 2.7733052
time: 0.46 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766655, upper bound: 2.7739303
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3896064, 2.3704360, -2.7561796, 2.7914894
1: -0.4820945, 3.3220072, -0.4801092, 3.3121889, -3.7942834, 3.8021164
2: -1.1904538, 2.3187947, -1.2162256, 2.2503803, -3.4408340, 3.5350204
3: -0.9715070, 2.7528343, -0.9699728, 2.7404900, -3.7119970, 3.7228072
4: -1.4044091, 3.0569649, -1.4223459, 3.0280967, -4.4325056, 4.4793110

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.26 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7733500, upper bound: 2.7747033
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767665, upper bound: 2.7762000
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3630810, 2.2252908, -2.6110344, 2.7649639
1: -0.4820945, 3.3220072, -0.4512913, 3.1082866, -3.5903811, 3.7732985
2: -1.1904538, 2.3187947, -1.1403358, 2.1198447, -3.3102984, 3.4591305
3: -0.9715070, 2.7528343, -0.9133595, 2.5548997, -3.5264068, 3.6661940
4: -1.4044091, 3.0569649, -1.3233099, 2.8539722, -4.2583814, 4.3802748

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7733500, upper bound: 2.7751604
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767665, upper bound: 2.7766965
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3896064, 2.3704360, -2.7584693, 2.8078470
1: -0.4848027, 3.3492410, -0.4801092, 3.3121889, -3.7969916, 3.8293502
2: -1.2020583, 2.3271937, -1.2162256, 2.2503803, -3.4524386, 3.5434194
3: -0.9766790, 2.7750435, -0.9699728, 2.7404900, -3.7171688, 3.7450163
4: -1.4156674, 3.0728226, -1.4223459, 3.0280967, -4.4437642, 4.4951687

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.27 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753946, upper bound: 2.7726556
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782713, upper bound: 2.7732835
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3630810, 2.2252908, -2.6133242, 2.7813215
1: -0.4848027, 3.3492410, -0.4512913, 3.1082866, -3.5930893, 3.8005323
2: -1.2020583, 2.3271937, -1.1403358, 2.1198447, -3.3219030, 3.4675295
3: -0.9766790, 2.7750435, -0.9133595, 2.5548997, -3.5315785, 3.6884031
4: -1.4156674, 3.0728226, -1.3233099, 2.8539722, -4.2696395, 4.3961325

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.27 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753946, upper bound: 2.7729598
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782713, upper bound: 2.7737568
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3974134, 2.3932111, -2.7789547, 2.7992964
1: -0.4820945, 3.3220072, -0.4861144, 3.3475065, -3.8296010, 3.8081217
2: -1.1904538, 2.3187947, -1.2313797, 2.2665799, -3.4570336, 3.5501745
3: -0.9715070, 2.7528343, -0.9823158, 2.7698097, -3.7413168, 3.7351501
4: -1.4044091, 3.0569649, -1.4410114, 3.0508230, -4.4552321, 4.4979763

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.26 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7719465, upper bound: 2.7765348
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753615, upper bound: 2.7779751
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3709612, 2.2503333, -2.6360769, 2.7728441
1: -0.4820945, 3.3220072, -0.4575120, 3.1468184, -3.6289129, 3.7795191
2: -1.1904538, 2.3187947, -1.1562662, 2.1376271, -3.3280809, 3.4750609
3: -0.9715070, 2.7528343, -0.9258730, 2.5864434, -3.5579505, 3.6787074
4: -1.4044091, 3.0569649, -1.3418519, 2.8791902, -4.2835994, 4.3988171

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.31 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7719465, upper bound: 2.7765348
time: 0.43 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753615, upper bound: 2.7779751
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3974134, 2.3932111, -2.7812445, 2.8156538
1: -0.4848027, 3.3492410, -0.4861144, 3.3475065, -3.8323092, 3.8353555
2: -1.2020583, 2.3271937, -1.2313797, 2.2665799, -3.4686382, 3.5585735
3: -0.9766790, 2.7750435, -0.9823158, 2.7698097, -3.7464886, 3.7573593
4: -1.4156674, 3.0728226, -1.4410114, 3.0508230, -4.4664903, 4.5138340

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.31 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739638, upper bound: 2.7733651
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767682, upper bound: 2.7739825
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3709612, 2.2503333, -2.6383667, 2.7892017
1: -0.4848027, 3.3492410, -0.4575120, 3.1468184, -3.6316211, 3.8067529
2: -1.2020583, 2.3271937, -1.1562662, 2.1376271, -3.3396854, 3.4834599
3: -0.9766790, 2.7750435, -0.9258730, 2.5864434, -3.5631223, 3.7009165
4: -1.4156674, 3.0728226, -1.3418519, 2.8791902, -4.2948575, 4.4146748

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.33 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739638, upper bound: 2.7733651
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767682, upper bound: 2.7739825
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4056197, 2.5140209, -2.9196405, 2.9196405
1: -0.5055367, 3.4809954, -0.5055367, 3.4809954, -3.9865322, 3.9865322
2: -1.2521493, 2.4233136, -1.2521493, 2.4233136, -3.6754630, 3.6754630
3: -1.0172695, 2.8950694, -1.0172695, 2.8950694, -3.9123387, 3.9123387
4: -1.4811087, 3.1972914, -1.4811087, 3.1972914, -4.6784000, 4.6784000

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3857437, 2.4018829, -2.8075025, 2.8997645
1: -0.5055367, 3.4809954, -0.4820945, 3.3220072, -3.8275437, 3.9630899
2: -1.2521493, 2.4233136, -1.1904538, 2.3187947, -3.5709441, 3.6137674
3: -1.0172695, 2.8950694, -0.9715070, 2.7528343, -3.7701039, 3.8665762
4: -1.4811087, 3.1972914, -1.4044091, 3.0569649, -4.5380735, 4.6017003

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693686, upper bound: 2.7602173
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7692605, upper bound: 2.7604589
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4056197, 2.5140209, -2.9259706, 2.9307890
1: -0.5089593, 3.5015557, -0.5055367, 3.4809954, -3.9899547, 4.0070925
2: -1.2644246, 2.4245837, -1.2521493, 2.4233136, -3.6877382, 3.6767330
3: -1.0243788, 2.9105000, -1.0172695, 2.8950694, -3.9194481, 3.9277697
4: -1.4927979, 3.2066495, -1.4811087, 3.1972914, -4.6900892, 4.6877584

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689538, upper bound: 2.7570467
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689836, upper bound: 2.7570490
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3857437, 2.4018829, -2.8138323, 2.9109130
1: -0.5089593, 3.5015557, -0.4820945, 3.3220072, -3.8309665, 3.9836502
2: -1.2644246, 2.4245837, -1.1904538, 2.3187947, -3.5832193, 3.6150374
3: -1.0243788, 2.9105000, -0.9715070, 2.7528343, -3.7772131, 3.8820071
4: -1.4927979, 3.2066495, -1.4044091, 3.0569649, -4.5497627, 4.6110587

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689537, upper bound: 2.7570466
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689537, upper bound: 2.7570489
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4119495, 2.5251694, -2.9307890, 2.9259706
1: -0.5055367, 3.4809954, -0.5089593, 3.5015557, -4.0070925, 3.9899547
2: -1.2521493, 2.4233136, -1.2644246, 2.4245837, -3.6767330, 3.6877382
3: -1.0172695, 2.8950694, -1.0243788, 2.9105000, -3.9277697, 3.9194481
4: -1.4811087, 3.1972914, -1.4927979, 3.2066495, -4.6877584, 4.6900892

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3880334, 2.4182405, -2.8238602, 2.9020543
1: -0.5055367, 3.4809954, -0.4848027, 3.3492410, -3.8547778, 3.9657981
2: -1.2521493, 2.4233136, -1.2020583, 2.3271937, -3.5793431, 3.6253719
3: -1.0172695, 2.8950694, -0.9766790, 2.7750435, -3.7923131, 3.8717484
4: -1.4811087, 3.1972914, -1.4156674, 3.0728226, -4.5539312, 4.6129589

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4119495, 2.5251694, -2.9371190, 2.9371190
1: -0.5089593, 3.5015557, -0.5089593, 3.5015557, -4.0105152, 4.0105152
2: -1.2644246, 2.4245837, -1.2644246, 2.4245837, -3.6890082, 3.6890082
3: -1.0243788, 2.9105000, -1.0243788, 2.9105000, -3.9348788, 3.9348788
4: -1.4927979, 3.2066495, -1.4927979, 3.2066495, -4.6994476, 4.6994476

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3880334, 2.4182405, -2.8301902, 2.9132028
1: -0.5089593, 3.5015557, -0.4848027, 3.3492410, -3.8582003, 3.9863584
2: -1.2644246, 2.4245837, -1.2020583, 2.3271937, -3.5916183, 3.6266420
3: -1.0243788, 2.9105000, -0.9766790, 2.7750435, -3.7994223, 3.8871789
4: -1.4927979, 3.2066495, -1.4156674, 3.0728226, -4.5656204, 4.6223168

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4056197, 2.5140209, -2.8997645, 2.8075025
1: -0.4820945, 3.3220072, -0.5055367, 3.4809954, -3.9630899, 3.8275437
2: -1.1904538, 2.3187947, -1.2521493, 2.4233136, -3.6137674, 3.5709441
3: -0.9715070, 2.7528343, -1.0172695, 2.8950694, -3.8665762, 3.7701039
4: -1.4044091, 3.0569649, -1.4811087, 3.1972914, -4.6017003, 4.5380735

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693673, upper bound: 2.7626228
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7641839, upper bound: 2.7609127
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3857437, 2.4018829, -2.7876265, 2.7876265
1: -0.4820945, 3.3220072, -0.4820945, 3.3220072, -3.8041017, 3.8041017
2: -1.1904538, 2.3187947, -1.1904538, 2.3187947, -3.5092485, 3.5092485
3: -0.9715070, 2.7528343, -0.9715070, 2.7528343, -3.7243414, 3.7243414
4: -1.4044091, 3.0569649, -1.4044091, 3.0569649, -4.4613738, 4.4613738

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7693673, upper bound: 2.7626228
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7641839, upper bound: 2.7609127
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4056197, 2.5140209, -2.9020543, 2.8238602
1: -0.4848027, 3.3492410, -0.5055367, 3.4809954, -3.9657981, 3.8547778
2: -1.2020583, 2.3271937, -1.2521493, 2.4233136, -3.6253719, 3.5793431
3: -0.9766790, 2.7750435, -1.0172695, 2.8950694, -3.8717484, 3.7923131
4: -1.4156674, 3.0728226, -1.4811087, 3.1972914, -4.6129589, 4.5539312

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7685957, upper bound: 2.7569821
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7605560, upper bound: 2.7557949
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3857437, 2.4018829, -2.7899163, 2.8039842
1: -0.4848027, 3.3492410, -0.4820945, 3.3220072, -3.8068099, 3.8313355
2: -1.2020583, 2.3271937, -1.1904538, 2.3187947, -3.5208530, 3.5176475
3: -0.9766790, 2.7750435, -0.9715070, 2.7528343, -3.7295132, 3.7465506
4: -1.4156674, 3.0728226, -1.4044091, 3.0569649, -4.4726324, 4.4772315

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7685957, upper bound: 2.7569821
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7605560, upper bound: 2.7557949
time: 0.51 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4119495, 2.5251694, -2.9109130, 2.8138323
1: -0.4820945, 3.3220072, -0.5089593, 3.5015557, -3.9836502, 3.8309665
2: -1.1904538, 2.3187947, -1.2644246, 2.4245837, -3.6150374, 3.5832193
3: -0.9715070, 2.7528343, -1.0243788, 2.9105000, -3.8820071, 3.7772131
4: -1.4044091, 3.0569649, -1.4927979, 3.2066495, -4.6110587, 4.5497627

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7579361
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3880334, 2.4182405, -2.8039842, 2.7899163
1: -0.4820945, 3.3220072, -0.4848027, 3.3492410, -3.8313355, 3.8068099
2: -1.1904538, 2.3187947, -1.2020583, 2.3271937, -3.5176475, 3.5208530
3: -0.9715070, 2.7528343, -0.9766790, 2.7750435, -3.7465506, 3.7295132
4: -1.4044091, 3.0569649, -1.4156674, 3.0728226, -4.4772315, 4.4726324

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7579361
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4119495, 2.5251694, -2.9132028, 2.8301902
1: -0.4848027, 3.3492410, -0.5089593, 3.5015557, -3.9863584, 3.8582003
2: -1.2020583, 2.3271937, -1.2644246, 2.4245837, -3.6266420, 3.5916183
3: -0.9766790, 2.7750435, -1.0243788, 2.9105000, -3.8871789, 3.7994223
4: -1.4156674, 3.0728226, -1.4927979, 3.2066495, -4.6223168, 4.5656204

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3880334, 2.4182405, -2.8062739, 2.8062739
1: -0.4848027, 3.3492410, -0.4848027, 3.3492410, -3.8340437, 3.8340437
2: -1.2020583, 2.3271937, -1.2020583, 2.3271937, -3.5292521, 3.5292521
3: -0.9766790, 2.7750435, -0.9766790, 2.7750435, -3.7517223, 3.7517223
4: -1.4156674, 3.0728226, -1.4156674, 3.0728226, -4.4884901, 4.4884901

Time for backsubstitution: 1.75 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=3.285133123397827
rel_dist={0: [-2.7803829052250393, 2.7803829052250393]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797632, upper bound: 2.7798534
time: 0.42 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7795992
time: 0.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 0, lower bound: -2.7797632, upper bound: 2.7798534
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.96
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7795992

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.4334733, 2.4970913, -0.4735588, 2.6049562, -3.0384295, 2.9706502
1: -0.5019389, 3.4698267, -0.5253949, 3.5894566, -4.0913954, 3.9952216
2: -1.2600799, 2.3818066, -1.2851930, 2.5367539, -3.7968338, 3.6669996
3: -1.0162834, 2.9394152, -1.0644845, 3.1261163, -4.1423998, 4.0038996
4: -1.5453744, 3.1090739, -1.5940905, 3.2114999, -4.7568741, 4.7031641

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7798534
time: 0.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797632, upper bound: 2.7798534
time: 0.41 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.5003434, 2.7480116, -3.2233658, 3.1779671
1: -0.5395899, 3.6900327, -0.5552917, 3.7822132, -4.3218031, 4.2453241
2: -1.3205540, 2.6000621, -1.3549172, 2.6739645, -3.9945185, 3.9549794
3: -1.0897965, 3.1838365, -1.1219391, 3.3105505, -4.4003468, 4.3057756
4: -1.6415637, 3.3140776, -1.7104982, 3.3804812, -5.0220451, 5.0245757

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757814
time: 0.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7798534
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -2.7797632, upper bound: 2.7798534
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757814
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.28
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.3976661, 2.3702574, -0.4141727, 2.5066614, -2.9043274, 2.7844300
1: -0.4774379, 3.3009636, -0.5037363, 3.4724970, -3.9499350, 3.8046999
2: -1.2029440, 2.2589540, -1.2499893, 2.4201710, -3.6231151, 3.5089433
3: -0.9665314, 2.7549174, -1.0151881, 2.8970599, -3.8635912, 3.7701054
4: -1.4378498, 2.9917743, -1.4859279, 3.1694007, -4.6072502, 4.4777021

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7798534
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7780246
time: 0.39 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.4029805, 2.3903747, -0.3942038, 2.3730016, -2.7759821, 2.7845783
1: -0.4820042, 3.3288662, -0.4782165, 3.2846050, -3.7666092, 3.8070827
2: -1.2137234, 2.2776916, -1.1793141, 2.2899625, -3.5036860, 3.4570057
3: -0.9756974, 2.7835200, -0.9670746, 2.7289331, -3.7046304, 3.7505946
4: -1.4569836, 3.0132043, -1.3914483, 3.0069008, -4.4638844, 4.4046526

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757544, upper bound: 2.7798534
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770219, upper bound: 2.7780428
time: 0.42 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.4753543, 2.6776235, -0.4855686, 2.6948020, -3.1701565, 3.1631923
1: -0.5395899, 3.6900327, -0.5438708, 3.7131903, -4.2527804, 4.2339034
2: -1.3205540, 2.6000621, -1.3294082, 2.6198123, -3.9403663, 3.9294703
3: -1.0897965, 3.1838365, -1.0992439, 3.2316928, -4.3214893, 4.2830801
4: -1.6415637, 3.3140776, -1.6643423, 3.3243432, -4.9659071, 4.9784198

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7753875
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757814
time: 0.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.4752761, 2.6773210, -0.4893742, 2.7133813, -3.1886573, 3.1666951
1: -0.5395244, 3.6896281, -0.5471199, 3.7470856, -4.2866101, 4.2367477
2: -1.3204033, 2.5997541, -1.3424731, 2.6225457, -3.9429491, 3.9422274
3: -1.0896666, 3.1833949, -1.1056707, 3.2536237, -4.3432903, 4.2890654
4: -1.6413004, 3.3137448, -1.6805805, 3.3433518, -4.9846525, 4.9943252

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7766929
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.24 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7798534
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7780246
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7757544, upper bound: 2.7798534
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7770219, upper bound: 2.7780428
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7753875
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757814
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7766929
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3851464, 2.3213916, -0.4141727, 2.5066614, -2.8918078, 2.7355642
1: -0.4669839, 3.2352452, -0.5037363, 3.4724970, -3.9394808, 3.7389815
2: -1.1786203, 2.2106924, -1.2499893, 2.4201710, -3.5987914, 3.4606817
3: -0.9456136, 2.6868510, -1.0151881, 2.8970599, -3.8426735, 3.7020392
4: -1.3952422, 2.9377618, -1.4859279, 3.1694007, -4.5646429, 4.4236898

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7763723
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7780246
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3928851, 2.3415949, -0.4141018, 2.5063624, -2.8992476, 2.7556968
1: -0.4731214, 3.2672031, -0.5036725, 3.4720984, -3.9452198, 3.7708755
2: -1.1931055, 2.2240765, -1.2498429, 2.4198759, -3.6129813, 3.4739194
3: -0.9582127, 2.7131007, -1.0150615, 2.8966455, -3.8548584, 3.7281623
4: -1.4110876, 2.9584844, -1.4856787, 3.1690710, -4.5801587, 4.4441633

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7763723
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7780246
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3900694, 2.3411717, -0.3942038, 2.3730016, -2.7630711, 2.7353754
1: -0.4715114, 3.2627015, -0.4782165, 3.2846050, -3.7561164, 3.7409182
2: -1.1892037, 2.2289414, -1.1793141, 2.2899625, -3.4791663, 3.4082556
3: -0.9546710, 2.7147241, -0.9670746, 2.7289331, -3.6836042, 3.6817987
4: -1.4139897, 2.9587419, -1.3914483, 3.0069008, -4.4208903, 4.3501902

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757544, upper bound: 2.7767681
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757544, upper bound: 2.7780428
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3989310, 2.3654022, -0.3941274, 2.3727050, -2.7716360, 2.7595296
1: -0.4783176, 3.3001723, -0.4781514, 3.2842073, -3.7625251, 3.7783237
2: -1.2059543, 2.2458639, -1.1791642, 2.2896750, -3.4956293, 3.4250281
3: -0.9687432, 2.7471230, -0.9669462, 2.7285159, -3.6972589, 3.7140694
4: -1.4337101, 2.9834573, -1.3911924, 3.0065706, -4.4402809, 4.3746500

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770219, upper bound: 2.7767681
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770219, upper bound: 2.7780428
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4380667, 2.5541043, -2.9715662, 2.9980319
1: -0.5153302, 3.5426629, -0.5146266, 3.5257249, -4.0410552, 4.0572896
2: -1.2751217, 2.4697607, -1.2626326, 2.4655423, -3.7406640, 3.7323933
3: -1.0362854, 2.9608207, -1.0393541, 2.9872773, -4.0235624, 4.0001745
4: -1.5216012, 3.2471507, -1.5360646, 3.1954689, -4.7170701, 4.7832155

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7753078
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7753078
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4479578, 2.5859699, -2.9847107, 2.8962805
1: -0.4923947, 3.3845534, -0.5217259, 3.5691609, -4.0615559, 3.9062793
2: -1.2141361, 2.3641450, -1.2793503, 2.4988418, -3.7129779, 3.6434953
3: -0.9919410, 2.8182275, -1.0537901, 3.0373750, -4.0293159, 3.8720176
4: -1.4456424, 3.1076982, -1.5649216, 3.2275004, -4.6731429, 4.6726198

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757544
time: 0.39 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757544
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.4173891, 2.5596557, -0.4414486, 2.5718119, -2.9892011, 3.0011044
1: -0.5152627, 3.5422533, -0.5178673, 3.5545766, -4.0698395, 4.0601206
2: -1.2749739, 2.4694438, -1.2756569, 2.4747121, -3.7496860, 3.7451007
3: -1.0361511, 2.9603953, -1.0457748, 3.0073490, -4.0434999, 4.0061703
4: -1.5213493, 3.2468047, -1.5507008, 3.2134900, -4.7348394, 4.7975054

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7766929
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7766929
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3986654, 2.4480238, -0.4515175, 2.6044040, -3.0030694, 2.8995414
1: -0.4923295, 3.3841541, -0.5250107, 3.5997119, -4.0920415, 3.9091649
2: -1.2139883, 2.3638554, -1.2925737, 2.5063608, -3.7203491, 3.6564291
3: -0.9918113, 2.8178139, -1.0603561, 3.0586283, -4.0504398, 3.8781700
4: -1.4453887, 3.1073673, -1.5800858, 3.2463772, -4.6917658, 4.6874533

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.29 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7763723
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7753078, upper bound: 2.7780246
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7763723
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7780246
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7757544, upper bound: 2.7767681
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7757544, upper bound: 2.7780428
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7770219, upper bound: 2.7767681
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7770219, upper bound: 2.7780428
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7753078
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7753078
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757544
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7795992, upper bound: 2.7757544
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7766929
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7766929
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -2.7767402, upper bound: 2.7767402

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3851464, 2.3213916, -0.4014583, 2.4555919, -2.8407383, 2.7228498
1: -0.4669839, 3.2352452, -0.4927686, 3.4043105, -3.8712943, 3.7280138
2: -1.1786203, 2.2106924, -1.2251759, 2.3678098, -3.5464301, 3.4358683
3: -0.9456136, 2.6868510, -0.9933085, 2.8243985, -3.7700121, 3.6801596
4: -1.3952422, 2.9377618, -1.4420398, 3.1132209, -4.5084629, 4.3798018

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782109
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782109
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3851464, 2.3213916, -0.4098997, 2.4738402, -2.8589866, 2.7312913
1: -0.4669839, 3.2352452, -0.4980948, 3.4338298, -3.9008136, 3.7333400
2: -1.1786203, 2.2106924, -1.2407064, 2.3778672, -3.5564876, 3.4513988
3: -0.9456136, 2.6868510, -1.0045066, 2.8533080, -3.7989216, 3.6913576
4: -1.3952422, 2.9377618, -1.4608200, 3.1313741, -4.5266161, 4.3985815

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3928851, 2.3415949, -0.4014583, 2.4555919, -2.8484769, 2.7430532
1: -0.4731214, 3.2672031, -0.4927686, 3.4043105, -3.8774319, 3.7599716
2: -1.1931055, 2.2240765, -1.2251759, 2.3678098, -3.5609152, 3.4492524
3: -0.9582127, 2.7131007, -0.9933085, 2.8243985, -3.7826114, 3.7064092
4: -1.4110876, 2.9584844, -1.4420398, 3.1132209, -4.5243087, 4.4005241

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7763723
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7763723
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3928851, 2.3415949, -0.4098997, 2.4738402, -2.8667254, 2.7514946
1: -0.4731214, 3.2672031, -0.4980948, 3.4338298, -3.9069512, 3.7652979
2: -1.1931055, 2.2240765, -1.2407064, 2.3778672, -3.5709727, 3.4647830
3: -0.9582127, 2.7131007, -1.0045066, 2.8533080, -3.8115206, 3.7176073
4: -1.4110876, 2.9584844, -1.4608200, 3.1313741, -4.5424619, 4.4193044

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7772540
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7772540
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3900694, 2.3411717, -0.3801980, 2.3226647, -2.7127342, 2.7213697
1: -0.4715114, 3.2627015, -0.4670255, 3.2172227, -3.6887341, 3.7297270
2: -1.1892037, 2.2289414, -1.1542571, 2.2398834, -3.4290872, 3.3831985
3: -0.9546710, 2.7147241, -0.9446915, 2.6566396, -3.6113105, 3.6594157
4: -1.4139897, 2.9587419, -1.3476346, 2.9511504, -4.3651400, 4.3063765

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782275
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782109
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3900694, 2.3411717, -0.3836220, 2.3417249, -2.7317944, 2.7247937
1: -0.4715114, 3.2627015, -0.4706959, 3.2480569, -3.7195683, 3.7333975
2: -1.1892037, 2.2289414, -1.1675966, 2.2518730, -3.4410768, 3.3965380
3: -0.9546710, 2.7147241, -0.9520016, 2.6862433, -3.6409144, 3.6667256
4: -1.4139897, 2.9587419, -1.3631822, 2.9707329, -4.3847227, 4.3219242

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7798534
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3989310, 2.3654022, -0.3801980, 2.3226647, -2.7215958, 2.7456002
1: -0.4783176, 3.3001723, -0.4670255, 3.2172227, -3.6955404, 3.7671978
2: -1.2059543, 2.2458639, -1.1542571, 2.2398834, -3.4458377, 3.4001210
3: -0.9687432, 2.7471230, -0.9446915, 2.6566396, -3.6253829, 3.6918144
4: -1.4337101, 2.9834573, -1.3476346, 2.9511504, -4.3848605, 4.3310919

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7764888
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7766429
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3989310, 2.3654022, -0.3836220, 2.3417249, -2.7406559, 2.7490242
1: -0.4783176, 3.3001723, -0.4706959, 3.2480569, -3.7263746, 3.7708683
2: -1.2059543, 2.2458639, -1.1675966, 2.2518730, -3.4578273, 3.4134605
3: -0.9687432, 2.7471230, -0.9520016, 2.6862433, -3.6549864, 3.6991246
4: -1.4337101, 2.9834573, -1.3631822, 2.9707329, -4.4044428, 4.3466396

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7773383
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7772838
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.3851464, 2.3213916, -2.7388535, 2.9451115
1: -0.5153302, 3.5426629, -0.4669839, 3.2352452, -3.7505755, 4.0096469
2: -1.2751217, 2.4697607, -1.1786203, 2.2106924, -3.4858141, 3.6483810
3: -1.0362854, 2.9608207, -0.9456136, 2.6868510, -3.7231364, 3.9064343
4: -1.5216012, 3.2471507, -1.3952422, 2.9377618, -4.4593630, 4.6423931

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
time: 0.37 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4174619, 2.5599651, -0.4156432, 2.4902687, -2.9077306, 2.9756083
1: -0.5153302, 3.5426629, -0.5001699, 3.4397397, -3.9550700, 4.0428329
2: -1.2751217, 2.4697607, -1.2297778, 2.4018970, -3.6770186, 3.6995385
3: -1.0362854, 2.9608207, -1.0092379, 2.8811445, -3.9174299, 3.9700584
4: -1.5216012, 3.2471507, -1.4760703, 3.1325667, -4.6541681, 4.7232208

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.3900694, 2.3411717, -2.7399125, 2.8383923
1: -0.4923947, 3.3845534, -0.4715114, 3.2627015, -3.7550962, 3.8560648
2: -1.2141361, 2.3641450, -1.1892037, 2.2289414, -3.4430776, 3.5533488
3: -0.9919410, 2.8182275, -0.9546710, 2.7147241, -3.7066650, 3.7728987
4: -1.4456424, 3.1076982, -1.4139897, 2.9587419, -4.4043841, 4.5216880

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
time: 0.52 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3987406, 2.4483228, -0.4254830, 2.5213263, -2.9200668, 2.8738058
1: -0.4923947, 3.3845534, -0.5071784, 3.4822540, -3.9746487, 3.8917317
2: -1.2141361, 2.3641450, -1.2461286, 2.4308486, -3.6449847, 3.6102736
3: -0.9919410, 2.8182275, -1.0235286, 2.9278283, -3.9197693, 3.8417561
4: -1.4456424, 3.1076982, -1.5042672, 3.1638393, -4.6094818, 4.6119652

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.4173891, 2.5596557, -0.3928851, 2.3415949, -2.7589841, 2.9525409
1: -0.5152627, 3.5422533, -0.4731214, 3.2672031, -3.7824657, 4.0153747
2: -1.2749739, 2.4694438, -1.1931055, 2.2240765, -3.4990504, 3.6625493
3: -1.0361511, 2.9603953, -0.9582127, 2.7131007, -3.7492518, 3.9186082
4: -1.5213493, 3.2468047, -1.4110876, 2.9584844, -4.4798336, 4.6578922

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7766929
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7758333
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.4173891, 2.5596557, -0.4177715, 2.5048292, -2.9222183, 2.9774272
1: -0.5152627, 3.5422533, -0.5025089, 3.4644697, -3.9797323, 4.0447621
2: -1.2749739, 2.4694438, -1.2412105, 2.4087973, -3.6837711, 3.7106543
3: -1.0361511, 2.9603953, -1.0136938, 2.8979239, -3.9340749, 3.9740891
4: -1.5213493, 3.2468047, -1.4873513, 3.1465032, -4.6678524, 4.7341561

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7766929
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7758333
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3986654, 2.4480238, -0.3989310, 2.3654022, -2.7640676, 2.8469548
1: -0.4923295, 3.3841541, -0.4783176, 3.3001723, -3.7925019, 3.8624716
2: -1.2139883, 2.3638554, -1.2059543, 2.2458639, -3.4598522, 3.5698097
3: -0.9918113, 2.8178139, -0.9687432, 2.7471230, -3.7389343, 3.7865572
4: -1.4453887, 3.1073673, -1.4337101, 2.9834573, -4.4288459, 4.5410776

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7767401
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7759173
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3986654, 2.4480238, -0.4278332, 2.5368888, -2.9355543, 2.8758569
1: -0.4923295, 3.3841541, -0.5096344, 3.5080438, -4.0003734, 3.8937883
2: -1.2139883, 2.3638554, -1.2579489, 2.4390063, -3.6529946, 3.6218042
3: -0.9918113, 2.8178139, -1.0282184, 2.9436550, -3.9354663, 3.8460321
4: -1.4453887, 3.1073673, -1.5163301, 3.1790090, -4.6243978, 4.6236973

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7767402
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7759173
time: 0.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.40 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782109
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782109
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7763723
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7763723
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7772540
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7772540
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782275
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7782109
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7798534
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7764888
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7766429
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7773383
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7772838
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7753078
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7780857, upper bound: 2.7757544
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7766929
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7758333
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7766929
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7752233, upper bound: 2.7758333
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7767401
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7759173
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7767402
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 0, lower bound: -2.7757814, upper bound: 2.7759173

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.4014583, 2.4555919, -2.8451984, 2.7718942
1: -0.4801092, 3.3121889, -0.4927686, 3.4043105, -3.8844197, 3.8049574
2: -1.2162256, 2.2503803, -1.2251759, 2.3678098, -3.5840354, 3.4755561
3: -0.9699728, 2.7404900, -0.9933085, 2.8243985, -3.7943714, 3.7337985
4: -1.4223459, 3.0280967, -1.4420398, 3.1132209, -4.5355668, 4.4701366

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4014583, 2.4555919, -2.8186729, 2.6267490
1: -0.4512913, 3.1082866, -0.4927686, 3.4043105, -3.8556018, 3.6010551
2: -1.1403358, 2.1198447, -1.2251759, 2.3678098, -3.5081456, 3.3450205
3: -0.9133595, 2.5548997, -0.9933085, 2.8243985, -3.7377582, 3.5482082
4: -1.3233099, 2.8539722, -1.4420398, 3.1132209, -4.4365311, 4.2960119

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.4098997, 2.4738402, -2.8634467, 2.7803357
1: -0.4801092, 3.3121889, -0.4980948, 3.4338298, -3.9139390, 3.8102837
2: -1.2162256, 2.2503803, -1.2407064, 2.3778672, -3.5940928, 3.4910867
3: -0.9699728, 2.7404900, -1.0045066, 2.8533080, -3.8232808, 3.7449965
4: -1.4223459, 3.0280967, -1.4608200, 3.1313741, -4.5537200, 4.4889164

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4098997, 2.4738402, -2.8369212, 2.6351905
1: -0.4512913, 3.1082866, -0.4980948, 3.4338298, -3.8851211, 3.6063814
2: -1.1403358, 2.1198447, -1.2407064, 2.3778672, -3.5182030, 3.3605511
3: -0.9133595, 2.5548997, -1.0045066, 2.8533080, -3.7666674, 3.5594063
4: -1.3233099, 2.8539722, -1.4608200, 3.1313741, -4.4546843, 4.3147922

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.4014583, 2.4555919, -2.8530054, 2.7946694
1: -0.4861144, 3.3475065, -0.4927686, 3.4043105, -3.8904250, 3.8402750
2: -1.2313797, 2.2665799, -1.2251759, 2.3678098, -3.5991895, 3.4917557
3: -0.9823158, 2.7698097, -0.9933085, 2.8243985, -3.8067143, 3.7631183
4: -1.4410114, 3.0508230, -1.4420398, 3.1132209, -4.5542326, 4.4928627

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7763723
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7763723
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4014583, 2.4555919, -2.8265531, 2.6517916
1: -0.4575120, 3.1468184, -0.4927686, 3.4043105, -3.8618224, 3.6395869
2: -1.1562662, 2.1376271, -1.2251759, 2.3678098, -3.5240760, 3.3628030
3: -0.9258730, 2.5864434, -0.9933085, 2.8243985, -3.7502716, 3.5797520
4: -1.3418519, 2.8791902, -1.4420398, 3.1132209, -4.4550729, 4.3212299

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7763723
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7763723
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.4098997, 2.4738402, -2.8712535, 2.8031108
1: -0.4861144, 3.3475065, -0.4980948, 3.4338298, -3.9199443, 3.8456013
2: -1.2313797, 2.2665799, -1.2407064, 2.3778672, -3.6092470, 3.5072863
3: -0.9823158, 2.7698097, -1.0045066, 2.8533080, -3.8356237, 3.7743163
4: -1.4410114, 3.0508230, -1.4608200, 3.1313741, -4.5723858, 4.5116429

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772540
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772541
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4098997, 2.4738402, -2.8448014, 2.6602330
1: -0.4575120, 3.1468184, -0.4980948, 3.4338298, -3.8913417, 3.6449132
2: -1.1562662, 2.1376271, -1.2407064, 2.3778672, -3.5341334, 3.3783336
3: -0.9258730, 2.5864434, -1.0045066, 2.8533080, -3.7791810, 3.5909500
4: -1.3418519, 2.8791902, -1.4608200, 3.1313741, -4.4732261, 4.3400102

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772541
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772541
time: 0.40 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3801980, 2.3226647, -2.7122712, 2.7506340
1: -0.4801092, 3.3121889, -0.4670255, 3.2172227, -3.6973319, 3.7792144
2: -1.2162256, 2.2503803, -1.1542571, 2.2398834, -3.4561090, 3.4046373
3: -0.9699728, 2.7404900, -0.9446915, 2.6566396, -3.6266124, 3.6851816
4: -1.4223459, 3.0280967, -1.3476346, 2.9511504, -4.3734961, 4.3757315

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782275
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782275
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3801980, 2.3226647, -2.6857457, 2.6054888
1: -0.4512913, 3.1082866, -0.4670255, 3.2172227, -3.6685140, 3.5753121
2: -1.1403358, 2.1198447, -1.1542571, 2.2398834, -3.3802192, 3.2741017
3: -0.9133595, 2.5548997, -0.9446915, 2.6566396, -3.5699992, 3.4995914
4: -1.3233099, 2.8539722, -1.3476346, 2.9511504, -4.2744603, 4.2016068

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3836220, 2.3417249, -2.7313313, 2.7540579
1: -0.4801092, 3.3121889, -0.4706959, 3.2480569, -3.7281661, 3.7828848
2: -1.2162256, 2.2503803, -1.1675966, 2.2518730, -3.4680986, 3.4179769
3: -0.9699728, 2.7404900, -0.9520016, 2.6862433, -3.6562161, 3.6924915
4: -1.4223459, 3.0280967, -1.3631822, 2.9707329, -4.3930788, 4.3912787

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3836220, 2.3417249, -2.7048059, 2.6089127
1: -0.4512913, 3.1082866, -0.4706959, 3.2480569, -3.6993482, 3.5789826
2: -1.1403358, 2.1198447, -1.1675966, 2.2518730, -3.3922088, 3.2874413
3: -0.9133595, 2.5548997, -0.9520016, 2.6862433, -3.5996027, 3.5069013
4: -1.3233099, 2.8539722, -1.3631822, 2.9707329, -4.2940426, 4.2171545

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7798534
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3801980, 2.3226647, -2.7200780, 2.7734091
1: -0.4861144, 3.3475065, -0.4670255, 3.2172227, -3.7033372, 3.8145320
2: -1.2313797, 2.2665799, -1.1542571, 2.2398834, -3.4712632, 3.4208369
3: -0.9823158, 2.7698097, -0.9446915, 2.6566396, -3.6389554, 3.7145014
4: -1.4410114, 3.0508230, -1.3476346, 2.9511504, -4.3921618, 4.3984575

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7764888
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7764888
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3801980, 2.3226647, -2.6936259, 2.6305313
1: -0.4575120, 3.1468184, -0.4670255, 3.2172227, -3.6747346, 3.6138439
2: -1.1562662, 2.1376271, -1.1542571, 2.2398834, -3.3961496, 3.2918842
3: -0.9258730, 2.5864434, -0.9446915, 2.6566396, -3.5825126, 3.5311351
4: -1.3418519, 2.8791902, -1.3476346, 2.9511504, -4.2930021, 4.2268248

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7766429
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7766429
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3836220, 2.3417249, -2.7391381, 2.7768331
1: -0.4861144, 3.3475065, -0.4706959, 3.2480569, -3.7341714, 3.8182025
2: -1.2313797, 2.2665799, -1.1675966, 2.2518730, -3.4832528, 3.4341764
3: -0.9823158, 2.7698097, -0.9520016, 2.6862433, -3.6685591, 3.7218113
4: -1.4410114, 3.0508230, -1.3631822, 2.9707329, -4.4117441, 4.4140053

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7773382
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7773383
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3836220, 2.3417249, -2.7126861, 2.6339552
1: -0.4575120, 3.1468184, -0.4706959, 3.2480569, -3.7055688, 3.6175144
2: -1.1562662, 2.1376271, -1.1675966, 2.2518730, -3.4081392, 3.3052237
3: -0.9258730, 2.5864434, -0.9520016, 2.6862433, -3.6121163, 3.5384450
4: -1.3418519, 2.8791902, -1.3631822, 2.9707329, -4.3125849, 4.2423725

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772839
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772839
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3851464, 2.3213916, -2.7270112, 2.8991673
1: -0.5055367, 3.4809954, -0.4669839, 3.2352452, -3.7407818, 3.9479792
2: -1.2521493, 2.4233136, -1.1786203, 2.2106924, -3.4628417, 3.6019340
3: -1.0172695, 2.8950694, -0.9456136, 2.6868510, -3.7041206, 3.8406830
4: -1.4811087, 3.1972914, -1.3952422, 2.9377618, -4.4188704, 4.5925336

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7751757
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7751757
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3851464, 2.3213916, -2.7333412, 2.9103158
1: -0.5089593, 3.5015557, -0.4669839, 3.2352452, -3.7442045, 3.9685395
2: -1.2644246, 2.4245837, -1.1786203, 2.2106924, -3.4751170, 3.6032040
3: -1.0243788, 2.9105000, -0.9456136, 2.6868510, -3.7112298, 3.8561137
4: -1.4927979, 3.2066495, -1.3952422, 2.9377618, -4.4305596, 4.6018915

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7751757
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7753078
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4156432, 2.4902687, -2.8958883, 2.9296641
1: -0.5055367, 3.4809954, -0.5001699, 3.4397397, -3.9452763, 3.9811654
2: -1.2521493, 2.4233136, -1.2297778, 2.4018970, -3.6540463, 3.6530914
3: -1.0172695, 2.8950694, -1.0092379, 2.8811445, -3.8984141, 3.9043074
4: -1.4811087, 3.1972914, -1.4760703, 3.1325667, -4.6136751, 4.6733618

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739603, upper bound: 2.7650250
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689393, upper bound: 2.7570490
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4156432, 2.4902687, -2.9022183, 2.9408126
1: -0.5089593, 3.5015557, -0.5001699, 3.4397397, -3.9486990, 4.0017257
2: -1.2644246, 2.4245837, -1.2297778, 2.4018970, -3.6663215, 3.6543615
3: -1.0243788, 2.9105000, -1.0092379, 2.8811445, -3.9055233, 3.9197378
4: -1.4927979, 3.2066495, -1.4760703, 3.1325667, -4.6253643, 4.6827197

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739603, upper bound: 2.7650250
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7689393, upper bound: 2.7570490
time: 0.40 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3900694, 2.3411717, -2.7269154, 2.7919524
1: -0.4820945, 3.3220072, -0.4715114, 3.2627015, -3.7447960, 3.7935185
2: -1.1904538, 2.3187947, -1.1892037, 2.2289414, -3.4193952, 3.5079985
3: -0.9715070, 2.7528343, -0.9546710, 2.7147241, -3.6862311, 3.7075052
4: -1.4044091, 3.0569649, -1.4139897, 2.9587419, -4.3631511, 4.4709544

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7752233
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782275, upper bound: 2.7756541
time: 0.38 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3900694, 2.3411717, -2.7292051, 2.8083100
1: -0.4848027, 3.3492410, -0.4715114, 3.2627015, -3.7475042, 3.8207524
2: -1.2020583, 2.3271937, -1.1892037, 2.2289414, -3.4309998, 3.5163975
3: -0.9766790, 2.7750435, -0.9546710, 2.7147241, -3.6914029, 3.7297144
4: -1.4156674, 3.0728226, -1.4139897, 2.9587419, -4.3744092, 4.4868121

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7752233
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782275, upper bound: 2.7756541
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4254830, 2.5213263, -2.9070699, 2.8273659
1: -0.4820945, 3.3220072, -0.5071784, 3.4822540, -3.9643486, 3.8291855
2: -1.1904538, 2.3187947, -1.2461286, 2.4308486, -3.6213024, 3.5649233
3: -0.9715070, 2.7528343, -1.0235286, 2.9278283, -3.8993354, 3.7763629
4: -1.4044091, 3.0569649, -1.5042672, 3.1638393, -4.5682483, 4.5612321

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7621634, upper bound: 2.7586477
time: 0.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7605560, upper bound: 2.7557949
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4254830, 2.5213263, -2.9093597, 2.8437235
1: -0.4848027, 3.3492410, -0.5071784, 3.4822540, -3.9670568, 3.8564196
2: -1.2020583, 2.3271937, -1.2461286, 2.4308486, -3.6329069, 3.5733223
3: -0.9766790, 2.7750435, -1.0235286, 2.9278283, -3.9045072, 3.7985721
4: -1.4156674, 3.0728226, -1.5042672, 3.1638393, -4.5795069, 4.5770898

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7621634, upper bound: 2.7586477
time: 0.44 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7605560, upper bound: 2.7557949
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3928851, 2.3415949, -2.7472146, 2.9069061
1: -0.5055367, 3.4809954, -0.4731214, 3.2672031, -3.7727399, 3.9541168
2: -1.2521493, 2.4233136, -1.1931055, 2.2240765, -3.4762259, 3.6164191
3: -1.0172695, 2.8950694, -0.9582127, 2.7131007, -3.7303700, 3.8532820
4: -1.4811087, 3.1972914, -1.4110876, 2.9584844, -4.4395933, 4.6083789

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7768912
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7768912
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3928851, 2.3415949, -2.7535443, 2.9180546
1: -0.5089593, 3.5015557, -0.4731214, 3.2672031, -3.7761624, 3.9746771
2: -1.2644246, 2.4245837, -1.1931055, 2.2240765, -3.4885011, 3.6176891
3: -1.0243788, 2.9105000, -0.9582127, 2.7131007, -3.7374794, 3.8687129
4: -1.4927979, 3.2066495, -1.4110876, 2.9584844, -4.4512825, 4.6177373

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7758239
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7758333
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.4177715, 2.5048292, -2.9104488, 2.9317925
1: -0.5055367, 3.4809954, -0.5025089, 3.4644697, -3.9700065, 3.9835043
2: -1.2521493, 2.4233136, -1.2412105, 2.4087973, -3.6609466, 3.6645241
3: -1.0172695, 2.8950694, -1.0136938, 2.8979239, -3.9151936, 3.9087632
4: -1.4811087, 3.1972914, -1.4873513, 3.1465032, -4.6276121, 4.6846428

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7672876, upper bound: 2.7649100
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526610, upper bound: 2.7524708
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.4177715, 2.5048292, -2.9167786, 2.9429410
1: -0.5089593, 3.5015557, -0.5025089, 3.4644697, -3.9734290, 4.0040646
2: -1.2644246, 2.4245837, -1.2412105, 2.4087973, -3.6732218, 3.6657941
3: -1.0243788, 2.9105000, -1.0136938, 2.8979239, -3.9223027, 3.9241939
4: -1.4927979, 3.2066495, -1.4873513, 3.1465032, -4.6393013, 4.6940007

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7672876, upper bound: 2.7649102
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526610, upper bound: 2.7524708
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.3989310, 2.3654022, -2.7511458, 2.8008139
1: -0.4820945, 3.3220072, -0.4783176, 3.3001723, -3.7822669, 3.8003249
2: -1.1904538, 2.3187947, -1.2059543, 2.2458639, -3.4363177, 3.5247490
3: -0.9715070, 2.7528343, -0.9687432, 2.7471230, -3.7186298, 3.7215776
4: -1.4044091, 3.0569649, -1.4337101, 2.9834573, -4.3878665, 4.4906750

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7770217
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764888, upper bound: 2.7770219
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.3989310, 2.3654022, -2.7534356, 2.8171716
1: -0.4848027, 3.3492410, -0.4783176, 3.3001723, -3.7849751, 3.8275585
2: -1.2020583, 2.3271937, -1.2059543, 2.2458639, -3.4479222, 3.5331481
3: -0.9766790, 2.7750435, -0.9687432, 2.7471230, -3.7238021, 3.7437868
4: -1.4156674, 3.0728226, -1.4337101, 2.9834573, -4.3991246, 4.5065327

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7758239
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764888, upper bound: 2.7759220
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.3857437, 2.4018829, -0.4278332, 2.5368888, -2.9226325, 2.8297162
1: -0.4820945, 3.3220072, -0.5096344, 3.5080438, -3.9901383, 3.8316417
2: -1.1904538, 2.3187947, -1.2579489, 2.4390063, -3.6294601, 3.5767436
3: -0.9715070, 2.7528343, -1.0282184, 2.9436550, -3.9151621, 3.7810526
4: -1.4044091, 3.0569649, -1.5163301, 3.1790090, -4.5834179, 4.5732951

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7558646, upper bound: 2.7576655
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.3880334, 2.4182405, -0.4278332, 2.5368888, -2.9249222, 2.8460736
1: -0.4848027, 3.3492410, -0.5096344, 3.5080438, -3.9928465, 3.8588753
2: -1.2020583, 2.3271937, -1.2579489, 2.4390063, -3.6410646, 3.5851426
3: -0.9766790, 2.7750435, -1.0282184, 2.9436550, -3.9203339, 3.8032618
4: -1.4156674, 3.0728226, -1.5163301, 3.1790090, -4.5946765, 4.5891528

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7558646, upper bound: 2.7576655
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.44 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.61 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797581
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7763723
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7763723
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7763723
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7763723
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772540
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772541
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772541
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772541
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782275
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782275
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7776871, upper bound: 2.7782109
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7798534
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7751757, upper bound: 2.7797361
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7764888
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7764888
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7766429
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7788504, upper bound: 2.7766429
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7773382
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7773383
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772839
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7768912, upper bound: 2.7772839
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7751757
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7751757
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7751757
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7753078
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7739603, upper bound: 2.7650250
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7689393, upper bound: 2.7570490
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7739603, upper bound: 2.7650250
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7689393, upper bound: 2.7570490
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7752233
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782275, upper bound: 2.7756541
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782109, upper bound: 2.7752233
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7782275, upper bound: 2.7756541
IS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7621634, upper bound: 2.7586477
IS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7605560, upper bound: 2.7557949
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7621634, upper bound: 2.7586477
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7605560, upper bound: 2.7557949
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7768912
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7768912
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7758239
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7758333
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7672876, upper bound: 2.7649100
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7526610, upper bound: 2.7524708
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7672876, upper bound: 2.7649102
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7526610, upper bound: 2.7524708
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7770217
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7764888, upper bound: 2.7770219
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7763723, upper bound: 2.7758239
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7764888, upper bound: 2.7759220
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7558646, upper bound: 2.7576655
IS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7558646, upper bound: 2.7576655
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.61
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3896064, 2.3705792, -2.7601857, 2.7600424
1: -0.4801092, 3.3121889, -0.4801092, 3.3123057, -3.7924149, 3.7922981
2: -1.2162256, 2.2503803, -1.2162256, 2.2505226, -3.4667482, 3.4666059
3: -0.9699728, 2.7404900, -0.9699728, 2.7406662, -3.7106390, 3.7104628
4: -1.4223459, 3.0280967, -1.4223459, 3.0282202, -4.4505663, 4.4504423

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747216, upper bound: 2.7750383
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742223, upper bound: 2.7742223
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.4056197, 2.5140209, -2.9036274, 2.7760556
1: -0.4801092, 3.3121889, -0.5055367, 3.4809954, -3.9611046, 3.8177257
2: -1.2162256, 2.2503803, -1.2521493, 2.4233136, -3.6395392, 3.5025296
3: -0.9699728, 2.7404900, -1.0172695, 2.8950694, -3.8650422, 3.7577596
4: -1.4223459, 3.0280967, -1.4811087, 3.1972914, -4.6196375, 4.5092053

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747216, upper bound: 2.7752291
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742223, upper bound: 2.7742223
time: 0.38 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3896064, 2.3705792, -2.7336602, 2.6148973
1: -0.4512913, 3.1082866, -0.4801092, 3.3123057, -3.7635970, 3.5883958
2: -1.1403358, 2.1198447, -1.2162256, 2.2505226, -3.3908584, 3.3360703
3: -0.9133595, 2.5548997, -0.9699728, 2.7406662, -3.6540256, 3.5248725
4: -1.3233099, 2.8539722, -1.4223459, 3.0282202, -4.3515301, 4.2763181

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.02 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723883, upper bound: 2.7750425
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757062, upper bound: 2.7762659
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4056197, 2.5140209, -2.8771019, 2.6309104
1: -0.4512913, 3.1082866, -0.5055367, 3.4809954, -3.9322867, 3.6138234
2: -1.1403358, 2.1198447, -1.2521493, 2.4233136, -3.5636494, 3.3719940
3: -0.9133595, 2.5548997, -1.0172695, 2.8950694, -3.8084288, 3.5721693
4: -1.3233099, 2.8539722, -1.4811087, 3.1972914, -4.5206013, 4.3350811

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.02 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723883, upper bound: 2.7750425
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757062, upper bound: 2.7762659
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3974134, 2.3933864, -2.7829928, 2.7678494
1: -0.4801092, 3.3121889, -0.4861144, 3.3476477, -3.8277569, 3.7983034
2: -1.2162256, 2.2503803, -1.2313797, 2.2667561, -3.4829817, 3.4817600
3: -0.9699728, 2.7404900, -0.9823158, 2.7700262, -3.7399991, 3.7228057
4: -1.4223459, 3.0280967, -1.4410114, 3.0509758, -4.4733219, 4.4691081

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7730267, upper bound: 2.7762925
time: 0.44 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725115, upper bound: 2.7754898
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.4119495, 2.5251694, -2.9147758, 2.7823853
1: -0.4801092, 3.3121889, -0.5089593, 3.5015557, -3.9816649, 3.8211482
2: -1.2162256, 2.2503803, -1.2644246, 2.4245837, -3.6408093, 3.5148048
3: -0.9699728, 2.7404900, -1.0243788, 2.9105000, -3.8804729, 3.7648687
4: -1.4223459, 3.0280967, -1.4927979, 3.2066495, -4.6289954, 4.5208945

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7730267, upper bound: 2.7777663
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725115, upper bound: 2.7769933
time: 0.41 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3974134, 2.3933864, -2.7564673, 2.6227040
1: -0.4512913, 3.1082866, -0.4861144, 3.3476477, -3.7989390, 3.5944011
2: -1.1403358, 2.1198447, -1.2313797, 2.2667561, -3.4070919, 3.3512244
3: -0.9133595, 2.5548997, -0.9823158, 2.7700262, -3.6833858, 3.5372155
4: -1.3233099, 2.8539722, -1.4410114, 3.0509758, -4.3742857, 4.2949839

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698031, upper bound: 2.7766945
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732919, upper bound: 2.7778781
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.4119495, 2.5251694, -2.8882504, 2.6372404
1: -0.4512913, 3.1082866, -0.5089593, 3.5015557, -3.9528470, 3.6172459
2: -1.1403358, 2.1198447, -1.2644246, 2.4245837, -3.5649195, 3.3842692
3: -0.9133595, 2.5548997, -1.0243788, 2.9105000, -3.8238597, 3.5792785
4: -1.3233099, 2.8539722, -1.4927979, 3.2066495, -4.5299597, 4.3467703

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698031, upper bound: 2.7766562
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7732919, upper bound: 2.7778781
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3896064, 2.3705792, -2.7679925, 2.7828176
1: -0.4861144, 3.3475065, -0.4801092, 3.3123057, -3.7984202, 3.8276157
2: -1.2313797, 2.2665799, -1.2162256, 2.2505226, -3.4819024, 3.4828055
3: -0.9823158, 2.7698097, -0.9699728, 2.7406662, -3.7229819, 3.7397826
4: -1.4410114, 3.0508230, -1.4223459, 3.0282202, -4.4692316, 4.4731688

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747216, upper bound: 2.7731774
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.4056197, 2.5140209, -2.9114342, 2.7988307
1: -0.4861144, 3.3475065, -0.5055367, 3.4809954, -3.9671099, 3.8530431
2: -1.2313797, 2.2665799, -1.2521493, 2.4233136, -3.6546934, 3.5187292
3: -0.9823158, 2.7698097, -1.0172695, 2.8950694, -3.8773851, 3.7870793
4: -1.4410114, 3.0508230, -1.4811087, 3.1972914, -4.6383028, 4.5319319

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747216, upper bound: 2.7732945
time: 0.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.42 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3896064, 2.3705792, -2.7415404, 2.6399398
1: -0.4575120, 3.1468184, -0.4801092, 3.3123057, -3.7698176, 3.6269276
2: -1.1562662, 2.1376271, -1.2162256, 2.2505226, -3.4067888, 3.3538527
3: -0.9258730, 2.5864434, -0.9699728, 2.7406662, -3.6665392, 3.5564163
4: -1.3418519, 2.8791902, -1.4223459, 3.0282202, -4.3700724, 4.3015361

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.07 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723883, upper bound: 2.7738077
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768347, upper bound: 2.7741484
time: 0.47 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4056197, 2.5140209, -2.8849821, 2.6559529
1: -0.4575120, 3.1468184, -0.5055367, 3.4809954, -3.9385073, 3.6523552
2: -1.1562662, 2.1376271, -1.2521493, 2.4233136, -3.5795798, 3.3897765
3: -0.9258730, 2.5864434, -1.0172695, 2.8950694, -3.8209424, 3.6037130
4: -1.3418519, 2.8791902, -1.4811087, 3.1972914, -4.5391436, 4.3602991

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.05 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740000, upper bound: 2.7738077
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768347, upper bound: 2.7741484
time: 0.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3974134, 2.3933864, -2.7907996, 2.7906246
1: -0.4861144, 3.3475065, -0.4861144, 3.3476477, -3.8337622, 3.8336210
2: -1.2313797, 2.2665799, -1.2313797, 2.2667561, -3.4981358, 3.4979596
3: -0.9823158, 2.7698097, -0.9823158, 2.7700262, -3.7523420, 3.7521255
4: -1.4410114, 3.0508230, -1.4410114, 3.0509758, -4.4919872, 4.4918346

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7730267, upper bound: 2.7734955
time: 0.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7729640
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.4119495, 2.5251694, -2.9225826, 2.8051605
1: -0.4861144, 3.3475065, -0.5089593, 3.5015557, -3.9876702, 3.8564658
2: -1.2313797, 2.2665799, -1.2644246, 2.4245837, -3.6559634, 3.5310044
3: -0.9823158, 2.7698097, -1.0243788, 2.9105000, -3.8928158, 3.7941885
4: -1.4410114, 3.0508230, -1.4927979, 3.2066495, -4.6476612, 4.5436211

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7730267, upper bound: 2.7745318
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7740142
time: 0.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3974134, 2.3933864, -2.7643476, 2.6477466
1: -0.4575120, 3.1468184, -0.4861144, 3.3476477, -3.8051596, 3.6329329
2: -1.1562662, 2.1376271, -1.2313797, 2.2667561, -3.4230223, 3.3690069
3: -0.9258730, 2.5864434, -0.9823158, 2.7700262, -3.6958992, 3.5687592
4: -1.3418519, 2.8791902, -1.4410114, 3.0509758, -4.3928280, 4.3202019

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.07 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724023, upper bound: 2.7746988
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748821, upper bound: 2.7750838
time: 0.44 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.4119495, 2.5251694, -2.8961306, 2.6622829
1: -0.4575120, 3.1468184, -0.5089593, 3.5015557, -3.9590676, 3.6557777
2: -1.1562662, 2.1376271, -1.2644246, 2.4245837, -3.5808499, 3.4020517
3: -0.9258730, 2.5864434, -1.0243788, 2.9105000, -3.8363731, 3.6108222
4: -1.3418519, 2.8791902, -1.4927979, 3.2066495, -4.5485015, 4.3719883

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.12 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7724023, upper bound: 2.7746988
time: 0.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748821, upper bound: 2.7750838
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3630810, 2.2252908, -2.6148973, 2.7335169
1: -0.4801092, 3.3121889, -0.4512913, 3.1082866, -3.5883958, 3.7634802
2: -1.2162256, 2.2503803, -1.1403358, 2.1198447, -3.3360703, 3.3907161
3: -0.9699728, 2.7404900, -0.9133595, 2.5548997, -3.5248725, 3.6538496
4: -1.4223459, 3.0280967, -1.3233099, 2.8539722, -4.2763181, 4.3514066

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.09 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723883, upper bound: 2.7750425
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759906, upper bound: 2.7762326
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3857437, 2.4018829, -2.7914894, 2.7561796
1: -0.4801092, 3.3121889, -0.4820945, 3.3220072, -3.8021164, 3.7942834
2: -1.2162256, 2.2503803, -1.1904538, 2.3187947, -3.5350204, 3.4408340
3: -0.9699728, 2.7404900, -0.9715070, 2.7528343, -3.7228072, 3.7119970
4: -1.4223459, 3.0280967, -1.4044091, 3.0569649, -4.4793110, 4.4325056

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.06 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7723883, upper bound: 2.7750425
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759906, upper bound: 2.7762326
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3630810, 2.2252908, -2.5883718, 2.5883718
1: -0.4512913, 3.1082866, -0.4512913, 3.1082866, -3.5595779, 3.5595779
2: -1.1403358, 2.1198447, -1.1403358, 2.1198447, -3.2601805, 3.2601805
3: -0.9133595, 2.5548997, -0.9133595, 2.5548997, -3.4682593, 3.4682593
4: -1.3233099, 2.8539722, -1.3233099, 2.8539722, -4.1772823, 4.1772823

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756165, upper bound: 2.7786018
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782101, upper bound: 2.7787504
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3857437, 2.4018829, -2.7649639, 2.6110344
1: -0.4512913, 3.1082866, -0.4820945, 3.3220072, -3.7732985, 3.5903811
2: -1.1403358, 2.1198447, -1.1904538, 2.3187947, -3.4591305, 3.3102984
3: -0.9133595, 2.5548997, -0.9715070, 2.7528343, -3.6661940, 3.5264068
4: -1.3233099, 2.8539722, -1.4044091, 3.0569649, -4.3802748, 4.2583814

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756165, upper bound: 2.7786018
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782101, upper bound: 2.7787504
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3709612, 2.2503333, -2.6399398, 2.7413971
1: -0.4801092, 3.3121889, -0.4575120, 3.1468184, -3.6269276, 3.7697008
2: -1.2162256, 2.2503803, -1.1562662, 2.1376271, -3.3538527, 3.4066465
3: -0.9699728, 2.7404900, -0.9258730, 2.5864434, -3.5564163, 3.6663630
4: -1.4223459, 3.0280967, -1.3418519, 2.8791902, -4.3015361, 4.3699484

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.13 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698031, upper bound: 2.7766101
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731538, upper bound: 2.7778003
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3896064, 2.3704360, -0.3880334, 2.4182405, -2.8078470, 2.7584693
1: -0.4801092, 3.3121889, -0.4848027, 3.3492410, -3.8293502, 3.7969916
2: -1.2162256, 2.2503803, -1.2020583, 2.3271937, -3.5434194, 3.4524386
3: -0.9699728, 2.7404900, -0.9766790, 2.7750435, -3.7450163, 3.7171688
4: -1.4223459, 3.0280967, -1.4156674, 3.0728226, -4.4951687, 4.4437642

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751823, upper bound: 2.7795667
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.57 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698031, upper bound: 2.7766101
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7731538, upper bound: 2.7778003
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3709612, 2.2503333, -2.6134143, 2.5962520
1: -0.4512913, 3.1082866, -0.4575120, 3.1468184, -3.5981097, 3.5657985
2: -1.1403358, 2.1198447, -1.1562662, 2.1376271, -3.2779629, 3.2761109
3: -0.9133595, 2.5548997, -0.9258730, 2.5864434, -3.4998031, 3.4807727
4: -1.3233099, 2.8539722, -1.3418519, 2.8791902, -4.2025003, 4.1958241

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7797776
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757299, upper bound: 2.7797776
time: 0.45 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3630810, 2.2252908, -0.3880334, 2.4182405, -2.7813215, 2.6133242
1: -0.4512913, 3.1082866, -0.4848027, 3.3492410, -3.8005323, 3.5930893
2: -1.1403358, 2.1198447, -1.2020583, 2.3271937, -3.4675295, 3.3219030
3: -0.9133595, 2.5548997, -0.9766790, 2.7750435, -3.6884031, 3.5315785
4: -1.3233099, 2.8539722, -1.4156674, 3.0728226, -4.3961325, 4.2696395

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7797776
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757299, upper bound: 2.7797776
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3630810, 2.2252908, -2.6227040, 2.7562921
1: -0.4861144, 3.3475065, -0.4512913, 3.1082866, -3.5944011, 3.7987978
2: -1.2313797, 2.2665799, -1.1403358, 2.1198447, -3.3512244, 3.4069157
3: -0.9823158, 2.7698097, -0.9133595, 2.5548997, -3.5372155, 3.6831694
4: -1.4410114, 3.0508230, -1.3233099, 2.8539722, -4.2949839, 4.3741331

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796801, upper bound: 2.7763613
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.48 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752252, upper bound: 2.7738806
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776269, upper bound: 2.7743509
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3857437, 2.4018829, -2.7992964, 2.7789547
1: -0.4861144, 3.3475065, -0.4820945, 3.3220072, -3.8081217, 3.8296010
2: -1.2313797, 2.2665799, -1.1904538, 2.3187947, -3.5501745, 3.4570336
3: -0.9823158, 2.7698097, -0.9715070, 2.7528343, -3.7351501, 3.7413168
4: -1.4410114, 3.0508230, -1.4044091, 3.0569649, -4.4979763, 4.4552321

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796801, upper bound: 2.7763613
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.58 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752252, upper bound: 2.7738806
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7776269, upper bound: 2.7743509
time: 0.42 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3630810, 2.2252908, -2.5962520, 2.6134143
1: -0.4575120, 3.1468184, -0.4512913, 3.1082866, -3.5657985, 3.5981097
2: -1.1562662, 2.1376271, -1.1403358, 2.1198447, -3.2761109, 3.2779629
3: -0.9258730, 2.5864434, -0.9133595, 2.5548997, -3.4807727, 3.4998031
4: -1.3418519, 2.8791902, -1.3233099, 2.8539722, -4.1958241, 4.2025003

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763854, upper bound: 2.7765369
time: 0.41 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788794, upper bound: 2.7766042
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3857437, 2.4018829, -2.7728441, 2.6360769
1: -0.4575120, 3.1468184, -0.4820945, 3.3220072, -3.7795191, 3.6289129
2: -1.1562662, 2.1376271, -1.1904538, 2.3187947, -3.4750609, 3.3280809
3: -0.9258730, 2.5864434, -0.9715070, 2.7528343, -3.6787074, 3.5579505
4: -1.3418519, 2.8791902, -1.4044091, 3.0569649, -4.3988171, 4.2835994

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7763854, upper bound: 2.7765369
time: 0.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788794, upper bound: 2.7766042
time: 0.47 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3709612, 2.2503333, -2.6477466, 2.7641723
1: -0.4861144, 3.3475065, -0.4575120, 3.1468184, -3.6329329, 3.8050184
2: -1.2313797, 2.2665799, -1.1562662, 2.1376271, -3.3690069, 3.4228461
3: -0.9823158, 2.7698097, -0.9258730, 2.5864434, -3.5687592, 3.6956828
4: -1.4410114, 3.0508230, -1.3418519, 2.8791902, -4.3202019, 4.3926749

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769447, upper bound: 2.7772489
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.59 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725042, upper bound: 2.7747498
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749263, upper bound: 2.7752162
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.3974134, 2.3932111, -0.3880334, 2.4182405, -2.8156538, 2.7812445
1: -0.4861144, 3.3475065, -0.4848027, 3.3492410, -3.8353555, 3.8323092
2: -1.2313797, 2.2665799, -1.2020583, 2.3271937, -3.5585735, 3.4686382
3: -0.9823158, 2.7698097, -0.9766790, 2.7750435, -3.7573593, 3.7464886
4: -1.4410114, 3.0508230, -1.4156674, 3.0728226, -4.5138340, 4.4664903

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769447, upper bound: 2.7772489
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.59 seconds

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725042, upper bound: 2.7747497
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749263, upper bound: 2.7752162
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3709612, 2.2503333, -2.6212945, 2.6212945
1: -0.4575120, 3.1468184, -0.4575120, 3.1468184, -3.6043303, 3.6043303
2: -1.1562662, 2.1376271, -1.1562662, 2.1376271, -3.2938933, 3.2938933
3: -0.9258730, 2.5864434, -0.9258730, 2.5864434, -3.5123165, 3.5123165
4: -1.3418519, 2.8791902, -1.3418519, 2.8791902, -4.2210422, 4.2210422

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764863, upper bound: 2.7772416
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767563, upper bound: 2.7772416
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.3709612, 2.2503333, -0.3880334, 2.4182405, -2.7892017, 2.6383667
1: -0.4575120, 3.1468184, -0.4848027, 3.3492410, -3.8067529, 3.6316211
2: -1.1562662, 2.1376271, -1.2020583, 2.3271937, -3.4834599, 3.3396854
3: -0.9258730, 2.5864434, -0.9766790, 2.7750435, -3.7009165, 3.5631223
4: -1.3418519, 2.8791902, -1.4156674, 3.0728226, -4.4146748, 4.2948575

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764863, upper bound: 2.7772415
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767563, upper bound: 2.7772415
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3896064, 2.3704360, -2.7760556, 2.9036274
1: -0.5055367, 3.4809954, -0.4801092, 3.3121889, -3.8177257, 3.9611046
2: -1.2521493, 2.4233136, -1.2162256, 2.2503803, -3.5025296, 3.6395392
3: -1.0172695, 2.8950694, -0.9699728, 2.7404900, -3.7577596, 3.8650422
4: -1.4811087, 3.1972914, -1.4223459, 3.0280967, -4.5092053, 4.6196375

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728250, upper bound: 2.7743349
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7762326, upper bound: 2.7757063
time: 0.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.4056197, 2.5140209, -0.3630810, 2.2252908, -2.6309104, 2.8771019
1: -0.5055367, 3.4809954, -0.4512913, 3.1082866, -3.6138234, 3.9322867
2: -1.2521493, 2.4233136, -1.1403358, 2.1198447, -3.3719940, 3.5636494
3: -1.0172695, 2.8950694, -0.9133595, 2.5548997, -3.5721693, 3.8084288
4: -1.4811087, 3.1972914, -1.3233099, 2.8539722, -4.3350811, 4.5206013

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.20 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728250, upper bound: 2.7743486
time: 0.42 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7762326, upper bound: 2.7757133
time: 0.45 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3896064, 2.3704360, -2.7823853, 2.9147758
1: -0.5089593, 3.5015557, -0.4801092, 3.3121889, -3.8211482, 3.9816649
2: -1.2644246, 2.4245837, -1.2162256, 2.2503803, -3.5148048, 3.6408093
3: -1.0243788, 2.9105000, -0.9699728, 2.7404900, -3.7648687, 3.8804729
4: -1.4927979, 3.2066495, -1.4223459, 3.0280967, -4.5208945, 4.6289954

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 5
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 10
type: A, layer: 3, pos: 18
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 6
type: A, layer: 3, pos: 13
type: A, layer: 3, pos: 25

Time for candidate selection: 2.21 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728250, upper bound: 2.7724801
time: 0.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778450, upper bound: 2.7731124
time: 0.48 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.4119495, 2.5251694, -0.3630810, 2.2252908, -2.6372404, 2.8882504
1: -0.5089593, 3.5015557, -0.4512913, 3.1082866, -3.6172459, 3.9528470
2: -1.2644246, 2.4245837, -1.1403358, 2.1198447, -3.3842692, 3.5649195
3: -1.0243788, 2.9105000, -0.9133595, 2.5548997, -3.5792785, 3.8238597
4: -1.4927979, 3.2066495, -1.3233099, 2.8539722, -4.3467703, 4.5299597

Time for backsubstitution: 1.68 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=3.285133123397827
rel_dist={0: [-2.780286344644136, 2.7802863446441357]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1151.74 seconds
