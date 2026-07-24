## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.7638016924


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331)
1: (-0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803)
2: (-1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662)
3: (-1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606)
4: (-1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.02 = 2.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735217, upper bound: 2.7756735
time: 0.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7804673
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.73 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -2.7735217, upper bound: 2.7756735
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -2.7804673, upper bound: 2.7804673

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.3937300, 2.2607875, -0.5017872, 2.7487946, -3.1425245, 2.7625747
1: -0.4659128, 3.1165721, -0.5561673, 3.7846758, -4.2505884, 3.6727395
2: -1.1388535, 2.1871104, -1.3571187, 2.6733608, -3.8122144, 3.5442290
3: -0.9433906, 2.6073360, -1.1238993, 3.3164425, -4.2598333, 3.7312355
4: -1.3197460, 2.8582294, -1.7159182, 3.3825500, -4.7022963, 4.5741477

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667264, upper bound: 2.7700536
time: 0.34 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7718051, upper bound: 2.7738750
time: 0.32 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.5007467, 2.7479298, -0.5095432, 2.7755899, -3.2763367, 3.2574730
1: -0.5553178, 3.7793705, -0.5611423, 3.8165379, -4.3718557, 4.3405128
2: -1.3528485, 2.6755908, -1.3674926, 2.7016737, -4.0545225, 4.0430832
3: -1.1222062, 3.3138933, -1.1338987, 3.3577619, -4.4799681, 4.4477921
4: -1.7101582, 3.3786907, -1.7360522, 3.4053361, -5.1154943, 5.1147428

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7704332
time: 0.35 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7804438
time: 0.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.26 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7667264, upper bound: 2.7700536
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7718051, upper bound: 2.7738750
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7704332
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.26
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7804438

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.2694138, 1.5103993, -0.5017872, 2.7487946, -3.0182085, 2.0121865
1: -0.3349953, 2.1516724, -0.5561673, 3.7846758, -4.1196709, 2.7078397
2: -0.8057680, 1.4441006, -1.3571187, 2.6733608, -3.4791288, 2.8012195
3: -0.7113398, 1.6636263, -1.1238993, 3.3164425, -4.0277824, 2.7875257
4: -0.8483682, 2.0235796, -1.7159182, 3.3825500, -4.2309179, 3.7394977

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
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

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521981, upper bound: 2.7537987
time: 0.35 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7494703, upper bound: 2.7541922
time: 0.34 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.3660870, 2.1625853, -0.5017872, 2.7487946, -3.1148815, 2.6643724
1: -0.4462218, 2.9898553, -0.5561673, 3.7846758, -4.2308979, 3.5460227
2: -1.0902050, 2.0946541, -1.3571187, 2.6733608, -3.7635658, 3.4517727
3: -0.9038171, 2.4661355, -1.1238993, 3.3164425, -4.2202597, 3.5900350
4: -1.2328745, 2.7640190, -1.7159182, 3.3825500, -4.6154246, 4.4799371

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7618593
time: 0.34 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7738750
time: 0.38 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.2679194, 1.6677933, -0.5095432, 2.7755899, -3.0435092, 2.1773365
1: -0.3506812, 2.3546071, -0.5611423, 3.8165379, -4.1672192, 2.9157493
2: -0.8583503, 1.6077589, -1.3674926, 2.7016737, -3.5600240, 2.9752514
3: -0.7267100, 1.8529313, -1.1338987, 3.3577619, -4.0844717, 2.9868300
4: -0.9143050, 2.2346778, -1.7360522, 3.4053361, -4.3196411, 3.9707298

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
time: 0.36 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7702911
time: 0.40 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.5095432, 2.7755899, -3.2503080, 3.1729670
1: -0.5381627, 3.6703260, -0.5611423, 3.8165379, -4.3547006, 4.2314682
2: -1.3116175, 2.5915504, -1.3674926, 2.7016737, -4.0132914, 3.9590430
3: -1.0878518, 3.1816847, -1.1338987, 3.3577619, -4.4456139, 4.3155832
4: -1.6326261, 3.2974410, -1.7360522, 3.4053361, -5.0379620, 5.0334930

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803173, upper bound: 2.7804438
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7804438
time: 0.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.27 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7521981, upper bound: 2.7537987
NS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7494703, upper bound: 2.7541922
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7618593
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7628589, upper bound: 2.7738750
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7675101, upper bound: 2.7702911
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7803173, upper bound: 2.7804438
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.27
Output dim: 0, lower bound: -2.7804438, upper bound: 2.7804438

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3660870, 2.1625853, -0.4759405, 2.6666312, -3.0327182, 2.6385257
1: -0.4462218, 2.9898553, -0.5391535, 3.6765852, -4.1228070, 3.5290089
2: -1.0902050, 2.0946541, -1.3161898, 2.5899267, -3.6801317, 3.4108438
3: -0.9038171, 2.4661355, -1.0898176, 3.1851838, -4.0890007, 3.5559530
4: -1.2328745, 2.7640190, -1.6389180, 3.3019912, -4.5348659, 4.4029369

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7470599, upper bound: 2.7607279
time: 0.36 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7626374, upper bound: 2.7682226
time: 0.34 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.5095432, 2.7755899, -2.9685907, 1.8188198
1: -0.2793456, 1.8499331, -0.5611423, 3.8165379, -4.0958834, 2.4110754
2: -0.6552293, 1.2896700, -1.3674926, 2.7016737, -3.3569031, 2.6571627
3: -0.5973468, 1.4119213, -1.1338987, 3.3577619, -3.9551086, 2.5458200
4: -0.6719720, 1.8049903, -1.7360522, 3.4053361, -4.0773082, 3.5410423

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
time: 0.36 seconds

## Relational analysis of NS_A2_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.5095432, 2.7755899, -2.9782934, 1.9343488
1: -0.2976533, 2.0236523, -0.5611423, 3.8165379, -4.1141911, 2.5847945
2: -0.7181041, 1.3894733, -1.3674926, 2.7016737, -3.4197779, 2.7569659
3: -0.6241174, 1.5397948, -1.1338987, 3.3577619, -3.9818792, 2.6736937
4: -0.7333497, 1.9524517, -1.7360522, 3.4053361, -4.1386857, 3.6885037

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7620793, upper bound: 2.7629830
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7702911
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.4453183, 2.6621878, -3.1369057, 3.1087420
1: -0.5381627, 3.6703260, -0.5362673, 3.6797814, -4.2179441, 4.2065935
2: -1.3116175, 2.5915504, -1.3237610, 2.5723453, -3.8839626, 3.9153113
3: -1.0878518, 3.1816847, -1.0783684, 3.1114309, -4.1992826, 4.2600532
4: -1.6326261, 3.2974410, -1.6110044, 3.3475635, -4.9801893, 4.9084454

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7672176
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7672176
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.4292831, 2.5398004, -3.0145183, 3.0927069
1: -0.5381627, 3.6703260, -0.5133139, 3.5072136, -4.0453763, 4.1836400
2: -1.3116175, 2.5915504, -1.2595012, 2.4519665, -3.7635841, 3.8510516
3: -1.0878518, 3.1816847, -1.0352175, 2.9516258, -4.0394773, 4.2169023
4: -1.6326261, 3.2974410, -1.5273011, 3.1979368, -4.8305626, 4.8247423

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7700858, upper bound: 2.7675101
time: 0.35 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7700858, upper bound: 2.7804438
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.27 seconds
NS_A1_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7470599, upper bound: 2.7607279
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7626374, upper bound: 2.7682226
NS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
NS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7663679, upper bound: 2.7669801
NS_A2_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7620793, upper bound: 2.7629830
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7702911
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7672176
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7672176
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7700858, upper bound: 2.7675101
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.27
Output dim: 0, lower bound: -2.7700858, upper bound: 2.7804438

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.3660870, 2.1625853, -0.3979207, 2.4372573, -2.8033442, 2.5605059
1: -0.4462218, 2.9898553, -0.4919404, 3.3703771, -3.8165989, 3.4817958
2: -1.0902050, 2.0946541, -1.2087388, 2.3540077, -3.4442127, 3.3033929
3: -0.9038171, 2.4661355, -0.9920961, 2.8034899, -3.7073069, 3.4582314
4: -1.2328745, 2.7640190, -1.4356111, 3.0962472, -4.3291216, 4.1996303

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7601399, upper bound: 2.7606044
time: 0.34 seconds

## Relational analysis of NS_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535570, upper bound: 2.7594770
time: 0.34 seconds

## BFS NS instance: NS_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.3917128, 2.3017175, -2.4947183, 1.7009892
1: -0.2793456, 1.8499331, -0.4645773, 3.1410573, -3.4204028, 2.3145103
2: -0.6552293, 1.2896700, -1.0995426, 2.2633626, -2.9185920, 2.3892126
3: -0.5973468, 1.4119213, -0.9485424, 2.7034569, -3.3008037, 2.3604636
4: -0.6719720, 1.8049903, -1.3363373, 2.8527780, -3.5247500, 3.1413276

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_A1_B1_B1

### Relational analysis result of NS_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7642129
time: 0.37 seconds

## Relational analysis of NS_A2_A1_A1_B1_B2

### Relational analysis result of NS_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7662691
time: 0.37 seconds

## BFS NS instance: NS_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.4442576, 2.5300765, -2.7230773, 1.7535341
1: -0.2793456, 1.8499331, -0.5091101, 3.4816647, -3.7610102, 2.3590431
2: -0.6552293, 1.2896700, -1.2315428, 2.4600618, -3.1152911, 2.5212128
3: -0.5973468, 1.4119213, -1.0322413, 3.0134051, -3.6107519, 2.4441626
4: -0.6719720, 1.8049903, -1.5232764, 3.1220119, -3.7939839, 3.3282666

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7642129
time: 0.32 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7662691
time: 0.37 seconds

## BFS NS instance: NS_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.4956162, 2.7313313, -2.9340348, 1.9204218
1: -0.2976533, 2.0236523, -0.5523267, 3.7618747, -4.0595279, 2.5759790
2: -0.7181041, 1.3894733, -1.3483021, 2.6578324, -3.3759365, 2.7377753
3: -0.6241174, 1.5397948, -1.1161582, 3.2839887, -3.9081061, 2.6559529
4: -0.7333497, 1.9524517, -1.6967287, 3.3673153, -4.1006651, 3.6491804

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7585579, upper bound: 2.7585579
time: 0.35 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7577765, upper bound: 2.7702911
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.2758048, 1.7312763, -2.2059941, 2.9392285
1: -0.5381627, 3.6703260, -0.3635933, 2.4716668, -3.0098295, 4.0339193
2: -1.3116175, 2.5915504, -0.9028287, 1.6458170, -2.9574347, 3.4943790
3: -1.0878518, 3.1816847, -0.7530521, 1.9307290, -3.0185809, 3.9347367
4: -1.6326261, 3.2974410, -0.9503052, 2.3278046, -3.9604306, 4.2477465

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7645539, upper bound: 2.7663238
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7668046
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.4250804, 2.5852566, -3.0599747, 3.0885043
1: -0.5381627, 3.6703260, -0.5211089, 3.5761673, -4.1143303, 4.1914349
2: -1.3116175, 2.5915504, -1.2865007, 2.4982405, -3.8098578, 3.8780510
3: -1.0878518, 3.1816847, -1.0481679, 3.0023003, -4.0901518, 4.2298527
4: -1.6326261, 3.2974410, -1.5452410, 3.2716055, -4.9042315, 4.8426819

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7645539, upper bound: 2.7801972
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7767273
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.2433186, 1.5781031, -2.0528212, 2.9067423
1: -0.5381627, 3.6703260, -0.3324639, 2.2505271, -2.7886899, 4.0027900
2: -1.3116175, 2.5915504, -0.8196230, 1.5147862, -2.8264036, 3.4111733
3: -1.0878518, 3.1816847, -0.6908419, 1.7496001, -2.8374519, 3.8725266
4: -1.6326261, 3.2974410, -0.8599036, 2.1438165, -3.7764425, 4.1573448

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7646859
time: 0.33 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7699478, upper bound: 2.7675101
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.4047078, 2.4600189, -2.9347367, 3.0681317
1: -0.5381627, 3.6703260, -0.4967405, 3.4003437, -3.9385064, 4.1670666
2: -1.3116175, 2.5915504, -1.2186558, 2.3767066, -3.6883240, 3.8102062
3: -1.0878518, 3.1816847, -1.0018195, 2.8326006, -3.9204524, 4.1835041
4: -1.6326261, 3.2974410, -1.4532027, 3.1177409, -4.7503672, 4.7506437

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7801972
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7699478, upper bound: 2.7801145
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.37 seconds
NS_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7601399, upper bound: 2.7606044
NS_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7535570, upper bound: 2.7594770
NS_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7642129
NS_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7662691
NS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7642129
NS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7662691
NS_A2_A1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7585579, upper bound: 2.7585579
NS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7577765, upper bound: 2.7702911
NS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7645539, upper bound: 2.7663238
NS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7668046
NS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7645539, upper bound: 2.7801972
NS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7767273
NS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7646859
NS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7699478, upper bound: 2.7675101
NS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7801972
NS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.7699478, upper bound: 2.7801145

## BFS NS instance: NS_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.3270343, 2.1658700, -2.3588707, 1.6363108
1: -0.2793456, 1.8499331, -0.4350435, 2.9883678, -3.2677133, 2.2849765
2: -0.6552293, 1.2896700, -1.0483774, 2.1126413, -2.7678707, 2.3380475
3: -0.5973468, 1.4119213, -0.8804526, 2.4416454, -3.0389922, 2.2923739
4: -0.6719720, 1.8049903, -1.1933124, 2.7831554, -3.4551275, 2.9983027

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7653318, upper bound: 2.7646799
time: 0.37 seconds

## Relational analysis of NS_A2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7653318, upper bound: 2.7664323
time: 0.36 seconds

## BFS NS instance: NS_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.3234184, 2.0675738, -2.2605746, 1.6326950
1: -0.2793456, 1.8499331, -0.4215800, 2.8456964, -3.1250420, 2.2715130
2: -0.6552293, 1.2896700, -1.0000257, 2.0370378, -2.6922672, 2.2896957
3: -0.5973468, 1.4119213, -0.8567761, 2.3449783, -2.9423251, 2.2686973
4: -0.6719720, 1.8049903, -1.1425889, 2.6601443, -3.3321164, 2.9475791

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534942, upper bound: 2.7534942
time: 0.36 seconds

## Relational analysis of NS_A2_A1_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7534942, upper bound: 2.7664323
time: 0.38 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.3958364, 2.4383278, -2.6313286, 1.7051128
1: -0.2793456, 1.8499331, -0.4911292, 3.3656516, -3.6449971, 2.3410623
2: -0.6552293, 1.2896700, -1.2006791, 2.3622475, -3.0174768, 2.4903491
3: -0.5973468, 1.4119213, -0.9917954, 2.8072424, -3.4045892, 2.4037166
4: -0.6719720, 1.8049903, -1.4238141, 3.0826614, -3.7546334, 3.2288043

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7631274
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7642129
time: 0.38 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.3647310, 2.2999167, -2.4929175, 1.6740074
1: -0.2793456, 1.8499331, -0.4616175, 3.1788001, -3.4581456, 2.3115506
2: -0.6552293, 1.2896700, -1.1243796, 2.2282844, -2.8835137, 2.4140496
3: -0.5973468, 1.4119213, -0.9345922, 2.6306338, -3.2279806, 2.3465135
4: -0.6719720, 1.8049903, -1.3199248, 2.9210746, -3.5930467, 3.1249151

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7645539
time: 0.38 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7662691
time: 0.37 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.4697683, 2.6504877, -2.8531911, 1.8945739
1: -0.2976533, 2.0236523, -0.5353363, 3.6540415, -3.9516950, 2.5589886
2: -0.7181041, 1.3894733, -1.3073822, 2.5746906, -3.2927947, 2.6968555
3: -0.6241174, 1.5397948, -1.0820575, 3.1532676, -3.7773850, 2.6218524
4: -0.7333497, 1.9524517, -1.6197194, 3.2872617, -4.0206113, 3.5721712

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7511064, upper bound: 2.7672198
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7583364, upper bound: 2.7699478
time: 0.35 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.1860043, 1.3466570, -1.8213750, 2.8494282
1: -0.5381627, 3.6703260, -0.2826974, 1.9239516, -2.4621143, 3.9530234
2: -1.3116175, 2.5915504, -0.6801631, 1.3064035, -2.6180210, 3.2717135
3: -1.0878518, 3.1816847, -0.5949054, 1.4550583, -2.5429101, 3.7765901
4: -1.6326261, 3.2974410, -0.6886064, 1.8628993, -3.4955254, 3.9860473

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7663238
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7663238
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.2265765, 1.5162649, -1.9909829, 2.8900003
1: -0.5381627, 3.6703260, -0.3197167, 2.1686816, -2.7068443, 3.9900427
2: -1.3116175, 2.5915504, -0.7828116, 1.4565961, -2.7682137, 3.3743620
3: -1.0878518, 3.1816847, -0.6725806, 1.6489701, -2.7368219, 3.8542652
4: -1.6326261, 3.2974410, -0.7932428, 2.0768647, -3.7094908, 4.0906839

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7574583
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7668046
time: 0.35 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.3076048, 2.0884678, -2.5631857, 2.9710286
1: -0.5381627, 3.6703260, -0.4196330, 2.8851950, -3.4233577, 4.0899591
2: -1.3116175, 2.5915504, -1.0115439, 2.0395269, -3.3511443, 3.6030941
3: -1.0878518, 3.1816847, -0.8498347, 2.3376288, -3.4254806, 4.0315194
4: -1.6326261, 3.2974410, -1.1316509, 2.7040546, -4.3366804, 4.4290919

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7801972
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.3753327, 2.3569636, -2.8316817, 3.0387564
1: -0.5381627, 3.6703260, -0.4759032, 3.2600689, -3.7982316, 4.1462293
2: -1.3116175, 2.5915504, -1.1634135, 2.2862463, -3.5978637, 3.7549639
3: -1.0878518, 3.1816847, -0.9616026, 2.6963146, -3.7841663, 4.1432872
4: -1.6326261, 3.2974410, -1.3583037, 3.0054598, -4.6380858, 4.6557446

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.1731156, 1.2429918, -1.7177098, 2.8365393
1: -0.5381627, 3.6703260, -0.2653253, 1.7709899, -2.3091526, 3.9356513
2: -1.3116175, 2.5915504, -0.6234457, 1.2220012, -2.5336187, 3.2149961
3: -1.0878518, 3.1816847, -0.5676874, 1.3276134, -2.4154651, 3.7493720
4: -1.6326261, 3.2974410, -0.6302408, 1.7394276, -3.3720536, 3.9276819

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7646859
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7646859
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.1845304, 1.3606682, -1.8353862, 2.8479543
1: -0.5381627, 3.6703260, -0.2824830, 1.9470716, -2.4852343, 3.9528089
2: -1.3116175, 2.5915504, -0.6872041, 1.3180709, -2.6296883, 3.2787545
3: -1.0878518, 3.1816847, -0.5939487, 1.4622965, -2.5501482, 3.7756333
4: -1.6326261, 3.2974410, -0.6902229, 1.8839808, -3.5166068, 3.9876637

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7692422, upper bound: 2.7675101
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7692422, upper bound: 2.7675101
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.3022718, 1.9925733, -2.4672914, 2.9656956
1: -0.5381627, 3.6703260, -0.4058715, 2.7465084, -3.2846711, 4.0761976
2: -1.3116175, 2.5915504, -0.9626346, 1.9593803, -3.2709978, 3.5541849
3: -1.0878518, 3.1816847, -0.8254330, 2.2351263, -3.3229780, 4.0071177
4: -1.6326261, 3.2974410, -1.0756483, 2.5819404, -4.2145662, 4.3730893

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7556823
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7801972
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.4747180, 2.6634238, -0.3389214, 2.2148643, -2.6895823, 3.0023451
1: -0.5381627, 3.6703260, -0.4439855, 3.0646944, -3.6028571, 4.1143112
2: -1.3116175, 2.5915504, -1.0810840, 2.1477108, -3.4593282, 3.6726344
3: -1.0878518, 3.1816847, -0.8988513, 2.5088573, -3.5967090, 4.0805359
4: -1.6326261, 3.2974410, -1.2414910, 2.8358724, -4.4684982, 4.5389318

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664013
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145
time: 0.40 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.42 seconds
NS_A2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7653318, upper bound: 2.7646799
NS_A2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7653318, upper bound: 2.7664323
NS_A2_A1_A1_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7534942, upper bound: 2.7534942
NS_A2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7534942, upper bound: 2.7664323
NS_A2_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7631274
NS_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7556823, upper bound: 2.7642129
NS_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7645539
NS_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7646860, upper bound: 2.7662691
NS_A2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7511064, upper bound: 2.7672198
NS_A2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7583364, upper bound: 2.7699478
NS_A2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7663238
NS_A2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7619114, upper bound: 2.7663238
NS_A2_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7574583
NS_A2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7668046
NS_A2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
NS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7801972
NS_A2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
NS_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
NS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7646859
NS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7662691, upper bound: 2.7646859
NS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7692422, upper bound: 2.7675101
NS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7692422, upper bound: 2.7675101
NS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7556823
NS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7801972
NS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664013
NS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.42
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145

## BFS NS instance: NS_A2_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1812261, 1.3317630, -0.3270343, 2.1658700, -2.3470960, 1.6587973
1: -0.2787997, 1.9029334, -0.4350435, 2.9883678, -3.2671676, 2.3379769
2: -0.6704855, 1.2932925, -1.0483774, 2.1126413, -2.7831268, 2.3416700
3: -0.5872393, 1.4343947, -0.8804526, 2.4416454, -3.0288846, 2.3148475
4: -0.6767123, 1.8457990, -1.1933124, 2.7831554, -3.4598677, 3.0391114

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7540983
time: 0.36 seconds

## Relational analysis of NS_A2_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7663743
time: 0.34 seconds

## BFS NS instance: NS_A2_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3270343, 2.1658700, -2.3329027, 1.5527585
1: -0.2606599, 1.7469907, -0.4350435, 2.9883678, -3.2490277, 2.1820340
2: -0.6121430, 1.2063824, -1.0483774, 2.1126413, -2.7247844, 2.2547598
3: -0.5581758, 1.3027804, -0.8804526, 2.4416454, -2.9998212, 2.1832330
4: -0.6156360, 1.7192602, -1.1933124, 2.7831554, -3.3987913, 2.9125726

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7551838
time: 0.35 seconds

## Relational analysis of NS_A2_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7674856
time: 0.33 seconds

## BFS NS instance: NS_A2_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1930007, 1.3092765, -0.3022718, 1.9925733, -2.1855741, 1.6115482
1: -0.2793456, 1.8499331, -0.4058715, 2.7465084, -3.0258539, 2.2558045
2: -0.6552293, 1.2896700, -0.9626346, 1.9593803, -2.6146097, 2.2523046
3: -0.5973468, 1.4119213, -0.8254330, 2.2351263, -2.8324730, 2.2373543
4: -0.6719720, 1.8049903, -1.0756483, 2.5819404, -3.2539124, 2.8806386

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7646799
time: 0.38 seconds

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7664323
time: 0.35 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3958364, 2.4383278, -2.6053605, 1.6215606
1: -0.2606599, 1.7469907, -0.4911292, 3.3656516, -3.6263115, 2.2381198
2: -0.6121430, 1.2063824, -1.2006791, 2.3622475, -2.9743905, 2.4070616
3: -0.5581758, 1.3027804, -0.9917954, 2.8072424, -3.3654182, 2.2945757
4: -0.6156360, 1.7192602, -1.4238141, 3.0826614, -3.6982975, 3.1430743

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7552702
time: 0.36 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7642129
time: 0.34 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1812261, 1.3317630, -0.3647310, 2.2999167, -2.4811428, 1.6964940
1: -0.2787997, 1.9029334, -0.4616175, 3.1788001, -3.4575996, 2.3645508
2: -0.6704855, 1.2932925, -1.1243796, 2.2282844, -2.8987699, 2.4176722
3: -0.5872393, 1.4343947, -0.9345922, 2.6306338, -3.2178731, 2.3689868
4: -0.6767123, 1.8457990, -1.3199248, 2.9210746, -3.5977869, 3.1657238

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7542386
time: 0.37 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7645539
time: 0.35 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3647310, 2.2999167, -2.4669495, 1.5904552
1: -0.2606599, 1.7469907, -0.4616175, 3.1788001, -3.4394600, 2.2086082
2: -0.6121430, 1.2063824, -1.1243796, 2.2282844, -2.8404274, 2.3307619
3: -0.5581758, 1.3027804, -0.9345922, 2.6306338, -3.1888096, 2.2373726
4: -0.6156360, 1.7192602, -1.3199248, 2.9210746, -3.5367107, 3.0391850

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7553241
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7662691
time: 0.35 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.4137641, 2.5419936, -2.7446971, 1.8385698
1: -0.2976533, 2.0236523, -0.5127166, 3.5182648, -3.8159180, 2.5363688
2: -0.7181041, 1.3894733, -1.2668126, 2.4548297, -3.1729338, 2.6562858
3: -0.6241174, 1.5397948, -1.0314655, 2.9391799, -3.5632973, 2.5712605
4: -0.7333497, 1.9524517, -1.5080035, 3.2306254, -3.9639750, 3.4604552

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7574954, upper bound: 2.7672198
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7574954, upper bound: 2.7672198
time: 0.38 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.3928633, 2.4223971, -2.6251006, 1.8176689
1: -0.2976533, 2.0236523, -0.4885578, 3.3497157, -3.6473689, 2.5122101
2: -0.7181041, 1.3894733, -1.2008266, 2.3411322, -3.0592363, 2.5903001
3: -0.6241174, 1.5397948, -0.9852172, 2.7804747, -3.4045920, 2.5250120
4: -0.7333497, 1.9524517, -1.4205528, 3.0822139, -3.8155637, 3.3730044

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7692422
time: 0.37 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7699478
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3570149, 2.1824479, -0.1860043, 1.3466570, -1.7036719, 2.3684523
1: -0.4407536, 2.9849753, -0.2826974, 1.9239516, -2.3647053, 3.2676728
2: -1.0423620, 2.1470139, -0.6801631, 1.3064035, -2.3487654, 2.8271770
3: -0.8984227, 2.5226510, -0.5949054, 1.4550583, -2.3534811, 3.1175563
4: -1.2236698, 2.7386091, -0.6886064, 1.8628993, -3.0865691, 3.4272156

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7573718
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7663238
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4083217, 2.4110599, -0.1860043, 1.3466570, -1.7549788, 2.5970643
1: -0.4853213, 3.3257933, -0.2826974, 1.9239516, -2.4092729, 3.6084907
2: -1.1735004, 2.3448300, -0.6801631, 1.3064035, -2.4799039, 3.0249932
3: -0.9847659, 2.8284349, -0.5949054, 1.4550583, -2.4398241, 3.4233403
4: -1.4156859, 3.0070877, -0.6886064, 1.8628993, -3.2785852, 3.6956940

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7573718
time: 0.34 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7663238
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.2265765, 1.5162649, -1.9132075, 2.6605561
1: -0.4911667, 3.3645411, -0.3197167, 2.1686816, -2.6598482, 3.6842577
2: -1.2047796, 2.3536205, -0.7828116, 1.4565961, -2.6613758, 3.1364322
3: -0.9906334, 2.7955449, -0.6725806, 1.6489701, -2.6396036, 3.4681253
4: -1.4297944, 3.0918128, -0.7932428, 2.0768647, -3.5066590, 3.8850555

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7668046
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7664635
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4169641, 2.5576580, -0.3076048, 2.0884678, -2.5054319, 2.8652627
1: -0.5152790, 3.5381331, -0.4196330, 2.8851950, -3.4004741, 3.9577661
2: -1.2717404, 2.4722149, -1.0115439, 2.0395269, -3.3112674, 3.4837589
3: -1.0365596, 2.9613614, -0.8498347, 2.3376288, -3.3741884, 3.8111961
4: -1.5201240, 3.2440412, -1.1316509, 2.7040546, -4.2241783, 4.3756924

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.3076048, 2.0884678, -2.4854105, 2.7415843
1: -0.4911667, 3.3645411, -0.4196330, 2.8851950, -3.3763616, 3.7841740
2: -1.2047796, 2.3536205, -1.0115439, 2.0395269, -3.2443066, 3.3651643
3: -0.9906334, 2.7955449, -0.8498347, 2.3376288, -3.3282623, 3.6453795
4: -1.4297944, 3.0918128, -1.1316509, 2.7040546, -4.1338491, 4.2234640

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7766884
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7785847
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4169641, 2.5576580, -0.3753327, 2.3569636, -2.7739277, 2.9329906
1: -0.5152790, 3.5381331, -0.4759032, 3.2600689, -3.7753479, 4.0140362
2: -1.2717404, 2.4722149, -1.1634135, 2.2862463, -3.5579867, 3.6356285
3: -1.0365596, 2.9613614, -0.9616026, 2.6963146, -3.7328742, 3.9229641
4: -1.5201240, 3.2440412, -1.3583037, 3.0054598, -4.5255837, 4.6023450

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.3753327, 2.3569636, -2.7539062, 2.8093123
1: -0.4911667, 3.3645411, -0.4759032, 3.2600689, -3.7512355, 3.8404441
2: -1.2047796, 2.3536205, -1.1634135, 2.2862463, -3.4910259, 3.5170341
3: -0.9906334, 2.7955449, -0.9616026, 2.6963146, -3.6869478, 3.7571473
4: -1.4297944, 3.0918128, -1.3583037, 3.0054598, -4.4352541, 4.4501166

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
time: 0.35 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3570149, 2.1824479, -0.1731156, 1.2429918, -1.6000067, 2.3555634
1: -0.4407536, 2.9849753, -0.2653253, 1.7709899, -2.2117436, 3.2503006
2: -1.0423620, 2.1470139, -0.6234457, 1.2220012, -2.2643633, 2.7704597
3: -0.8984227, 2.5226510, -0.5676874, 1.3276134, -2.2260361, 3.0903382
4: -1.2236698, 2.7386091, -0.6302408, 1.7394276, -2.9630973, 3.3688498

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7556823
time: 0.35 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7646859
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4083217, 2.4110599, -0.1731156, 1.2429918, -1.6513135, 2.5841753
1: -0.4853213, 3.3257933, -0.2653253, 1.7709899, -2.2563112, 3.5911186
2: -1.1735004, 2.3448300, -0.6234457, 1.2220012, -2.3955016, 2.9682758
3: -0.9847659, 2.8284349, -0.5676874, 1.3276134, -2.3123794, 3.3961225
4: -1.4156859, 3.0070877, -0.6302408, 1.7394276, -3.1551135, 3.6373286

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7556823
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7646859
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3570149, 2.1824479, -0.1845304, 1.3606682, -1.7176831, 2.3669782
1: -0.4407536, 2.9849753, -0.2824830, 1.9470716, -2.3878253, 3.2674584
2: -1.0423620, 2.1470139, -0.6872041, 1.3180709, -2.3604329, 2.8342180
3: -0.8984227, 2.5226510, -0.5939487, 1.4622965, -2.3607192, 3.1165996
4: -1.2236698, 2.7386091, -0.6902229, 1.8839808, -3.1076505, 3.4288321

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7575121
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7675101
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4083217, 2.4110599, -0.1845304, 1.3606682, -1.7689899, 2.5955901
1: -0.4853213, 3.3257933, -0.2824830, 1.9470716, -2.4323928, 3.6082764
2: -1.1735004, 2.3448300, -0.6872041, 1.3180709, -2.4915714, 3.0320342
3: -0.9847659, 2.8284349, -0.5939487, 1.4622965, -2.4470625, 3.4223838
4: -1.4156859, 3.0070877, -0.6902229, 1.8839808, -3.2996666, 3.6973104

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7556823
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7665755
time: 0.46 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4169641, 2.5576580, -0.3022718, 1.9925733, -2.4095373, 2.8599298
1: -0.5152790, 3.5381331, -0.4058715, 2.7465084, -3.2617874, 3.9440045
2: -1.2717404, 2.4722149, -0.9626346, 1.9593803, -3.2311206, 3.4348495
3: -1.0365596, 2.9613614, -0.8254330, 2.2351263, -3.2716858, 3.7867944
4: -1.5201240, 3.2440412, -1.0756483, 2.5819404, -4.1020641, 4.3196898

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.3022718, 1.9925733, -2.3895159, 2.7362514
1: -0.4911667, 3.3645411, -0.4058715, 2.7465084, -3.2376750, 3.7704124
2: -1.2047796, 2.3536205, -0.9626346, 1.9593803, -3.1641598, 3.3162551
3: -0.9906334, 2.7955449, -0.8254330, 2.2351263, -3.2257595, 3.6209779
4: -1.4297944, 3.0918128, -1.0756483, 2.5819404, -4.0117350, 4.1674614

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7767387
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7785847
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4169641, 2.5576580, -0.3389214, 2.2148643, -2.6318283, 2.8965793
1: -0.5152790, 3.5381331, -0.4439855, 3.0646944, -3.5799735, 3.9821186
2: -1.2717404, 2.4722149, -1.0810840, 2.1477108, -3.4194512, 3.5532990
3: -1.0365596, 2.9613614, -0.8988513, 2.5088573, -3.5454168, 3.8602128
4: -1.5201240, 3.2440412, -1.2414910, 2.8358724, -4.3559961, 4.4855323

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664013
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.3389214, 2.2148643, -2.6118069, 2.7729011
1: -0.4911667, 3.3645411, -0.4439855, 3.0646944, -3.5558610, 3.8085265
2: -1.2047796, 2.3536205, -1.0810840, 2.1477108, -3.3524904, 3.4347045
3: -0.9906334, 2.7955449, -0.8988513, 2.5088573, -3.4994907, 3.6943960
4: -1.4297944, 3.0918128, -1.2414910, 2.8358724, -4.2656670, 4.3333039

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145
time: 0.38 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.46 seconds
NS_A2_A1_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7540983
NS_A2_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7663743
NS_A2_A1_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7551838
NS_A2_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7674856
NS_A2_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7646799
NS_A2_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7524087, upper bound: 2.7664323
NS_A2_A1_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7552702
NS_A2_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7642129
NS_A2_A1_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7542386
NS_A2_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7645539
NS_A2_A1_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7553241
NS_A2_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7662691
NS_A2_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7574954, upper bound: 2.7672198
NS_A2_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7574954, upper bound: 2.7672198
NS_A2_A1_A2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7692422
NS_A2_A1_A2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7674908, upper bound: 2.7699478
NS_A2_A2_B1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7573718
NS_A2_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7663238
NS_A2_A2_B1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7573718
NS_A2_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7663238
NS_A2_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7668046
NS_A2_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7664635
NS_A2_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
NS_A2_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
NS_A2_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7766884
NS_A2_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7785847
NS_A2_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
NS_A2_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664009
NS_A2_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
NS_A2_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7767273
NS_A2_A2_B2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7556823
NS_A2_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7646859
NS_A2_A2_B2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7556823
NS_A2_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7646859
NS_A2_A2_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7575121
NS_A2_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7675101
NS_A2_A2_B2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7556823
NS_A2_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7600119, upper bound: 2.7665755
NS_A2_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
NS_A2_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7720995, upper bound: 2.7664014
NS_A2_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7767387
NS_A2_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7667468, upper bound: 2.7785847
NS_A2_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664014
NS_A2_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7664013
NS_A2_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145
NS_A2_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 0, lower bound: -2.7664009, upper bound: 2.7801145

## BFS NS instance: NS_A2_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1812261, 1.3317630, -0.3076048, 2.0884678, -2.2696939, 1.6393678
1: -0.2787997, 1.9029334, -0.4196330, 2.8851950, -3.1639948, 2.3225665
2: -0.6704855, 1.2932925, -1.0115439, 2.0395269, -2.7100124, 2.3048363
3: -0.5872393, 1.4343947, -0.8498347, 2.3376288, -2.9248681, 2.2842293
4: -0.6767123, 1.8457990, -1.1316509, 2.7040546, -3.3807669, 2.9774499

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
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

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_A1_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3076048, 2.0884678, -2.2555006, 1.5333290
1: -0.2606599, 1.7469907, -0.4196330, 2.8851950, -3.1458549, 2.1666236
2: -0.6121430, 1.2063824, -1.0115439, 2.0395269, -2.6516700, 2.2179263
3: -0.5581758, 1.3027804, -0.8498347, 2.3376288, -2.8958046, 2.1526151
4: -0.6156360, 1.7192602, -1.1316509, 2.7040546, -3.3196907, 2.8509111

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1812261, 1.3317630, -0.3022718, 1.9925733, -2.1737993, 1.6340349
1: -0.2787997, 1.9029334, -0.4058715, 2.7465084, -3.0253081, 2.3088050
2: -0.6704855, 1.2932925, -0.9626346, 1.9593803, -2.6298656, 2.2559271
3: -0.5872393, 1.4343947, -0.8254330, 2.2351263, -2.8223655, 2.2598276
4: -0.6767123, 1.8457990, -1.0756483, 2.5819404, -3.2586527, 2.9214473

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7652067, upper bound: 2.7646799
time: 0.37 seconds

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7572198, upper bound: 2.7547890
time: 0.36 seconds

## BFS NS instance: NS_A2_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3022718, 1.9925733, -2.1596060, 1.5279961
1: -0.2606599, 1.7469907, -0.4058715, 2.7465084, -3.0071683, 2.1528621
2: -0.6121430, 1.2063824, -0.9626346, 1.9593803, -2.5715232, 2.1690168
3: -0.5581758, 1.3027804, -0.8254330, 2.2351263, -2.7933021, 2.1282134
4: -0.6156360, 1.7192602, -1.0756483, 2.5819404, -3.1975765, 2.7949085

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7652067, upper bound: 2.7664323
time: 0.35 seconds

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7572198, upper bound: 2.7558745
time: 0.37 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3753327, 2.3569636, -2.5239964, 1.6010569
1: -0.2606599, 1.7469907, -0.4759032, 3.2600689, -3.5207288, 2.2228937
2: -0.6121430, 1.2063824, -1.1634135, 2.2862463, -2.8983893, 2.3697958
3: -0.5581758, 1.3027804, -0.9616026, 2.6963146, -3.2544904, 2.2643828
4: -0.6156360, 1.7192602, -1.3583037, 3.0054598, -3.6210957, 3.0775638

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7474518, upper bound: 2.7642129
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7573026
time: 0.41 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1812261, 1.3317630, -0.3389214, 2.2148643, -2.3960903, 1.6706845
1: -0.2787997, 1.9029334, -0.4439855, 3.0646944, -3.3434939, 2.3469188
2: -0.6704855, 1.2932925, -1.0810840, 2.1477108, -2.8181963, 2.3743765
3: -0.5872393, 1.4343947, -0.8988513, 2.5088573, -3.0960965, 2.3332460
4: -0.6767123, 1.8457990, -1.2414910, 2.8358724, -3.5125847, 3.0872898

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7581899, upper bound: 2.7645539
time: 0.37 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7556474, upper bound: 2.7553128
time: 0.38 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3389214, 2.2148643, -2.3818970, 1.5646456
1: -0.2606599, 1.7469907, -0.4439855, 3.0646944, -3.3253543, 2.1909761
2: -0.6121430, 1.2063824, -1.0810840, 2.1477108, -2.7598538, 2.2874665
3: -0.5581758, 1.3027804, -0.8988513, 2.5088573, -3.0670331, 2.2016315
4: -0.6156360, 1.7192602, -1.2414910, 2.8358724, -3.4515085, 2.9607511

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7477324, upper bound: 2.7662691
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7494220, upper bound: 2.7563983
time: 0.41 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.3020771, 2.0642514, -2.2669549, 1.7268827
1: -0.2976533, 2.0236523, -0.4143897, 2.8539455, -3.1515989, 2.4380419
2: -0.7181041, 1.3894733, -1.0009217, 2.0149243, -2.7330284, 2.3903952
3: -0.6241174, 1.5397948, -0.8396599, 2.3050339, -2.9291513, 2.3794546
4: -0.7333497, 1.9524517, -1.1124036, 2.6775739, -3.4109235, 3.0648553

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534307, upper bound: 2.7582079
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533198, upper bound: 2.7578819
time: 0.39 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.3640199, 2.3124781, -2.5151815, 1.7888255
1: -0.2976533, 2.0236523, -0.4675400, 3.2033932, -3.5010467, 2.4911923
2: -0.7181041, 1.3894733, -1.1439898, 2.2436988, -2.9618030, 2.5334630
3: -0.6241174, 1.5397948, -0.9452005, 2.6351333, -3.2592506, 2.4849954
4: -0.7333497, 1.9524517, -1.3219600, 2.9653451, -3.6986947, 3.2744117

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534307, upper bound: 2.7582079
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B1_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533198, upper bound: 2.7578819
time: 0.40 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.2970527, 1.9705740, -2.1732776, 1.7218584
1: -0.2976533, 2.0236523, -0.4009827, 2.7181237, -3.0157771, 2.4246349
2: -0.7181041, 1.3894733, -0.9531471, 1.9358556, -2.6539598, 2.3426204
3: -0.6241174, 1.5397948, -0.8158574, 2.2048943, -2.8290117, 2.3556523
4: -0.7333497, 1.9524517, -1.0591745, 2.5576968, -3.2910466, 3.0116262

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7512427, upper bound: 2.7596463
time: 0.46 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7583897, upper bound: 2.7608972
time: 0.37 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2027035, 1.4248056, -0.3282322, 2.1751082, -2.3778117, 1.7530379
1: -0.2976533, 2.0236523, -0.4355537, 3.0112066, -3.3088598, 2.4592061
2: -0.7181041, 1.3894733, -1.0623658, 2.1097383, -2.8278425, 2.4518390
3: -0.6241174, 1.5397948, -0.8816080, 2.4524479, -3.0765653, 2.4214029
4: -0.7333497, 1.9524517, -1.2070264, 2.7982173, -3.5315671, 3.1594782

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7623244, upper bound: 2.7596463
time: 0.41 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_B2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7583897, upper bound: 2.7608972
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.1860043, 1.3466570, -1.6426914, 2.1549425
1: -0.4008421, 2.7146063, -0.2826974, 1.9239516, -2.3247938, 2.9973037
2: -0.9505634, 1.9361818, -0.6801631, 1.3064035, -2.2569671, 2.6163449
3: -0.8154755, 2.2027259, -0.5949054, 1.4550583, -2.2705338, 2.7976313
4: -1.0562243, 2.5568314, -0.6886064, 1.8628993, -2.9191236, 3.2454376

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7594008, upper bound: 2.7631652
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7598854, upper bound: 2.7590865
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.1860043, 1.3466570, -1.6780013, 2.3740208
1: -0.4382458, 3.0278041, -0.2826974, 1.9239516, -2.3621974, 3.3105016
2: -1.0667892, 2.1236391, -0.6801631, 1.3064035, -2.3731928, 2.8038023
3: -0.8872414, 2.4708958, -0.5949054, 1.4550583, -2.3422997, 3.0658011
4: -1.2173010, 2.8092000, -0.6886064, 1.8628993, -3.0802002, 3.4978065

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7580684, upper bound: 2.7622837
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B1_B1_B1_A2_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577393, upper bound: 2.7581125
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.2265765, 1.5162649, -1.8122993, 2.1955147
1: -0.4008421, 2.7146063, -0.3197167, 2.1686816, -2.5695238, 3.0343230
2: -0.9505634, 1.9361818, -0.7828116, 1.4565961, -2.4071596, 2.7189934
3: -0.8154755, 2.2027259, -0.6725806, 1.6489701, -2.4644456, 2.8753066
4: -1.0562243, 2.5568314, -0.7932428, 2.0768647, -3.1330891, 3.3500743

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7593531, upper bound: 2.7623229
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7597378, upper bound: 2.7582820
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.2265765, 1.5162649, -1.8476092, 2.4145930
1: -0.4382458, 3.0278041, -0.3197167, 2.1686816, -2.6069274, 3.3475208
2: -1.0667892, 2.1236391, -0.7828116, 1.4565961, -2.5233853, 2.9064507
3: -0.8872414, 2.4708958, -0.6725806, 1.6489701, -2.5362115, 3.1434765
4: -1.2173010, 2.8092000, -0.7932428, 2.0768647, -3.2941656, 3.6024427

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7593531, upper bound: 2.7622837
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7597378, upper bound: 2.7581125
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.3031371, 2.0717506, -0.3076048, 2.0884678, -2.3916049, 2.3793554
1: -0.4158132, 2.8628929, -0.4196330, 2.8851950, -3.3010082, 3.2825260
2: -1.0028709, 2.0236433, -1.0115439, 2.0395269, -3.0423980, 3.0351872
3: -0.8422414, 2.3150473, -0.8498347, 2.3376288, -3.1798701, 3.1648819
4: -1.1176394, 2.6857204, -1.1316509, 2.7040546, -3.8216939, 3.8173714

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562183, upper bound: 2.7550932
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7584490, upper bound: 2.7550857
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.3076048, 2.0884678, -2.4550650, 2.6332493
1: -0.4696044, 3.2193704, -0.4196330, 2.8851950, -3.3547993, 3.6390033
2: -1.1476570, 2.2576227, -1.0115439, 2.0395269, -3.1871839, 3.2691665
3: -0.9491717, 2.6523957, -0.8498347, 2.3376288, -3.2868004, 3.5022304
4: -1.3314362, 2.9754171, -1.1316509, 2.7040546, -4.0354910, 4.1070681

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667467, upper bound: 2.7660594
time: 0.35 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.2973586, 2.0464444, -2.4433870, 2.7313380
1: -0.4911667, 3.3645411, -0.4104910, 2.8293266, -3.3204932, 3.7750320
2: -1.2047796, 2.3536205, -0.9908106, 1.9982010, -3.2029805, 3.3444312
3: -0.9906334, 2.7955449, -0.8320768, 2.2798955, -3.2705288, 3.6276217
4: -1.4297944, 3.0918128, -1.0959523, 2.6567075, -4.0865021, 4.1877651

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7528209, upper bound: 2.7507118
time: 0.44 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.3086333, 2.0736451, -2.4705877, 2.7426128
1: -0.4911667, 3.3645411, -0.4168339, 2.8735497, -3.3647163, 3.7813749
2: -1.2047796, 2.3536205, -1.0126864, 2.0177553, -3.2225349, 3.3663068
3: -0.9906334, 2.7955449, -0.8451283, 2.3225791, -3.3132124, 3.6406732
4: -1.4297944, 3.0918128, -1.1221292, 2.6829336, -4.1127281, 4.2139421

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7484555
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.3031371, 2.0717506, -0.3753327, 2.3569636, -2.6601007, 2.4470835
1: -0.4158132, 2.8628929, -0.4759032, 3.2600689, -3.6758821, 3.3387961
2: -1.0028709, 2.0236433, -1.1634135, 2.2862463, -3.2891173, 3.1870568
3: -0.8422414, 2.3150473, -0.9616026, 2.6963146, -3.5385561, 3.2766500
4: -1.1176394, 2.6857204, -1.3583037, 3.0054598, -4.1230993, 4.0440240

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7662881, upper bound: 2.7660701
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.3753327, 2.3569636, -2.7235608, 2.7009773
1: -0.4696044, 3.2193704, -0.4759032, 3.2600689, -3.7296734, 3.6952734
2: -1.1476570, 2.2576227, -1.1634135, 2.2862463, -3.4339032, 3.4210362
3: -0.9491717, 2.6523957, -0.9616026, 2.6963146, -3.6454864, 3.6139984
4: -1.3314362, 2.9754171, -1.3583037, 3.0054598, -4.3368959, 4.3337207

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549331, upper bound: 2.7550932
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549331, upper bound: 2.7550090
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.3753327, 2.3569636, -2.6529980, 2.3442707
1: -0.4008421, 2.7146063, -0.4759032, 3.2600689, -3.6609111, 3.1905093
2: -0.9505634, 1.9361818, -1.1634135, 2.2862463, -3.2368097, 3.0995953
3: -0.8154755, 2.2027259, -0.9616026, 2.6963146, -3.5117900, 3.1643286
4: -1.0562243, 2.5568314, -1.3583037, 3.0054598, -4.0616841, 3.9151349

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7696262, upper bound: 2.7746064
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.3753327, 2.3569636, -2.6883078, 2.5633492
1: -0.4382458, 3.0278041, -0.4759032, 3.2600689, -3.6983147, 3.5037074
2: -1.0667892, 2.1236391, -1.1634135, 2.2862463, -3.3530354, 3.2870526
3: -0.8872414, 2.4708958, -0.9616026, 2.6963146, -3.5835559, 3.4324985
4: -1.2173010, 2.8092000, -1.3583037, 3.0054598, -4.2227607, 4.1675038

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7692350
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533981, upper bound: 2.7588491
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.1731156, 1.2429918, -1.5390263, 2.1420536
1: -0.4008421, 2.7146063, -0.2653253, 1.7709899, -2.1718321, 2.9799316
2: -0.9505634, 1.9361818, -0.6234457, 1.2220012, -2.1725645, 2.5596275
3: -0.8154755, 2.2027259, -0.5676874, 1.3276134, -2.1430888, 2.7704134
4: -1.0562243, 2.5568314, -0.6302408, 1.7394276, -2.7956519, 3.1870723

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7652272
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7547890, upper bound: 2.7572198
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.1731156, 1.2429918, -1.5743361, 2.3611319
1: -0.4382458, 3.0278041, -0.2653253, 1.7709899, -2.2092357, 3.2931294
2: -1.0667892, 2.1236391, -0.6234457, 1.2220012, -2.2887902, 2.7470849
3: -0.8872414, 2.4708958, -0.5676874, 1.3276134, -2.2148547, 3.0385833
4: -1.2173010, 2.8092000, -0.6302408, 1.7394276, -2.9567285, 3.4394407

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7646015
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7553128, upper bound: 2.7553580
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.1845304, 1.3606682, -1.6567025, 2.1534686
1: -0.4008421, 2.7146063, -0.2824830, 1.9470716, -2.3479137, 2.9970894
2: -0.9505634, 1.9361818, -0.6872041, 1.3180709, -2.2686343, 2.6233859
3: -0.8154755, 2.2027259, -0.5939487, 1.4622965, -2.2777719, 2.7966747
4: -1.0562243, 2.5568314, -0.6902229, 1.8839808, -2.9402051, 3.2470541

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7595488, upper bound: 2.7630590
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7601947, upper bound: 2.7593250
time: 0.44 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.1845304, 1.3606682, -1.6920124, 2.3725467
1: -0.4382458, 3.0278041, -0.2824830, 1.9470716, -2.3853173, 3.3102870
2: -1.0667892, 2.1236391, -0.6872041, 1.3180709, -2.3848600, 2.8108432
3: -0.8872414, 2.4708958, -0.5939487, 1.4622965, -2.3495378, 3.0648446
4: -1.2173010, 2.8092000, -0.6902229, 1.8839808, -3.1012816, 3.4994230

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7500113, upper bound: 2.7619548
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7476467, upper bound: 2.7577091
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.3031371, 2.0717506, -0.3022718, 1.9925733, -2.2957103, 2.3740225
1: -0.4158132, 2.8628929, -0.4058715, 2.7465084, -3.1623216, 3.2687645
2: -1.0028709, 2.0236433, -0.9626346, 1.9593803, -2.9622512, 2.9862778
3: -0.8422414, 2.3150473, -0.8254330, 2.2351263, -3.0773678, 3.1404803
4: -1.1176394, 2.6857204, -1.0756483, 2.5819404, -3.6995797, 3.7613688

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7596142, upper bound: 2.7538151
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7594406, upper bound: 2.7535808
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.3022718, 1.9925733, -2.3591704, 2.6279163
1: -0.4696044, 3.2193704, -0.4058715, 2.7465084, -3.2161126, 3.6252418
2: -1.1476570, 2.2576227, -0.9626346, 1.9593803, -3.1070373, 3.2202573
3: -0.9491717, 2.6523957, -0.8254330, 2.2351263, -3.1842980, 3.4778287
4: -1.3314362, 2.9754171, -1.0756483, 2.5819404, -3.9133766, 4.0510654

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7721467, upper bound: 2.7662918
time: 0.33 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.2913533, 1.9494442, -2.3463869, 2.7253327
1: -0.4911667, 3.3645411, -0.3965143, 2.6892331, -3.1803997, 3.7610555
2: -1.2047796, 2.3536205, -0.9412345, 1.9158479, -3.1206274, 3.2948551
3: -0.9906334, 2.7955449, -0.8072340, 2.1748338, -3.1654673, 3.6027789
4: -1.4297944, 3.0918128, -1.0388029, 2.5333719, -3.9631662, 4.1306157

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798951, upper bound: 2.7767387
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798951, upper bound: 2.7767387
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3969426, 2.4339795, -0.2971243, 1.9722085, -2.3691511, 2.7311039
1: -0.4911667, 3.3645411, -0.4001294, 2.7280126, -3.2191792, 3.7646704
2: -1.2047796, 2.3536205, -0.9578750, 1.9310058, -3.1357856, 3.3114955
3: -0.9906334, 2.7955449, -0.8142713, 2.2070019, -3.1976352, 3.6098161
4: -1.4297944, 3.0918128, -1.0563898, 2.5555344, -3.9853287, 4.1482029

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782579, upper bound: 2.7785847
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782579, upper bound: 2.7785847
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.3031371, 2.0717506, -0.3389214, 2.2148643, -2.5180013, 2.4106722
1: -0.4158132, 2.8628929, -0.4439855, 3.0646944, -3.4805076, 3.3068783
2: -1.0028709, 2.0236433, -1.0810840, 2.1477108, -3.1505818, 3.1047273
3: -0.8422414, 2.3150473, -0.8988513, 2.5088573, -3.3510985, 3.2138987
4: -1.1176394, 2.6857204, -1.2414910, 2.8358724, -3.9535117, 3.9272113

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7670873, upper bound: 2.7660701
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.3389214, 2.2148643, -2.5814614, 2.6645660
1: -0.4696044, 3.2193704, -0.4439855, 3.0646944, -3.5342989, 3.6633558
2: -1.1476570, 2.2576227, -1.0810840, 2.1477108, -3.2953677, 3.3387067
3: -0.9491717, 2.6523957, -0.8988513, 2.5088573, -3.4580288, 3.5512471
4: -1.3314362, 2.9754171, -1.2414910, 2.8358724, -4.1673088, 4.2169080

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7596142, upper bound: 2.7538151
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7591664, upper bound: 2.7535808
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.3389214, 2.2148643, -2.5108986, 2.3078594
1: -0.4008421, 2.7146063, -0.4439855, 3.0646944, -3.4655366, 3.1585917
2: -0.9505634, 1.9361818, -1.0810840, 2.1477108, -3.0982742, 3.0172658
3: -0.8154755, 2.2027259, -0.8988513, 2.5088573, -3.3243327, 3.1015773
4: -1.0562243, 2.5568314, -1.2414910, 2.8358724, -3.8920968, 3.7983222

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756868, upper bound: 2.7801145
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771159, upper bound: 2.7785020
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.3389214, 2.2148643, -2.5462084, 2.5269380
1: -0.4382458, 3.0278041, -0.4439855, 3.0646944, -3.5029402, 3.4717896
2: -1.0667892, 2.1236391, -1.0810840, 2.1477108, -3.2145000, 3.2047231
3: -0.8872414, 2.4708958, -0.8988513, 2.5088573, -3.3960986, 3.3697472
4: -1.2173010, 2.8092000, -1.2414910, 2.8358724, -4.0531735, 4.0506911

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7753216
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7783620
time: 0.40 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.03 seconds
NS_A2_A1_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7652067, upper bound: 2.7646799
NS_A2_A1_A1_B1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7572198, upper bound: 2.7547890
NS_A2_A1_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7652067, upper bound: 2.7664323
NS_A2_A1_A1_B1_B2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7572198, upper bound: 2.7558745
NS_A2_A1_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7474518, upper bound: 2.7642129
NS_A2_A1_A1_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7492933, upper bound: 2.7573026
NS_A2_A1_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7581899, upper bound: 2.7645539
NS_A2_A1_A1_B2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7556474, upper bound: 2.7553128
NS_A2_A1_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7477324, upper bound: 2.7662691
NS_A2_A1_A1_B2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7494220, upper bound: 2.7563983
NS_A2_A1_A2_B2_B2_B1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7534307, upper bound: 2.7582079
NS_A2_A1_A2_B2_B2_B1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7533198, upper bound: 2.7578819
NS_A2_A1_A2_B2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7534307, upper bound: 2.7582079
NS_A2_A1_A2_B2_B2_B1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7533198, upper bound: 2.7578819
NS_A2_A1_A2_B2_B2_B2_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7512427, upper bound: 2.7596463
NS_A2_A1_A2_B2_B2_B2_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7583897, upper bound: 2.7608972
NS_A2_A1_A2_B2_B2_B2_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7623244, upper bound: 2.7596463
NS_A2_A1_A2_B2_B2_B2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7583897, upper bound: 2.7608972
NS_A2_A2_B1_B1_B1_A1_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7594008, upper bound: 2.7631652
NS_A2_A2_B1_B1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7598854, upper bound: 2.7590865
NS_A2_A2_B1_B1_B1_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7580684, upper bound: 2.7622837
NS_A2_A2_B1_B1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7577393, upper bound: 2.7581125
NS_A2_A2_B1_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7593531, upper bound: 2.7623229
NS_A2_A2_B1_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7597378, upper bound: 2.7582820
NS_A2_A2_B1_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7593531, upper bound: 2.7622837
NS_A2_A2_B1_B1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7597378, upper bound: 2.7581125
NS_A2_A2_B1_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7562183, upper bound: 2.7550932
NS_A2_A2_B1_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7584490, upper bound: 2.7550857
NS_A2_A2_B1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7717726, upper bound: 2.7662918
NS_A2_A2_B1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7667467, upper bound: 2.7660594
NS_A2_A2_B1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
NS_A2_A2_B1_B2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7528209, upper bound: 2.7507118
NS_A2_A2_B1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
NS_A2_A2_B1_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7478606, upper bound: 2.7484555
NS_A2_A2_B1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7662881, upper bound: 2.7660701
NS_A2_A2_B1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7659609, upper bound: 2.7659609
NS_A2_A2_B1_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7549331, upper bound: 2.7550932
NS_A2_A2_B1_B2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7549331, upper bound: 2.7550090
NS_A2_A2_B1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7678628, upper bound: 2.7763965
NS_A2_A2_B1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7696262, upper bound: 2.7746064
NS_A2_A2_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7692350
NS_A2_A2_B1_B2_B2_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7533981, upper bound: 2.7588491
NS_A2_A2_B2_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7646798, upper bound: 2.7652272
NS_A2_A2_B2_B1_B1_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7547890, upper bound: 2.7572198
NS_A2_A2_B2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7631274, upper bound: 2.7646015
NS_A2_A2_B2_B1_B1_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7553128, upper bound: 2.7553580
NS_A2_A2_B2_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7595488, upper bound: 2.7630590
NS_A2_A2_B2_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7601947, upper bound: 2.7593250
NS_A2_A2_B2_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7500113, upper bound: 2.7619548
NS_A2_A2_B2_B1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7476467, upper bound: 2.7577091
NS_A2_A2_B2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7596142, upper bound: 2.7538151
NS_A2_A2_B2_B2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7594406, upper bound: 2.7535808
NS_A2_A2_B2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7721467, upper bound: 2.7662918
NS_A2_A2_B2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
NS_A2_A2_B2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7798951, upper bound: 2.7767387
NS_A2_A2_B2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7798951, upper bound: 2.7767387
NS_A2_A2_B2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7782579, upper bound: 2.7785847
NS_A2_A2_B2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7782579, upper bound: 2.7785847
NS_A2_A2_B2_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7670873, upper bound: 2.7660701
NS_A2_A2_B2_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
NS_A2_A2_B2_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7596142, upper bound: 2.7538151
NS_A2_A2_B2_B2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7591664, upper bound: 2.7535808
NS_A2_A2_B2_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7756868, upper bound: 2.7801145
NS_A2_A2_B2_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7771159, upper bound: 2.7785020
NS_A2_A2_B2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7753216
NS_A2_A2_B2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.03
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7783620

## BFS NS instance: NS_A2_A1_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1812261, 1.3317630, -0.2859867, 1.9069934, -2.0882194, 1.6177497
1: -0.2787997, 1.9029334, -0.3881981, 2.6300483, -2.9088478, 2.2911315
2: -0.6704855, 1.2932925, -0.9169891, 1.8772290, -2.5477145, 2.2102816
3: -0.5872393, 1.4343947, -0.7919734, 2.1231287, -2.7103679, 2.2263680
4: -0.6767123, 1.8457990, -0.9999120, 2.4782417, -3.1549540, 2.8457110

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A1_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.2859867, 1.9069934, -2.0740261, 1.5117109
1: -0.2606599, 1.7469907, -0.3881981, 2.6300483, -2.8907082, 2.1351888
2: -0.6121430, 1.2063824, -0.9169891, 1.8772290, -2.4893720, 2.1233716
3: -0.5581758, 1.3027804, -0.7919734, 2.1231287, -2.6813045, 2.0947537
4: -0.6156360, 1.7192602, -0.9999120, 2.4782417, -3.0938778, 2.7191722

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7436562, upper bound: 2.7504401
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A1_B1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7436562, upper bound: 2.7558745
time: 0.35 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3601642, 2.2756541, -2.4426868, 1.5858884
1: -0.2606599, 1.7469907, -0.4592577, 3.1483088, -3.4089687, 2.2062483
2: -0.6121430, 1.2063824, -1.1209748, 2.2070556, -2.8191986, 2.3273573
3: -0.5581758, 1.3027804, -0.9298903, 2.5834980, -3.1416738, 2.2326708
4: -0.6156360, 1.7192602, -1.2861826, 2.9092791, -3.5249152, 3.0054429

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7376370, upper bound: 2.7518682
time: 0.39 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7376370, upper bound: 2.7573026
time: 0.41 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1812261, 1.3317630, -0.3231725, 2.1327963, -2.3140223, 1.6549355
1: -0.2787997, 1.9029334, -0.4273107, 2.9528756, -3.2316751, 2.3302441
2: -0.6704855, 1.2932925, -1.0381293, 2.0690100, -2.7394955, 2.3314219
3: -0.5872393, 1.4343947, -0.8672407, 2.3985398, -2.9857790, 2.3016355
4: -0.6767123, 1.8457990, -1.1689031, 2.7397261, -3.4164383, 3.0147021

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577978, upper bound: 2.7470599
time: 0.41 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577978, upper bound: 2.7613545
time: 0.37 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1670328, 1.2257242, -0.3231725, 2.1327963, -2.2998290, 1.5488967
1: -0.2606599, 1.7469907, -0.4273107, 2.9528756, -3.2135355, 2.1743014
2: -0.6121430, 1.2063824, -1.0381293, 2.0690100, -2.6811531, 2.2445116
3: -0.5581758, 1.3027804, -0.8672407, 2.3985398, -2.9567156, 2.1700211
4: -0.6156360, 1.7192602, -1.1689031, 2.7397261, -3.3553619, 2.8881633

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7407950, upper bound: 2.7509639
time: 0.38 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7407950, upper bound: 2.7563983
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.2973586, 2.0464444, -2.4130416, 2.6230030
1: -0.4696044, 3.2193704, -0.4104910, 2.8293266, -3.2989311, 3.6298614
2: -1.1476570, 2.2576227, -0.9908106, 1.9982010, -3.1458580, 3.2484334
3: -0.9491717, 2.6523957, -0.8320768, 2.2798955, -3.2290673, 3.4844725
4: -1.3314362, 2.9754171, -1.0959523, 2.6567075, -3.9881437, 4.0713692

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7544897, upper bound: 2.7533442
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.3086333, 2.0736451, -2.4402423, 2.6342778
1: -0.4696044, 3.2193704, -0.4168339, 2.8735497, -3.3431540, 3.6362042
2: -1.1476570, 2.2576227, -1.0126864, 2.0177553, -3.1654124, 3.2703090
3: -0.9491717, 2.6523957, -0.8451283, 2.3225791, -3.2717509, 3.4975240
4: -1.3314362, 2.9754171, -1.1221292, 2.6829336, -4.0143700, 4.0975466

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515212, upper bound: 2.7513427
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3789127, 2.3446078, -0.2973586, 2.0464444, -2.4253571, 2.6419663
1: -0.4728479, 3.2426538, -0.4104910, 2.8293266, -3.3021746, 3.6531448
2: -1.1574149, 2.2680767, -0.9908106, 1.9982010, -3.1556158, 3.2588873
3: -0.9554390, 2.6743917, -0.8320768, 2.2798955, -3.2353344, 3.5064685
4: -1.3494930, 2.9870510, -1.0959523, 2.6567075, -4.0062008, 4.0830030

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
time: 0.45 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3789127, 2.3446078, -0.3086333, 2.0736451, -2.4525578, 2.6532412
1: -0.4728479, 3.2426538, -0.4168339, 2.8735497, -3.3463976, 3.6594877
2: -1.1574149, 2.2680767, -1.0126864, 2.0177553, -3.1751702, 3.2807631
3: -0.9554390, 2.6743917, -0.8451283, 2.3225791, -3.2780180, 3.5195200
4: -1.3494930, 2.9870510, -1.1221292, 2.6829336, -4.0324268, 4.1091805

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2909694, 2.0227432, -0.3753327, 2.3569636, -2.6479330, 2.3980761
1: -0.4052326, 2.7972746, -0.4759032, 3.2600689, -3.6653016, 3.2731776
2: -0.9782555, 1.9761380, -1.1634135, 2.2862463, -3.2645018, 3.1395516
3: -0.8216132, 2.2480628, -0.9616026, 2.6963146, -3.5179276, 3.2096653
4: -1.0757260, 2.6311049, -1.3583037, 3.0054598, -4.0811858, 3.9894085

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7616702, upper bound: 2.7672633
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7603289, upper bound: 2.7651148
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.3042246, 2.0575032, -0.3753327, 2.3569636, -2.6611881, 2.4328361
1: -0.4130379, 2.8520617, -0.4759032, 3.2600689, -3.6731067, 3.3279648
2: -1.0041143, 2.0024467, -1.1634135, 2.2862463, -3.2903605, 3.1658602
3: -0.8375547, 2.3009794, -0.9616026, 2.6963146, -3.5338693, 3.2625818
4: -1.1082883, 2.6651578, -1.3583037, 3.0054598, -4.1137481, 4.0234613

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7634116, upper bound: 2.7651577
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7625498, upper bound: 2.7635387
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.2852379, 1.9261826, -0.3753327, 2.3569636, -2.6422014, 2.3015153
1: -0.3915178, 2.6576438, -0.4759032, 3.2600689, -3.6515868, 3.1335468
2: -0.9292547, 1.8941118, -1.1634135, 2.2862463, -3.2155008, 3.0575252
3: -0.7973023, 2.1429307, -0.9616026, 2.6963146, -3.4936168, 3.1045332
4: -1.0195756, 2.5085783, -1.3583037, 3.0054598, -4.0250354, 3.8668818

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7616800, upper bound: 2.7673473
time: 0.35 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7517690, upper bound: 2.7591097
time: 0.44 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.3753327, 2.3569636, -2.6501577, 2.3316183
1: -0.3966470, 2.7066140, -0.4759032, 3.2600689, -3.6567159, 3.1825171
2: -0.9499824, 1.9157126, -1.1634135, 2.2862463, -3.2362287, 3.0791261
3: -0.8073892, 2.1858242, -0.9616026, 2.6963146, -3.5037038, 3.1474266
4: -1.0437701, 2.5382841, -1.3583037, 3.0054598, -4.0492296, 3.8965878

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628420, upper bound: 2.7645789
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.3753327, 2.3569636, -2.6724212, 2.4809117
1: -0.4214576, 2.9155774, -0.4759032, 3.2600689, -3.6815267, 3.3914804
2: -1.0235951, 2.0444477, -1.1634135, 2.2862463, -3.3098414, 3.2078612
3: -0.8554188, 2.3600154, -0.9616026, 2.6963146, -3.5517335, 3.3216181
4: -1.1441643, 2.7125952, -1.3583037, 3.0054598, -4.1496239, 4.0708990

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7626503, upper bound: 2.7635633
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.2796247, 1.8826946, -0.1731156, 1.2429918, -1.5226165, 2.0558102
1: -0.3829975, 2.5972061, -0.2653253, 1.7709899, -2.1539874, 2.8625314
2: -0.9045649, 1.8533102, -0.6234457, 1.2220012, -2.1265659, 2.4767561
3: -0.7816354, 2.0899224, -0.5676874, 1.3276134, -2.1092486, 2.6576099
4: -0.9799299, 2.4523220, -0.6302408, 1.7394276, -2.7193575, 3.0825629

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7455668, upper bound: 2.7430950
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7455667, upper bound: 2.7572198
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.1731156, 1.2429918, -1.5584493, 2.2786944
1: -0.4214576, 2.9155774, -0.2653253, 1.7709899, -2.1924477, 3.1809027
2: -1.0235951, 2.0444477, -0.6234457, 1.2220012, -2.2455964, 2.6678934
3: -0.8554188, 2.3600154, -0.5676874, 1.3276134, -2.1830320, 2.9277029
4: -1.1441643, 2.7125952, -0.6302408, 1.7394276, -2.8835919, 3.3428359

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7474877, upper bound: 2.7417944
time: 0.46 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7474876, upper bound: 2.7553580
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.2913533, 1.9494442, -2.3160415, 2.6169977
1: -0.4696044, 3.2193704, -0.3965143, 2.6892331, -3.1588373, 3.6158848
2: -1.1476570, 2.2576227, -0.9412345, 1.9158479, -3.0635049, 3.1988573
3: -0.9491717, 2.6523957, -0.8072340, 2.1748338, -3.1240053, 3.4596298
4: -1.3314362, 2.9754171, -1.0388029, 2.5333719, -3.8648081, 4.0142202

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577854, upper bound: 2.7520659
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575793, upper bound: 2.7518331
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3665972, 2.3256445, -0.2971243, 1.9722085, -2.3388057, 2.6227689
1: -0.4696044, 3.2193704, -0.4001294, 2.7280126, -3.1976171, 3.6194997
2: -1.1476570, 2.2576227, -0.9578750, 1.9310058, -3.0786629, 3.2154977
3: -0.9491717, 2.6523957, -0.8142713, 2.2070019, -3.1561737, 3.4666672
4: -1.3314362, 2.9754171, -1.0563898, 2.5555344, -3.8869705, 4.0318069

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.2913533, 1.9494442, -2.2454786, 2.2602916
1: -0.4008421, 2.7146063, -0.3965143, 2.6892331, -3.0900750, 3.1111207
2: -0.9505634, 1.9361818, -0.9412345, 1.9158479, -2.8664112, 2.8774161
3: -0.8154755, 2.2027259, -0.8072340, 2.1748338, -2.9903092, 3.0099599
4: -1.0562243, 2.5568314, -1.0388029, 2.5333719, -3.5895963, 3.5956342

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.2913533, 1.9494442, -2.2807884, 2.4793696
1: -0.4382458, 3.0278041, -0.3965143, 2.6892331, -3.1274788, 3.4243183
2: -1.0667892, 2.1236391, -0.9412345, 1.9158479, -2.9826369, 3.0648737
3: -0.8872414, 2.4708958, -0.8072340, 2.1748338, -3.0620751, 3.2781298
4: -1.2173010, 2.8092000, -1.0388029, 2.5333719, -3.7506728, 3.8480029

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2960344, 1.9689381, -0.2971243, 1.9722085, -2.2682428, 2.2660623
1: -0.4008421, 2.7146063, -0.4001294, 2.7280126, -3.1288548, 3.1147356
2: -0.9505634, 1.9361818, -0.9578750, 1.9310058, -2.8815694, 2.8940568
3: -0.8154755, 2.2027259, -0.8142713, 2.2070019, -3.0224774, 3.0169973
4: -1.0562243, 2.5568314, -1.0563898, 2.5555344, -3.6117587, 3.6132212

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7785847
time: 0.47 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7776084
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.2971243, 1.9722085, -2.3035526, 2.4851408
1: -0.4382458, 3.0278041, -0.4001294, 2.7280126, -3.1662583, 3.4279335
2: -1.0667892, 2.1236391, -0.9578750, 1.9310058, -2.9977951, 3.0815141
3: -0.8872414, 2.4708958, -0.8142713, 2.2070019, -3.0942433, 3.2851672
4: -1.2173010, 2.8092000, -1.0563898, 2.5555344, -3.7728353, 3.8655899

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7785847
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7776084
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2909694, 2.0227432, -0.3389214, 2.2148643, -2.5058336, 2.3616648
1: -0.4052326, 2.7972746, -0.4439855, 3.0646944, -3.4699271, 3.2412601
2: -0.9782555, 1.9761380, -1.0810840, 2.1477108, -3.1259663, 3.0572219
3: -0.8216132, 2.2480628, -0.8988513, 2.5088573, -3.3304706, 3.1469140
4: -1.0757260, 2.6311049, -1.2414910, 2.8358724, -3.9115984, 3.8725958

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7735331, upper bound: 2.7768214
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7589985, upper bound: 2.7626461
time: 0.49 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.3042246, 2.0575032, -0.3389214, 2.2148643, -2.5190887, 2.3964248
1: -0.4130379, 2.8520617, -0.4439855, 3.0646944, -3.4777322, 3.2960472
2: -1.0041143, 2.0024467, -1.0810840, 2.1477108, -3.1518250, 3.0835307
3: -0.8375547, 2.3009794, -0.8988513, 2.5088573, -3.3464119, 3.1998305
4: -1.1082883, 2.6651578, -1.2414910, 2.8358724, -3.9441607, 3.9066486

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7755064
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7608470, upper bound: 2.7603791
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.2852379, 1.9261826, -0.3389214, 2.2148643, -2.5001020, 2.2651041
1: -0.3915178, 2.6576438, -0.4439855, 3.0646944, -3.4562123, 3.1016293
2: -0.9292547, 1.8941118, -1.0810840, 2.1477108, -3.0769653, 2.9751959
3: -0.7973023, 2.1429307, -0.8988513, 2.5088573, -3.3061595, 3.0417819
4: -1.0195756, 2.5085783, -1.2414910, 2.8358724, -3.8554480, 3.7500691

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.3389214, 2.2148643, -2.5080583, 2.2952070
1: -0.3966470, 2.7066140, -0.4439855, 3.0646944, -3.4613414, 3.1505995
2: -0.9499824, 1.9157126, -1.0810840, 2.1477108, -3.0976932, 2.9967966
3: -0.8073892, 2.1858242, -0.8988513, 2.5088573, -3.3162465, 3.0846753
4: -1.0437701, 2.5382841, -1.2414910, 2.8358724, -3.8796425, 3.7797751

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7751746
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7754416
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.3029714, 1.9758003, -2.3071446, 2.4909878
1: -0.4382458, 3.0278041, -0.4003568, 2.7667799, -3.2050257, 3.4281609
2: -1.0667892, 2.1236391, -0.9991634, 1.8891554, -2.9559445, 3.1228025
3: -0.8872414, 2.4708958, -0.8155400, 2.2180204, -3.1052618, 3.2864356
4: -1.2173010, 2.8092000, -1.1026624, 2.5684884, -3.7857895, 3.9118624

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753440, upper bound: 2.7753213
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764309, upper bound: 2.7751748
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3313442, 2.1880164, -0.3140926, 2.1232934, -2.4546375, 2.5021091
1: -0.4382458, 3.0278041, -0.4233531, 2.9424019, -3.3806477, 3.4511571
2: -1.0667892, 2.1236391, -1.0367260, 2.0574467, -3.1242359, 3.1603651
3: -0.8872414, 2.4708958, -0.8564682, 2.3797078, -3.2669492, 3.3273640
4: -1.2173010, 2.8092000, -1.1613970, 2.7425013, -3.9598022, 3.9705970

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748215, upper bound: 2.7783620
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756245, upper bound: 2.7754416
time: 0.40 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.70 seconds
NS_A2_A1_A1_B1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7436562, upper bound: 2.7504401
NS_A2_A1_A1_B1_B2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7436562, upper bound: 2.7558745
NS_A2_A1_A1_B2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7376370, upper bound: 2.7518682
NS_A2_A1_A1_B2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7376370, upper bound: 2.7573026
NS_A2_A1_A1_B2_B2_A1_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7577978, upper bound: 2.7470599
NS_A2_A1_A1_B2_B2_A1_B2_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7577978, upper bound: 2.7613545
NS_A2_A1_A1_B2_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7407950, upper bound: 2.7509639
NS_A2_A1_A1_B2_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7407950, upper bound: 2.7563983
NS_A2_A2_B1_B2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7544897, upper bound: 2.7533442
NS_A2_A2_B1_B2_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7566259, upper bound: 2.7533368
NS_A2_A2_B1_B2_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7515212, upper bound: 2.7513427
NS_A2_A2_B1_B2_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7515240, upper bound: 2.7515980
NS_A2_A2_B1_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
NS_A2_A2_B1_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
NS_A2_A2_B1_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
NS_A2_A2_B1_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
NS_A2_A2_B1_B2_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7616702, upper bound: 2.7672633
NS_A2_A2_B1_B2_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7603289, upper bound: 2.7651148
NS_A2_A2_B1_B2_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7634116, upper bound: 2.7651577
NS_A2_A2_B1_B2_B2_A1_A1_A2_A2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7625498, upper bound: 2.7635387
NS_A2_A2_B1_B2_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7616800, upper bound: 2.7673473
NS_A2_A2_B1_B2_B2_A2_A1_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7517690, upper bound: 2.7591097
NS_A2_A2_B1_B2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
NS_A2_A2_B1_B2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7697089, upper bound: 2.7748574
NS_A2_A2_B1_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
NS_A2_A2_B1_B2_B2_A2_A2_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7626503, upper bound: 2.7635633
NS_A2_A2_B2_B1_B1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7455668, upper bound: 2.7430950
NS_A2_A2_B2_B1_B1_A1_A2_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7455667, upper bound: 2.7572198
NS_A2_A2_B2_B1_B1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7474877, upper bound: 2.7417944
NS_A2_A2_B2_B1_B1_A2_A2_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7474876, upper bound: 2.7553580
NS_A2_A2_B2_B2_B1_A1_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7577854, upper bound: 2.7520659
NS_A2_A2_B2_B2_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7575793, upper bound: 2.7518331
NS_A2_A2_B2_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
NS_A2_A2_B2_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7667610, upper bound: 2.7659609
NS_A2_A2_B2_B2_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
NS_A2_A2_B2_B2_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
NS_A2_A2_B2_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
NS_A2_A2_B2_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7778822, upper bound: 2.7767387
NS_A2_A2_B2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7785847
NS_A2_A2_B2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7776084
NS_A2_A2_B2_B2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7785847
NS_A2_A2_B2_B2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7757543, upper bound: 2.7776084
NS_A2_A2_B2_B2_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7735331, upper bound: 2.7768214
NS_A2_A2_B2_B2_B2_A1_A1_A1_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7589985, upper bound: 2.7626461
NS_A2_A2_B2_B2_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7755064
NS_A2_A2_B2_B2_B2_A1_A1_A2_B2, status: Status.VERIFIED, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7608470, upper bound: 2.7603791
NS_A2_A2_B2_B2_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
NS_A2_A2_B2_B2_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
NS_A2_A2_B2_B2_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7751746
NS_A2_A2_B2_B2_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7746754, upper bound: 2.7754416
NS_A2_A2_B2_B2_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7753440, upper bound: 2.7753213
NS_A2_A2_B2_B2_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7764309, upper bound: 2.7751748
NS_A2_A2_B2_B2_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7748215, upper bound: 2.7783620
NS_A2_A2_B2_B2_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.70
Output dim: 0, lower bound: -2.7756245, upper bound: 2.7754416

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2796247, 1.8826946, -0.2973586, 2.0464444, -2.3260691, 2.1800532
1: -0.3829975, 2.5972061, -0.4104910, 2.8293266, -3.2123241, 3.0076971
2: -0.9045649, 1.8533102, -0.9908106, 1.9982010, -2.9027658, 2.8441210
3: -0.7816354, 2.0899224, -0.8320768, 2.2798955, -3.0615311, 2.9219992
4: -0.9799299, 2.4523220, -1.0959523, 2.6567075, -3.6366374, 3.5482743

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7744474
time: 0.36 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7744474
time: 0.45 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.2973586, 2.0464444, -2.3619020, 2.4029374
1: -0.4214576, 2.9155774, -0.4104910, 2.8293266, -3.2507844, 3.3260684
2: -1.0235951, 2.0444477, -0.9908106, 1.9982010, -3.0217962, 3.0352583
3: -0.8554188, 2.3600154, -0.8320768, 2.2798955, -3.1353145, 3.1920922
4: -1.1441643, 2.7125952, -1.0959523, 2.6567075, -3.8008718, 3.8085475

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
time: 0.38 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.2796247, 1.8826946, -0.3086333, 2.0736451, -2.3532698, 2.1913280
1: -0.3829975, 2.5972061, -0.4168339, 2.8735497, -3.2565472, 3.0140400
2: -0.9045649, 1.8533102, -1.0126864, 2.0177553, -2.9223201, 2.8659966
3: -0.7816354, 2.0899224, -0.8451283, 2.3225791, -3.1042147, 2.9350507
4: -0.9799299, 2.4523220, -1.1221292, 2.6829336, -3.6628635, 3.5744512

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7771505
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7756437
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.3086333, 2.0736451, -2.3891027, 2.4142122
1: -0.4214576, 2.9155774, -0.4168339, 2.8735497, -3.2950072, 3.3324113
2: -1.0235951, 2.0444477, -1.0126864, 2.0177553, -3.0413504, 3.0571342
3: -0.8554188, 2.3600154, -0.8451283, 2.3225791, -3.1779981, 3.2051437
4: -1.1441643, 2.7125952, -1.1221292, 2.6829336, -3.8270979, 3.8347244

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
time: 0.37 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
time: 0.38 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2776344, 1.9485888, -0.3753327, 2.3569636, -2.6345980, 2.3239217
1: -0.3899742, 2.6961823, -0.4759032, 3.2600689, -3.6500430, 3.1720853
2: -0.9390552, 1.9043281, -1.1634135, 2.2862463, -3.2253015, 3.0677416
3: -0.7928967, 2.1503923, -0.9616026, 2.6963146, -3.4892113, 3.1119947
4: -1.0102851, 2.5407043, -1.3583037, 3.0054598, -4.0157452, 3.8990078

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7616702, upper bound: 2.7672633
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7616702, upper bound: 2.7672633
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.3021082, 2.0485234, -0.3753327, 2.3569636, -2.6590719, 2.4238563
1: -0.4073578, 2.8168266, -0.4759032, 3.2600689, -3.6674266, 3.2927299
2: -0.9709806, 2.0237799, -1.1634135, 2.2862463, -3.2572269, 3.1871934
3: -0.8272771, 2.2859435, -0.9616026, 2.6963146, -3.5235915, 3.2475462
4: -1.0761096, 2.6191840, -1.3583037, 3.0054598, -4.0815697, 3.9774876

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7603289, upper bound: 2.7651148
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7603289, upper bound: 2.7651148
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.2900513, 1.9800816, -0.3753327, 2.3569636, -2.6470151, 2.3554144
1: -0.3972396, 2.7473166, -0.4759032, 3.2600689, -3.6573086, 3.2232199
2: -0.9633340, 1.9280410, -1.1634135, 2.2862463, -3.2495804, 3.0914545
3: -0.8077149, 2.1986573, -0.9616026, 2.6963146, -3.5040295, 3.1602597
4: -1.0401150, 2.5716171, -1.3583037, 3.0054598, -4.0455747, 3.9299207

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7634116, upper bound: 2.7651577
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_A1_A2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7634116, upper bound: 2.7651577
time: 0.40 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2693014, 1.8415074, -0.3753327, 2.3569636, -2.6262650, 2.2168403
1: -0.3738544, 2.5427141, -0.4759032, 3.2600689, -3.6339233, 3.0186172
2: -0.8839430, 1.8124728, -1.1634135, 2.2862463, -3.1701894, 2.9758863
3: -0.7638226, 2.0339918, -0.9616026, 2.6963146, -3.4601371, 2.9955945
4: -0.9475492, 2.4048069, -1.3583037, 3.0054598, -3.9530091, 3.7631106

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7616800, upper bound: 2.7673473
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7616800, upper bound: 2.7673473
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.3629161, 2.3064585, -2.5996525, 2.3192017
1: -0.3966470, 2.7066140, -0.4652137, 3.1939573, -3.5906043, 3.1718278
2: -0.9499824, 1.9157126, -1.1392360, 2.2357109, -3.1856933, 3.0549486
3: -0.8073892, 2.1858242, -0.9405786, 2.6256416, -3.4330308, 3.1264029
4: -1.0437701, 2.5382841, -1.3152579, 2.9512329, -3.9950030, 3.8535419

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
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

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.3708772, 2.3199959, -2.6131899, 2.3271627
1: -0.3966470, 2.7066140, -0.4697720, 3.2195220, -3.6161690, 3.1763859
2: -0.9499824, 1.9157126, -1.1537372, 2.2423575, -3.1923399, 3.0694499
3: -0.8073892, 2.1858242, -0.9502540, 2.6516135, -3.4590027, 3.1360781
4: -1.0437701, 2.5382841, -1.3322339, 2.9660292, -4.0097990, 3.8705180

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.3629161, 2.3064585, -2.6219161, 2.4684949
1: -0.4214576, 2.9155774, -0.4652137, 3.1939573, -3.6154151, 3.3807912
2: -1.0235951, 2.0444477, -1.1392360, 2.2357109, -3.2593060, 3.1836836
3: -0.8554188, 2.3600154, -0.9405786, 2.6256416, -3.4810605, 3.3005941
4: -1.1441643, 2.7125952, -1.3152579, 2.9512329, -4.0953970, 4.0278530

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_A2_A1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3536609, 2.2733560, -0.2971243, 1.9722085, -2.3258696, 2.5704803
1: -0.4585382, 3.1511228, -0.4001294, 2.7280126, -3.1865509, 3.5512521
2: -1.1225939, 2.2052293, -0.9578750, 1.9310058, -3.0535998, 3.1631043
3: -0.9274352, 2.5793722, -0.8142713, 2.2070019, -3.1344371, 3.3936434
4: -1.2869700, 2.9193895, -1.0563898, 2.5555344, -3.8425045, 3.9757793

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3630877, 2.2926280, -0.2971243, 1.9722085, -2.3352962, 2.5897522
1: -0.4641411, 3.1831021, -0.4001294, 2.7280126, -3.1921537, 3.5832314
2: -1.1396843, 2.2169552, -0.9578750, 1.9310058, -3.0706902, 3.1748302
3: -0.9391701, 2.6123335, -0.8142713, 2.2070019, -3.1461720, 3.4266047
4: -1.3082904, 2.9394026, -1.0563898, 2.5555344, -3.8638248, 3.9957924

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2852379, 1.9261826, -0.2913533, 1.9494442, -2.2346821, 2.2175360
1: -0.3915178, 2.6576438, -0.3965143, 2.6892331, -3.0807509, 3.0541582
2: -0.9292547, 1.8941118, -0.9412345, 1.9158479, -2.8451025, 2.8353462
3: -0.7973023, 2.1429307, -0.8072340, 2.1748338, -2.9721360, 2.9501648
4: -1.0195756, 2.5085783, -1.0388029, 2.5333719, -3.5529475, 3.5473812

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.2913533, 1.9494442, -2.2426381, 2.2476389
1: -0.3966470, 2.7066140, -0.3965143, 2.6892331, -3.0858800, 3.1031284
2: -0.9499824, 1.9157126, -0.9412345, 1.9158479, -2.8658304, 2.8569469
3: -0.8073892, 2.1858242, -0.8072340, 2.1748338, -2.9822230, 2.9930582
4: -1.0437701, 2.5382841, -1.0388029, 2.5333719, -3.5771420, 3.5770869

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.3182679, 2.1371770, -0.2913533, 1.9494442, -2.2677121, 2.4285302
1: -0.4268867, 2.9595909, -0.3965143, 2.6892331, -3.1161199, 3.3561053
2: -1.0412936, 2.0729208, -0.9412345, 1.9158479, -2.9571414, 3.0141554
3: -0.8644805, 2.3999858, -0.8072340, 2.1748338, -3.0393143, 3.2072198
4: -1.1726522, 2.7533128, -1.0388029, 2.5333719, -3.7060242, 3.7921157

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.3276167, 2.1572509, -0.2913533, 1.9494442, -2.2770610, 2.4486041
1: -0.4325817, 2.9916034, -0.3965143, 2.6892331, -3.1218147, 3.3881178
2: -1.0583760, 2.0858326, -0.9412345, 1.9158479, -2.9742239, 3.0270672
3: -0.8765814, 2.4330173, -0.8072340, 2.1748338, -3.0514152, 3.2402513
4: -1.1936884, 2.7738743, -1.0388029, 2.5333719, -3.7270603, 3.8126771

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.2852379, 1.9261826, -0.2971243, 1.9722085, -2.2574463, 2.2233069
1: -0.3915178, 2.6576438, -0.4001294, 2.7280126, -3.1195304, 3.0577731
2: -0.9292547, 1.8941118, -0.9578750, 1.9310058, -2.8602605, 2.8519869
3: -0.7973023, 2.1429307, -0.8142713, 2.2070019, -3.0043042, 2.9572020
4: -1.0195756, 2.5085783, -1.0563898, 2.5555344, -3.5751100, 3.5649681

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.2971243, 1.9722085, -2.2654026, 2.2534099
1: -0.3966470, 2.7066140, -0.4001294, 2.7280126, -3.1246595, 3.1067433
2: -0.9499824, 1.9157126, -0.9578750, 1.9310058, -2.8809881, 2.8735876
3: -0.8073892, 2.1858242, -0.8142713, 2.2070019, -3.0143912, 3.0000954
4: -1.0437701, 2.5382841, -1.0563898, 2.5555344, -3.5993044, 3.5946739

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.3182679, 2.1371770, -0.2971243, 1.9722085, -2.2904763, 2.4343014
1: -0.4268867, 2.9595909, -0.4001294, 2.7280126, -3.1548994, 3.3597202
2: -1.0412936, 2.0729208, -0.9578750, 1.9310058, -2.9722996, 3.0307958
3: -0.8644805, 2.3999858, -0.8142713, 2.2070019, -3.0714824, 3.2142572
4: -1.1726522, 2.7533128, -1.0563898, 2.5555344, -3.7281866, 3.8097026

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3276167, 2.1572509, -0.2971243, 1.9722085, -2.2998252, 2.4543753
1: -0.4325817, 2.9916034, -0.4001294, 2.7280126, -3.1605942, 3.3917327
2: -1.0583760, 2.0858326, -0.9578750, 1.9310058, -2.9893818, 3.0437076
3: -0.8765814, 2.4330173, -0.8142713, 2.2070019, -3.0835834, 3.2472887
4: -1.1936884, 2.7738743, -1.0563898, 2.5555344, -3.7492228, 3.8302641

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_B2_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_B2_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2909694, 2.0227432, -0.3231725, 2.1327963, -2.4237657, 2.3459158
1: -0.4052326, 2.7972746, -0.4273107, 2.9528756, -3.3581083, 3.2245853
2: -0.9782555, 1.9761380, -1.0381293, 2.0690100, -3.0472655, 3.0142674
3: -0.8216132, 2.2480628, -0.8672407, 2.3985398, -3.2201529, 3.1153035
4: -1.0757260, 2.6311049, -1.1689031, 2.7397261, -3.8154521, 3.8000081

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7589985, upper bound: 2.7626461
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7589985, upper bound: 2.7626461
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3042246, 2.0575032, -0.3231725, 2.1327963, -2.4370208, 2.3806758
1: -0.4130379, 2.8520617, -0.4273107, 2.9528756, -3.3659134, 3.2793725
2: -1.0041143, 2.0024467, -1.0381293, 2.0690100, -3.0731244, 3.0405760
3: -0.8375547, 2.3009794, -0.8672407, 2.3985398, -3.2360945, 3.1682200
4: -1.1082883, 2.6651578, -1.1689031, 2.7397261, -3.8480144, 3.8340609

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7743022
time: 0.47 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7747715
time: 0.44 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2852379, 1.9261826, -0.3029714, 1.9758003, -2.2610383, 2.2291541
1: -0.3915178, 2.6576438, -0.4003568, 2.7667799, -3.1582978, 3.0580006
2: -0.9292547, 1.8941118, -0.9991634, 1.8891554, -2.8184099, 2.8932753
3: -0.7973023, 2.1429307, -0.8155400, 2.2180204, -3.0153227, 2.9584708
4: -1.0195756, 2.5085783, -1.1026624, 2.5684884, -3.5880640, 3.6112409

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746772
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
time: 0.49 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2852379, 1.9261826, -0.3140926, 2.1232934, -2.4085312, 2.2402751
1: -0.3915178, 2.6576438, -0.4233531, 2.9424019, -3.3339198, 3.0809970
2: -0.9292547, 1.8941118, -1.0367260, 2.0574467, -2.9867015, 2.9308376
3: -0.7973023, 2.1429307, -0.8564682, 2.3797078, -3.1770101, 2.9993989
4: -1.0195756, 2.5085783, -1.1613970, 2.7425013, -3.7620769, 3.6699753

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783336
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.3029714, 1.9758003, -2.2689943, 2.2592568
1: -0.3966470, 2.7066140, -0.4003568, 2.7667799, -3.1634269, 3.1069708
2: -0.9499824, 1.9157126, -0.9991634, 1.8891554, -2.8391378, 2.9148760
3: -0.8073892, 2.1858242, -0.8155400, 2.2180204, -3.0254097, 3.0013642
4: -1.0437701, 2.5382841, -1.1026624, 2.5684884, -3.6122584, 3.6409464

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741407
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746774
time: 0.45 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2931940, 1.9562856, -0.3140926, 2.1232934, -2.4164875, 2.2703781
1: -0.3966470, 2.7066140, -0.4233531, 2.9424019, -3.3390489, 3.1299672
2: -0.9499824, 1.9157126, -1.0367260, 2.0574467, -3.0074291, 2.9524386
3: -0.8073892, 2.1858242, -0.8564682, 2.3797078, -3.1870971, 3.0422924
4: -1.0437701, 2.5382841, -1.1613970, 2.7425013, -3.7862713, 3.6996810

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7613019, upper bound: 2.7613651
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7742810
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7747346
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3182679, 2.1371770, -0.3029714, 1.9758003, -2.2940683, 2.4401484
1: -0.4268867, 2.9595909, -0.4003568, 2.7667799, -3.1936667, 3.3599477
2: -1.0412936, 2.0729208, -0.9991634, 1.8891554, -2.9304490, 3.0720842
3: -0.8644805, 2.3999858, -0.8155400, 2.2180204, -3.0825009, 3.2155256
4: -1.1726522, 2.7533128, -1.1026624, 2.5684884, -3.7411406, 3.8559752

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753440, upper bound: 2.7744466
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753440, upper bound: 2.7751748
time: 0.44 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3276167, 2.1572509, -0.3029714, 1.9758003, -2.3034170, 2.4602222
1: -0.4325817, 2.9916034, -0.4003568, 2.7667799, -3.1993616, 3.3919601
2: -1.0583760, 2.0858326, -0.9991634, 1.8891554, -2.9475312, 3.0849960
3: -0.8765814, 2.4330173, -0.8155400, 2.2180204, -3.0946019, 3.2485571
4: -1.1936884, 2.7738743, -1.1026624, 2.5684884, -3.7621768, 3.8765368

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764309, upper bound: 2.7744466
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7764309, upper bound: 2.7751746
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3182679, 2.1371770, -0.3140926, 2.1232934, -2.4415612, 2.4512696
1: -0.4268867, 2.9595909, -0.4233531, 2.9424019, -3.3692887, 3.3829441
2: -1.0412936, 2.0729208, -1.0367260, 2.0574467, -3.0987403, 3.1096468
3: -0.8644805, 2.3999858, -0.8564682, 2.3797078, -3.2441883, 3.2564540
4: -1.1726522, 2.7533128, -1.1613970, 2.7425013, -3.9151535, 3.9147098

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7603189, upper bound: 2.7640982
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7510546, upper bound: 2.7536455
time: 0.39 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3276167, 2.1572509, -0.3140926, 2.1232934, -2.4509101, 2.4713435
1: -0.4325817, 2.9916034, -0.4233531, 2.9424019, -3.3749835, 3.4149566
2: -1.0583760, 2.0858326, -1.0367260, 2.0574467, -3.1158228, 3.1225586
3: -0.8765814, 2.4330173, -0.8564682, 2.3797078, -3.2562892, 3.2894855
4: -1.1936884, 2.7738743, -1.1613970, 2.7425013, -3.9361897, 3.9352713

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
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

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756245, upper bound: 2.7745399
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756245, upper bound: 2.7754416
time: 0.42 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 2.77 seconds
NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7744474
NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7744474
NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7770627, upper bound: 2.7744474
NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7771505
NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7756437
NS_A2_A2_B1_B2_B1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
NS_A2_A2_B1_B2_B1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7756523, upper bound: 2.7771505
NS_A2_A2_B1_B2_B2_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7616702, upper bound: 2.7672633
NS_A2_A2_B1_B2_B2_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7616702, upper bound: 2.7672633
NS_A2_A2_B1_B2_B2_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7603289, upper bound: 2.7651148
NS_A2_A2_B1_B2_B2_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7603289, upper bound: 2.7651148
NS_A2_A2_B1_B2_B2_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7634116, upper bound: 2.7651577
NS_A2_A2_B1_B2_B2_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7634116, upper bound: 2.7651577
NS_A2_A2_B1_B2_B2_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7616800, upper bound: 2.7673473
NS_A2_A2_B1_B2_B2_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7616800, upper bound: 2.7673473
NS_A2_A2_B1_B2_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
NS_A2_A2_B1_B2_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7644547, upper bound: 2.7657279
NS_A2_A2_B2_B2_B2_A1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7589985, upper bound: 2.7626461
NS_A2_A2_B2_B2_B2_A1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7589985, upper bound: 2.7626461
NS_A2_A2_B2_B2_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7743022
NS_A2_A2_B2_B2_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7761991, upper bound: 2.7747715
NS_A2_A2_B2_B2_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746772
NS_A2_A2_B2_B2_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7753308
NS_A2_A2_B2_B2_B2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783336
NS_A2_A2_B2_B2_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7783335
NS_A2_A2_B2_B2_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7741407
NS_A2_A2_B2_B2_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7746774
NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7742810
NS_A2_A2_B2_B2_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7741109, upper bound: 2.7747346
NS_A2_A2_B2_B2_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7753440, upper bound: 2.7744466
NS_A2_A2_B2_B2_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7753440, upper bound: 2.7751748
NS_A2_A2_B2_B2_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7764309, upper bound: 2.7744466
NS_A2_A2_B2_B2_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7764309, upper bound: 2.7751746
NS_A2_A2_B2_B2_B2_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7603189, upper bound: 2.7640982
NS_A2_A2_B2_B2_B2_A2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7510546, upper bound: 2.7536455
NS_A2_A2_B2_B2_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7756245, upper bound: 2.7745399
NS_A2_A2_B2_B2_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.77
Output dim: 0, lower bound: -2.7756245, upper bound: 2.7754416

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2693014, 1.8415074, -0.2973586, 2.0464444, -2.3157458, 2.1388659
1: -0.3738544, 2.5427141, -0.4104910, 2.8293266, -3.2031810, 2.9532051
2: -0.8839430, 1.8124728, -0.9908106, 1.9982010, -2.8821440, 2.8032835
3: -0.7638226, 2.0339918, -0.8320768, 2.2798955, -3.0437181, 2.8660686
4: -0.9475492, 2.4048069, -1.0959523, 2.6567075, -3.6042566, 3.5007591

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7746846
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7746846
time: 0.36 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.2763575, 1.8685887, -0.2973586, 2.0464444, -2.3228021, 2.1659472
1: -0.3783313, 2.5875869, -0.4104910, 2.8293266, -3.2076578, 2.9980779
2: -0.9030623, 1.8312291, -0.9908106, 1.9982010, -2.9012632, 2.8220396
3: -0.7726189, 2.0697749, -0.8320768, 2.2798955, -3.0525146, 2.9018517
4: -0.9667039, 2.4314411, -1.0959523, 2.6567075, -3.6234114, 3.5273933

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7746846
time: 0.39 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755102, upper bound: 2.7746846
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.2841836, 1.9730074, -2.2884650, 2.3897624
1: -0.4214576, 2.9155774, -0.3954064, 2.7292490, -3.1507068, 3.3109837
2: -1.0235951, 2.0444477, -0.9519596, 1.9270574, -2.9506526, 2.9964073
3: -0.8554188, 2.3600154, -0.8037341, 2.1830540, -3.0384727, 3.1637495
4: -1.1441643, 2.7125952, -1.0309584, 2.5671232, -3.7112875, 3.7435536

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756990, upper bound: 2.7744474
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756990, upper bound: 2.7744474
time: 0.48 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.3092373, 2.0740407, -2.3894982, 2.4148161
1: -0.4214576, 2.9155774, -0.4123492, 2.8505728, -3.2720304, 3.3279266
2: -1.0235951, 2.0444477, -0.9831481, 2.0508795, -3.0744746, 3.0275958
3: -0.8554188, 2.3600154, -0.8371741, 2.3228741, -3.1782928, 3.1971896
4: -1.1441643, 2.7125952, -1.0971978, 2.6443734, -3.7885377, 3.8097930

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756990, upper bound: 2.7744474
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756990, upper bound: 2.7744474
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.2693014, 1.8415074, -0.3086333, 2.0736451, -2.3429465, 2.1501408
1: -0.3738544, 2.5427141, -0.4168339, 2.8735497, -3.2474041, 2.9595480
2: -0.8839430, 1.8124728, -1.0126864, 2.0177553, -2.9016981, 2.8251591
3: -0.7638226, 2.0339918, -0.8451283, 2.3225791, -3.0864017, 2.8791201
4: -0.9475492, 2.4048069, -1.1221292, 2.6829336, -3.6304827, 3.5269361

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7773864
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7773862
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.2763575, 1.8685887, -0.3086333, 2.0736451, -2.3500028, 2.1772220
1: -0.3783313, 2.5875869, -0.4168339, 2.8735497, -3.2518811, 3.0044208
2: -0.9030623, 1.8312291, -1.0126864, 2.0177553, -2.9208176, 2.8439155
3: -0.7726189, 2.0697749, -0.8451283, 2.3225791, -3.0951982, 2.9149032
4: -0.9667039, 2.4314411, -1.1221292, 2.6829336, -3.6496375, 3.5535703

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7759374
time: 0.41 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7740071, upper bound: 2.7759374
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3154575, 2.1055789, -0.2944492, 1.9961758, -2.3116333, 2.4000282
1: -0.4214576, 2.9155774, -0.4010615, 2.7686687, -3.1901264, 3.3166389
2: -1.0235951, 2.0444477, -0.9719038, 1.9433281, -2.9669232, 3.0163515
3: -0.8554188, 2.3600154, -0.8153162, 2.2201476, -3.0755663, 3.1753316
4: -1.1441643, 2.7125952, -1.0539017, 2.5894523, -3.7336166, 3.7664969

Time for backsubstitution: 1.77 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.52 + 418.49 = 421.01 seconds
