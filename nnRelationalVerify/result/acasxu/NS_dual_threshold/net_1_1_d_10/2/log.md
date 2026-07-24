## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 1.0495482984


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3035725, 0.8847764, -0.3035725, 0.8847764, -1.1883489, 1.1883489)
1: (-0.5660125, 1.0933844, -0.5660125, 1.0933844, -1.6593968, 1.6593965)
2: (-0.4826685, 1.2412479, -0.4826685, 1.2412479, -1.7239163, 1.7239164)
3: (-0.9617165, 1.2755736, -0.9617165, 1.2755736, -2.2372897, 2.2372897)
4: (-0.8354526, 1.4994075, -0.8354526, 1.4994075, -2.3348601, 2.3348601)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 0.97 = 1.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0558836, upper bound: 1.0558836

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511832
time: 0.26 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.59 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511832
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.2352601, 0.7538407, -0.3035725, 0.8847764, -1.1200366, 1.0574131
1: -0.4706453, 0.9488738, -0.5660125, 1.0933844, -1.5640295, 1.5148864
2: -0.3915833, 1.0723588, -0.4826685, 1.2412479, -1.6328310, 1.5550274
3: -0.8320177, 1.0878556, -0.9617165, 1.2755736, -2.1075912, 2.0495720
4: -0.7029035, 1.3031529, -0.8354526, 1.4994075, -2.2023110, 2.1386056

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.26 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.26 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -0.3035725, 0.8847764, -2.2119637, 3.0038373
1: -1.8266034, 2.9257355, -0.5660125, 1.0933844, -2.9199877, 3.4917479
2: -1.7866864, 3.3100519, -0.4826685, 1.2412479, -3.0279343, 3.7927201
3: -2.3538351, 3.8103127, -0.9617165, 1.2755736, -3.6294079, 4.7720289
4: -2.7129741, 3.8588223, -0.8354526, 1.4994075, -4.2123814, 4.6942744

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.28 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.30 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.30
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.2352601, 0.7538407, -0.2352601, 0.7538407, -0.9891008, 0.9891006
1: -0.4706453, 0.9488738, -0.4706453, 0.9488738, -1.4195192, 1.4195192
2: -0.3915833, 1.0723588, -0.3915833, 1.0723588, -1.4639422, 1.4639422
3: -0.8320177, 1.0878556, -0.8320177, 1.0878556, -1.9198732, 1.9198732
4: -0.7029035, 1.3031529, -0.7029035, 1.3031529, -2.0060563, 2.0060563

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513471, upper bound: 1.0483656
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0552122, upper bound: 1.0511409
time: 0.32 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.2352601, 0.7538407, -1.3271873, 2.7002649, -2.9355247, 2.0810280
1: -0.4706453, 0.9488738, -1.8266034, 2.9257355, -3.3963809, 2.7754772
2: -0.3915833, 1.0723588, -1.7866864, 3.3100519, -3.7016354, 2.8590453
3: -0.8320177, 1.0878556, -2.3538351, 3.8103127, -4.6423302, 3.4416907
4: -0.7029035, 1.3031529, -2.7129741, 3.8588223, -4.5617256, 4.0161266

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511352
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522625, upper bound: 1.0507646
time: 0.38 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
time: 0.28 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.37 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507511, upper bound: 1.0507166
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.36 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0513471, upper bound: 1.0483656
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0552122, upper bound: 1.0511409
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511352
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0522625, upper bound: 1.0507646
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0507166, upper bound: 1.0507511
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0507511, upper bound: 1.0507166
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.36
Output dim: 0, lower bound: -1.0507030, upper bound: 1.0507030

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.2352601, 0.7538407, -0.8688198, 0.8147694
1: -0.2967805, 0.7322532, -0.4706453, 0.9488738, -1.2456543, 1.2028985
2: -0.2199608, 0.8229321, -0.3915833, 1.0723588, -1.2923197, 1.2145154
3: -0.5989540, 0.8103240, -0.8320177, 1.0878556, -1.6868094, 1.6423416
4: -0.4461793, 0.9936326, -0.7029035, 1.3031529, -1.7493322, 1.6965361

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0530660, upper bound: 1.0518593
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535924, upper bound: 1.0517559
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540940, upper bound: 1.0541796
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2312334, 0.7488964, -0.2352601, 0.7538407, -0.9850740, 0.9841565
1: -0.4649892, 0.9427366, -0.4706453, 0.9488738, -1.4138629, 1.4133818
2: -0.3858119, 1.0657110, -0.3915833, 1.0723588, -1.4581707, 1.4572943
3: -0.8264768, 1.0796602, -0.8320177, 1.0878556, -1.9143324, 1.9116778
4: -0.6945735, 1.2946483, -0.7029035, 1.3031529, -1.9977264, 1.9975514

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553682, upper bound: 1.0538194
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538548, upper bound: 1.0538548
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2121267, 0.7201707, -1.3271873, 2.7002649, -2.9123917, 2.0473580
1: -0.4377390, 0.9047500, -1.8266034, 2.9257355, -3.3634744, 2.7313535
2: -0.3605663, 1.0264064, -1.7866864, 3.3100519, -3.6706183, 2.8130927
3: -0.7864621, 1.0343318, -2.3538351, 3.8103127, -4.5967746, 3.3881669
4: -0.6582627, 1.2474420, -2.7129741, 3.8588223, -4.5170841, 3.9604161

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0489561, upper bound: 1.0405855
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0509748
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535849, upper bound: 1.0510412
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2998355, 0.9004637, -1.3111556, 2.6662879, -2.9661231, 2.2116191
1: -0.5571232, 1.1422836, -1.8055077, 2.8884964, -3.4456196, 2.9477911
2: -0.4746163, 1.2865998, -1.7664661, 3.2709689, -3.7455852, 3.0530658
3: -0.9696201, 1.2997241, -2.3283195, 3.7628031, -4.7324228, 3.6280432
4: -0.8469371, 1.4981039, -2.6840961, 3.8145638, -4.6615005, 4.1822000

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507602, upper bound: 1.0497692
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515450, upper bound: 1.0506707
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -0.2121267, 0.7201707, -2.0473580, 2.9123917
1: -1.8266034, 2.9257355, -0.4377390, 0.9047500, -2.7313535, 3.3634744
2: -1.7866864, 3.3100519, -0.3605663, 1.0264064, -2.8130927, 3.6706183
3: -2.3538351, 3.8103127, -0.7864621, 1.0343318, -3.3881669, 4.5967746
4: -2.7129741, 3.8588223, -0.6582627, 1.2474420, -3.9604161, 4.5170846

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0405855, upper bound: 1.0489561
time: 0.27 seconds

## Relational analysis of NS_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0549353
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510412, upper bound: 1.0535849
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.3111556, 2.6662879, -0.2998355, 0.9004637, -2.2116189, 2.9661233
1: -1.8055077, 2.8884964, -0.5571232, 1.1422836, -2.9477909, 3.4456196
2: -1.7664661, 3.2709689, -0.4746163, 1.2865998, -3.0530658, 3.7455852
3: -2.3283195, 3.7628031, -0.9696201, 1.2997241, -3.6280432, 4.7324224
4: -2.6840961, 3.8145638, -0.8469371, 1.4981039, -4.1822000, 4.6615000

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0497692, upper bound: 1.0507602
time: 0.36 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506707, upper bound: 1.0515450
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.3108072, 2.6636837, -1.3271873, 2.7002649, -4.0110717, 3.9908710
1: -1.8037789, 2.8828609, -1.8266034, 2.9257355, -4.7295141, 4.7094636
2: -1.7677765, 3.2624910, -1.7866864, 3.3100519, -5.0778284, 5.0491767
3: -2.3203149, 3.7582724, -2.3538351, 3.8103127, -6.1306272, 6.1121073
4: -2.6851826, 3.8056340, -2.7129741, 3.8588223, -6.5440044, 6.5186081

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487246, upper bound: 1.0489573
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0478075, upper bound: 1.0476278
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.3019793, 2.7201226, -1.3111556, 2.6662879, -3.9682670, 4.0312781
1: -1.7827781, 2.9920893, -1.8055077, 2.8884964, -4.6712747, 4.7975965
2: -1.7184958, 3.3870904, -1.7664661, 3.2709689, -4.9894648, 5.1535559
3: -2.3666272, 3.8082108, -2.3283195, 3.7628031, -6.1294303, 6.1365304
4: -2.6377411, 3.8847511, -2.6840961, 3.8145638, -6.4523048, 6.5688472

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0493781, upper bound: 1.0497338
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0486990, upper bound: 1.0486990
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.38 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0535924, upper bound: 1.0517559
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0540940, upper bound: 1.0541796
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0553682, upper bound: 1.0538194
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0538548, upper bound: 1.0538548
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0509748
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0535849, upper bound: 1.0510412
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0507602, upper bound: 1.0497692
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0515450, upper bound: 1.0506707
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0549353
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0510412, upper bound: 1.0535849
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0497692, upper bound: 1.0507602
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0506707, upper bound: 1.0515450
NS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0487246, upper bound: 1.0489573
NS_A2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0478075, upper bound: 1.0476278
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0493781, upper bound: 1.0497338
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 1.38
Output dim: 0, lower bound: -1.0486990, upper bound: 1.0486990

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.1999423, 0.7006042, -0.8155833, 0.7794516
1: -0.2967805, 0.7322532, -0.4179226, 0.8853608, -1.1821413, 1.1501758
2: -0.2199608, 0.8229321, -0.3408493, 0.9954937, -1.2154546, 1.1637814
3: -0.5989540, 0.8103240, -0.7652031, 1.0039408, -1.6028948, 1.5755270
4: -0.4461793, 0.9936326, -0.6261964, 1.2012441, -1.6474234, 1.6198289

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535637, upper bound: 1.0504967
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535924, upper bound: 1.0517559
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0532250, upper bound: 1.0513534
time: 0.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534079, upper bound: 1.0517310
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535924, upper bound: 1.0516925
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.2121059, 0.7206463, -0.8356254, 0.7916151
1: -0.2967805, 0.7322532, -0.4365718, 0.9153755, -1.2121559, 1.1688250
2: -0.2199608, 0.8229321, -0.3573927, 1.0247047, -1.2446655, 1.1803248
3: -0.5989540, 0.8103240, -0.7891545, 1.0367652, -1.6357192, 1.5994785
4: -0.4461793, 0.9936326, -0.6509602, 1.2383894, -1.6845686, 1.6445928

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540746, upper bound: 1.0539902
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527503, upper bound: 1.0534314
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.1979212, 0.6986737, -0.2352601, 0.7538407, -0.9517618, 0.9339339
1: -0.4156915, 0.8787559, -0.4706453, 0.9488738, -1.3645654, 1.3494012
2: -0.3415477, 0.9951348, -0.3915833, 1.0723588, -1.4139066, 1.3867182
3: -0.7555668, 1.0017967, -0.8320177, 1.0878556, -1.8434224, 1.8338144
4: -0.6297117, 1.2078556, -0.7029035, 1.3031529, -1.9328642, 1.9107591

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550931, upper bound: 1.0518475
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508516, upper bound: 1.0508642
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.2328679, 0.7384187, -0.2352601, 0.7538407, -0.9867086, 0.9736788
1: -0.4674057, 0.9188583, -0.4706453, 0.9488738, -1.4162796, 1.3895036
2: -0.3912686, 1.0532951, -0.3915833, 1.0723588, -1.4636275, 1.4448785
3: -0.8202137, 1.0752358, -0.8320177, 1.0878556, -1.9080693, 1.9072535
4: -0.7066647, 1.2945275, -0.7029035, 1.3031529, -2.0098176, 1.9974310

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0534958, upper bound: 1.0519323
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515733, upper bound: 1.0515733
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -1.3271873, 2.7002649, -2.8853905, 2.0055187
1: -0.3969364, 0.8526834, -1.8266034, 2.9257355, -3.3226719, 2.6792867
2: -0.3242711, 0.9644121, -1.7866864, 3.3100519, -3.6343226, 2.7510986
3: -0.7251614, 0.9704387, -2.3538351, 3.8103127, -4.5354738, 3.3242738
4: -0.6038694, 1.1718525, -2.7129741, 3.8588223, -4.4626908, 3.8848267

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0504677
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0491670
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.3271873, 2.7002649, -2.9125235, 2.0368783
1: -0.4381096, 0.8818882, -1.8266034, 2.9257355, -3.3638449, 2.7084916
2: -0.3631817, 1.0126708, -1.7866864, 3.3100519, -3.6732335, 2.7993572
3: -0.7779223, 1.0290604, -2.3538351, 3.8103127, -4.5882349, 3.3828955
4: -0.6656082, 1.2431200, -2.7129741, 3.8588223, -4.5244298, 3.9560940

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.2699254, 0.8529468, -1.3111556, 2.6662879, -2.9362133, 2.1641021
1: -0.5107998, 1.0817231, -1.8055077, 2.8884964, -3.3992960, 2.8872306
2: -0.4349182, 1.2233939, -1.7664661, 3.2709689, -3.7058871, 2.9898598
3: -0.9060124, 1.2251596, -2.3283195, 3.7628031, -4.6688156, 3.5534792
4: -0.7885253, 1.4212955, -2.6840961, 3.8145638, -4.6030893, 4.1053915

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0492938
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499365, upper bound: 1.0483732
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.3204554, 0.9300180, -1.3111556, 2.6662879, -2.9867432, 2.2411735
1: -0.5845242, 1.1713222, -1.8055077, 2.8884964, -3.4730206, 2.9768300
2: -0.5056426, 1.3326001, -1.7664661, 3.2709689, -3.7766116, 3.0990663
3: -1.0023541, 1.3492649, -2.3283195, 3.7628031, -4.7651567, 3.6775842
4: -0.8973903, 1.5542920, -2.6840961, 3.8145638, -4.7119541, 4.2383881

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507073, upper bound: 1.0499679
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500790, upper bound: 1.0486672
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -0.1851257, 0.6783313, -2.0055187, 2.8853903
1: -1.8266034, 2.9257355, -0.3969364, 0.8526834, -2.6792867, 3.3226719
2: -1.7866864, 3.3100519, -0.3242711, 0.9644121, -2.7510986, 3.6343231
3: -2.3538351, 3.8103127, -0.7251614, 0.9704387, -3.3242738, 4.5354738
4: -2.7129741, 3.8588223, -0.6038694, 1.1718525, -3.8848267, 4.4626908

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544712
time: 0.27 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -0.2122585, 0.7096910, -2.0368783, 2.9125235
1: -1.8266034, 2.9257355, -0.4381096, 0.8818882, -2.7084916, 3.3638451
2: -1.7866864, 3.3100519, -0.3631817, 1.0126708, -2.7993572, 3.6732330
3: -2.3538351, 3.8103127, -0.7779223, 1.0290604, -3.3828955, 4.5882349
4: -2.7129741, 3.8588223, -0.6656082, 1.2431200, -3.9560940, 4.5244298

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -1.3111556, 2.6662879, -0.2699254, 0.8529468, -2.1641021, 2.9362133
1: -1.8055077, 2.8884964, -0.5107998, 1.0817231, -2.8872302, 3.3992963
2: -1.7664661, 3.2709689, -0.4349182, 1.2233939, -2.9898601, 3.7058871
3: -2.3283195, 3.7628031, -0.9060124, 1.2251596, -3.5534790, 4.6688156
4: -2.6840961, 3.8145638, -0.7885253, 1.4212955, -4.1053915, 4.6030893

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0499365
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -1.3111556, 2.6662879, -0.3204554, 0.9300180, -2.2411735, 2.9867432
1: -1.8055077, 2.8884964, -0.5845242, 1.1713222, -2.9768300, 3.4730206
2: -1.7664661, 3.2709689, -0.5056426, 1.3326001, -3.0990663, 3.7766116
3: -2.3283195, 3.7628031, -1.0023541, 1.3492649, -3.6775842, 4.7651567
4: -2.6840961, 3.8145638, -0.8973903, 1.5542920, -4.2383881, 4.7119541

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0500790
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3019793, 2.7201226, -1.2485261, 2.5536690, -3.8556483, 3.9686484
1: -1.7827781, 2.9920893, -1.7301784, 2.7653055, -4.5480833, 4.7222676
2: -1.7184958, 3.3870904, -1.6982117, 3.1253252, -4.8438206, 5.0853014
3: -2.3666272, 3.8082108, -2.2293622, 3.6074800, -5.9741073, 6.0375729
4: -2.6377411, 3.8847511, -2.5856273, 3.6531067, -6.2908478, 6.4703784

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0484131, upper bound: 1.0483951
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484131, upper bound: 1.0496440
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.50 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0534079, upper bound: 1.0517310
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0535924, upper bound: 1.0516925
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0540746, upper bound: 1.0539902
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0527503, upper bound: 1.0534314
NS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0550931, upper bound: 1.0518475
NS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0508516, upper bound: 1.0508642
NS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0534958, upper bound: 1.0519323
NS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0515733, upper bound: 1.0515733
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0504677
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0491670
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0504533, upper bound: 1.0492938
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0499365, upper bound: 1.0483732
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0507073, upper bound: 1.0499679
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0500790, upper bound: 1.0486672
NS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
NS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544712
NS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
NS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0499365
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0483732, upper bound: 1.0500790
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0484131, upper bound: 1.0483951
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.50
Output dim: 0, lower bound: -1.0484131, upper bound: 1.0496440

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.1451201, 0.6235247, -0.7385038, 0.7246294
1: -0.2967805, 0.7322532, -0.3407811, 0.7839735, -1.0807539, 1.0730343
2: -0.2199608, 0.8229321, -0.2641292, 0.8805671, -1.1005280, 1.0870612
3: -0.5989540, 0.8103240, -0.6625596, 0.8773333, -1.4762874, 1.4728836
4: -0.4461793, 0.9936326, -0.5096070, 1.0692580, -1.5154372, 1.5032396

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.1824228, 0.6756839, -0.7906630, 0.7619320
1: -0.2967805, 0.7322532, -0.3915790, 0.8555044, -1.1522849, 1.1238322
2: -0.2199608, 0.8229321, -0.3154528, 0.9602969, -1.1802577, 1.1383848
3: -0.5989540, 0.8103240, -0.7325594, 0.9634358, -1.5623897, 1.5428834
4: -0.4461793, 0.9936326, -0.5876749, 1.1569469, -1.6031262, 1.5813075

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.2052484, 0.7111984, -0.8261776, 0.7847576
1: -0.2967805, 0.7322532, -0.4266216, 0.9039569, -1.2007375, 1.1588748
2: -0.2199608, 0.8229321, -0.3478366, 1.0107183, -1.2306792, 1.1707687
3: -0.5989540, 0.8103240, -0.7762232, 1.0212073, -1.6201613, 1.5865471
4: -0.4461793, 0.9936326, -0.6368161, 1.2210026, -1.6671818, 1.6304487

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540746, upper bound: 1.0539902
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522968, upper bound: 1.0505129
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0540746, upper bound: 1.0539902
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.3563843, 0.8891286, -1.0041078, 0.9358935
1: -0.2967805, 0.7322532, -0.6305937, 1.1183617, -1.4151422, 1.3628469
2: -0.2199608, 0.8229321, -0.5588591, 1.2487665, -1.4687274, 1.3817911
3: -0.5989540, 0.8103240, -1.0174500, 1.3245214, -1.9234754, 1.8277739
4: -0.4461793, 0.9936326, -0.9522445, 1.5367720, -1.9829513, 1.9458771

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0494066, upper bound: 1.0510647
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491234, upper bound: 1.0504277
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519738, upper bound: 1.0534314
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519738, upper bound: 1.0534314
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1792509, 0.6726518, -0.2352601, 0.7538407, -0.9330916, 0.9079119
1: -0.3888319, 0.8461026, -0.4706453, 0.9488738, -1.3377056, 1.3167479
2: -0.3158575, 0.9569659, -0.3915833, 1.0723588, -1.3882163, 1.3485492
3: -0.7172773, 0.9604697, -0.8320177, 1.0878556, -1.8051329, 1.7924874
4: -0.5917992, 1.1613320, -0.7029035, 1.3031529, -1.8949521, 1.8642355

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548306, upper bound: 1.0510817
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548723, upper bound: 1.0508626
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.2649441, 0.8246372, -0.2285286, 0.7422867, -1.0072308, 1.0531657
1: -0.5047384, 1.0513973, -0.4600134, 0.9343750, -1.4391130, 1.5114107
2: -0.4291972, 1.1954941, -0.3819052, 1.0572844, -1.4864815, 1.5773993
3: -0.8983393, 1.1942482, -0.8200858, 1.0694902, -1.9678295, 2.0143340
4: -0.7814043, 1.4037045, -0.6887528, 1.2839000, -2.0653043, 2.0924573

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505401, upper bound: 1.0502919
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502519, upper bound: 1.0500871
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.2085951, 0.7061945, -0.2352601, 0.7538407, -0.9624357, 0.9414546
1: -0.4330689, 0.8780445, -0.4706453, 0.9488738, -1.3819427, 1.3486898
2: -0.3577549, 1.0080304, -0.3915833, 1.0723588, -1.4301138, 1.3996137
3: -0.7733389, 1.0228337, -0.8320177, 1.0878556, -1.8611945, 1.8548514
4: -0.6577712, 1.2365112, -0.7029035, 1.3031529, -1.9609239, 1.9394147

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519147, upper bound: 1.0509997
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514975, upper bound: 1.0507857
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.2995591, 0.8546940, -0.2285286, 0.7422867, -1.0418458, 1.0832226
1: -0.5560523, 1.0767007, -0.4600134, 0.9343750, -1.4904270, 1.5367141
2: -0.4788076, 1.2320927, -0.3819052, 1.0572844, -1.5360919, 1.6139979
3: -0.9559603, 1.2597048, -0.8200858, 1.0694902, -2.0254505, 2.0797906
4: -0.8567076, 1.4731483, -0.6887528, 1.2839000, -2.1406076, 2.1619010

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507416, upper bound: 1.0505925
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503784, upper bound: 1.0503784
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -1.2651416, 2.5885944, -2.7737200, 1.9434729
1: -0.3969364, 0.8526834, -1.7519391, 2.8037324, -3.2006688, 2.6046224
2: -0.3242711, 0.9644121, -1.7190838, 3.1656418, -3.4899130, 2.6834960
3: -0.7251614, 0.9704387, -2.2556522, 3.6565268, -4.3816881, 3.2260909
4: -0.6038694, 1.1718525, -2.6154146, 3.6987345, -4.3026037, 3.7872672

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543338, upper bound: 1.0504677
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0504677
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -1.2914166, 2.6484618, -2.8335872, 1.9697480
1: -0.3969364, 0.8526834, -1.7824845, 2.8723085, -3.2692444, 2.6351678
2: -0.3242711, 0.9644121, -1.7462265, 3.2360384, -3.5603094, 2.7106385
3: -0.7251614, 0.9704387, -2.2972631, 3.7364984, -4.4616594, 3.2677019
4: -0.6038694, 1.1718525, -2.6558306, 3.7713811, -4.3752503, 3.8276830

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541095, upper bound: 1.0491670
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544712, upper bound: 1.0491670
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.2651416, 2.5885944, -2.8008530, 1.9748325
1: -0.4381096, 0.8818882, -1.7519391, 2.8037324, -3.2418420, 2.6338272
2: -0.3631817, 1.0126708, -1.7190838, 3.1656418, -3.5288234, 2.7317545
3: -0.7779223, 1.0290604, -2.2556522, 3.6565268, -4.4344492, 3.2847126
4: -0.6656082, 1.2431200, -2.6154146, 3.6987345, -4.3643427, 3.8585346

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.2914166, 2.6484618, -2.8607204, 2.0011077
1: -0.4381096, 0.8818882, -1.7824845, 2.8723085, -3.3104179, 2.6643727
2: -0.3631817, 1.0126708, -1.7462265, 3.2360384, -3.5992198, 2.7588973
3: -0.7779223, 1.0290604, -2.2972631, 3.7364984, -4.5144205, 3.3263235
4: -0.6656082, 1.2431200, -2.6558306, 3.7713811, -4.4369893, 3.8989506

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2699254, 0.8529468, -1.2485261, 2.5536690, -2.8235943, 2.1014729
1: -0.5107998, 1.0817231, -1.7301784, 2.7653055, -3.2761047, 2.8119016
2: -0.4349182, 1.2233939, -1.6982117, 3.1253252, -3.5602434, 2.9216056
3: -0.9060124, 1.2251596, -2.2293622, 3.6074800, -4.5134926, 3.4545219
4: -0.7885253, 1.4212955, -2.5856273, 3.6531067, -4.4416318, 4.0069227

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503338, upper bound: 1.0488995
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2699254, 0.8529468, -1.2754259, 2.6143732, -2.8842983, 2.1283722
1: -0.5107998, 1.0817231, -1.7613282, 2.8349187, -3.3457179, 2.8430512
2: -0.4349182, 1.2233939, -1.7258899, 3.1967897, -3.6317079, 2.9492838
3: -0.9060124, 1.2251596, -2.2717423, 3.6887112, -4.5947237, 3.4969020
4: -0.7885253, 1.4212955, -2.6267328, 3.7268844, -4.5154095, 4.0480280

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498917, upper bound: 1.0482416
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3204554, 0.9300180, -1.2485261, 2.5536690, -2.8741243, 2.1785440
1: -0.5845242, 1.1713222, -1.7301784, 2.7653055, -3.3498292, 2.9015007
2: -0.5056426, 1.3326001, -1.6982117, 3.1253252, -3.6309679, 3.0308118
3: -1.0023541, 1.3492649, -2.2293622, 3.6074800, -4.6098342, 3.5786271
4: -0.8973903, 1.5542920, -2.5856273, 3.6531067, -4.5504971, 4.1399193

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3204554, 0.9300180, -1.2754259, 2.6143732, -2.9348283, 2.2054439
1: -0.5845242, 1.1713222, -1.7613282, 2.8349187, -3.4194427, 2.9326506
2: -0.5056426, 1.3326001, -1.7258899, 3.1967897, -3.7024324, 3.0584900
3: -1.0023541, 1.3492649, -2.2717423, 3.6887112, -4.6910653, 3.6210070
4: -0.8973903, 1.5542920, -2.6267328, 3.7268844, -4.6242747, 4.1810246

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0482167
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0486524
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1851257, 0.6783313, -1.9434729, 2.7737203
1: -1.7519391, 2.8037324, -0.3969364, 0.8526834, -2.6046224, 3.2006688
2: -1.7190838, 3.1656418, -0.3242711, 0.9644121, -2.6834960, 3.4899130
3: -2.2556522, 3.6565268, -0.7251614, 0.9704387, -3.2260909, 4.3816881
4: -2.6154146, 3.6987345, -0.6038694, 1.1718525, -3.7872672, 4.3026037

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543338
time: 0.35 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1.2914166, 2.6484618, -0.1851257, 0.6783313, -1.9697480, 2.8335872
1: -1.7824845, 2.8723085, -0.3969364, 0.8526834, -2.6351678, 3.2692449
2: -1.7462265, 3.2360384, -0.3242711, 0.9644121, -2.7106385, 3.5603094
3: -2.2972631, 3.7364984, -0.7251614, 0.9704387, -3.2677019, 4.4616594
4: -2.6558306, 3.7713811, -0.6038694, 1.1718525, -3.8276830, 4.3752503

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0541095
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544712
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.2122585, 0.7096910, -1.9748325, 2.8008530
1: -1.7519391, 2.8037324, -0.4381096, 0.8818882, -2.6338272, 3.2418418
2: -1.7190838, 3.1656418, -0.3631817, 1.0126708, -2.7317545, 3.5288229
3: -2.2556522, 3.6565268, -0.7779223, 1.0290604, -3.2847126, 4.4344492
4: -2.6154146, 3.6987345, -0.6656082, 1.2431200, -3.8585346, 4.3643427

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.28 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.2914166, 2.6484618, -0.2122585, 0.7096910, -2.0011077, 2.8607204
1: -1.7824845, 2.8723085, -0.4381096, 0.8818882, -2.6643727, 3.3104179
2: -1.7462265, 3.2360384, -0.3631817, 1.0126708, -2.7588973, 3.5992193
3: -2.2972631, 3.7364984, -0.7779223, 1.0290604, -3.3263235, 4.5144205
4: -2.6558306, 3.7713811, -0.6656082, 1.2431200, -3.8989506, 4.4369893

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1.2485261, 2.5536690, -0.2699254, 0.8529468, -2.1014726, 2.8235943
1: -1.7301784, 2.7653055, -0.5107998, 1.0817231, -2.8119009, 3.2761045
2: -1.6982117, 3.1253252, -0.4349182, 1.2233939, -2.9216056, 3.5602434
3: -2.2293622, 3.6074800, -0.9060124, 1.2251596, -3.4545219, 4.5134926
4: -2.5856273, 3.6531067, -0.7885253, 1.4212955, -4.0069227, 4.4416318

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0488995, upper bound: 1.0503338
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504370
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1.2754259, 2.6143732, -0.2699254, 0.8529468, -2.1283724, 2.8842986
1: -1.7613282, 2.8349187, -0.5107998, 1.0817231, -2.8430510, 3.3457179
2: -1.7258899, 3.1967897, -0.4349182, 1.2233939, -2.9492836, 3.6317079
3: -2.2717423, 3.6887112, -0.9060124, 1.2251596, -3.4969020, 4.5947237
4: -2.6267328, 3.7268844, -0.7885253, 1.4212955, -4.0480280, 4.5154095

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0482416, upper bound: 1.0498917
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.2485261, 2.5536690, -0.3204554, 0.9300180, -2.1785440, 2.8741241
1: -1.7301784, 2.7653055, -0.5845242, 1.1713222, -2.9015007, 3.3498294
2: -1.6982117, 3.1253252, -0.5056426, 1.3326001, -3.0308118, 3.6309679
3: -2.2293622, 3.6074800, -1.0023541, 1.3492649, -3.5786271, 4.6098342
4: -2.5856273, 3.6531067, -0.8973903, 1.5542920, -4.1399193, 4.5504971

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0498760
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0500790
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.2754259, 2.6143732, -0.3204554, 0.9300180, -2.2054439, 2.9348283
1: -1.7613282, 2.8349187, -0.5845242, 1.1713222, -2.9326506, 3.4194427
2: -1.7258899, 3.1967897, -0.5056426, 1.3326001, -3.0584898, 3.7024324
3: -2.2717423, 3.6887112, -1.0023541, 1.3492649, -3.6210070, 4.6910653
4: -2.6267328, 3.7268844, -0.8973903, 1.5542920, -4.1810246, 4.6242747

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0491766
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0500790
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.3900077, 2.7811232, -1.2485261, 2.5536690, -3.9436765, 4.0296488
1: -1.8534610, 3.0589395, -1.7301784, 2.7653055, -4.6187668, 4.7891178
2: -1.7888100, 3.4775665, -1.6982117, 3.1253252, -4.9141350, 5.1757784
3: -2.4571981, 3.9132597, -2.2293622, 3.6074800, -6.0646782, 6.1426210
4: -2.7346168, 3.9865932, -2.5856273, 3.6531067, -6.3877230, 6.5722208

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0435239, upper bound: 1.0433981
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
time: 0.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.16 seconds
NS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0522968, upper bound: 1.0505129
NS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0540746, upper bound: 1.0539902
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0519738, upper bound: 1.0534314
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0519738, upper bound: 1.0534314
NS_A1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0548306, upper bound: 1.0510817
NS_A1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0548723, upper bound: 1.0508626
NS_A1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0505401, upper bound: 1.0502919
NS_A1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0502519, upper bound: 1.0500871
NS_A1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0519147, upper bound: 1.0509997
NS_A1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0514975, upper bound: 1.0507857
NS_A1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0507416, upper bound: 1.0505925
NS_A1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0503784, upper bound: 1.0503784
NS_A1_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0543338, upper bound: 1.0504677
NS_A1_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0504677
NS_A1_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0541095, upper bound: 1.0491670
NS_A1_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0544712, upper bound: 1.0491670
NS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
NS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0512565, upper bound: 1.0490802
NS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
NS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
NS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
NS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0504370, upper bound: 1.0492938
NS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0498917, upper bound: 1.0482416
NS_A1_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
NS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
NS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0498760, upper bound: 1.0486672
NS_A1_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0482167
NS_A1_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0491766, upper bound: 1.0486524
NS_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543338
NS_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
NS_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0541095
NS_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0491670, upper bound: 1.0544712
NS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
NS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
NS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
NS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
NS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504370
NS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0492938, upper bound: 1.0504533
NS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0482416, upper bound: 1.0498917
NS_A2_B1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
NS_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0498760
NS_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0486672, upper bound: 1.0500790
NS_A2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0491766
NS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0500790
NS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
NS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.16
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.1553145, 0.6463247, -0.7613039, 0.7348238
1: -0.2967805, 0.7322532, -0.3591110, 0.8135738, -1.1103542, 1.0913641
2: -0.2199608, 0.8229321, -0.2793274, 0.9168239, -1.1367848, 1.1022594
3: -0.5989540, 0.8103240, -0.6974133, 0.9111461, -1.5101001, 1.5077373
4: -0.4461793, 0.9936326, -0.5338272, 1.1137388, -1.5599180, 1.5274599

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518497, upper bound: 1.0501072
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522968, upper bound: 1.0505129
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0522968, upper bound: 1.0505129
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1149792, 0.5795093, -0.1826628, 0.6760664, -0.7910455, 0.7621720
1: -0.2967805, 0.7322532, -0.3921647, 0.8608914, -1.1576719, 1.1244179
2: -0.2199608, 0.8229321, -0.3148580, 0.9619305, -1.1818912, 1.1377900
3: -0.5989540, 0.8103240, -0.7321308, 0.9645033, -1.5634573, 1.5424547
4: -0.4461793, 0.9936326, -0.5864717, 1.1600819, -1.6062611, 1.5801044

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537968, upper bound: 1.0539112
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535709, upper bound: 1.0539902
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535709, upper bound: 1.0539902
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1093142, 0.5719484, -0.3563843, 0.8891286, -0.9984429, 0.9283327
1: -0.2884543, 0.7235280, -0.6305937, 1.1183617, -1.4068160, 1.3541217
2: -0.2119465, 0.8112619, -0.5588591, 1.2487665, -1.4607130, 1.3701210
3: -0.5873810, 0.7978517, -1.0174500, 1.3245214, -1.9119024, 1.8153017
4: -0.4342344, 0.9781362, -0.9522445, 1.5367720, -1.9710064, 1.9303808

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0527186
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0534314
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2213382, 0.6948912, -0.3563843, 0.8891286, -1.1104667, 1.0512755
1: -0.4437678, 0.8638017, -0.6305937, 1.1183617, -1.5621295, 1.4943954
2: -0.3757644, 0.9738847, -0.5588591, 1.2487665, -1.6245309, 1.5327438
3: -0.7661010, 1.0106292, -1.0174500, 1.3245214, -2.0906224, 2.0280790
4: -0.6784313, 1.2178802, -0.9522445, 1.5367720, -2.2152033, 2.1701248

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0522379
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0534314
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1792509, 0.6726518, -0.1999423, 0.7006042, -0.8798552, 0.8725941
1: -0.3888319, 0.8461026, -0.4179226, 0.8853608, -1.2741927, 1.2640252
2: -0.3158575, 0.9569659, -0.3408493, 0.9954937, -1.3113512, 1.2978152
3: -0.7172773, 0.9604697, -0.7652031, 1.0039408, -1.7212181, 1.7256727
4: -0.5917992, 1.1613320, -0.6261964, 1.2012441, -1.7930434, 1.7875284

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548306, upper bound: 1.0510215
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544462, upper bound: 1.0510817
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548306, upper bound: 1.0510817
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1792509, 0.6726518, -0.2121059, 0.7206463, -0.8998972, 0.8847576
1: -0.3888319, 0.8461026, -0.4365718, 0.9153755, -1.3042073, 1.2826744
2: -0.3158575, 0.9569659, -0.3573927, 1.0247047, -1.3405621, 1.3143586
3: -0.7172773, 0.9604697, -0.7891545, 1.0367652, -1.7540425, 1.7496243
4: -0.5917992, 1.1613320, -0.6509602, 1.2383894, -1.8301885, 1.8122922

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0507336
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0508626
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2649441, 0.8246372, -0.1937883, 0.6910462, -0.9559904, 1.0184255
1: -0.5047384, 1.0513973, -0.4080151, 0.8736901, -1.3784285, 1.4594125
2: -0.4291972, 1.1954941, -0.3316857, 0.9828165, -1.4120138, 1.5271797
3: -0.8983393, 1.1942482, -0.7543924, 0.9884150, -1.8867543, 1.9486406
4: -0.7814043, 1.4037045, -0.6126174, 1.1841940, -1.9655982, 2.0163219

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504513, upper bound: 1.0501119
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505401, upper bound: 1.0502919
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505401, upper bound: 1.0502919
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2649441, 0.8246372, -0.2052861, 0.7094477, -0.9743919, 1.0299233
1: -0.5047384, 1.0513973, -0.4257671, 0.9013068, -1.4060452, 1.4771644
2: -0.4291972, 1.1954941, -0.3475279, 1.0098722, -1.4390694, 1.5430219
3: -0.8983393, 1.1942482, -0.7770597, 1.0189474, -1.9172866, 1.9713080
4: -0.7814043, 1.4037045, -0.6364758, 1.2197304, -2.0011346, 2.0401802

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0499605
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0500871
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2085951, 0.7061945, -0.1999423, 0.7006042, -0.9091994, 0.9061369
1: -0.4330689, 0.8780445, -0.4179226, 0.8853608, -1.3184297, 1.2959671
2: -0.3577549, 1.0080304, -0.3408493, 0.9954937, -1.3532486, 1.3488797
3: -0.7733389, 1.0228337, -0.7652031, 1.0039408, -1.7772797, 1.7880368
4: -0.6577712, 1.2365112, -0.6261964, 1.2012441, -1.8590152, 1.8627076

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514975, upper bound: 1.0505740
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0514975, upper bound: 1.0505740
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2085951, 0.7061945, -0.2121059, 0.7206463, -0.9292414, 0.9183004
1: -0.4330689, 0.8780445, -0.4365718, 0.9153755, -1.3484443, 1.3146163
2: -0.3577549, 1.0080304, -0.3573927, 1.0247047, -1.3824596, 1.3654231
3: -0.7733389, 1.0228337, -0.7891545, 1.0367652, -1.8101041, 1.8119881
4: -0.6577712, 1.2365112, -0.6509602, 1.2383894, -1.8961606, 1.8874714

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0506591
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0507100
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2995591, 0.8546940, -0.1937883, 0.6910462, -0.9906054, 1.0484823
1: -0.5560523, 1.0767007, -0.4080151, 0.8736901, -1.4297425, 1.4847158
2: -0.4788076, 1.2320927, -0.3316857, 0.9828165, -1.4616241, 1.5637784
3: -0.9559603, 1.2597048, -0.7543924, 0.9884150, -1.9443753, 2.0140972
4: -0.8567076, 1.4731483, -0.6126174, 1.1841940, -2.0409017, 2.0857658

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501667, upper bound: 1.0501667
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501667, upper bound: 1.0501667
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2995591, 0.8546940, -0.2052861, 0.7094477, -1.0090069, 1.0599802
1: -0.5560523, 1.0767007, -0.4257671, 0.9013068, -1.4573591, 1.5024678
2: -0.4788076, 1.2320927, -0.3475279, 1.0098722, -1.4886798, 1.5796206
3: -0.9559603, 1.2597048, -0.7770597, 1.0189474, -1.9749076, 2.0367646
4: -0.8567076, 1.4731483, -0.6364758, 1.2197304, -2.0764380, 2.1096241

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0502519
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0503784
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -1.2651416, 2.5885944, -2.7982364, 1.9753036
1: -0.4267260, 0.8983275, -1.7519391, 2.8037324, -3.2304583, 2.6502666
2: -0.3558276, 1.0129061, -1.7190838, 3.1656418, -3.5214691, 2.7319899
3: -0.7740620, 1.0201761, -2.2556522, 3.6565268, -4.4305887, 3.2758284
4: -0.6563050, 1.2186592, -2.6154146, 3.6987345, -4.3550396, 3.8340738

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0543338, upper bound: 1.0504137
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -1.2651416, 2.5885944, -2.7632530, 1.9312744
1: -0.3825408, 0.8379728, -1.7519391, 2.8037324, -3.1862731, 2.5899119
2: -0.3095305, 0.9459749, -1.7190838, 3.1656418, -3.4751723, 2.6650586
3: -0.7079530, 0.9492630, -2.2556522, 3.6565268, -4.3644800, 3.2049153
4: -0.5819045, 1.1485283, -2.6154146, 3.6987345, -4.2806392, 3.7639430

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0504076
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -1.2914166, 2.6484618, -2.8581033, 2.0015788
1: -0.4267260, 0.8983275, -1.7824845, 2.8723085, -3.2990344, 2.6808119
2: -0.3558276, 1.0129061, -1.7462265, 3.2360384, -3.5918655, 2.7591326
3: -0.7740620, 1.0201761, -2.2972631, 3.7364984, -4.5105600, 3.3174391
4: -0.6563050, 1.2186592, -2.6558306, 3.7713811, -4.4276857, 3.8744898

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0541095, upper bound: 1.0491135
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0436953, upper bound: 1.0420088
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -1.2914166, 2.6484618, -2.8231201, 1.9575493
1: -0.3825408, 0.8379728, -1.7824845, 2.8723085, -3.2548492, 2.6204572
2: -0.3095305, 0.9459749, -1.7462265, 3.2360384, -3.5455689, 2.6922016
3: -0.7079530, 0.9492630, -2.2972631, 3.7364984, -4.4444509, 3.2465262
4: -0.5819045, 1.1485283, -2.6558306, 3.7713811, -4.3532858, 3.8043590

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544712, upper bound: 1.0491131
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490575, upper bound: 1.0439313
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1806883, 0.6643587, -1.2651416, 2.5885944, -2.7692828, 1.9295003
1: -0.3907865, 0.8332261, -1.7519391, 2.8037324, -3.1945188, 2.5851653
2: -0.3165183, 0.9454506, -1.7190838, 3.1656418, -3.4821601, 2.6645343
3: -0.7177029, 0.9569116, -2.2556522, 3.6565268, -4.3742294, 3.2125638
4: -0.5945891, 1.1518157, -2.6154146, 3.6987345, -4.2933235, 3.7672303

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1905538, 0.6793638, -1.2651416, 2.5885944, -2.7791481, 1.9445055
1: -0.4057388, 0.8529248, -1.7519391, 2.8037324, -3.2094710, 2.6048639
2: -0.3302413, 0.9696531, -1.7190838, 3.1656418, -3.4958830, 2.6887369
3: -0.7378664, 0.9809170, -2.2556522, 3.6565268, -4.3943930, 3.2365692
4: -0.6153858, 1.1839089, -2.6154146, 3.6987345, -4.3141203, 3.7993236

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.2661704, 2.5898392, -2.8020978, 1.9758613
1: -0.4381096, 0.8818882, -1.7478166, 2.8040323, -3.2421417, 2.6297047
2: -0.3631817, 1.0126708, -1.7160549, 3.1630225, -3.5262039, 2.7287257
3: -0.7779223, 1.0290604, -2.2481234, 3.6546252, -4.4325476, 3.2771838
4: -0.6656082, 1.2431200, -2.6115189, 3.6906371, -4.3562450, 3.8546388

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.3370345, 2.6126482, -2.8249068, 2.0467255
1: -0.4381096, 0.8818882, -1.7949092, 2.8271534, -3.2652628, 2.6767974
2: -0.3631817, 1.0126708, -1.7629795, 3.2088265, -3.5720079, 2.7756503
3: -0.7779223, 1.0290604, -2.3119197, 3.7054596, -4.4833817, 3.3409801
4: -0.6656082, 1.2431200, -2.6737659, 3.7467427, -4.4123507, 3.9168859

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.2699254, 0.8529468, -1.2491606, 2.5527794, -2.8227048, 2.1021070
1: -0.5107998, 1.0817231, -1.7293634, 2.7617764, -3.2725759, 2.8110857
2: -0.4349182, 1.2233939, -1.7005792, 3.1189947, -3.5539129, 2.9239726
3: -0.9060124, 1.2251596, -2.2224197, 3.6054430, -4.5114551, 3.4475794
4: -0.7885253, 1.4212955, -2.5882497, 3.6463084, -4.4348335, 4.0095453

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.2699254, 0.8529468, -1.2271219, 2.5771050, -2.8470304, 2.0800686
1: -0.5107998, 1.0817231, -1.6880672, 2.8331177, -3.3439171, 2.7697897
2: -0.4349182, 1.2233939, -1.6324024, 3.2002738, -3.6351919, 2.8557963
3: -0.9060124, 1.2251596, -2.2409055, 3.6114621, -4.5174747, 3.4660652
4: -0.7885253, 1.4212955, -2.5104861, 3.6776853, -4.4662104, 3.9317815

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.2699254, 0.8529468, -1.1687024, 2.3932176, -2.6631429, 2.0216491
1: -0.5107998, 1.0817231, -1.6323254, 2.6166544, -3.1274538, 2.7140479
2: -0.4349182, 1.2233939, -1.5996003, 2.9338965, -3.3688147, 2.8229938
3: -0.9060124, 1.2251596, -2.1112137, 3.3825173, -4.2885299, 3.3363733
4: -0.7885253, 1.4212955, -2.4405456, 3.4329128, -4.2214384, 3.8618402

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2847042, 0.8710025, -1.2485261, 2.5536690, -2.8383732, 2.1195285
1: -0.5346295, 1.1044348, -1.7301784, 2.7653055, -3.2999346, 2.8346133
2: -0.4548847, 1.2482237, -1.6982117, 3.1253252, -3.5802100, 2.9464355
3: -0.9340510, 1.2600147, -2.2293622, 3.6074800, -4.5415306, 3.4893768
4: -0.8210288, 1.4488646, -2.5856273, 3.6531067, -4.4741354, 4.0344920

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2971591, 0.8949012, -1.2485261, 2.5536690, -2.8508282, 2.1434269
1: -0.5497876, 1.1363558, -1.7301784, 2.7653055, -3.3150928, 2.8665342
2: -0.4700940, 1.2826568, -1.6982117, 3.1253252, -3.5954187, 2.9808683
3: -0.9585164, 1.2952633, -2.2293622, 3.6074800, -4.5659962, 3.5246255
4: -0.8436151, 1.4857011, -2.5856273, 3.6531067, -4.4967217, 4.0713282

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.2096419, 0.7101620, -1.9753036, 2.7982364
1: -1.7519391, 2.8037324, -0.4267260, 0.8983275, -2.6502666, 3.2304583
2: -1.7190838, 3.1656418, -0.3558276, 1.0129061, -2.7319899, 3.5214691
3: -2.2556522, 3.6565268, -0.7740620, 1.0201761, -3.2758284, 4.4305887
4: -2.6154146, 3.6987345, -0.6563050, 1.2186592, -3.8340738, 4.3550391

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504137, upper bound: 1.0543338
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543338
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1746585, 0.6661327, -1.9312744, 2.7632530
1: -1.7519391, 2.8037324, -0.3825408, 0.8379728, -2.5899119, 3.1862731
2: -1.7190838, 3.1656418, -0.3095305, 0.9459749, -2.6650586, 3.4751723
3: -2.2556522, 3.6565268, -0.7079530, 0.9492630, -3.2049153, 4.3644800
4: -2.6154146, 3.6987345, -0.5819045, 1.1485283, -3.7639430, 4.2806382

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504076, upper bound: 1.0544438
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.2914166, 2.6484618, -0.2096419, 0.7101620, -2.0015788, 2.8581035
1: -1.7824845, 2.8723085, -0.4267260, 0.8983275, -2.6808119, 3.2990346
2: -1.7462265, 3.2360384, -0.3558276, 1.0129061, -2.7591326, 3.5918658
3: -2.2972631, 3.7364984, -0.7740620, 1.0201761, -3.3174391, 4.5105605
4: -2.6558306, 3.7713811, -0.6563050, 1.2186592, -3.8744898, 4.4276853

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491135, upper bound: 1.0541095
time: 0.28 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0420088, upper bound: 1.0436953
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.2914166, 2.6484618, -0.1746585, 0.6661327, -1.9575493, 2.8231204
1: -1.7824845, 2.8723085, -0.3825408, 0.8379728, -2.6204572, 3.2548492
2: -1.7462265, 3.2360384, -0.3095305, 0.9459749, -2.6922016, 3.5455689
3: -2.2972631, 3.7364984, -0.7079530, 0.9492630, -3.2465262, 4.4444513
4: -2.6558306, 3.7713811, -0.5819045, 1.1485283, -3.8043590, 4.3532848

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0491131, upper bound: 1.0544712
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0439313, upper bound: 1.0490575
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1806883, 0.6643587, -1.9295003, 2.7692828
1: -1.7519391, 2.8037324, -0.3907865, 0.8332261, -2.5851653, 3.1945190
2: -1.7190838, 3.1656418, -0.3165183, 0.9454506, -2.6645343, 3.4821596
3: -2.2556522, 3.6565268, -0.7177029, 0.9569116, -3.2125638, 4.3742294
4: -2.6154146, 3.6987345, -0.5945891, 1.1518157, -3.7672303, 4.2933235

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1905538, 0.6793638, -1.9445055, 2.7791481
1: -1.7519391, 2.8037324, -0.4057388, 0.8529248, -2.6048639, 3.2094712
2: -1.7190838, 3.1656418, -0.3302413, 0.9696531, -2.6887369, 3.4958830
3: -2.2556522, 3.6565268, -0.7378664, 0.9809170, -3.2365692, 4.3943930
4: -2.6154146, 3.6987345, -0.6153858, 1.1839089, -3.7993236, 4.3141203

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1.2661704, 2.5898392, -0.2122585, 0.7096910, -1.9758613, 2.8020978
1: -1.7478166, 2.8040323, -0.4381096, 0.8818882, -2.6297047, 3.2421417
2: -1.7160549, 3.1630225, -0.3631817, 1.0126708, -2.7287257, 3.5262041
3: -2.2481234, 3.6546252, -0.7779223, 1.0290604, -3.2771838, 4.4325476
4: -2.6115189, 3.6906371, -0.6656082, 1.2431200, -3.8546388, 4.3562450

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.3370345, 2.6126482, -0.2122585, 0.7096910, -2.0467255, 2.8249068
1: -1.7949092, 2.8271534, -0.4381096, 0.8818882, -2.6767974, 3.2652631
2: -1.7629795, 3.2088265, -0.3631817, 1.0126708, -2.7756503, 3.5720081
3: -2.3119197, 3.7054596, -0.7779223, 1.0290604, -3.3409801, 4.4833817
4: -2.6737659, 3.7467427, -0.6656082, 1.2431200, -3.9168859, 4.4123507

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
time: 0.41 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1.2491606, 2.5527794, -0.2699254, 0.8529468, -2.1021070, 2.8227046
1: -1.7293634, 2.7617764, -0.5107998, 1.0817231, -2.8110857, 3.2725754
2: -1.7005792, 3.1189947, -0.4349182, 1.2233939, -2.9239731, 3.5539129
3: -2.2224197, 3.6054430, -0.9060124, 1.2251596, -3.4475791, 4.5114546
4: -2.5882497, 3.6463084, -0.7885253, 1.4212955, -4.0095453, 4.4348335

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1.2271219, 2.5771050, -0.2699254, 0.8529468, -2.0800686, 2.8470304
1: -1.6880672, 2.8331177, -0.5107998, 1.0817231, -2.7697899, 3.3439171
2: -1.6324024, 3.2002738, -0.4349182, 1.2233939, -2.8557963, 3.6351919
3: -2.2409055, 3.6114621, -0.9060124, 1.2251596, -3.4660652, 4.5174742
4: -2.5104861, 3.6776853, -0.7885253, 1.4212955, -3.9317815, 4.4662099

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1.1687024, 2.3932176, -0.2699254, 0.8529468, -2.0216489, 2.6631429
1: -1.6323254, 2.6166544, -0.5107998, 1.0817231, -2.7140477, 3.1274536
2: -1.5996003, 2.9338965, -0.4349182, 1.2233939, -2.8229942, 3.3688147
3: -2.1112137, 3.3825173, -0.9060124, 1.2251596, -3.3363733, 4.2885294
4: -2.4405456, 3.4329128, -0.7885253, 1.4212955, -3.8618405, 4.2214384

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.2485261, 2.5536690, -0.2847042, 0.8710025, -2.1195285, 2.8383732
1: -1.7301784, 2.7653055, -0.5346295, 1.1044348, -2.8346133, 3.2999349
2: -1.6982117, 3.1253252, -0.4548847, 1.2482237, -2.9464350, 3.5802097
3: -2.2293622, 3.6074800, -0.9340510, 1.2600147, -3.4893765, 4.5415306
4: -2.5856273, 3.6531067, -0.8210288, 1.4488646, -4.0344920, 4.4741354

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0505552
time: 0.35 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0505552
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.2485261, 2.5536690, -0.2971591, 0.8949012, -2.1434271, 2.8508279
1: -1.7301784, 2.7653055, -0.5497876, 1.1363558, -2.8665342, 3.3150930
2: -1.6982117, 3.1253252, -0.4700940, 1.2826568, -2.9808683, 3.5954192
3: -2.2293622, 3.6074800, -0.9585164, 1.2952633, -3.5246255, 4.5659957
4: -2.5856273, 3.6531067, -0.8436151, 1.4857011, -4.0713282, 4.4967217

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
time: 0.42 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.3217049, 2.5799780, -0.3204554, 0.9300180, -2.2517228, 2.9004331
1: -1.7742481, 2.7913742, -0.5845242, 1.1713222, -2.9455705, 3.3758984
2: -1.7435780, 3.1709807, -0.5056426, 1.3326001, -3.0761781, 3.6766233
3: -2.2867398, 3.6599166, -1.0023541, 1.3492649, -3.6360044, 4.6622701
4: -2.6460733, 3.7037711, -0.8973903, 1.5542920, -4.2003651, 4.6011615

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0500325
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0500790
time: 0.40 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1.2961659, 2.6475687, -1.2485261, 2.5536690, -3.8498349, 3.8960946
1: -1.7545326, 2.9115472, -1.7301784, 2.7653055, -4.5198383, 4.6417255
2: -1.6955929, 3.3053660, -1.6982117, 3.1253252, -4.8209171, 5.0035777
3: -2.3274460, 3.7219892, -2.2293622, 3.6074800, -5.9349260, 5.9513512
4: -2.6058412, 3.7953048, -2.5856273, 3.6531067, -6.2589474, 6.3809309

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.3605993, 2.7310739, -1.2485261, 2.5536690, -3.9142683, 3.9795997
1: -1.8164110, 3.0072777, -1.7301784, 2.7653055, -4.5817165, 4.7374563
2: -1.7567012, 3.4051988, -1.6982117, 3.1253252, -4.8820267, 5.1034107
3: -2.4038906, 3.8425276, -2.2293622, 3.6074800, -6.0113707, 6.0718899
4: -2.6877170, 3.9016829, -2.5856273, 3.6531067, -6.3408237, 6.4873095

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
time: 0.35 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.68 seconds
NS_A1_B1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0522968, upper bound: 1.0505129
NS_A1_B1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0522968, upper bound: 1.0505129
NS_A1_B1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0535709, upper bound: 1.0539902
NS_A1_B1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0535709, upper bound: 1.0539902
NS_A1_B1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0527186
NS_A1_B1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0534314
NS_A1_B1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0522379
NS_A1_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519421, upper bound: 1.0534314
NS_A1_B1_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0544462, upper bound: 1.0510817
NS_A1_B1_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0548306, upper bound: 1.0510817
NS_A1_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0507336
NS_A1_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0508626
NS_A1_B1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0505401, upper bound: 1.0502919
NS_A1_B1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0505401, upper bound: 1.0502919
NS_A1_B1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0499605
NS_A1_B1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0500871
NS_A1_B1_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0514975, upper bound: 1.0505740
NS_A1_B1_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0514975, upper bound: 1.0505740
NS_A1_B1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0506591
NS_A1_B1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0507100
NS_A1_B1_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0501667, upper bound: 1.0501667
NS_A1_B1_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0501667, upper bound: 1.0501667
NS_A1_B1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0502519
NS_A1_B1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0503784
NS_A1_B2_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
NS_A1_B2_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
NS_A1_B2_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
NS_A1_B2_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
NS_A1_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0541095, upper bound: 1.0491135
NS_A1_B2_A1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0436953, upper bound: 1.0420088
NS_A1_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0544712, upper bound: 1.0491131
NS_A1_B2_A1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0490575, upper bound: 1.0439313
NS_A1_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
NS_A1_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
NS_A1_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
NS_A1_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0519357, upper bound: 1.0503808
NS_A1_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
NS_A1_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486297
NS_A1_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
NS_A1_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503540, upper bound: 1.0486761
NS_A1_B2_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
NS_A1_B2_A2_A1_B2_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0444293, upper bound: 1.0431251
NS_A1_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
NS_A1_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
NS_A1_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
NS_A1_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0505552, upper bound: 1.0499679
NS_A2_B1_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
NS_A2_B1_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543338
NS_A2_B1_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
NS_A2_B1_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
NS_A2_B1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0491135, upper bound: 1.0541095
NS_A2_B1_B1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0420088, upper bound: 1.0436953
NS_A2_B1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0491131, upper bound: 1.0544712
NS_A2_B1_B1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0439313, upper bound: 1.0490575
NS_A2_B1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
NS_A2_B1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
NS_A2_B1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
NS_A2_B1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0503808, upper bound: 1.0519357
NS_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
NS_A2_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
NS_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
NS_A2_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0512565
NS_A2_B1_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
NS_A2_B1_B2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0431251, upper bound: 1.0444293
NS_A2_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0505552
NS_A2_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0505552
NS_A2_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
NS_A2_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0499679, upper bound: 1.0507073
NS_A2_B1_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0500325
NS_A2_B1_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0479227, upper bound: 1.0500790
NS_A2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
NS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
NS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440
NS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.68
Output dim: 0, lower bound: -1.0490225, upper bound: 1.0496440

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1093142, 0.5719484, -0.1553145, 0.6463247, -0.7556390, 0.7272630
1: -0.2884543, 0.7235280, -0.3591110, 0.8135738, -1.1020281, 1.0826390
2: -0.2119465, 0.8112619, -0.2793274, 0.9168239, -1.1287704, 1.0905893
3: -0.5873810, 0.7978517, -0.6974133, 0.9111461, -1.4985271, 1.4952650
4: -0.4342344, 0.9781362, -0.5338272, 1.1137388, -1.5479732, 1.5119634

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2213382, 0.6948912, -0.1553145, 0.6463247, -0.8676628, 0.8502058
1: -0.4437678, 0.8638017, -0.3591110, 0.8135738, -1.2573416, 1.2229127
2: -0.3757644, 0.9738847, -0.2793274, 0.9168239, -1.2925882, 1.2532121
3: -0.7661010, 1.0106292, -0.6974133, 0.9111461, -1.6772470, 1.7080425
4: -0.6784313, 1.2178802, -0.5338272, 1.1137388, -1.7921700, 1.7517074

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1093142, 0.5719484, -0.1826628, 0.6760664, -0.7853807, 0.7546111
1: -0.2884543, 0.7235280, -0.3921647, 0.8608914, -1.1493456, 1.1156927
2: -0.2119465, 0.8112619, -0.3148580, 0.9619305, -1.1738770, 1.1261199
3: -0.5873810, 0.7978517, -0.7321308, 0.9645033, -1.5518844, 1.5299824
4: -0.4342344, 0.9781362, -0.5864717, 1.1600819, -1.5943162, 1.5646079

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520955, upper bound: 1.0519421
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520955, upper bound: 1.0539902
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2213382, 0.6948912, -0.1826628, 0.6760664, -0.8974046, 0.8775539
1: -0.4437678, 0.8638017, -0.3921647, 0.8608914, -1.3046591, 1.2559664
2: -0.3757644, 0.9738847, -0.3148580, 0.9619305, -1.3376949, 1.2887427
3: -0.7661010, 1.0106292, -0.7321308, 0.9645033, -1.7306044, 1.7427599
4: -0.6784313, 1.2178802, -0.5864717, 1.1600819, -1.8385131, 1.8043519

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520955, upper bound: 1.0519421
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520955, upper bound: 1.0539902
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1093142, 0.5719484, -0.2039456, 0.6722066, -0.7815209, 0.7758940
1: -0.2884543, 0.7235280, -0.4182327, 0.8502618, -1.1387161, 1.1417607
2: -0.2119465, 0.8112619, -0.3480552, 0.9428558, -1.1548023, 1.1593171
3: -0.5873810, 0.7978517, -0.7352566, 0.9777575, -1.5651385, 1.5331082
4: -0.4342344, 0.9781362, -0.6373773, 1.1677426, -1.6019771, 1.6155136

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 8

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1093142, 0.5719484, -0.3533702, 0.8853041, -0.9946184, 0.9253186
1: -0.2884543, 0.7235280, -0.6266433, 1.1134070, -1.4018613, 1.3501713
2: -0.2119465, 0.8112619, -0.5533800, 1.2433376, -1.4552841, 1.3646419
3: -0.5873810, 0.7978517, -1.0132209, 1.3180169, -1.9053979, 1.8110726
4: -0.4342344, 0.9781362, -0.9455986, 1.5302014, -1.9644358, 1.9237349

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2213382, 0.6948912, -0.2039456, 0.6722066, -0.8935448, 0.8988369
1: -0.4437678, 0.8638017, -0.4182327, 0.8502618, -1.2940296, 1.2820344
2: -0.3757644, 0.9738847, -0.3480552, 0.9428558, -1.3186202, 1.3219399
3: -0.7661010, 1.0106292, -0.7352566, 0.9777575, -1.7438585, 1.7458857
4: -0.6784313, 1.2178802, -0.6373773, 1.1677426, -1.8461739, 1.8552575

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2213382, 0.6948912, -0.3533702, 0.8853041, -1.1066422, 1.0482614
1: -0.4437678, 0.8638017, -0.6266433, 1.1134070, -1.5571748, 1.4904450
2: -0.3757644, 0.9738847, -0.5533800, 1.2433376, -1.6191020, 1.5272647
3: -0.7661010, 1.0106292, -1.0132209, 1.3180169, -2.0841179, 2.0238500
4: -0.6784313, 1.2178802, -0.9455986, 1.5302014, -2.2086327, 2.1634789

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2078650, 0.7069069, -0.1999423, 0.7006042, -0.9084692, 0.9068493
1: -0.4240694, 0.8942963, -0.4179226, 0.8853608, -1.3094302, 1.3122189
2: -0.3533992, 1.0083507, -0.3408493, 0.9954937, -1.3488929, 1.3492000
3: -0.7703033, 1.0153725, -0.7652031, 1.0039408, -1.7742441, 1.7805755
4: -0.6526360, 1.2132375, -0.6261964, 1.2012441, -1.8538802, 1.8394339

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527157, upper bound: 1.0510817
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527157, upper bound: 1.0510817
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1687989, 0.6604357, -0.1999423, 0.7006042, -0.8694031, 0.8603780
1: -0.3743765, 0.8313762, -0.4179226, 0.8853608, -1.2597374, 1.2492988
2: -0.3010136, 0.9384998, -0.3408493, 0.9954937, -1.2965074, 1.2793491
3: -0.7000468, 0.9392474, -0.7652031, 1.0039408, -1.7039876, 1.7044504
4: -0.5697083, 1.1379783, -0.6261964, 1.2012441, -1.7709525, 1.7641747

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0490199, upper bound: 1.0398891
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527157, upper bound: 1.0510817
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0527157, upper bound: 1.0510817
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.1792509, 0.6726518, -0.1796174, 0.6729426, -0.8521936, 0.8522692
1: -0.3888319, 0.8461026, -0.3886608, 0.8546216, -1.2434535, 1.2347634
2: -0.3158575, 0.9569659, -0.3137678, 0.9577398, -1.2735972, 1.2707337
3: -0.7172773, 0.9604697, -0.7210382, 0.9618257, -1.6791030, 1.6815079
4: -0.5917992, 1.1613320, -0.5868647, 1.1561475, -1.7479467, 1.7481967

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0506814
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533638, upper bound: 1.0505881
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0507336
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.1792509, 0.6726518, -0.2148956, 0.7115479, -0.8907988, 0.8875474
1: -0.3888319, 0.8461026, -0.4401824, 0.8935100, -1.2823420, 1.2862850
2: -0.3158575, 0.9569659, -0.3639436, 1.0145798, -1.3304372, 1.3209095
3: -0.7172773, 0.9604697, -0.7847203, 1.0327761, -1.7500534, 1.7451900
4: -0.5917992, 1.1613320, -0.6645471, 1.2404692, -1.8322685, 1.8258791

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0508259
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533638, upper bound: 1.0507801
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0546347, upper bound: 1.0508626
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.2649441, 0.8246372, -0.1797266, 0.6730186, -0.9379627, 1.0043638
1: -0.5047384, 1.0513973, -0.3887473, 0.8507720, -1.3555104, 1.4401447
2: -0.4291972, 1.1954941, -0.3128836, 0.9551901, -1.3843873, 1.5083777
3: -0.8983393, 1.1942482, -0.7242810, 0.9599463, -1.8582855, 1.9185292
4: -0.7814043, 1.4037045, -0.5851926, 1.1522350, -1.9336393, 1.9888971

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## BFS NS instance: NS_A1_B1_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.2649441, 0.8246372, -0.2493654, 0.7847975, -1.0497416, 1.0740026
1: -0.5047384, 1.0513973, -0.4858735, 1.0028971, -1.5076356, 1.5372708
2: -0.4291972, 1.1954941, -0.4051812, 1.1277900, -1.5569872, 1.6006753
3: -0.8983393, 1.1942482, -0.8688342, 1.1415675, -2.0399067, 2.0630825
4: -0.7814043, 1.4037045, -0.7417756, 1.3354181, -2.1168222, 2.1454802

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.2649441, 0.8246372, -0.1733778, 0.6628710, -0.9278151, 0.9980150
1: -0.5047384, 1.0513973, -0.3785492, 0.8422645, -1.3470030, 1.4299465
2: -0.4291972, 1.1954941, -0.3045089, 0.9439083, -1.3731055, 1.5000030
3: -0.8983393, 1.1942482, -0.7094678, 0.9455806, -1.8439199, 1.9037160
4: -0.7814043, 1.4037045, -0.5732061, 1.1381915, -1.9195957, 1.9769106

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499171, upper bound: 1.0498457
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0499605
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0499605
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2649441, 0.8246372, -0.2089669, 0.7024295, -0.9673737, 1.0336040
1: -0.5047384, 1.0513973, -0.4306579, 0.8829746, -1.3877130, 1.4820552
2: -0.4291972, 1.1954941, -0.3551179, 1.0026535, -1.4318507, 1.5506120
3: -0.8983393, 1.1942482, -0.7743858, 1.0177509, -1.9160901, 1.9686339
4: -0.7814043, 1.4037045, -0.6515192, 1.2241321, -2.0055363, 2.0552237

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499171, upper bound: 1.0499838
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0500871
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499605, upper bound: 1.0500871
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1772242, 0.6612130, -0.1999423, 0.7006042, -0.8778284, 0.8611554
1: -0.3860259, 0.8298028, -0.4179226, 0.8853608, -1.2713867, 1.2477254
2: -0.3113425, 0.9413394, -0.3408493, 0.9954937, -1.3068362, 1.2821887
3: -0.7134979, 0.9513810, -0.7652031, 1.0039408, -1.7174387, 1.7165840
4: -0.5871243, 1.1458344, -0.6261964, 1.2012441, -1.7883685, 1.7720308

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519147, upper bound: 1.0509997
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519147, upper bound: 1.0509997
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1868522, 0.6758040, -0.1999423, 0.7006042, -0.8874564, 0.8757464
1: -0.4006263, 0.8490487, -0.4179226, 0.8853608, -1.2859871, 1.2669713
2: -0.3247458, 0.9649436, -0.3408493, 0.9954937, -1.3202395, 1.3057928
3: -0.7331902, 0.9748261, -0.7652031, 1.0039408, -1.7371311, 1.7400291
4: -0.6074159, 1.1772201, -0.6261964, 1.2012441, -1.8086600, 1.8034165

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519147, upper bound: 1.0509997
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519147, upper bound: 1.0509997
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.2085951, 0.7061945, -0.1796174, 0.6729426, -0.8815378, 0.8858119
1: -0.4330689, 0.8780445, -0.3886608, 0.8546216, -1.2876904, 1.2667053
2: -0.3577549, 1.0080304, -0.3137678, 0.9577398, -1.3154947, 1.3217982
3: -0.7733389, 1.0228337, -0.7210382, 0.9618257, -1.7351646, 1.7438719
4: -0.6577712, 1.2365112, -0.5868647, 1.1561475, -1.8139186, 1.8233759

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0506591
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0506591
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.2085951, 0.7061945, -0.2148956, 0.7115479, -0.9201430, 0.9210901
1: -0.4330689, 0.8780445, -0.4401824, 0.8935100, -1.3265789, 1.3182269
2: -0.3577549, 1.0080304, -0.3639436, 1.0145798, -1.3723347, 1.3719740
3: -0.7733389, 1.0228337, -0.7847203, 1.0327761, -1.8061150, 1.8075540
4: -0.6577712, 1.2365112, -0.6645471, 1.2404692, -1.8982403, 1.9010584

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0507100
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512062, upper bound: 1.0507100
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2663195, 0.8055804, -0.1937883, 0.6910462, -0.9573657, 0.9993687
1: -0.5091786, 1.0243922, -0.4080151, 0.8736901, -1.3828688, 1.4324074
2: -0.4302250, 1.1573812, -0.3316857, 0.9828165, -1.4130416, 1.4890668
3: -0.8929978, 1.1819116, -0.7543924, 0.9884150, -1.8814127, 1.9363040
4: -0.7828243, 1.3741795, -0.6126174, 1.1841940, -1.9670182, 1.9867969

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505839, upper bound: 1.0505925
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505839, upper bound: 1.0505925
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2771621, 0.8229077, -0.1937883, 0.6910462, -0.9682083, 1.0166960
1: -0.5219501, 1.0464377, -0.4080151, 0.8736901, -1.3956403, 1.4544529
2: -0.4440638, 1.1858789, -0.3316857, 0.9828165, -1.4268804, 1.5175645
3: -0.9141124, 1.2082267, -0.7543924, 0.9884150, -1.9025273, 1.9626191
4: -0.8039485, 1.4075160, -0.6126174, 1.1841940, -1.9881425, 2.0201335

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505839, upper bound: 1.0505925
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505839, upper bound: 1.0505925
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.2995591, 0.8546940, -0.1733778, 0.6628710, -0.9624301, 1.0280719
1: -0.5560523, 1.0767007, -0.3785492, 0.8422645, -1.3983169, 1.4552499
2: -0.4788076, 1.2320927, -0.3045089, 0.9439083, -1.4227159, 1.5366017
3: -0.9559603, 1.2597048, -0.7094678, 0.9455806, -1.9015410, 1.9691726
4: -0.8567076, 1.4731483, -0.5732061, 1.1381915, -1.9948990, 2.0463545

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0502519
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0502519
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2995591, 0.8546940, -0.2089669, 0.7024295, -1.0019886, 1.0636609
1: -0.5560523, 1.0767007, -0.4306579, 0.8829746, -1.4390268, 1.5073586
2: -0.4788076, 1.2320927, -0.3551179, 1.0026535, -1.4814610, 1.5872107
3: -0.9559603, 1.2597048, -0.7743858, 1.0177509, -1.9737113, 2.0340905
4: -0.8567076, 1.4731483, -0.6515192, 1.2241321, -2.0808396, 2.1246676

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0503784
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500871, upper bound: 1.0503784
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.1848080, 0.6680613, -1.2651416, 2.5885944, -2.7734025, 1.9332029
1: -0.3889396, 0.8493978, -1.7519391, 2.8037324, -3.1926720, 2.6013370
2: -0.3200649, 0.9513507, -1.7190838, 3.1656418, -3.4857063, 2.6704345
3: -0.7211692, 0.9553143, -2.2556522, 3.6565268, -4.3776960, 3.2109666
4: -0.6010383, 1.1388059, -2.6154146, 3.6987345, -4.2997727, 3.7542205

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.1877181, 0.6809893, -1.2651416, 2.5885944, -2.7763126, 1.9461309
1: -0.3939159, 0.8700165, -1.7519391, 2.8037324, -3.1976483, 2.6219554
2: -0.3215793, 0.9708202, -1.7190838, 3.1656418, -3.4872212, 2.6899040
3: -0.7350662, 0.9733512, -2.2556522, 3.6565268, -4.3915930, 3.2290034
4: -0.6044438, 1.1596342, -2.6154146, 3.6987345, -4.3031783, 3.7750487

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.1443180, 0.6224142, -1.2651416, 2.5885944, -2.7329125, 1.8875558
1: -0.3365273, 0.7882043, -1.7519391, 2.8037324, -3.1402597, 2.5401433
2: -0.2643731, 0.8791410, -1.7190838, 3.1656418, -3.4300148, 2.5982246
3: -0.6507289, 0.8804431, -2.2556522, 3.6565268, -4.3072557, 3.1360953
4: -0.5131175, 1.0606205, -2.6154146, 3.6987345, -4.2118521, 3.6760352

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.1530081, 0.6367402, -1.2651416, 2.5885944, -2.7416024, 1.9018818
1: -0.3504792, 0.8092135, -1.7519391, 2.8037324, -3.1542115, 2.5611525
2: -0.2763952, 0.9036225, -1.7190838, 3.1656418, -3.4420371, 2.6227064
3: -0.6688560, 0.9029907, -2.2556522, 3.6565268, -4.3253827, 3.1586428
4: -0.5312647, 1.0901577, -2.6154146, 3.6987345, -4.2299991, 3.7055724

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528144, upper bound: 1.0504677
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -1.1855485, 2.4290857, -2.6387277, 1.8957106
1: -0.4267260, 0.8983275, -1.6548533, 2.6556582, -3.0823841, 2.5531807
2: -0.3558276, 1.0129061, -1.6208439, 2.9758430, -3.3316703, 2.6337500
3: -0.7740620, 1.0201761, -2.1393950, 3.4326754, -4.2067375, 3.1595712
4: -0.6563050, 1.2186592, -2.4708536, 3.4807491, -4.1370540, 3.6895127

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526076, upper bound: 1.0484853
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0526076, upper bound: 1.0491135
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -1.1855485, 2.4290857, -2.6037440, 1.8516812
1: -0.3825408, 0.8379728, -1.6548533, 2.6556582, -3.0381989, 2.4928260
2: -0.3095305, 0.9459749, -1.6208439, 2.9758430, -3.2853734, 2.5668187
3: -0.7079530, 0.9492630, -2.1393950, 3.4326754, -4.1406283, 3.0886581
4: -0.5819045, 1.1485283, -2.4708536, 3.4807491, -4.0626535, 3.6193819

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538616, upper bound: 1.0485942
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538616, upper bound: 1.0491131
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1806883, 0.6643587, -1.2491606, 2.5527794, -2.7334676, 1.9135194
1: -0.3907865, 0.8332261, -1.7293634, 2.7617764, -3.1525621, 2.5625896
2: -0.3165183, 0.9454506, -1.7005792, 3.1189947, -3.4355125, 2.6460297
3: -0.7177029, 0.9569116, -2.2224197, 3.6054430, -4.3231449, 3.1793313
4: -0.5945891, 1.1518157, -2.5882497, 3.6463084, -4.2408977, 3.7400653

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1806883, 0.6643587, -1.2271219, 2.5771050, -2.7577934, 1.8914807
1: -0.3907865, 0.8332261, -1.6880672, 2.8331177, -3.2239041, 2.5212934
2: -0.3165183, 0.9454506, -1.6324024, 3.2002738, -3.5167918, 2.5778530
3: -0.7177029, 0.9569116, -2.2409055, 3.6114621, -4.3291645, 3.1978171
4: -0.5945891, 1.1518157, -2.5104861, 3.6776853, -4.2722745, 3.6623018

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1905538, 0.6793638, -1.2491606, 2.5527794, -2.7433333, 1.9285245
1: -0.4057388, 0.8529248, -1.7293634, 2.7617764, -3.1675148, 2.5822883
2: -0.3302413, 0.9696531, -1.7005792, 3.1189947, -3.4492354, 2.6702323
3: -0.7378664, 0.9809170, -2.2224197, 3.6054430, -4.3433089, 3.2033367
4: -0.6153858, 1.1839089, -2.5882497, 3.6463084, -4.2616940, 3.7721586

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1905538, 0.6793638, -1.2271219, 2.5771050, -2.7676587, 1.9064858
1: -0.4057388, 0.8529248, -1.6880672, 2.8331177, -3.2388561, 2.5409920
2: -0.3302413, 0.9696531, -1.6324024, 3.2002738, -3.5305150, 2.6020555
3: -0.7378664, 0.9809170, -2.2409055, 3.6114621, -4.3493280, 3.2218225
4: -0.6153858, 1.1839089, -2.5104861, 3.6776853, -4.2930708, 3.6943951

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.2500529, 2.5539076, -2.7661657, 1.9597440
1: -0.4381096, 0.8818882, -1.7254283, 2.7621200, -3.2002294, 2.6073165
2: -0.3631817, 1.0126708, -1.6970711, 3.1160312, -3.4792123, 2.7097418
3: -0.7779223, 1.0290604, -2.2155218, 3.6032507, -4.3811731, 3.2445822
4: -0.6656082, 1.2431200, -2.5835793, 3.6378174, -4.3034258, 3.8266993

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.2422146, 2.6061690, -2.8184276, 1.9519055
1: -0.4381096, 0.8818882, -1.7000232, 2.8640509, -3.3021603, 2.5819113
2: -0.3631817, 1.0126708, -1.6440988, 3.2362671, -3.5994482, 2.6567695
3: -0.7779223, 1.0290604, -2.2584476, 3.6456170, -4.4235392, 3.2875080
4: -0.6656082, 1.2431200, -2.5312793, 3.7108212, -4.3764296, 3.7743993

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.3211722, 2.5754597, -2.7877183, 2.0308633
1: -0.4381096, 0.8818882, -1.7724123, 2.7839189, -3.2220283, 2.6543005
2: -0.3631817, 1.0126708, -1.7444291, 3.1608489, -3.5240300, 2.7570999
3: -0.7779223, 1.0290604, -2.2792263, 3.6524487, -4.4303708, 3.3082867
4: -0.6656082, 1.2431200, -2.6457515, 3.6928382, -4.3584461, 3.8888714

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.2122585, 0.7096910, -1.3605993, 2.7310739, -2.9433324, 2.0702903
1: -0.4381096, 0.8818882, -1.8164110, 3.0072777, -3.4453864, 2.6982992
2: -0.3631817, 1.0126708, -1.7567012, 3.4051988, -3.7683797, 2.7693720
3: -0.7779223, 1.0290604, -2.4038906, 3.8425276, -4.6204500, 3.4329510
4: -0.6656082, 1.2431200, -2.6877170, 3.9016829, -4.5672903, 3.9308369

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2847042, 0.8710025, -1.2491606, 2.5527794, -2.8374836, 2.1201630
1: -0.5346295, 1.1044348, -1.7293634, 2.7617764, -3.2964058, 2.8337984
2: -0.4548847, 1.2482237, -1.7005792, 3.1189947, -3.5738792, 2.9488025
3: -0.9340510, 1.2600147, -2.2224197, 3.6054430, -4.5394936, 3.4824338
4: -0.8210288, 1.4488646, -2.5882497, 3.6463084, -4.4673371, 4.0371141

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2847042, 0.8710025, -1.2271219, 2.5771050, -2.8618093, 2.0981245
1: -0.5346295, 1.1044348, -1.6880672, 2.8331177, -3.3677473, 2.7925019
2: -0.4548847, 1.2482237, -1.6324024, 3.2002738, -3.6551585, 2.8806257
3: -0.9340510, 1.2600147, -2.2409055, 3.6114621, -4.5455132, 3.5009203
4: -0.8210288, 1.4488646, -2.5104861, 3.6776853, -4.4987140, 3.9593506

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2971591, 0.8949012, -1.2491606, 2.5527794, -2.8499384, 2.1440616
1: -0.5497876, 1.1363558, -1.7293634, 2.7617764, -3.3115637, 2.8657193
2: -0.4700940, 1.2826568, -1.7005792, 3.1189947, -3.5890880, 2.9832356
3: -0.9585164, 1.2952633, -2.2224197, 3.6054430, -4.5639582, 3.5176830
4: -0.8436151, 1.4857011, -2.5882497, 3.6463084, -4.4899235, 4.0739508

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2971591, 0.8949012, -1.2271219, 2.5771050, -2.8742642, 2.1220231
1: -0.5497876, 1.1363558, -1.6880672, 2.8331177, -3.3829050, 2.8244228
2: -0.4700940, 1.2826568, -1.6324024, 3.2002738, -3.6703672, 2.9150586
3: -0.9585164, 1.2952633, -2.2409055, 3.6114621, -4.5699782, 3.5361688
4: -0.8436151, 1.4857011, -2.5104861, 3.6776853, -4.5213003, 3.9961872

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1848080, 0.6680613, -1.9332029, 2.7734025
1: -1.7519391, 2.8037324, -0.3889396, 0.8493978, -2.6013370, 3.1926718
2: -1.7190838, 3.1656418, -0.3200649, 0.9513507, -2.6704345, 3.4857066
3: -2.2556522, 3.6565268, -0.7211692, 0.9553143, -3.2109666, 4.3776960
4: -2.6154146, 3.6987345, -0.6010383, 1.1388059, -3.7542205, 4.2997727

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1877181, 0.6809893, -1.9461309, 2.7763124
1: -1.7519391, 2.8037324, -0.3939159, 0.8700165, -2.6219554, 3.1976483
2: -1.7190838, 3.1656418, -0.3215793, 0.9708202, -2.6899040, 3.4872212
3: -2.2556522, 3.6565268, -0.7350662, 0.9733512, -3.2290034, 4.3915930
4: -2.6154146, 3.6987345, -0.6044438, 1.1596342, -3.7750487, 4.3031783

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543338
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0543338
time: 0.39 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1443180, 0.6224142, -1.8875558, 2.7329125
1: -1.7519391, 2.8037324, -0.3365273, 0.7882043, -2.5401433, 3.1402597
2: -1.7190838, 3.1656418, -0.2643731, 0.8791410, -2.5982246, 3.4300148
3: -2.2556522, 3.6565268, -0.6507289, 0.8804431, -3.1360953, 4.3072557
4: -2.6154146, 3.6987345, -0.5131175, 1.0606205, -3.6760352, 4.2118521

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0528144
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.2651416, 2.5885944, -0.1530081, 0.6367402, -1.9018818, 2.7416024
1: -1.7519391, 2.8037324, -0.3504792, 0.8092135, -2.5611525, 3.1542115
2: -1.7190838, 3.1656418, -0.2763952, 0.9036225, -2.6227064, 3.4420366
3: -2.2556522, 3.6565268, -0.6688560, 0.9029907, -3.1586428, 4.3253827
4: -2.6154146, 3.6987345, -0.5312647, 1.0901577, -3.7055724, 4.2299991

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504677, upper bound: 1.0544438
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.1855485, 2.4290857, -0.2096419, 0.7101620, -1.8957106, 2.6387277
1: -1.6548533, 2.6556582, -0.4267260, 0.8983275, -2.5531807, 3.0823841
2: -1.6208439, 2.9758430, -0.3558276, 1.0129061, -2.6337500, 3.3316703
3: -2.1393950, 3.4326754, -0.7740620, 1.0201761, -3.1595712, 4.2067375
4: -2.4708536, 3.4807491, -0.6563050, 1.2186592, -3.6895127, 4.1370540

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484853, upper bound: 1.0526076
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0484853, upper bound: 1.0526076
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.1855485, 2.4290857, -0.1746585, 0.6661327, -1.8516812, 2.6037443
1: -1.6548533, 2.6556582, -0.3825408, 0.8379728, -2.4928260, 3.0381989
2: -1.6208439, 2.9758430, -0.3095305, 0.9459749, -2.5668187, 3.2853734
3: -2.1393950, 3.4326754, -0.7079530, 0.9492630, -3.0886581, 4.1406283
4: -2.4708536, 3.4807491, -0.5819045, 1.1485283, -3.6193819, 4.0626535

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0485942, upper bound: 1.0538616
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0485942, upper bound: 1.0544712
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.2491606, 2.5527794, -0.1806883, 0.6643587, -1.9135194, 2.7334673
1: -1.7293634, 2.7617764, -0.3907865, 0.8332261, -2.5625896, 3.1525624
2: -1.7005792, 3.1189947, -0.3165183, 0.9454506, -2.6460297, 3.4355125
3: -2.2224197, 3.6054430, -0.7177029, 0.9569116, -3.1793313, 4.3231454
4: -2.5882497, 3.6463084, -0.5945891, 1.1518157, -3.7400653, 4.2408972

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.2271219, 2.5771050, -0.1806883, 0.6643587, -1.8914807, 2.7577932
1: -1.6880672, 2.8331177, -0.3907865, 0.8332261, -2.5212934, 3.2239041
2: -1.6324024, 3.2002738, -0.3165183, 0.9454506, -2.5778530, 3.5167918
3: -2.2409055, 3.6114621, -0.7177029, 0.9569116, -3.1978171, 4.3291645
4: -2.5104861, 3.6776853, -0.5945891, 1.1518157, -3.6623018, 4.2722740

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.2491606, 2.5527794, -0.1905538, 0.6793638, -1.9285245, 2.7433331
1: -1.7293634, 2.7617764, -0.4057388, 0.8529248, -2.5822883, 3.1675153
2: -1.7005792, 3.1189947, -0.3302413, 0.9696531, -2.6702323, 3.4492357
3: -2.2224197, 3.6054430, -0.7378664, 0.9809170, -3.2033367, 4.3433089
4: -2.5882497, 3.6463084, -0.6153858, 1.1839089, -3.7721586, 4.2616940

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.2271219, 2.5771050, -0.1905538, 0.6793638, -1.9064858, 2.7676587
1: -1.6880672, 2.8331177, -0.4057388, 0.8529248, -2.5409920, 3.2388563
2: -1.6324024, 3.2002738, -0.3302413, 0.9696531, -2.6020555, 3.5305150
3: -2.2409055, 3.6114621, -0.7378664, 0.9809170, -3.2218225, 4.3493285
4: -2.5104861, 3.6776853, -0.6153858, 1.1839089, -3.6943951, 4.2930708

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1.2661704, 2.5898392, -0.1806883, 0.6643587, -1.9305291, 2.7705274
1: -1.7478166, 2.8040323, -0.3907865, 0.8332261, -2.5810428, 3.1948185
2: -1.7160549, 3.1630225, -0.3165183, 0.9454506, -2.6615055, 3.4795403
3: -2.2481234, 3.6546252, -0.7177029, 0.9569116, -3.2050350, 4.3723278
4: -2.6115189, 3.6906371, -0.5945891, 1.1518157, -3.7633345, 4.2852263

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.35 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1.2661704, 2.5898392, -0.1905538, 0.6793638, -1.9455342, 2.7803931
1: -1.7478166, 2.8040323, -0.4057388, 0.8529248, -2.6007414, 3.2097709
2: -1.7160549, 3.1630225, -0.3302413, 0.9696531, -2.6857080, 3.4932637
3: -2.2481234, 3.6546252, -0.7378664, 0.9809170, -3.2290404, 4.3924913
4: -2.6115189, 3.6906371, -0.6153858, 1.1839089, -3.7954278, 4.3060226

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.35 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0486297, upper bound: 1.0503540
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3370345, 2.6126482, -0.1806883, 0.6643587, -2.0013933, 2.7933364
1: -1.7949092, 2.8271534, -0.3907865, 0.8332261, -2.6281354, 3.2179399
2: -1.7629795, 3.2088265, -0.3165183, 0.9454506, -2.7084301, 3.5253444
3: -2.3119197, 3.7054596, -0.7177029, 0.9569116, -3.2688313, 4.4231625
4: -2.6737659, 3.7467427, -0.5945891, 1.1518157, -3.8255816, 4.3413315

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512100
time: 0.39 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.3370345, 2.6126482, -0.1905538, 0.6793638, -2.0163984, 2.8032019
1: -1.7949092, 2.8271534, -0.4057388, 0.8529248, -2.6478341, 3.2328923
2: -1.7629795, 3.2088265, -0.3302413, 0.9696531, -2.7326326, 3.5390675
3: -2.3119197, 3.7054596, -0.7378664, 0.9809170, -3.2928367, 4.4433260
4: -2.6737659, 3.7467427, -0.6153858, 1.1839089, -3.8576748, 4.3621283

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512100
time: 0.39 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0490802, upper bound: 1.0512565
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.2491606, 2.5527794, -0.2847042, 0.8710025, -2.1201632, 2.8374836
1: -1.7293634, 2.7617764, -0.5346295, 1.1044348, -2.8337982, 3.2964056
2: -1.7005792, 3.1189947, -0.4548847, 1.2482237, -2.9488025, 3.5738792
3: -2.2224197, 3.6054430, -0.9340510, 1.2600147, -3.4824338, 4.5394936
4: -2.5882497, 3.6463084, -0.8210288, 1.4488646, -4.0371141, 4.4673371

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.2271219, 2.5771050, -0.2847042, 0.8710025, -2.0981245, 2.8618093
1: -1.6880672, 2.8331177, -0.5346295, 1.1044348, -2.7925019, 3.3677473
2: -1.6324024, 3.2002738, -0.4548847, 1.2482237, -2.8806257, 3.6551583
3: -2.2409055, 3.6114621, -0.9340510, 1.2600147, -3.5009201, 4.5455122
4: -2.5104861, 3.6776853, -0.8210288, 1.4488646, -3.9593506, 4.4987135

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.2491606, 2.5527794, -0.2971591, 0.8949012, -2.1440616, 2.8499384
1: -1.7293634, 2.7617764, -0.5497876, 1.1363558, -2.8657188, 3.3115640
2: -1.7005792, 3.1189947, -0.4700940, 1.2826568, -2.9832356, 3.5890882
3: -2.2224197, 3.6054430, -0.9585164, 1.2952633, -3.5176828, 4.5639582
4: -2.5882497, 3.6463084, -0.8436151, 1.4857011, -4.0739508, 4.4899235

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.2271219, 2.5771050, -0.2971591, 0.8949012, -2.1220231, 2.8742642
1: -1.6880672, 2.8331177, -0.5497876, 1.1363558, -2.8244226, 3.3829055
2: -1.6324024, 3.2002738, -0.4700940, 1.2826568, -2.9150589, 3.6703675
3: -2.2409055, 3.6114621, -0.9585164, 1.2952633, -3.5361688, 4.5699778
4: -2.5104861, 3.6776853, -0.8436151, 1.4857011, -3.9961872, 4.5212998

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -1.3211722, 2.5754597, -0.3204554, 0.9300180, -2.2511902, 2.8959148
1: -1.7724123, 2.7839189, -0.5845242, 1.1713222, -2.9437346, 3.3684430
2: -1.7444291, 3.1608489, -0.5056426, 1.3326001, -3.0770292, 3.6664915
3: -2.2792263, 3.6524487, -1.0023541, 1.3492649, -3.6284912, 4.6548028
4: -2.6457515, 3.6928382, -0.8973903, 1.5542920, -4.2000427, 4.5902286

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -1.3605993, 2.7310739, -0.3204554, 0.9300180, -2.2906172, 3.0515289
1: -1.8164110, 3.0072777, -0.5845242, 1.1713222, -2.9877334, 3.5918016
2: -1.7567012, 3.4051988, -0.5056426, 1.3326001, -3.0893013, 3.9108415
3: -2.4038906, 3.8425276, -1.0023541, 1.3492649, -3.7531555, 4.8448820
4: -2.6877170, 3.9016829, -0.8973903, 1.5542920, -4.2420092, 4.7990727

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1.2961659, 2.6475687, -1.2491606, 2.5527794, -3.8489451, 3.8967285
1: -1.7545326, 2.9115472, -1.7293634, 2.7617764, -4.5163088, 4.6409106
2: -1.6955929, 3.3053660, -1.7005792, 3.1189947, -4.8145866, 5.0059452
3: -2.3274460, 3.7219892, -2.2224197, 3.6054430, -5.9328880, 5.9444089
4: -2.6058412, 3.7953048, -2.5882497, 3.6463084, -6.2521496, 6.3835530

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1.2961659, 2.6475687, -1.2271219, 2.5771050, -3.8732710, 3.8746905
1: -1.7545326, 2.9115472, -1.6880672, 2.8331177, -4.5876503, 4.5996141
2: -1.6955929, 3.3053660, -1.6324024, 3.2002738, -4.8958664, 4.9377685
3: -2.3274460, 3.7219892, -2.2409055, 3.6114621, -5.9389081, 5.9628940
4: -2.6058412, 3.7953048, -2.5104861, 3.6776853, -6.2835264, 6.3057909

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_A2_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1.3605993, 2.7310739, -1.2491606, 2.5527794, -3.9133787, 3.9802341
1: -1.8164110, 3.0072777, -1.7293634, 2.7617764, -4.5781875, 4.7366409
2: -1.7567012, 3.4051988, -1.7005792, 3.1189947, -4.8756957, 5.1057777
3: -2.4038906, 3.8425276, -2.2224197, 3.6054430, -6.0093327, 6.0649471
4: -2.6877170, 3.9016829, -2.5882497, 3.6463084, -6.3340254, 6.4899311

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A2_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.3605993, 2.7310739, -1.2271219, 2.5771050, -3.9377043, 3.9581957
1: -1.8164110, 3.0072777, -1.6880672, 2.8331177, -4.6495285, 4.6953444
2: -1.7567012, 3.4051988, -1.6324024, 3.2002738, -4.9569750, 5.0376010
3: -2.4038906, 3.8425276, -2.2409055, 3.6114621, -6.0153522, 6.0834332
4: -2.6877170, 3.9016829, -2.5104861, 3.6776853, -6.3654022, 6.4121690

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.76 + 266.50 = 268.26 seconds
