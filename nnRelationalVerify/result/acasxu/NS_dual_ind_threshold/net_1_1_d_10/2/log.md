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
execution time: IAR + RelationalAnalysis = 0.77 + 0.97 = 1.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1.0558836, upper bound: 1.0558836

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511832
time: 0.25 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.57 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -1.0553496, upper bound: 1.0511832
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.57
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.2352601, 0.7538407, -0.3035725, 0.8847764, -1.1200366, 1.0574131
1: -0.4706453, 0.9488738, -0.5660125, 1.0933844, -1.5640295, 1.5148864
2: -0.3915833, 1.0723588, -0.4826685, 1.2412479, -1.6328310, 1.5550274
3: -0.8320177, 1.0878556, -0.9617165, 1.2755736, -2.1075912, 2.0495720
4: -0.7029035, 1.3031529, -0.8354526, 1.4994075, -2.2023110, 2.1386056

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

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
time: 0.27 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -0.3035725, 0.8847764, -2.2119637, 3.0038373
1: -1.8266034, 2.9257355, -0.5660125, 1.0933844, -2.9199877, 3.4917479
2: -1.7866864, 3.3100519, -0.4826685, 1.2412479, -3.0279343, 3.7927201
3: -2.3538351, 3.8103127, -0.9617165, 1.2755736, -3.6294079, 4.7720289
4: -2.7129741, 3.8588223, -0.8354526, 1.4994075, -4.2123814, 4.6942744

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
time: 0.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.45 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.45
Output dim: 0, lower bound: -1.0507647, upper bound: 1.0507647

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.2352601, 0.7538407, -0.2352601, 0.7538407, -0.9891008, 0.9891006
1: -0.4706453, 0.9488738, -0.4706453, 0.9488738, -1.4195192, 1.4195192
2: -0.3915833, 1.0723588, -0.3915833, 1.0723588, -1.4639422, 1.4639422
3: -0.8320177, 1.0878556, -0.8320177, 1.0878556, -1.9198732, 1.9198732
4: -0.7029035, 1.3031529, -0.7029035, 1.3031529, -2.0060563, 2.0060563

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.28 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.2352601, 0.7538407, -1.3271873, 2.7002649, -2.9355247, 2.0810280
1: -0.4706453, 0.9488738, -1.8266034, 2.9257355, -3.3963809, 2.7754772
2: -0.3915833, 1.0723588, -1.7866864, 3.3100519, -3.7016354, 2.8590453
3: -0.8320177, 1.0878556, -2.3538351, 3.8103127, -4.6423302, 3.4416907
4: -0.7029035, 1.3031529, -2.7129741, 3.8588223, -4.5617256, 4.0161266

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.31 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -0.2352601, 0.7538407, -2.0810280, 2.9355249
1: -1.8266034, 2.9257355, -0.4706453, 0.9488738, -2.7754772, 3.3963809
2: -1.7866864, 3.3100519, -0.3915833, 1.0723588, -2.8590453, 3.7016351
3: -2.3538351, 3.8103127, -0.8320177, 1.0878556, -3.4416907, 4.6423302
4: -2.7129741, 3.8588223, -0.7029035, 1.3031529, -4.0161266, 4.5617256

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269
time: 0.33 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.3271873, 2.7002649, -1.3271873, 2.7002649, -4.0274525, 4.0274525
1: -1.8266034, 2.9257355, -1.8266034, 2.9257355, -4.7523389, 4.7523384
2: -1.7866864, 3.3100519, -1.7866864, 3.3100519, -5.0967383, 5.0967379
3: -2.3538351, 3.8103127, -2.3538351, 3.8103127, -6.1641479, 6.1641479
4: -2.7129741, 3.8588223, -2.7129741, 3.8588223, -6.5717964, 6.5717964

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
time: 0.28 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.39 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0510210
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0489732, upper bound: 1.0494879
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 1.39
Output dim: 0, lower bound: -1.0479269, upper bound: 1.0479269

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2013956, 0.7023641, -0.2352601, 0.7538407, -0.9552363, 0.9376242
1: -0.4206367, 0.8831170, -0.4706453, 0.9488738, -1.3695104, 1.3537623
2: -0.3466128, 1.0003821, -0.3915833, 1.0723588, -1.4189715, 1.3919654
3: -0.7602279, 1.0080743, -0.8320177, 1.0878556, -1.8480835, 1.8400919
4: -0.6370696, 1.2147777, -0.7029035, 1.3031529, -1.9402225, 1.9176812

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2363607, 0.7416377, -0.2352601, 0.7538407, -0.9902014, 0.9768978
1: -0.4722191, 0.9222932, -0.4706453, 0.9488738, -1.4210930, 1.3929385
2: -0.3964722, 1.0577438, -0.3915833, 1.0723588, -1.4688311, 1.4493271
3: -0.8244337, 1.0810699, -0.8320177, 1.0878556, -1.9122893, 1.9130876
4: -0.7141775, 1.3009543, -0.7029035, 1.3031529, -2.0173304, 2.0038579

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2013956, 0.7023641, -1.3271873, 2.7002649, -2.9016604, 2.0295515
1: -0.4206367, 0.8831170, -1.8266034, 2.9257355, -3.3463721, 2.7097204
2: -0.3466128, 1.0003821, -1.7866864, 3.3100519, -3.6566644, 2.7870684
3: -0.7602279, 1.0080743, -2.3538351, 3.8103127, -4.5705404, 3.3619094
4: -0.6370696, 1.2147777, -2.7129741, 3.8588223, -4.4958920, 3.9277518

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507504
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494336, upper bound: 1.0441275
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2363607, 0.7416377, -1.3271873, 2.7002649, -2.9366255, 2.0688250
1: -0.4722191, 0.9222932, -1.8266034, 2.9257355, -3.3979545, 2.7488966
2: -0.3964722, 1.0577438, -1.7866864, 3.3100519, -3.7065241, 2.8444302
3: -0.8244337, 1.0810699, -2.3538351, 3.8103127, -4.6347466, 3.4349051
4: -0.7141775, 1.3009543, -2.7129741, 3.8588223, -4.5730000, 4.0139284

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0510581
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.76 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0539857
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0539857, upper bound: 1.0540170
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507504
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0494336, upper bound: 1.0441275
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.76
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0510581

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2013956, 0.7023641, -0.2013956, 0.7023641, -0.9037597, 0.9037597
1: -0.4206367, 0.8831170, -0.4206367, 0.8831170, -1.3037536, 1.3037536
2: -0.3466128, 1.0003821, -0.3466128, 1.0003821, -1.3469949, 1.3469949
3: -0.7602279, 1.0080743, -0.7602279, 1.0080743, -1.7683022, 1.7683022
4: -0.6370696, 1.2147777, -0.6370696, 1.2147777, -1.8518473, 1.8518473

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0512581
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2013956, 0.7023641, -0.2363607, 0.7416377, -0.9430333, 0.9387248
1: -0.4206367, 0.8831170, -0.4722191, 0.9222932, -1.3429298, 1.3553361
2: -0.3466128, 1.0003821, -0.3964722, 1.0577438, -1.4043566, 1.3968543
3: -0.7602279, 1.0080743, -0.8244337, 1.0810699, -1.8412979, 1.8325080
4: -0.6370696, 1.2147777, -0.7141775, 1.3009543, -1.9380239, 1.9289553

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0518924
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2363607, 0.7416377, -0.2013956, 0.7023641, -0.9387248, 0.9430333
1: -0.4722191, 0.9222932, -0.4206367, 0.8831170, -1.3553361, 1.3429298
2: -0.3964722, 1.0577438, -0.3466128, 1.0003821, -1.3968543, 1.4043566
3: -0.8244337, 1.0810699, -0.7602279, 1.0080743, -1.8325080, 1.8412979
4: -0.7141775, 1.3009543, -0.6370696, 1.2147777, -1.9289553, 1.9380239

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2363607, 0.7416377, -0.2363607, 0.7416377, -0.9779984, 0.9779984
1: -0.4722191, 0.9222932, -0.4722191, 0.9222932, -1.3945123, 1.3945123
2: -0.3964722, 1.0577438, -0.3964722, 1.0577438, -1.4542160, 1.4542160
3: -0.8244337, 1.0810699, -0.8244337, 1.0810699, -1.9055036, 1.9055036
4: -0.7141775, 1.3009543, -0.7141775, 1.3009543, -2.0151320, 2.0151320

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2013956, 0.7023641, -1.2275646, 2.4774826, -2.6788778, 1.9299287
1: -0.4206367, 0.8831170, -1.7005773, 2.7084031, -3.1290398, 2.5836942
2: -0.3466128, 1.0003821, -1.6626792, 3.0481279, -3.3947406, 2.6630611
3: -0.7602279, 1.0080743, -2.1923127, 3.5047750, -4.2650032, 3.2003870
4: -0.6370696, 1.2147777, -2.5291798, 3.5632858, -4.2003555, 3.7439575

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507291
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0506076, upper bound: 1.0494973
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2363607, 0.7416377, -1.2564423, 2.6100669, -2.8464274, 1.9980800
1: -0.4722191, 0.9222932, -1.7448046, 2.8423195, -3.3145382, 2.6670978
2: -0.3964722, 1.0577438, -1.6990671, 3.2052026, -3.6016748, 2.7568109
3: -0.8244337, 1.0810699, -2.2675142, 3.6723115, -4.4967451, 3.3485842
4: -0.7141775, 1.3009543, -2.5895221, 3.7267532, -4.4409308, 3.8904765

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2363607, 0.7416377, -1.3136601, 2.6670325, -2.9033928, 2.0552979
1: -0.4722191, 0.9222932, -1.8104784, 2.8914104, -3.3636293, 2.7327716
2: -0.3964722, 1.0577438, -1.7713401, 3.2708874, -3.6673596, 2.8290839
3: -0.8244337, 1.0810699, -2.3325043, 3.7654533, -4.5898871, 3.4135742
4: -0.7141775, 1.3009543, -2.6896212, 3.8165128, -4.5306902, 3.9905756

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
time: 0.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.56 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0512581
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0518924
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0549353, upper bound: 1.0507291
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0506076, upper bound: 1.0494973
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0513845, upper bound: 1.0496232
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.56
Output dim: 0, lower bound: -1.0535985, upper bound: 1.0510581

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -0.2013956, 0.7023641, -0.8874898, 0.8797269
1: -0.3969364, 0.8526834, -0.4206367, 0.8831170, -1.2800534, 1.2733200
2: -0.3242711, 0.9644121, -0.3466128, 1.0003821, -1.3246531, 1.3110249
3: -0.7251614, 0.9704387, -0.7602279, 1.0080743, -1.7332357, 1.7306666
4: -0.6038694, 1.1718525, -0.6370696, 1.2147777, -1.8186471, 1.8089221

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -0.1952423, 0.6923982, -0.9598503, 1.0229695
1: -0.5075184, 1.0548140, -0.4107184, 0.8708251, -1.3783436, 1.4655324
2: -0.4327361, 1.2001319, -0.3375845, 0.9867133, -1.4194494, 1.5377164
3: -0.9017408, 1.1985904, -0.7488785, 0.9919778, -1.8937185, 1.9474690
4: -0.7864016, 1.4086894, -0.6237999, 1.1971152, -1.9835168, 2.0324893

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -0.2363607, 0.7416377, -0.9267634, 0.9146920
1: -0.3969364, 0.8526834, -0.4722191, 0.9222932, -1.3192296, 1.3249025
2: -0.3242711, 0.9644121, -0.3964722, 1.0577438, -1.3820149, 1.3608843
3: -0.7251614, 0.9704387, -0.8244337, 1.0810699, -1.8062314, 1.7948724
4: -0.6038694, 1.1718525, -0.7141775, 1.3009543, -1.9048238, 1.8860300

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -0.2305530, 0.7326385, -1.0000906, 1.0582801
1: -0.5075184, 1.0548140, -0.4628580, 0.9118950, -1.4194134, 1.5176719
2: -0.4327361, 1.2001319, -0.3878351, 1.0457006, -1.4784367, 1.5879670
3: -0.9017408, 1.1985904, -0.8142614, 1.0661757, -1.9679165, 2.0128517
4: -0.7864016, 1.4086894, -0.7014952, 1.2842600, -2.0706615, 2.1101847

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0508939
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0508939
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -0.2013956, 0.7023641, -0.9654967, 0.9712744
1: -0.5045100, 0.9689885, -0.4206367, 0.8831170, -1.3876270, 1.3896251
2: -0.4292829, 1.0956014, -0.3466128, 1.0003821, -1.4296650, 1.4422143
3: -0.8731403, 1.1296451, -0.7602279, 1.0080743, -1.8812146, 1.8898730
4: -0.7632787, 1.3396218, -0.6370696, 1.2147777, -1.9780564, 1.9766914

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -0.2013956, 0.7023641, -0.9278631, 0.9303675
1: -0.4573010, 0.9075529, -0.4206367, 0.8831170, -1.3404180, 1.3281896
2: -0.3811812, 1.0387152, -0.3466128, 1.0003821, -1.3815633, 1.3853281
3: -0.8068940, 1.0590353, -0.7602279, 1.0080743, -1.8149683, 1.8192632
4: -0.6912529, 1.2759079, -0.6370696, 1.2147777, -1.9060307, 1.9129775

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552524
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -0.2363607, 0.7416377, -1.0047703, 1.0062394
1: -0.5045100, 0.9689885, -0.4722191, 0.9222932, -1.4268032, 1.4412076
2: -0.4292829, 1.0956014, -0.3964722, 1.0577438, -1.4870267, 1.4920737
3: -0.8731403, 1.1296451, -0.8244337, 1.0810699, -1.9542103, 1.9540788
4: -0.7632787, 1.3396218, -0.7141775, 1.3009543, -2.0642331, 2.0537994

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -0.2363607, 0.7416377, -0.9671367, 0.9653326
1: -0.4573010, 0.9075529, -0.4722191, 0.9222932, -1.3795942, 1.3797719
2: -0.3811812, 1.0387152, -0.3964722, 1.0577438, -1.4389250, 1.4351875
3: -0.8068940, 1.0590353, -0.8244337, 1.0810699, -1.8879640, 1.8834690
4: -0.6912529, 1.2759079, -0.7141775, 1.3009543, -1.9922073, 1.9900854

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -1.2275646, 2.4774826, -2.6626081, 1.9058959
1: -0.3969364, 0.8526834, -1.7005773, 2.7084031, -3.1053393, 2.5532606
2: -0.3242711, 0.9644121, -1.6626792, 3.0481279, -3.3723989, 2.6270914
3: -0.7251614, 0.9704387, -2.1923127, 3.5047750, -4.2299366, 3.1627514
4: -0.6038694, 1.1718525, -2.5291798, 3.5632858, -4.1671553, 3.7010322

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0487347, upper bound: 1.0399865
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -1.2111990, 2.4424505, -2.7099028, 2.0389261
1: -0.5075184, 1.0548140, -1.6786456, 2.6705396, -3.1780581, 2.7334595
2: -0.4327361, 1.2001319, -1.6420074, 3.0072918, -3.4400275, 2.8421392
3: -0.9017408, 1.1985904, -2.1648109, 3.4560356, -4.3577766, 3.3634014
4: -0.7864016, 1.4086894, -2.4996901, 3.5167820, -4.3031836, 3.9083796

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -1.2564423, 2.6100669, -2.8731995, 2.0263212
1: -0.5045100, 0.9689885, -1.7448046, 2.8423195, -3.3468294, 2.7137930
2: -0.4292829, 1.0956014, -1.6990671, 3.2052026, -3.6344855, 2.7946687
3: -0.8731403, 1.1296451, -2.2675142, 3.6723115, -4.5454516, 3.3971593
4: -0.7632787, 1.3396218, -2.5895221, 3.7267532, -4.4900312, 3.9291439

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -1.2564423, 2.6100669, -2.8355658, 1.9854143
1: -0.4573010, 0.9075529, -1.7448046, 2.8423195, -3.2996206, 2.6523576
2: -0.3811812, 1.0387152, -1.6990671, 3.2052026, -3.5863838, 2.7377825
3: -0.8068940, 1.0590353, -2.2675142, 3.6723115, -4.4792051, 3.3265495
4: -0.6912529, 1.2759079, -2.5895221, 3.7267532, -4.4180059, 3.8654299

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -1.3136601, 2.6670325, -2.9301651, 2.0835390
1: -0.5045100, 0.9689885, -1.8104784, 2.8914104, -3.3959203, 2.7794669
2: -0.4292829, 1.0956014, -1.7713401, 3.2708874, -3.7001703, 2.8669415
3: -0.8731403, 1.1296451, -2.3325043, 3.7654533, -4.6385937, 3.4621494
4: -0.7632787, 1.3396218, -2.6896212, 3.8165128, -4.5797911, 4.0292430

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500353
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500353
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -1.3136601, 2.6670325, -2.8925314, 2.0426321
1: -0.4573010, 0.9075529, -1.8104784, 2.8914104, -3.3487115, 2.7180314
2: -0.3811812, 1.0387152, -1.7713401, 3.2708874, -3.6520681, 2.8100553
3: -0.8068940, 1.0590353, -2.3325043, 3.7654533, -4.5723467, 3.3915396
4: -0.6912529, 1.2759079, -2.6896212, 3.8165128, -4.5077658, 3.9655290

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0507191
time: 0.32 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.95 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0502805
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0508607, upper bound: 1.0508939
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0508939
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0502805, upper bound: 1.0508939
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0554253
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552524
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0539330, upper bound: 1.0540170
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0495715
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0496232
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500353
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0500353
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0498831
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.95
Output dim: 0, lower bound: -1.0512050, upper bound: 1.0507191

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -0.1851257, 0.6783313, -0.8634571, 0.8634571
1: -0.3969364, 0.8526834, -0.3969364, 0.8526834, -1.2496197, 1.2496197
2: -0.3242711, 0.9644121, -0.3242711, 0.9644121, -1.2886832, 1.2886832
3: -0.7251614, 0.9704387, -0.7251614, 0.9704387, -1.6956002, 1.6956002
4: -0.6038694, 1.1718525, -0.6038694, 1.1718525, -1.7757219, 1.7757219

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538202, upper bound: 1.0511968
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0512581
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -0.2674521, 0.8277271, -1.0128528, 0.9457834
1: -0.3969364, 0.8526834, -0.5075184, 1.0548140, -1.4517504, 1.3602018
2: -0.3242711, 0.9644121, -0.4327361, 1.2001319, -1.5244030, 1.3971481
3: -0.7251614, 0.9704387, -0.9017408, 1.1985904, -1.9237518, 1.8721795
4: -0.6038694, 1.1718525, -0.7864016, 1.4086894, -2.0125589, 1.9582541

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538202, upper bound: 1.0511968
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0512581
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -0.1851257, 0.6783313, -0.9457834, 1.0128528
1: -0.5075184, 1.0548140, -0.3969364, 0.8526834, -1.3602018, 1.4517504
2: -0.4327361, 1.2001319, -0.3242711, 0.9644121, -1.3971481, 1.5244030
3: -0.9017408, 1.1985904, -0.7251614, 0.9704387, -1.8721795, 1.9237518
4: -0.7864016, 1.4086894, -0.6038694, 1.1718525, -1.9582541, 2.0125589

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -0.2674521, 0.8277271, -1.0951792, 1.0951792
1: -0.5075184, 1.0548140, -0.5075184, 1.0548140, -1.5623324, 1.5623324
2: -0.4327361, 1.2001319, -0.4327361, 1.2001319, -1.6328681, 1.6328681
3: -0.9017408, 1.1985904, -0.9017408, 1.1985904, -2.1003313, 2.1003313
4: -0.7864016, 1.4086894, -0.7864016, 1.4086894, -2.1950910, 2.1950910

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -0.2122585, 0.7096910, -0.8948168, 0.8905898
1: -0.3969364, 0.8526834, -0.4381096, 0.8818882, -1.2788246, 1.2907929
2: -0.3242711, 0.9644121, -0.3631817, 1.0126708, -1.3369418, 1.3275938
3: -0.7251614, 0.9704387, -0.7779223, 1.0290604, -1.7542218, 1.7483611
4: -0.6038694, 1.1718525, -0.6656082, 1.2431200, -1.8469894, 1.8374606

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520929, upper bound: 1.0510832
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550305, upper bound: 1.0508972
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -0.3022564, 0.8574660, -1.0425918, 0.9805877
1: -0.3969364, 0.8526834, -0.5588838, 1.0796481, -1.4765846, 1.4115672
2: -0.3242711, 0.9644121, -0.4826456, 1.2362378, -1.5605088, 1.4470577
3: -0.7251614, 0.9704387, -0.9591278, 1.2640674, -1.9892288, 1.9295666
4: -0.6038694, 1.1718525, -0.8620855, 1.4785453, -2.0824146, 2.0339379

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520929, upper bound: 1.0510832
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550305, upper bound: 1.0508972
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -0.2122585, 0.7096910, -0.9771431, 1.0399857
1: -0.5075184, 1.0548140, -0.4381096, 0.8818882, -1.3894066, 1.4929236
2: -0.4327361, 1.2001319, -0.3631817, 1.0126708, -1.4454069, 1.5633136
3: -0.9017408, 1.1985904, -0.7779223, 1.0290604, -1.9308012, 1.9765127
4: -0.7864016, 1.4086894, -0.6656082, 1.2431200, -2.0295215, 2.0742974

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -0.3022564, 0.8574660, -1.1249182, 1.1299834
1: -0.5075184, 1.0548140, -0.5588838, 1.0796481, -1.5871665, 1.6136978
2: -0.4327361, 1.2001319, -0.4826456, 1.2362378, -1.6689739, 1.6827774
3: -0.9017408, 1.1985904, -0.9591278, 1.2640674, -2.1658082, 2.1577182
4: -0.7864016, 1.4086894, -0.8620855, 1.4785453, -2.2649469, 2.2707748

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -0.2350464, 0.7415904, -1.0047231, 1.0049253
1: -0.5045100, 0.9689885, -0.4636297, 0.9372106, -1.4417207, 1.4326181
2: -0.4292829, 1.0956014, -0.3912313, 1.0568323, -1.4861152, 1.4868327
3: -0.8731403, 1.1296451, -0.8212245, 1.0728223, -1.9459627, 1.9508696
4: -0.7632787, 1.3396218, -0.7068326, 1.2774920, -2.0407708, 2.0464544

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0554253
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0547487
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -0.1909873, 0.6899648, -0.9530973, 0.9608661
1: -0.5045100, 0.9689885, -0.4062611, 0.8681383, -1.3726482, 1.3752496
2: -0.4292829, 1.0956014, -0.3319388, 0.9815091, -1.4107921, 1.4275403
3: -0.8731403, 1.1296451, -0.7429183, 0.9865521, -1.8596925, 1.8725634
4: -0.7632787, 1.3396218, -0.6151925, 1.1911318, -1.9544106, 1.9548143

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0554253
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0547487
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -0.1851257, 0.6783313, -0.9038303, 0.9140977
1: -0.4573010, 0.9075529, -0.3969364, 0.8526834, -1.3099844, 1.3044894
2: -0.3811812, 1.0387152, -0.3242711, 0.9644121, -1.3455933, 1.3629863
3: -0.8068940, 1.0590353, -0.7251614, 0.9704387, -1.7773328, 1.7841967
4: -0.6912529, 1.2759079, -0.6038694, 1.1718525, -1.8631054, 1.8797773

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2193296, 0.7198079, -0.2674521, 0.8277271, -1.0470567, 0.9872600
1: -0.4474986, 0.8969982, -0.5075184, 1.0548140, -1.5023125, 1.4045166
2: -0.3720031, 1.0267047, -0.4327361, 1.2001319, -1.5721350, 1.4594407
3: -0.7963632, 1.0437112, -0.9017408, 1.1985904, -1.9949536, 1.9454520
4: -0.6777812, 1.2588216, -0.7864016, 1.4086894, -2.0864706, 2.0452232

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -0.2631326, 0.7698788, -1.0330114, 1.0330114
1: -0.5045100, 0.9689885, -0.5045100, 0.9689885, -1.4734986, 1.4734986
2: -0.4292829, 1.0956014, -0.4292829, 1.0956014, -1.5248843, 1.5248843
3: -0.8731403, 1.1296451, -0.8731403, 1.1296451, -2.0027854, 2.0027854
4: -0.7632787, 1.3396218, -0.7632787, 1.3396218, -2.1029005, 2.1029005

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -0.2254990, 0.7289720, -0.9921045, 0.9953778
1: -0.5045100, 0.9689885, -0.4573010, 0.9075529, -1.4120629, 1.4262896
2: -0.4292829, 1.0956014, -0.3811812, 1.0387152, -1.4679981, 1.4767827
3: -0.8731403, 1.1296451, -0.8068940, 1.0590353, -1.9321756, 1.9365392
4: -0.7632787, 1.3396218, -0.6912529, 1.2759079, -2.0391865, 2.0308747

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -0.2631326, 0.7698788, -0.9953778, 0.9921045
1: -0.4573010, 0.9075529, -0.5045100, 0.9689885, -1.4262896, 1.4120629
2: -0.3811812, 1.0387152, -0.4292829, 1.0956014, -1.4767827, 1.4679981
3: -0.8068940, 1.0590353, -0.8731403, 1.1296451, -1.9365392, 1.9321756
4: -0.6912529, 1.2759079, -0.7632787, 1.3396218, -2.0308747, 2.0391865

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -0.2254990, 0.7289720, -0.9544709, 0.9544709
1: -0.4573010, 0.9075529, -0.4573010, 0.9075529, -1.3648539, 1.3648539
2: -0.3811812, 1.0387152, -0.3811812, 1.0387152, -1.4198965, 1.4198965
3: -0.8068940, 1.0590353, -0.8068940, 1.0590353, -1.8659294, 1.8659294
4: -0.6912529, 1.2759079, -0.6912529, 1.2759079, -1.9671608, 1.9671608

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -1.2136881, 2.4463463, -2.6314721, 1.8920195
1: -0.3969364, 0.8526834, -1.6815057, 2.6714954, -3.0684319, 2.5341890
2: -0.3242711, 0.9644121, -1.6466799, 3.0085113, -3.3327823, 2.6110921
3: -0.7251614, 0.9704387, -2.1654463, 3.4600346, -4.1851959, 3.1358850
4: -0.6038694, 1.1718525, -2.5056057, 3.5195577, -4.1234274, 3.6774583

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520680, upper bound: 1.0491056
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520680, upper bound: 1.0491135
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1851257, 0.6783313, -1.1447992, 2.3595920, -2.5447176, 1.8231306
1: -0.3969364, 0.8526834, -1.5842650, 2.6178081, -3.0147445, 2.4369483
2: -0.3242711, 0.9644121, -1.5295095, 2.9590945, -3.2833657, 2.4939218
3: -0.7251614, 0.9704387, -2.1012015, 3.3203290, -4.0454903, 3.0716403
4: -0.6038694, 1.1718525, -2.3559012, 3.4138608, -4.0177302, 3.5277538

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520680, upper bound: 1.0491056
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544711, upper bound: 1.0491135
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -1.2136881, 2.4463463, -2.7137985, 2.0414152
1: -0.5075184, 1.0548140, -1.6815057, 2.6714954, -3.1790137, 2.7363195
2: -0.4327361, 1.2001319, -1.6466799, 3.0085113, -3.4412475, 2.8468118
3: -0.9017408, 1.1985904, -2.1654463, 3.4600346, -4.3617754, 3.3640366
4: -0.7864016, 1.4086894, -2.5056057, 3.5195577, -4.3059592, 3.9142952

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2674521, 0.8277271, -1.1447992, 2.3595920, -2.6270442, 1.9725263
1: -0.5075184, 1.0548140, -1.5842650, 2.6178081, -3.1253266, 2.6390791
2: -0.4327361, 1.2001319, -1.5295095, 2.9590945, -3.3918307, 2.7296414
3: -0.9017408, 1.1985904, -2.1012015, 3.3203290, -4.2220697, 3.2997918
4: -0.7864016, 1.4086894, -2.3559012, 3.4138608, -4.2002621, 3.7645907

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -1.2317941, 2.5485647, -2.8116972, 2.0016730
1: -0.5045100, 0.9689885, -1.7097392, 2.7696524, -3.2741623, 2.6787276
2: -0.4292829, 1.0956014, -1.6688566, 3.1283512, -3.5576341, 2.7644582
3: -0.8731403, 1.1296451, -2.2169631, 3.5870750, -4.4602156, 3.3466082
4: -0.7632787, 1.3396218, -2.5448000, 3.6422343, -4.4055128, 3.8844218

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -1.3011413, 2.5799091, -2.8430414, 2.0710201
1: -0.5045100, 0.9689885, -1.7600503, 2.8037138, -3.3082237, 2.7290387
2: -0.4292829, 1.0956014, -1.7155523, 3.1846526, -3.6139355, 2.8111539
3: -0.8731403, 1.1296451, -2.2817159, 3.6491742, -4.5223141, 3.4113610
4: -0.7632787, 1.3396218, -2.6121106, 3.7078054, -4.4710827, 3.9517324

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -1.2317941, 2.5485647, -2.7740636, 1.9607661
1: -0.4573010, 0.9075529, -1.7097392, 2.7696524, -3.2269535, 2.6172922
2: -0.3811812, 1.0387152, -1.6688566, 3.1283512, -3.5095320, 2.7075720
3: -0.8068940, 1.0590353, -2.2169631, 3.5870750, -4.3939691, 3.2759984
4: -0.6912529, 1.2759079, -2.5448000, 3.6422343, -4.3334870, 3.8207078

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -1.3011413, 2.5799091, -2.8054080, 2.0301132
1: -0.4573010, 0.9075529, -1.7600503, 2.8037138, -3.2610145, 2.6676033
2: -0.3811812, 1.0387152, -1.7155523, 3.1846526, -3.5658336, 2.7542677
3: -0.8068940, 1.0590353, -2.2817159, 3.6491742, -4.4560671, 3.3407512
4: -0.6912529, 1.2759079, -2.6121106, 3.7078054, -4.3990583, 3.8880186

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -1.2890360, 2.6080267, -2.8711593, 2.0589149
1: -0.5045100, 0.9689885, -1.7757368, 2.8224382, -3.3269482, 2.7447252
2: -0.4292829, 1.0956014, -1.7412896, 3.1970921, -3.6263750, 2.8368912
3: -0.8731403, 1.1296451, -2.2828507, 3.6829684, -4.5561085, 3.4124959
4: -0.7632787, 1.3396218, -2.6450953, 3.7347975, -4.4980764, 3.9847171

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0492136
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2631326, 0.7698788, -1.3504868, 2.6251640, -2.8882966, 2.1203656
1: -0.5045100, 0.9689885, -1.8179705, 2.8394029, -3.3439131, 2.7869589
2: -0.4292829, 1.0956014, -1.7806034, 3.2355962, -3.6648791, 2.8762050
3: -0.8731403, 1.1296451, -2.3383503, 3.7266910, -4.5998311, 3.4679954
4: -0.7632787, 1.3396218, -2.7018437, 3.7814958, -4.5447741, 4.0414658

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512876, upper bound: 1.0493899
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -1.2890360, 2.6080267, -2.8335257, 2.0180080
1: -0.4573010, 0.9075529, -1.7757368, 2.8224382, -3.2797394, 2.6832898
2: -0.3811812, 1.0387152, -1.7412896, 3.1970921, -3.5782728, 2.7800050
3: -0.8068940, 1.0590353, -2.2828507, 3.6829684, -4.4898624, 3.3418860
4: -0.6912529, 1.2759079, -2.6450953, 3.7347975, -4.4260502, 3.9210033

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498314, upper bound: 1.0490003
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0487533
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2254990, 0.7289720, -1.3504868, 2.6251640, -2.8506629, 2.0794587
1: -0.4573010, 0.9075529, -1.8179705, 2.8394029, -3.2967038, 2.7255235
2: -0.3811812, 1.0387152, -1.7806034, 3.2355962, -3.6167774, 2.8193188
3: -0.8068940, 1.0590353, -2.3383503, 3.7266910, -4.5335836, 3.3973856
4: -0.6912529, 1.2759079, -2.7018437, 3.7814958, -4.4727488, 3.9777517

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0504519, upper bound: 1.0491027
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0488068
time: 0.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.61 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0538202, upper bound: 1.0511968
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0512581
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0538202, upper bound: 1.0511968
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0512581
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0501450, upper bound: 1.0502010
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0520929, upper bound: 1.0510832
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0550305, upper bound: 1.0508972
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0520929, upper bound: 1.0510832
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0550305, upper bound: 1.0508972
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0508939, upper bound: 1.0508607
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0520680, upper bound: 1.0491056
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0520680, upper bound: 1.0491135
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0520680, upper bound: 1.0491056
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0544711, upper bound: 1.0491135
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0505847, upper bound: 1.0494973
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0498314, upper bound: 1.0490003
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0487533
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0504519, upper bound: 1.0491027
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.61
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0488068

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -0.1851257, 0.6783313, -0.8879732, 0.8952878
1: -0.4267260, 0.8983275, -0.3969364, 0.8526834, -1.2794094, 1.2952639
2: -0.3558276, 1.0129061, -0.3242711, 0.9644121, -1.3202397, 1.3371772
3: -0.7740620, 1.0201761, -0.7251614, 0.9704387, -1.7445006, 1.7453375
4: -0.6563050, 1.2186592, -0.6038694, 1.1718525, -1.8281574, 1.8225286

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -0.1851257, 0.6783313, -0.8529898, 0.8512585
1: -0.3825408, 0.8379728, -0.3969364, 0.8526834, -1.2352242, 1.2349092
2: -0.3095305, 0.9459749, -0.3242711, 0.9644121, -1.2739426, 1.2702460
3: -0.7079530, 0.9492630, -0.7251614, 0.9704387, -1.6783917, 1.6744244
4: -0.5819045, 1.1485283, -0.6038694, 1.1718525, -1.7537570, 1.7523978

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -0.2674521, 0.8277271, -1.0373690, 0.9776142
1: -0.4267260, 0.8983275, -0.5075184, 1.0548140, -1.4815400, 1.4058459
2: -0.3558276, 1.0129061, -0.4327361, 1.2001319, -1.5559595, 1.4456422
3: -0.7740620, 1.0201761, -0.9017408, 1.1985904, -1.9726524, 1.9219168
4: -0.6563050, 1.2186592, -0.7864016, 1.4086894, -2.0649943, 2.0050607

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0538202, upper bound: 1.0509606
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0533586, upper bound: 1.0508971
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -0.2674521, 0.8277271, -1.0023856, 0.9335849
1: -0.3825408, 0.8379728, -0.5075184, 1.0548140, -1.4373548, 1.3454912
2: -0.3095305, 0.9459749, -0.4327361, 1.2001319, -1.5096624, 1.3787110
3: -0.7079530, 0.9492630, -0.9017408, 1.1985904, -1.9065434, 1.8510039
4: -0.5819045, 1.1485283, -0.7864016, 1.4086894, -1.9905939, 1.9349300

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0510182
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0548845, upper bound: 1.0509748
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -0.1851257, 0.6783313, -0.8612425, 0.9051466
1: -0.3864989, 0.9190384, -0.3969364, 0.8526834, -1.2391822, 1.3159748
2: -0.3115277, 1.0421524, -0.3242711, 0.9644121, -1.2759398, 1.3664235
3: -0.7443157, 1.0138530, -0.7251614, 0.9704387, -1.7147545, 1.7390144
4: -0.6034802, 1.2099588, -0.6038694, 1.1718525, -1.7753327, 1.8138282

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509606, upper bound: 1.0538202
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0510182, upper bound: 1.0550090
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -0.1851257, 0.6783313, -0.9277602, 0.9795976
1: -0.4813044, 1.0119214, -0.3969364, 0.8526834, -1.3339877, 1.4088578
2: -0.4077803, 1.1496273, -0.3242711, 0.9644121, -1.3721924, 1.4738984
3: -0.8642187, 1.1482675, -0.7251614, 0.9704387, -1.8346574, 1.8734289
4: -0.7469358, 1.3539220, -0.6038694, 1.1718525, -1.9187883, 1.9577914

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508971, upper bound: 1.0533586
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0548845
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -0.2674521, 0.8277271, -1.0106382, 0.9874730
1: -0.3864989, 0.9190384, -0.5075184, 1.0548140, -1.4413129, 1.4265568
2: -0.3115277, 1.0421524, -0.4327361, 1.2001319, -1.5116596, 1.4748886
3: -0.7443157, 1.0138530, -0.9017408, 1.1985904, -1.9429061, 1.9155937
4: -0.6034802, 1.2099588, -0.7864016, 1.4086894, -2.0121696, 1.9963604

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -0.2674521, 0.8277271, -1.0771561, 1.0619240
1: -0.4813044, 1.0119214, -0.5075184, 1.0548140, -1.5361184, 1.5194398
2: -0.4077803, 1.1496273, -0.4327361, 1.2001319, -1.6079122, 1.5823634
3: -0.8642187, 1.1482675, -0.9017408, 1.1985904, -2.0628090, 2.0500083
4: -0.7469358, 1.3539220, -0.7864016, 1.4086894, -2.1556253, 2.1403236

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -0.2122585, 0.7096910, -0.8643661, 0.8467072
1: -0.3508692, 0.8027984, -0.4381096, 0.8818882, -1.2327573, 1.2409080
2: -0.2792361, 0.8971295, -0.3631817, 1.0126708, -1.2919068, 1.2603111
3: -0.6675518, 0.9013090, -0.7779223, 1.0290604, -1.6966121, 1.6792313
4: -0.5352525, 1.0835073, -0.6656082, 1.2431200, -1.7783724, 1.7491155

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528513, upper bound: 1.0520946
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528721, upper bound: 1.0520802
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528721, upper bound: 1.0520802
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -0.2122585, 0.7096910, -0.8730345, 0.8610685
1: -0.3648036, 0.8238323, -0.4381096, 0.8818882, -1.2466917, 1.2619419
2: -0.2912067, 0.9218535, -0.3631817, 1.0126708, -1.3038775, 1.2850351
3: -0.6862502, 0.9238169, -0.7779223, 1.0290604, -1.7153106, 1.7017392
4: -0.5532730, 1.1132669, -0.6656082, 1.2431200, -1.7963929, 1.7788751

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528513, upper bound: 1.0520685
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0552339, upper bound: 1.0520802
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0552339, upper bound: 1.0520802
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -0.3022564, 0.8574660, -1.0121411, 0.9367051
1: -0.3508692, 0.8027984, -0.5588838, 1.0796481, -1.4305173, 1.3616822
2: -0.2792361, 0.8971295, -0.4826456, 1.2362378, -1.5154738, 1.3797750
3: -0.6675518, 0.9013090, -0.9591278, 1.2640674, -1.9316192, 1.8604368
4: -0.5352525, 1.0835073, -0.8620855, 1.4785453, -2.0137978, 1.9455928

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524382, upper bound: 1.0508499
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524548, upper bound: 1.0506986
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524548, upper bound: 1.0508972
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -0.3022564, 0.8574660, -1.0208095, 0.9510663
1: -0.3648036, 0.8238323, -0.5588838, 1.0796481, -1.4444517, 1.3827161
2: -0.2912067, 0.9218535, -0.4826456, 1.2362378, -1.5274445, 1.4044991
3: -0.6862502, 0.9238169, -0.9591278, 1.2640674, -1.9503176, 1.8829447
4: -0.5532730, 1.1132669, -0.8620855, 1.4785453, -2.0318184, 1.9753524

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0550305, upper bound: 1.0508632
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549914, upper bound: 1.0506986
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0549914, upper bound: 1.0508972
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2013567, 0.6970940, -0.1851257, 0.6783313, -0.8796880, 0.8822198
1: -0.4231896, 0.8673437, -0.3969364, 0.8526834, -1.2758729, 1.2642801
2: -0.3478338, 0.9940906, -0.3242711, 0.9644121, -1.3122458, 1.3183616
3: -0.7603652, 1.0071222, -0.7251614, 0.9704387, -1.7308040, 1.7322836
4: -0.6426137, 1.2188928, -0.6038694, 1.1718525, -1.8144662, 1.8227623

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551354
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552524
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2909008, 0.8438774, -0.1851257, 0.6783313, -0.9692321, 1.0290031
1: -0.5446191, 1.0634959, -0.3969364, 0.8526834, -1.3973025, 1.4604323
2: -0.4665691, 1.2150321, -0.3242711, 0.9644121, -1.4309812, 1.5393032
3: -0.9410769, 1.2407138, -0.7251614, 0.9704387, -1.9115157, 1.9658753
4: -0.8382176, 1.4520541, -0.6038694, 1.1718525, -2.0100701, 2.0559235

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551354
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552524
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2013567, 0.6970940, -0.2674521, 0.8277271, -1.0290837, 0.9645461
1: -0.4231896, 0.8673437, -0.5075184, 1.0548140, -1.4780036, 1.3748621
2: -0.3478338, 0.9940906, -0.4327361, 1.2001319, -1.5479656, 1.4268267
3: -0.7603652, 1.0071222, -0.9017408, 1.1985904, -1.9589556, 1.9088629
4: -0.6426137, 1.2188928, -0.7864016, 1.4086894, -2.0513031, 2.0052943

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2909008, 0.8438774, -0.2674521, 0.8277271, -1.1186279, 1.1113296
1: -0.5446191, 1.0634959, -0.5075184, 1.0548140, -1.5994332, 1.5710143
2: -0.4665691, 1.2150321, -0.4327361, 1.2001319, -1.6667011, 1.6477683
3: -0.9410769, 1.2407138, -0.9017408, 1.1985904, -2.1396673, 2.1424546
4: -0.8382176, 1.4520541, -0.7864016, 1.4086894, -2.2469070, 2.2384558

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -1.2136881, 2.4463463, -2.6010213, 1.8481368
1: -0.3508692, 0.8027984, -1.6815057, 2.6714954, -3.0223646, 2.4843040
2: -0.2792361, 0.8971295, -1.6466799, 3.0085113, -3.2877474, 2.5438094
3: -0.6675518, 0.9013090, -2.1654463, 3.4600346, -4.1275859, 3.0667553
4: -0.5352525, 1.0835073, -2.5056057, 3.5195577, -4.0548100, 3.5891130

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487068
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487068
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -1.2136881, 2.4463463, -2.6096897, 1.8624980
1: -0.3648036, 0.8238323, -1.6815057, 2.6714954, -3.0362988, 2.5053380
2: -0.2912067, 0.9218535, -1.6466799, 3.0085113, -3.2997179, 2.5685334
3: -0.6862502, 0.9238169, -2.1654463, 3.4600346, -4.1462851, 3.0892632
4: -0.5532730, 1.1132669, -2.5056057, 3.5195577, -4.0728307, 3.6188726

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487052
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0491135
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -1.1447992, 2.3595920, -2.5142670, 1.7792479
1: -0.3508692, 0.8027984, -1.5842650, 2.6178081, -2.9686770, 2.3870635
2: -0.2792361, 0.8971295, -1.5295095, 2.9590945, -3.2383306, 2.4266391
3: -0.6675518, 0.9013090, -2.1012015, 3.3203290, -3.9878802, 3.0025105
4: -0.5352525, 1.0835073, -2.3559012, 3.4138608, -3.9491129, 3.4394085

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520692, upper bound: 1.0491056
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520692, upper bound: 1.0491056
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -1.1447992, 2.3595920, -2.5229354, 1.7936091
1: -0.3648036, 0.8238323, -1.5842650, 2.6178081, -2.9826114, 2.4080973
2: -0.2912067, 0.9218535, -1.5295095, 2.9590945, -3.2503009, 2.4513631
3: -0.6862502, 0.9238169, -2.1012015, 3.3203290, -4.0065794, 3.0250185
4: -0.5532730, 1.1132669, -2.3559012, 3.4138608, -3.9671338, 3.4691682

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0491135
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0491135
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -1.2136881, 2.4463463, -2.6292572, 1.9337090
1: -0.3864989, 0.9190384, -1.6815057, 2.6714954, -3.0579944, 2.6005440
2: -0.3115277, 1.0421524, -1.6466799, 3.0085113, -3.3200390, 2.6888323
3: -0.7443157, 1.0138530, -2.1654463, 3.4600346, -4.2043505, 3.1792994
4: -0.6034802, 1.2099588, -2.5056057, 3.5195577, -4.1230378, 3.7155645

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -1.2136881, 2.4463463, -2.6957753, 2.0081601
1: -0.4813044, 1.0119214, -1.6815057, 2.6714954, -3.1527998, 2.6934271
2: -0.4077803, 1.1496273, -1.6466799, 3.0085113, -3.4162912, 2.7963071
3: -0.8642187, 1.1482675, -2.1654463, 3.4600346, -4.3242531, 3.3137138
4: -0.7469358, 1.3539220, -2.5056057, 3.5195577, -4.2664933, 3.8595276

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -1.1447992, 2.3595920, -2.5425029, 1.8648201
1: -0.3864989, 0.9190384, -1.5842650, 2.6178081, -3.0043070, 2.5033035
2: -0.3115277, 1.0421524, -1.5295095, 2.9590945, -3.2706223, 2.5716619
3: -0.7443157, 1.0138530, -2.1012015, 3.3203290, -4.0646448, 3.1150546
4: -0.6034802, 1.2099588, -2.3559012, 3.4138608, -4.0173407, 3.5658600

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -1.1447992, 2.3595920, -2.6090207, 1.9392711
1: -0.4813044, 1.0119214, -1.5842650, 2.6178081, -3.0991123, 2.5961864
2: -0.4077803, 1.1496273, -1.5295095, 2.9590945, -3.3668747, 2.6791368
3: -0.8642187, 1.1482675, -2.1012015, 3.3203290, -4.1845474, 3.2494690
4: -0.7469358, 1.3539220, -2.3559012, 3.4138608, -4.1607962, 3.7098231

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1935697, 0.6831827, -1.2890360, 2.6080267, -2.8015962, 1.9722188
1: -0.4094878, 0.8581718, -1.7757368, 2.8224382, -3.2319260, 2.6339087
2: -0.3339639, 0.9709776, -1.7412896, 3.1970921, -3.5310555, 2.7122672
3: -0.7463301, 0.9851730, -2.2828507, 3.6829684, -4.4292984, 3.2680237
4: -0.6192465, 1.1830521, -2.6450953, 3.7347975, -4.3540440, 3.8281474

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499730, upper bound: 1.0487554
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499730, upper bound: 1.0487554
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.2037730, 0.6988031, -1.2890360, 2.6080267, -2.8117998, 1.9878391
1: -0.4249563, 0.8786784, -1.7757368, 2.8224382, -3.2473946, 2.6544151
2: -0.3480631, 0.9957969, -1.7412896, 3.1970921, -3.5451550, 2.7370865
3: -0.7669204, 1.0105076, -2.2828507, 3.6829684, -4.4498882, 3.2933583
4: -0.6409768, 1.2159789, -2.6450953, 3.7347975, -4.3757744, 3.8610742

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0487554
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0487554
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1935697, 0.6831827, -1.3504868, 2.6251640, -2.8187337, 2.0336695
1: -0.4094878, 0.8581718, -1.8179705, 2.8394029, -3.2488909, 2.6761422
2: -0.3339639, 0.9709776, -1.7806034, 3.2355962, -3.5695596, 2.7515810
3: -0.7463301, 0.9851730, -2.3383503, 3.7266910, -4.4730206, 3.3235233
4: -0.6192465, 1.1830521, -2.7018437, 3.7814958, -4.4007425, 3.8848958

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499909, upper bound: 1.0488068
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0499909, upper bound: 1.0488068
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.2037730, 0.6988031, -1.3504868, 2.6251640, -2.8289371, 2.0492897
1: -0.4249563, 0.8786784, -1.8179705, 2.8394029, -3.2643590, 2.6966488
2: -0.3480631, 0.9957969, -1.7806034, 3.2355962, -3.5836592, 2.7764003
3: -0.7669204, 1.0105076, -2.3383503, 3.7266910, -4.4936104, 3.3488579
4: -0.6409768, 1.2159789, -2.7018437, 3.7814958, -4.4224725, 3.9178226

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0488068
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0488068
time: 0.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.65 seconds
NS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
NS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555443
NS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
NS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0555443, upper bound: 1.0555617
NS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0538202, upper bound: 1.0509606
NS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0533586, upper bound: 1.0508971
NS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0550090, upper bound: 1.0510182
NS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0548845, upper bound: 1.0509748
NS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0509606, upper bound: 1.0538202
NS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0510182, upper bound: 1.0550090
NS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0508971, upper bound: 1.0533586
NS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0509748, upper bound: 1.0548845
NS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
NS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
NS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
NS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0501055, upper bound: 1.0501055
NS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0528721, upper bound: 1.0520802
NS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0528721, upper bound: 1.0520802
NS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0552339, upper bound: 1.0520802
NS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0552339, upper bound: 1.0520802
NS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0524548, upper bound: 1.0506986
NS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0524548, upper bound: 1.0508972
NS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0549914, upper bound: 1.0506986
NS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0549914, upper bound: 1.0508972
NS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551354
NS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552524
NS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0551354
NS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0518924, upper bound: 1.0552524
NS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487068
NS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487068
NS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487052
NS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0491135
NS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0520692, upper bound: 1.0491056
NS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0520692, upper bound: 1.0491056
NS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0491135
NS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0544438, upper bound: 1.0491135
NS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
NS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
NS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0485268
NS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0494371, upper bound: 1.0494973
NS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
NS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
NS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0502910, upper bound: 1.0488995
NS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0498452, upper bound: 1.0482416
NS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0499730, upper bound: 1.0487554
NS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0499730, upper bound: 1.0487554
NS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0487554
NS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0487554
NS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0499909, upper bound: 1.0488068
NS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0499909, upper bound: 1.0488068
NS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0488068
NS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.65
Output dim: 0, lower bound: -1.0500875, upper bound: 1.0488068

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -0.2096419, 0.7101620, -0.9198039, 0.9198039
1: -0.4267260, 0.8983275, -0.4267260, 0.8983275, -1.3250535, 1.3250535
2: -0.3558276, 1.0129061, -0.3558276, 1.0129061, -1.3687336, 1.3687336
3: -0.7740620, 1.0201761, -0.7740620, 1.0201761, -1.7942381, 1.7942381
4: -0.6563050, 1.2186592, -0.6563050, 1.2186592, -1.8749641, 1.8749641

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0550976
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553212, upper bound: 1.0553205
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -0.1746585, 0.6661327, -0.8757746, 0.8848205
1: -0.4267260, 0.8983275, -0.3825408, 0.8379728, -1.2646987, 1.2808683
2: -0.3558276, 1.0129061, -0.3095305, 0.9459749, -1.3018025, 1.3224366
3: -0.7740620, 1.0201761, -0.7079530, 0.9492630, -1.7233250, 1.7281290
4: -0.6563050, 1.2186592, -0.5819045, 1.1485283, -1.8048333, 1.8005637

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0550976
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553212, upper bound: 1.0553205
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -0.2096419, 0.7101620, -0.8848205, 0.8757746
1: -0.3825408, 0.8379728, -0.4267260, 0.8983275, -1.2808683, 1.2646987
2: -0.3095305, 0.9459749, -0.3558276, 1.0129061, -1.3224366, 1.3018025
3: -0.7079530, 0.9492630, -0.7740620, 1.0201761, -1.7281290, 1.7233250
4: -0.5819045, 1.1485283, -0.6563050, 1.2186592, -1.8005637, 1.8048333

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0552327
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553204, upper bound: 1.0553496
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -0.1746585, 0.6661327, -0.8407912, 0.8407912
1: -0.3825408, 0.8379728, -0.3825408, 0.8379728, -1.2205136, 1.2205136
2: -0.3095305, 0.9459749, -0.3095305, 0.9459749, -1.2555054, 1.2555054
3: -0.7079530, 0.9492630, -0.7079530, 0.9492630, -1.6572161, 1.6572161
4: -0.5819045, 1.1485283, -0.5819045, 1.1485283, -1.7304329, 1.7304329

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0529589, upper bound: 1.0552327
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0553204, upper bound: 1.0553496
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -0.1829112, 0.7200209, -0.9296628, 0.8930733
1: -0.4267260, 0.8983275, -0.3864989, 0.9190384, -1.3457644, 1.2848264
2: -0.3558276, 1.0129061, -0.3115277, 1.0421524, -1.3979800, 1.3244338
3: -0.7740620, 1.0201761, -0.7443157, 1.0138530, -1.7879150, 1.7644918
4: -0.6563050, 1.2186592, -0.6034802, 1.2099588, -1.8662637, 1.8221394

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0515432, upper bound: 1.0506374
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0535025, upper bound: 1.0505798
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.2096419, 0.7101620, -0.2494289, 0.7944719, -1.0041137, 0.9595910
1: -0.4267260, 0.8983275, -0.4813044, 1.0119214, -1.4386474, 1.3796319
2: -0.3558276, 1.0129061, -0.4077803, 1.1496273, -1.5054549, 1.4206864
3: -0.7740620, 1.0201761, -0.8642187, 1.1482675, -1.9223294, 1.8843946
4: -0.6563050, 1.2186592, -0.7469358, 1.3539220, -2.0102270, 1.9655950

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511796, upper bound: 1.0505705
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0530358, upper bound: 1.0505114
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -0.1829112, 0.7200209, -0.8946794, 0.8490440
1: -0.3825408, 0.8379728, -0.3864989, 0.9190384, -1.3015792, 1.2244717
2: -0.3095305, 0.9459749, -0.3115277, 1.0421524, -1.3516829, 1.2575027
3: -0.7079530, 0.9492630, -0.7443157, 1.0138530, -1.7218059, 1.6935787
4: -0.5819045, 1.1485283, -0.6034802, 1.2099588, -1.7918633, 1.7520086

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520364, upper bound: 1.0506850
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0547872, upper bound: 1.0507009
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1746585, 0.6661327, -0.2494289, 0.7944719, -0.9691303, 0.9155617
1: -0.3825408, 0.8379728, -0.4813044, 1.0119214, -1.3944622, 1.3192772
2: -0.3095305, 0.9459749, -0.4077803, 1.1496273, -1.4591578, 1.3537552
3: -0.7079530, 0.9492630, -0.8642187, 1.1482675, -1.8562205, 1.8134817
4: -0.5819045, 1.1485283, -0.7469358, 1.3539220, -1.9358265, 1.8954642

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519132, upper bound: 1.0506554
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0546610, upper bound: 1.0506729
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -0.2096419, 0.7101620, -0.8930733, 0.9296628
1: -0.3864989, 0.9190384, -0.4267260, 0.8983275, -1.2848264, 1.3457644
2: -0.3115277, 1.0421524, -0.3558276, 1.0129061, -1.3244338, 1.3979800
3: -0.7443157, 1.0138530, -0.7740620, 1.0201761, -1.7644918, 1.7879150
4: -0.6034802, 1.2099588, -0.6563050, 1.2186592, -1.8221394, 1.8662637

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -0.1746585, 0.6661327, -0.8490440, 0.8946794
1: -0.3864989, 0.9190384, -0.3825408, 0.8379728, -1.2244717, 1.3015792
2: -0.3115277, 1.0421524, -0.3095305, 0.9459749, -1.2575027, 1.3516829
3: -0.7443157, 1.0138530, -0.7079530, 0.9492630, -1.6935787, 1.7218059
4: -0.6034802, 1.2099588, -0.5819045, 1.1485283, -1.7520086, 1.7918633

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0409310, upper bound: 1.0487554
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -0.2096419, 0.7101620, -0.9595910, 1.0041137
1: -0.4813044, 1.0119214, -0.4267260, 0.8983275, -1.3796319, 1.4386474
2: -0.4077803, 1.1496273, -0.3558276, 1.0129061, -1.4206864, 1.5054549
3: -0.8642187, 1.1482675, -0.7740620, 1.0201761, -1.8843946, 1.9223294
4: -0.7469358, 1.3539220, -0.6563050, 1.2186592, -1.9655950, 2.0102270

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -0.1746585, 0.6661327, -0.9155617, 0.9691303
1: -0.4813044, 1.0119214, -0.3825408, 0.8379728, -1.3192772, 1.3944622
2: -0.4077803, 1.1496273, -0.3095305, 0.9459749, -1.3537552, 1.4591578
3: -0.8642187, 1.1482675, -0.7079530, 0.9492630, -1.8134817, 1.8562205
4: -0.7469358, 1.3539220, -0.5819045, 1.1485283, -1.8954642, 1.9358265

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1.0407075, upper bound: 1.0484164
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -0.1829112, 0.7200209, -0.9029321, 0.9029321
1: -0.3864989, 0.9190384, -0.3864989, 0.9190384, -1.3055373, 1.3055373
2: -0.3115277, 1.0421524, -0.3115277, 1.0421524, -1.3536801, 1.3536801
3: -0.7443157, 1.0138530, -0.7443157, 1.0138530, -1.7581687, 1.7581687
4: -0.6034802, 1.2099588, -0.6034802, 1.2099588, -1.8134390, 1.8134390

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -0.2494289, 0.7944719, -0.9773831, 0.9694498
1: -0.3864989, 0.9190384, -0.4813044, 1.0119214, -1.3984203, 1.4003428
2: -0.3115277, 1.0421524, -0.4077803, 1.1496273, -1.4611551, 1.4499327
3: -0.7443157, 1.0138530, -0.8642187, 1.1482675, -1.8925833, 1.8780715
4: -0.6034802, 1.2099588, -0.7469358, 1.3539220, -1.9574022, 1.9568946

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -0.1829112, 0.7200209, -0.9694498, 0.9773831
1: -0.4813044, 1.0119214, -0.3864989, 0.9190384, -1.4003428, 1.3984203
2: -0.4077803, 1.1496273, -0.3115277, 1.0421524, -1.4499327, 1.4611551
3: -0.8642187, 1.1482675, -0.7443157, 1.0138530, -1.8780715, 1.8925833
4: -0.7469358, 1.3539220, -0.6034802, 1.2099588, -1.9568946, 1.9574022

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -0.2494289, 0.7944719, -1.0439007, 1.0439007
1: -0.4813044, 1.0119214, -0.4813044, 1.0119214, -1.4932258, 1.4932258
2: -0.4077803, 1.1496273, -0.4077803, 1.1496273, -1.5574076, 1.5574076
3: -0.8642187, 1.1482675, -0.8642187, 1.1482675, -2.0124862, 2.0124862
4: -0.7469358, 1.3539220, -0.7469358, 1.3539220, -2.1008577, 2.1008577

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -0.1806883, 0.6643587, -0.8190338, 0.8151370
1: -0.3508692, 0.8027984, -0.3907865, 0.8332261, -1.1840954, 1.1935849
2: -0.2792361, 0.8971295, -0.3165183, 0.9454506, -1.2246866, 1.2136478
3: -0.6675518, 0.9013090, -0.7177029, 0.9569116, -1.6244633, 1.6190119
4: -0.5352525, 1.0835073, -0.5945891, 1.1518157, -1.6870681, 1.6780964

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -0.1905538, 0.6793638, -0.8340389, 0.8250024
1: -0.3508692, 0.8027984, -0.4057388, 0.8529248, -1.2037940, 1.2085372
2: -0.2792361, 0.8971295, -0.3302413, 0.9696531, -1.2488892, 1.2273707
3: -0.6675518, 0.9013090, -0.7378664, 0.9809170, -1.6484687, 1.6391754
4: -0.5352525, 1.0835073, -0.6153858, 1.1839089, -1.7191614, 1.6988931

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -0.1806883, 0.6643587, -0.8277022, 0.8294982
1: -0.3648036, 0.8238323, -0.3907865, 0.8332261, -1.1980298, 1.2146188
2: -0.2912067, 0.9218535, -0.3165183, 0.9454506, -1.2366574, 1.2383718
3: -0.6862502, 0.9238169, -0.7177029, 0.9569116, -1.6431618, 1.6415198
4: -0.5532730, 1.1132669, -0.5945891, 1.1518157, -1.7050886, 1.7078561

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -0.1905538, 0.6793638, -0.8427073, 0.8393637
1: -0.3648036, 0.8238323, -0.4057388, 0.8529248, -1.2177284, 1.2295711
2: -0.2912067, 0.9218535, -0.3302413, 0.9696531, -1.2608598, 1.2520947
3: -0.6862502, 0.9238169, -0.7378664, 0.9809170, -1.6671672, 1.6616833
4: -0.5532730, 1.1132669, -0.6153858, 1.1839089, -1.7371819, 1.7286527

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -0.2689438, 0.8082717, -0.9629468, 0.9033924
1: -0.3508692, 0.8027984, -0.5119697, 1.0272677, -1.3781369, 1.3147681
2: -0.2792361, 0.8971295, -0.4340174, 1.1612520, -1.4404881, 1.3311470
3: -0.6675518, 0.9013090, -0.8960808, 1.1860839, -1.8536357, 1.7973897
4: -0.5352525, 1.0835073, -0.7881429, 1.3786812, -1.9139336, 1.8716502

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -0.2798769, 0.8257407, -0.9804158, 0.9143256
1: -0.3508692, 0.8027984, -0.5248469, 1.0494590, -1.4003282, 1.3276453
2: -0.2792361, 0.8971295, -0.4479707, 1.1899524, -1.4691885, 1.3451002
3: -0.6675518, 0.9013090, -0.9173980, 1.2125832, -1.8801349, 1.8187070
4: -0.5352525, 1.0835073, -0.8094229, 1.4129685, -1.9482210, 1.8929302

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -0.2689438, 0.8082717, -0.9716152, 0.9177537
1: -0.3648036, 0.8238323, -0.5119697, 1.0272677, -1.3920712, 1.3358021
2: -0.2912067, 0.9218535, -0.4340174, 1.1612520, -1.4524587, 1.3558710
3: -0.6862502, 0.9238169, -0.8960808, 1.1860839, -1.8723341, 1.8198977
4: -0.5532730, 1.1132669, -0.7881429, 1.3786812, -1.9319541, 1.9014099

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -0.2798769, 0.8257407, -0.9890841, 0.9286869
1: -0.3648036, 0.8238323, -0.5248469, 1.0494590, -1.4142625, 1.3486792
2: -0.2912067, 0.9218535, -0.4479707, 1.1899524, -1.4811591, 1.3698242
3: -0.6862502, 0.9238169, -0.9173980, 1.2125832, -1.8988334, 1.8412149
4: -0.5532730, 1.1132669, -0.8094229, 1.4129685, -1.9662415, 1.9226898

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.2013567, 0.6970940, -0.2096419, 0.7101620, -0.9115187, 0.9067359
1: -0.4231896, 0.8673437, -0.4267260, 0.8983275, -1.3215171, 1.2940696
2: -0.3478338, 0.9940906, -0.3558276, 1.0129061, -1.3607398, 1.3499181
3: -0.7603652, 1.0071222, -0.7740620, 1.0201761, -1.7805413, 1.7811842
4: -0.6426137, 1.2188928, -0.6563050, 1.2186592, -1.8612728, 1.8751978

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0555396
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0552327
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0473507, upper bound: 1.0526091
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.2013567, 0.6970940, -0.1746585, 0.6661327, -0.8674895, 0.8717525
1: -0.4231896, 0.8673437, -0.3825408, 0.8379728, -1.2611624, 1.2498845
2: -0.3478338, 0.9940906, -0.3095305, 0.9459749, -1.2938087, 1.3036211
3: -0.7603652, 1.0071222, -0.7079530, 0.9492630, -1.7096283, 1.7150751
4: -0.6426137, 1.2188928, -0.5819045, 1.1485283, -1.7911420, 1.8007973

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0537152, upper bound: 1.0555396
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0528525, upper bound: 1.0552327
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520802, upper bound: 1.0553110
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2909008, 0.8438774, -0.2096419, 0.7101620, -1.0010629, 1.0535192
1: -0.5446191, 1.0634959, -0.4267260, 0.8983275, -1.4429467, 1.4902219
2: -0.4665691, 1.2150321, -0.3558276, 1.0129061, -1.4794753, 1.5708597
3: -0.9410769, 1.2407138, -0.7740620, 1.0201761, -1.9612529, 2.0147758
4: -0.8382176, 1.4520541, -0.6563050, 1.2186592, -2.0568767, 2.1083591

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516505, upper bound: 1.0551354
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511174, upper bound: 1.0545827
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508131, upper bound: 1.0544705
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2909008, 0.8438774, -0.1746585, 0.6661327, -0.9570335, 1.0185359
1: -0.5446191, 1.0634959, -0.3825408, 0.8379728, -1.3825920, 1.4460367
2: -0.4665691, 1.2150321, -0.3095305, 0.9459749, -1.4125440, 1.5245626
3: -0.9410769, 1.2407138, -0.7079530, 0.9492630, -1.8903400, 1.9486668
4: -0.8382176, 1.4520541, -0.5819045, 1.1485283, -1.9867460, 2.0339587

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516505, upper bound: 1.0552524
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0511174, upper bound: 1.0549914
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0508131, upper bound: 1.0550306
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -1.1942506, 2.3986688, -2.5533438, 1.8286992
1: -0.3508692, 0.8027984, -1.6547589, 2.6153526, -2.9662213, 2.4575572
2: -0.2792361, 0.8971295, -1.6229236, 2.9506650, -3.2299011, 2.5200531
3: -0.6675518, 0.9013090, -2.1280761, 3.3928285, -4.0603795, 3.0293851
4: -0.5352525, 1.0835073, -2.4708157, 3.4560237, -3.9912763, 3.5543230

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487068
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0487068
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -1.2338083, 2.3774090, -2.5320840, 1.8682569
1: -0.3508692, 0.8027984, -1.6776686, 2.5895333, -2.9404023, 2.4804668
2: -0.2792361, 0.8971295, -1.6476820, 2.9423103, -3.2215464, 2.5448115
3: -0.6675518, 0.9013090, -2.1502376, 3.3911312, -4.0586824, 3.0515466
4: -0.5352525, 1.0835073, -2.5040121, 3.4527168, -3.9879694, 3.5875194

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0491056
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513804, upper bound: 1.0491056
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -1.1942506, 2.3986688, -2.5620122, 1.8430605
1: -0.3648036, 0.8238323, -1.6547589, 2.6153526, -2.9801559, 2.4785912
2: -0.2912067, 0.9218535, -1.6229236, 2.9506650, -3.2418716, 2.5447772
3: -0.6862502, 0.9238169, -2.1280761, 3.3928285, -4.0790787, 3.0518930
4: -0.5532730, 1.1132669, -2.4708157, 3.4560237, -4.0092964, 3.5840826

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513933, upper bound: 1.0487052
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513933, upper bound: 1.0487052
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -1.2338083, 2.3774090, -2.5407524, 1.8826182
1: -0.3648036, 0.8238323, -1.6776686, 2.5895333, -2.9543369, 2.5015008
2: -0.2912067, 0.9218535, -1.6476820, 2.9423103, -3.2335167, 2.5695355
3: -0.6862502, 0.9238169, -2.1502376, 3.3911312, -4.0773811, 3.0740545
4: -0.5532730, 1.1132669, -2.5040121, 3.4527168, -4.0059900, 3.6172791

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513933, upper bound: 1.0491135
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513933, upper bound: 1.0491135
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -1.0775944, 2.2407463, -2.3954210, 1.7120430
1: -0.3508692, 0.8027984, -1.5017793, 2.4878531, -2.8387220, 2.3045778
2: -0.2792361, 0.8971295, -1.4541702, 2.8047833, -3.0840192, 2.3512998
3: -0.6675518, 0.9013090, -1.9936779, 3.1547756, -3.8223271, 2.8949869
4: -0.5352525, 1.0835073, -2.2470565, 3.2434559, -3.7787085, 3.3305638

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520692, upper bound: 1.0491056
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520680, upper bound: 1.0491056
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1546751, 0.6344486, -1.0981882, 2.3075790, -2.4622540, 1.7326368
1: -0.3508692, 0.8027984, -1.5334218, 2.5611386, -2.9120076, 2.3362203
2: -0.2792361, 0.8971295, -1.4820399, 2.8812847, -3.1605208, 2.3791695
3: -0.6675518, 0.9013090, -2.0440249, 3.2370739, -3.9046254, 2.9453340
4: -0.5352525, 1.0835073, -2.2893934, 3.3254902, -3.8607426, 3.3729007

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520692, upper bound: 1.0491056
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520692, upper bound: 1.0491056
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -1.0775944, 2.2407463, -2.4040892, 1.7264043
1: -0.3648036, 0.8238323, -1.5017793, 2.4878531, -2.8526566, 2.3256116
2: -0.2912067, 0.9218535, -1.4541702, 2.8047833, -3.0959897, 2.3760238
3: -0.6862502, 0.9238169, -1.9936779, 3.1547756, -3.8410258, 2.9174948
4: -0.5532730, 1.1132669, -2.2470565, 3.2434559, -3.7967288, 3.3603234

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520809, upper bound: 1.0491135
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520809, upper bound: 1.0491135
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1633435, 0.6488099, -1.0981882, 2.3075790, -2.4709222, 1.7469981
1: -0.3648036, 0.8238323, -1.5334218, 2.5611386, -2.9259422, 2.3572540
2: -0.2912067, 0.9218535, -1.4820399, 2.8812847, -3.1724913, 2.4038935
3: -0.6862502, 0.9238169, -2.0440249, 3.2370739, -3.9233241, 2.9678419
4: -0.5532730, 1.1132669, -2.2893934, 3.3254902, -3.8787632, 3.4026604

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 44

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520809, upper bound: 1.0491135
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0521018, upper bound: 1.0491135
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -1.0775944, 2.2407463, -2.4236569, 1.7976153
1: -0.3864989, 0.9190384, -1.5017793, 2.4878531, -2.8743520, 2.4208179
2: -0.3115277, 1.0421524, -1.4541702, 2.8047833, -3.1163111, 2.4963226
3: -0.7443157, 1.0138530, -1.9936779, 3.1547756, -3.8990912, 3.0075307
4: -0.6034802, 1.2099588, -2.2470565, 3.2434559, -3.8469362, 3.4570153

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1829112, 0.7200209, -1.0981882, 2.3075790, -2.4904900, 1.8182091
1: -0.3864989, 0.9190384, -1.5334218, 2.5611386, -2.9476376, 2.4524603
2: -0.3115277, 1.0421524, -1.4820399, 2.8812847, -3.1928124, 2.5241923
3: -0.7443157, 1.0138530, -2.0440249, 3.2370739, -3.9813895, 3.0578780
4: -0.6034802, 1.2099588, -2.2893934, 3.3254902, -3.9289703, 3.4993522

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -1.0775944, 2.2407463, -2.4901748, 1.8720663
1: -0.4813044, 1.0119214, -1.5017793, 2.4878531, -2.9691575, 2.5137007
2: -0.4077803, 1.1496273, -1.4541702, 2.8047833, -3.2125633, 2.6037974
3: -0.8642187, 1.1482675, -1.9936779, 3.1547756, -4.0189943, 3.1419454
4: -0.7469358, 1.3539220, -2.2470565, 3.2434559, -3.9903917, 3.6009784

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2494289, 0.7944719, -1.0981882, 2.3075790, -2.5570078, 1.8926600
1: -0.4813044, 1.0119214, -1.5334218, 2.5611386, -3.0424430, 2.5453432
2: -0.4077803, 1.1496273, -1.4820399, 2.8812847, -3.2890649, 2.6316671
3: -0.8642187, 1.1482675, -2.0440249, 3.2370739, -4.1012926, 3.1922925
4: -0.7469358, 1.3539220, -2.2893934, 3.3254902, -4.0724254, 3.6433153

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1935697, 0.6831827, -1.2280920, 2.4976134, -2.6911826, 1.9112747
1: -0.4094878, 0.8581718, -1.7021303, 2.7018881, -3.1113758, 2.5603023
2: -0.3339639, 0.9709776, -1.6748042, 3.0545073, -3.3884711, 2.6457818
3: -0.7463301, 0.9851730, -2.1857715, 3.5306742, -4.2770042, 3.1709445
4: -0.6192465, 1.1830521, -2.5491729, 3.5767035, -4.1959500, 3.7322249

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1935697, 0.6831827, -1.2518184, 2.5562680, -2.7498374, 1.9350011
1: -0.4094878, 0.8581718, -1.7313824, 2.7695041, -3.1789918, 2.5895543
2: -0.3339639, 0.9709776, -1.7005339, 3.1234226, -3.4573860, 2.6715114
3: -0.7463301, 0.9851730, -2.2261362, 3.6091881, -4.3555174, 3.2113092
4: -0.6192465, 1.1830521, -2.5878334, 3.6477928, -4.2670393, 3.7708855

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.2037730, 0.6988031, -1.2280920, 2.4976134, -2.7013865, 1.9268950
1: -0.4249563, 0.8786784, -1.7021303, 2.7018881, -3.1268444, 2.5808086
2: -0.3480631, 0.9957969, -1.6748042, 3.0545073, -3.4025702, 2.6706011
3: -0.7669204, 1.0105076, -2.1857715, 3.5306742, -4.2975945, 3.1962790
4: -0.6409768, 1.2159789, -2.5491729, 3.5767035, -4.2176805, 3.7651517

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.2037730, 0.6988031, -1.2518184, 2.5562680, -2.7600410, 1.9506215
1: -0.4249563, 0.8786784, -1.7313824, 2.7695041, -3.1944599, 2.6100607
2: -0.3480631, 0.9957969, -1.7005339, 3.1234226, -3.4714856, 2.6963308
3: -0.7669204, 1.0105076, -2.2261362, 3.6091881, -4.3761086, 3.2366438
4: -0.6409768, 1.2159789, -2.5878334, 3.6477928, -4.2887692, 3.8038123

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1935697, 0.6831827, -1.2646008, 2.5050335, -2.6986027, 1.9477835
1: -0.4094878, 0.8581718, -1.7316601, 2.7091794, -3.1186671, 2.5898318
2: -0.3339639, 0.9709776, -1.7030411, 3.0801201, -3.4140832, 2.6740186
3: -0.7463301, 0.9851730, -2.2229700, 3.5556083, -4.3019385, 3.2081430
4: -0.6192465, 1.1830521, -2.5911546, 3.6092997, -4.2285461, 3.7742066

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1935697, 0.6831827, -1.3221376, 2.5785892, -2.7721586, 2.0053203
1: -0.4094878, 0.8581718, -1.7777612, 2.7921562, -3.2016439, 2.6359329
2: -0.3339639, 0.9709776, -1.7462506, 3.1683240, -3.5022879, 2.7172282
3: -0.7463301, 0.9851730, -2.2886705, 3.6591811, -4.4055114, 3.2738435
4: -0.6192465, 1.1830521, -2.6495557, 3.7025068, -4.3217535, 3.8326077

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.2037730, 0.6988031, -1.2646008, 2.5050335, -2.7088065, 1.9634038
1: -0.4249563, 0.8786784, -1.7316601, 2.7091794, -3.1341357, 2.6103384
2: -0.3480631, 0.9957969, -1.7030411, 3.0801201, -3.4281831, 2.6988380
3: -0.7669204, 1.0105076, -2.2229700, 3.5556083, -4.3225288, 3.2334776
4: -0.6409768, 1.2159789, -2.5911546, 3.6092997, -4.2502761, 3.8071334

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.2037730, 0.6988031, -1.3221376, 2.5785892, -2.7823622, 2.0209408
1: -0.4249563, 0.8786784, -1.7777612, 2.7921562, -3.2171121, 2.6564395
2: -0.3480631, 0.9957969, -1.7462506, 3.1683240, -3.5163870, 2.7420475
3: -0.7669204, 1.0105076, -2.2886705, 3.6591811, -4.4261017, 3.2991781
4: -0.6409768, 1.2159789, -2.6495557, 3.7025068, -4.3434834, 3.8655345

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.74 + 230.25 = 231.99 seconds
