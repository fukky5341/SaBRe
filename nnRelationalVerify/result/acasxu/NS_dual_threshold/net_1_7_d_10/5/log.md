## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 175.108430440685


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007)
1: (-119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691)
2: (-175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027)
3: (-100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469)
4: (-160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.87 + 2.21 = 3.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -175.1171863, upper bound: 175.1171863

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1168227, upper bound: 175.1169184
time: 0.85 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.65 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -175.1168227, upper bound: 175.1169184
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -43.8347244, 168.1222839, -42.7666740, 164.0580750, -207.8927917, 210.8889465
1: -119.0844269, 383.1814270, -116.1888351, 373.8872681, -492.9716797, 499.3702698
2: -175.4753113, 326.4086914, -171.1952057, 318.6710815, -494.1463928, 497.6038513
3: -100.4854202, 409.6264343, -98.0361176, 399.6714783, -500.1568909, 507.6625366
4: -160.2841492, 286.0808105, -156.4088135, 279.2815247, -439.5656433, 442.4896240

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.73 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.81 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -42.6489410, 163.5203400, -52.2043495, 199.7965088, -242.4454498, 215.7246704
1: -115.7870789, 372.5597229, -140.6551361, 457.9893799, -573.7764282, 513.2148438
2: -170.7633057, 317.6326904, -205.7594604, 391.9710693, -562.7343750, 523.3921509
3: -97.7277603, 398.4181824, -118.6919250, 487.8189392, -585.5466919, 517.1101074
4: -156.0076752, 278.3982849, -188.5289154, 343.2115479, -499.2192383, 466.9270935

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.88 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.59 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -42.7666740, 164.0580750, -42.7666740, 164.0580750, -206.8247375, 206.8247375
1: -116.1888351, 373.8872681, -116.1888351, 373.8872681, -490.0761108, 490.0760803
2: -171.1952057, 318.6710815, -171.1952057, 318.6710815, -489.8662415, 489.8662720
3: -98.0361176, 399.6714783, -98.0361176, 399.6714783, -497.7075806, 497.7075806
4: -156.4088135, 279.2815247, -156.4088135, 279.2815247, -435.6903381, 435.6903381

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1092564, upper bound: 175.1106865
time: 0.84 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1078721, upper bound: 175.1077912
time: 0.90 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -52.2043495, 199.7965088, -42.7666740, 164.0580750, -216.2624207, 242.5631714
1: -140.6551361, 457.9893799, -116.1888351, 373.8872681, -514.5424194, 574.1782227
2: -205.7594604, 391.9710693, -171.1952057, 318.6710815, -524.4305420, 563.1662598
3: -118.6919250, 487.8189392, -98.0361176, 399.6714783, -518.3634033, 585.8550415
4: -188.5289154, 343.2115479, -156.4088135, 279.2815247, -467.8103333, 499.6203613

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148387, upper bound: 175.1149938
time: 1.42 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1147328, upper bound: 175.1149769
time: 0.78 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -42.7666740, 164.0580750, -52.2043495, 199.7965088, -242.5631714, 216.2624207
1: -116.1888351, 373.8872681, -140.6551361, 457.9893799, -574.1782227, 514.5424194
2: -171.1952057, 318.6710815, -205.7594604, 391.9710693, -563.1662598, 524.4305420
3: -98.0361176, 399.6714783, -118.6919250, 487.8189392, -585.8550415, 518.3634033
4: -156.4088135, 279.2815247, -188.5289154, 343.2115479, -499.6203613, 467.8103638

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1148761, upper bound: 175.1146283
time: 1.32 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1141261, upper bound: 175.1141261
time: 0.97 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -52.2043495, 199.7965088, -52.2043495, 199.7965088, -252.0008545, 252.0008545
1: -140.6551361, 457.9893799, -140.6551361, 457.9893799, -598.6444702, 598.6444702
2: -205.7594604, 391.9710693, -205.7594604, 391.9710693, -597.7305298, 597.7305298
3: -118.6919250, 487.8189392, -118.6919250, 487.8189392, -606.5108643, 606.5108643
4: -188.5289154, 343.2115479, -188.5289154, 343.2115479, -531.7404175, 531.7404785

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1091965, upper bound: 175.1106866
time: 0.96 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077891, upper bound: 175.1077956
time: 1.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.97 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1092564, upper bound: 175.1106865
NS_B1_A1_A2, status: Status.VERIFIED, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1078721, upper bound: 175.1077912
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1148387, upper bound: 175.1149938
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1147328, upper bound: 175.1149769
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1148761, upper bound: 175.1146283
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1141261, upper bound: 175.1141261
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1091965, upper bound: 175.1106866
NS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 2.97
Output dim: 0, lower bound: -175.1077891, upper bound: 175.1077956

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -42.6816635, 163.7285156, -42.7666740, 164.0580750, -206.7397308, 206.4951630
1: -115.9523773, 373.1490479, -116.1888351, 373.8872681, -489.8396606, 489.3378601
2: -170.8125763, 318.0581970, -171.1952057, 318.6710815, -489.4836426, 489.2533875
3: -97.8353500, 398.8726196, -98.0361176, 399.6714783, -497.5068359, 496.9087524
4: -156.0698090, 278.7397156, -156.4088135, 279.2815247, -435.3512878, 435.1485291

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1078773, upper bound: 175.1078773
time: 0.82 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1078773, upper bound: 175.1078773
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -51.8010750, 198.2330780, -41.3389091, 158.5311737, -210.3322449, 239.5719910
1: -139.5649414, 454.3887939, -112.3375778, 361.1010437, -500.6659851, 566.7263794
2: -204.2322998, 388.9531250, -165.8764496, 307.7619019, -511.9941711, 554.8295898
3: -117.7784195, 484.0117188, -94.8152008, 386.2667847, -504.0451660, 578.8267212
4: -187.1174774, 340.5682068, -151.4770203, 269.7766418, -456.8941040, 492.0451355

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1092564, upper bound: 175.1106865
time: 0.75 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1078721, upper bound: 175.1077866
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -51.2478409, 196.1268005, -45.8709335, 175.9414673, -227.1893005, 241.9977264
1: -138.0597992, 449.6985779, -124.4324875, 402.0142517, -540.0739746, 574.1309814
2: -201.9375000, 384.9766235, -183.3883972, 341.2222900, -543.1597900, 568.3649902
3: -116.5218506, 479.0079956, -104.9526367, 429.9045410, -546.4263916, 583.9606323
4: -185.0434265, 337.0954285, -167.5778046, 299.1374817, -484.1809082, 504.6732178

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1091838, upper bound: 175.1106831
time: 0.74 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1078118, upper bound: 175.1077971
time: 1.18 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -41.3389091, 158.5311737, -51.8010750, 198.2330780, -239.5719910, 210.3322449
1: -112.3375778, 361.1010437, -139.5649414, 454.3887939, -566.7263794, 500.6659851
2: -165.8764496, 307.7619019, -204.2322998, 388.9531250, -554.8295898, 511.9941711
3: -94.8152008, 386.2667847, -117.7784195, 484.0117188, -578.8267212, 504.0451660
4: -151.4770203, 269.7766418, -187.1174774, 340.5682068, -492.0451355, 456.8941040

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1106866, upper bound: 175.1092564
time: 1.01 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077866, upper bound: 175.1078721
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -45.8709335, 175.9414673, -51.2478409, 196.1268005, -241.9977264, 227.1893005
1: -124.4324875, 402.0142517, -138.0597992, 449.6985779, -574.1309814, 540.0739746
2: -183.3883972, 341.2222900, -201.9375000, 384.9766235, -568.3649902, 543.1597900
3: -104.9526367, 429.9045410, -116.5218506, 479.0079956, -583.9606323, 546.4263916
4: -167.5778046, 299.1374817, -185.0434265, 337.0954285, -504.6732178, 484.1809082

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1106831, upper bound: 175.1091838
time: 1.01 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077866, upper bound: 175.1078118
time: 0.91 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -52.1151962, 199.4456329, -52.2043495, 199.7965088, -251.9117126, 251.6499786
1: -140.4091187, 457.2044678, -140.6551361, 457.9893799, -598.3983765, 597.8596191
2: -205.3594666, 391.3057556, -205.7594604, 391.9710693, -597.3305664, 597.0651245
3: -118.4839630, 486.9721375, -118.6919250, 487.8189392, -606.3029175, 605.6640625
4: -188.1714478, 342.6252747, -188.5289154, 343.2115479, -531.3829956, 531.1541748

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077945, upper bound: 175.1077956
time: 0.80 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077945, upper bound: 175.1077956
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.55 seconds
NS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1078773, upper bound: 175.1078773
NS_B1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1078773, upper bound: 175.1078773
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1092564, upper bound: 175.1106865
NS_B1_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1078721, upper bound: 175.1077866
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1091838, upper bound: 175.1106831
NS_B1_A2_B2_A2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1078118, upper bound: 175.1077971
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1106866, upper bound: 175.1092564
NS_B2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1077866, upper bound: 175.1078721
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1106831, upper bound: 175.1091838
NS_B2_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1077866, upper bound: 175.1078118
NS_B2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1077945, upper bound: 175.1077956
NS_B2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -175.1077945, upper bound: 175.1077956

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -51.7116013, 197.8811493, -41.3389091, 158.5311737, -210.2427673, 239.2200623
1: -139.3179016, 453.6016846, -112.3375778, 361.1010437, -500.4189453, 565.9392090
2: -203.8301697, 388.2861023, -165.8764496, 307.7619019, -511.5920715, 554.1625366
3: -117.5695953, 483.1619263, -94.8152008, 386.2667847, -503.8363647, 577.9770508
4: -186.7581787, 339.9801636, -151.4770203, 269.7766418, -456.5348206, 491.4571533

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1079806, upper bound: 175.1102331
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1086064, upper bound: 175.1104376
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -51.1584816, 195.7756958, -45.8709335, 175.9414673, -227.0999298, 241.6466370
1: -137.8123322, 448.9141846, -124.4324875, 402.0142517, -539.8265991, 573.3466797
2: -201.5367889, 384.3106384, -183.3883972, 341.2222900, -542.7590332, 567.6990356
3: -116.3130493, 478.1615295, -104.9526367, 429.9045410, -546.2175903, 583.1141357
4: -184.6852875, 336.5086975, -167.5778046, 299.1374817, -483.8227539, 504.0864868

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1079146, upper bound: 175.1102520
time: 0.78 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1082483, upper bound: 175.1103346
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -41.3389091, 158.5311737, -51.7116013, 197.8811493, -239.2200623, 210.2427673
1: -112.3375778, 361.1010437, -139.3179016, 453.6016846, -565.9391479, 500.4189453
2: -165.8764496, 307.7619019, -203.8301697, 388.2861023, -554.1625366, 511.5920715
3: -94.8152008, 386.2667847, -117.5695953, 483.1619263, -577.9770508, 503.8363647
4: -151.4770203, 269.7766418, -186.7581787, 339.9801636, -491.4571533, 456.5348206

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1102331, upper bound: 175.1079806
time: 0.93 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1104375, upper bound: 175.1086064
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -45.8709335, 175.9414673, -51.1584816, 195.7756958, -241.6466370, 227.0999298
1: -124.4324875, 402.0142517, -137.8123322, 448.9141846, -573.3466797, 539.8265991
2: -183.3883972, 341.2222900, -201.5367889, 384.3106384, -567.6990356, 542.7590942
3: -104.9526367, 429.9045410, -116.3130493, 478.1615295, -583.1141357, 546.2175903
4: -167.5778046, 299.1374817, -184.6852875, 336.5086975, -504.0864868, 483.8227539

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1102520, upper bound: 175.1079146
time: 0.78 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1103346, upper bound: 175.1082483
time: 1.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.03 seconds
NS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1079806, upper bound: 175.1102331
NS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1086064, upper bound: 175.1104376
NS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1079146, upper bound: 175.1102520
NS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1082483, upper bound: 175.1103346
NS_B2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1102331, upper bound: 175.1079806
NS_B2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1104375, upper bound: 175.1086064
NS_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1102520, upper bound: 175.1079146
NS_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.03
Output dim: 0, lower bound: -175.1103346, upper bound: 175.1082483

## BFS NS instance: NS_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -50.5545349, 193.4322205, -39.2191963, 150.4057465, -200.9602661, 232.6514130
1: -136.0657349, 443.7417603, -106.4535751, 342.5932007, -478.6589355, 550.1953125
2: -198.2966309, 380.2545166, -156.3977814, 292.4544678, -490.7510986, 536.6522827
3: -114.7866287, 472.3861389, -89.7849503, 366.3431396, -481.1297607, 562.1710815
4: -181.8890381, 332.8321533, -143.0945435, 256.2891541, -438.1781921, 475.9266968

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1077274, upper bound: 175.1101525
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1079208, upper bound: 175.1102330
time: 0.98 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -50.6400986, 193.7667999, -43.7814865, 168.0317688, -218.6718445, 237.5482788
1: -136.1913300, 444.2044373, -117.8903275, 384.4804993, -520.6718140, 562.0947876
2: -199.2961884, 380.6669617, -172.8881378, 330.6353455, -529.9315186, 553.5551147
3: -114.9564743, 473.1630554, -99.5192871, 409.2031860, -524.1596680, 572.6823730
4: -182.7192535, 333.3411255, -158.6732178, 289.4013672, -472.1206055, 492.0142822

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1085736, upper bound: 175.1104177
time: 0.79 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1085736, upper bound: 175.1104376
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -50.0050964, 191.3409729, -43.6090851, 167.3502350, -217.3553314, 234.9500275
1: -134.5643768, 439.1088867, -118.2372742, 382.4390259, -517.0034180, 557.3461304
2: -196.0205688, 376.3040771, -173.2625122, 325.0066833, -521.0272217, 549.5665894
3: -113.5285416, 467.4355469, -99.6585693, 408.7284851, -522.2570190, 567.0941162
4: -179.8417816, 329.3847046, -158.5839996, 284.8521118, -464.6939087, 487.9686890

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1077274, upper bound: 175.1102264
time: 0.98 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1078776, upper bound: 175.1102520
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -50.0358772, 191.4662170, -48.0940170, 184.5474548, -234.5833282, 239.5602112
1: -134.5444489, 439.0816956, -129.5681915, 423.3705444, -557.9149780, 568.6499023
2: -196.8028259, 376.3222351, -189.6818848, 362.3751221, -559.1779175, 566.0039673
3: -113.5767975, 467.6905518, -109.3105011, 450.9009399, -564.4776611, 577.0009766
4: -180.4652252, 329.5490417, -174.0693054, 317.2913818, -497.7565918, 503.6183472

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1082431, upper bound: 175.1103084
time: 0.66 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1082431, upper bound: 175.1103346
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -39.2191963, 150.4057465, -50.5545349, 193.4322205, -232.6514130, 200.9602661
1: -106.4535751, 342.5932007, -136.0657349, 443.7417603, -550.1953125, 478.6589355
2: -156.3977814, 292.4544678, -198.2966309, 380.2545166, -536.6522217, 490.7510986
3: -89.7849503, 366.3431396, -114.7866287, 472.3861389, -562.1710815, 481.1297607
4: -143.0945435, 256.2891541, -181.8890381, 332.8321533, -475.9266968, 438.1781921

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1101525, upper bound: 175.1077274
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1102329, upper bound: 175.1079208
time: 0.86 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -43.7814865, 168.0317688, -50.6400986, 193.7667999, -237.5482788, 218.6718445
1: -117.8903275, 384.4804993, -136.1913300, 444.2044373, -562.0947876, 520.6718140
2: -172.8881378, 330.6353455, -199.2961884, 380.6669617, -553.5551147, 529.9314575
3: -99.5192871, 409.2031860, -114.9564743, 473.1630554, -572.6823730, 524.1596680
4: -158.6732178, 289.4013672, -182.7192535, 333.3411255, -492.0142822, 472.1206055

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1104178, upper bound: 175.1085736
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1104178, upper bound: 175.1086064
time: 1.04 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -43.6090851, 167.3502350, -50.0050964, 191.3409729, -234.9500275, 217.3553314
1: -118.2372742, 382.4390259, -134.5643768, 439.1088867, -557.3461304, 517.0033569
2: -173.2625122, 325.0066833, -196.0205688, 376.3040771, -549.5665894, 521.0272217
3: -99.6585693, 408.7284851, -113.5285416, 467.4355469, -567.0941162, 522.2570190
4: -158.5839996, 284.8521118, -179.8417816, 329.3847046, -487.9686890, 464.6939087

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1101525, upper bound: 175.1077689
time: 0.94 seconds

## Relational analysis of NS_B2_A1_A2_B1_A1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1102520, upper bound: 175.1078776
time: 1.93 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -48.0940170, 184.5474548, -50.0358772, 191.4662170, -239.5602264, 234.5833282
1: -129.5681915, 423.3705444, -134.5444489, 439.0816956, -568.6499023, 557.9149780
2: -189.6818848, 362.3751221, -196.8028259, 376.3222351, -566.0040283, 559.1779175
3: -109.3105011, 450.9009399, -113.5767975, 467.6905518, -577.0009766, 564.4776611
4: -174.0693054, 317.2913818, -180.4652252, 329.5490417, -503.6183472, 497.7565918

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1103085, upper bound: 175.1082431
time: 0.82 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1103085, upper bound: 175.1082483
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.67 seconds
NS_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1077274, upper bound: 175.1101525
NS_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1079208, upper bound: 175.1102330
NS_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1085736, upper bound: 175.1104177
NS_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1085736, upper bound: 175.1104376
NS_B1_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1077274, upper bound: 175.1102264
NS_B1_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1078776, upper bound: 175.1102520
NS_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1082431, upper bound: 175.1103084
NS_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1082431, upper bound: 175.1103346
NS_B2_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1101525, upper bound: 175.1077274
NS_B2_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1102329, upper bound: 175.1079208
NS_B2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1104178, upper bound: 175.1085736
NS_B2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1104178, upper bound: 175.1086064
NS_B2_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1101525, upper bound: 175.1077689
NS_B2_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1102520, upper bound: 175.1078776
NS_B2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1103085, upper bound: 175.1082431
NS_B2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.67
Output dim: 0, lower bound: -175.1103085, upper bound: 175.1082483

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -50.0133514, 191.3534241, -37.1877785, 142.5817566, -192.5951080, 228.5411987
1: -134.5356140, 439.1558533, -100.9189758, 324.7163086, -459.2518311, 540.0748291
2: -195.7950592, 376.4314880, -148.0071106, 277.0126953, -472.8077393, 524.4385986
3: -113.4945755, 467.3587036, -85.1052017, 347.0308228, -460.5253906, 552.4638062
4: -179.7091980, 329.4500427, -135.4534760, 242.7190399, -422.4282227, 464.9034729

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1058412, upper bound: 175.1088862
time: 0.83 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1073906, upper bound: 175.1101041
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -50.1261063, 191.7990570, -38.1325684, 146.2725220, -196.3986206, 229.9316101
1: -134.8771210, 440.0725098, -103.4257050, 333.3331299, -468.2102661, 543.4982300
2: -196.4867859, 377.1958923, -151.6874390, 284.7179871, -481.2047119, 528.8833008
3: -113.7814026, 468.4150391, -87.2283630, 356.2762756, -470.0576782, 555.6432495
4: -180.2232056, 330.1417847, -138.8651886, 249.4823303, -429.7054749, 469.0069580

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1069933, upper bound: 175.1083043
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1069933, upper bound: 175.1102329
time: 0.83 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -49.5331764, 189.5534515, -43.7814865, 168.0317688, -217.5649261, 233.3349304
1: -133.2194672, 435.0702820, -117.8903275, 384.4804993, -517.6998901, 552.9605713
2: -193.5777283, 373.0771484, -172.8881378, 330.6353455, -524.2130127, 545.9652710
3: -112.3526382, 462.9354553, -99.5192871, 409.2031860, -521.5558472, 562.4547119
4: -177.7906342, 326.4732971, -158.6732178, 289.4013672, -467.1920166, 485.1464539

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1072130, upper bound: 175.1098995
time: 1.15 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1079179, upper bound: 175.1104177
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -54.6163330, 209.3759460, -43.7814865, 168.0317688, -222.6481018, 253.1574249
1: -146.1183472, 481.7488403, -117.8903275, 384.4804993, -530.5986938, 599.6391602
2: -211.9063110, 415.3855286, -172.8881378, 330.6353455, -542.5416260, 588.2736816
3: -123.3006287, 510.7327271, -99.5192871, 409.2031860, -532.5037231, 610.2520142
4: -195.0705261, 363.2395630, -158.6732178, 289.4013672, -484.4718933, 521.9127808

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1072130, upper bound: 175.1100508
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1079179, upper bound: 175.1102847
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -49.4877510, 189.3600616, -42.0362015, 161.2504883, -210.7382050, 231.3962708
1: -133.0959473, 434.7307129, -113.9261780, 368.5407715, -501.6366882, 548.6568604
2: -193.6129608, 372.6583557, -166.6977081, 312.9549561, -506.5679321, 539.3560791
3: -112.2892151, 462.6279907, -96.0392990, 393.7713013, -506.0605164, 558.6672974
4: -177.7465515, 326.1584473, -152.6450806, 274.2651672, -452.0116272, 478.8035278

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1068576, upper bound: 175.1081714
time: 0.72 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102264
time: 0.81 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -49.5620232, 189.6665192, -42.5559044, 163.3253479, -212.8873291, 232.2224274
1: -133.3395844, 435.3222046, -115.3303528, 373.3948364, -506.7344055, 550.6524048
2: -194.1568756, 373.1517639, -168.7579803, 317.4428406, -511.5997314, 541.9097290
3: -112.4922867, 463.3413391, -97.2007523, 399.0090942, -511.5013733, 560.5421143
4: -178.1272430, 326.6115112, -154.5298309, 278.2167664, -456.3439941, 481.1412964

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1069462, upper bound: 175.1081714
time: 0.77 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1069462, upper bound: 175.1102520
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -48.9917030, 187.5122375, -48.0940170, 184.5474548, -233.5391541, 235.6062622
1: -131.7478027, 430.5056763, -129.5681915, 423.3705444, -555.1183472, 560.0738525
2: -191.3944092, 369.1946411, -189.6818848, 362.3751221, -553.7695312, 558.8765259
3: -111.1235428, 458.0423279, -109.3105011, 450.9009399, -562.0243530, 567.3527832
4: -175.8094025, 323.0913086, -174.0693054, 317.2913818, -493.1007385, 497.1606140

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1074764, upper bound: 175.1100979
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1078739, upper bound: 175.1103085
time: 0.97 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -54.0062332, 207.0627747, -48.0940170, 184.5474548, -238.5536804, 255.1567841
1: -144.4248505, 476.6517334, -129.5681915, 423.3705444, -567.7954102, 606.2199097
2: -209.3488007, 411.0442810, -189.6818848, 362.3751221, -571.7238159, 600.7260132
3: -121.8874130, 505.1653137, -109.3105011, 450.9009399, -572.7883301, 614.4757080
4: -192.7769928, 359.4469910, -174.0693054, 317.2913818, -510.0683594, 533.5162964

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1074764, upper bound: 175.1101595
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1078739, upper bound: 175.1102877
time: 1.09 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -37.1877785, 142.5817566, -50.0133514, 191.3534241, -228.5411987, 192.5951080
1: -100.9189758, 324.7163086, -134.5356140, 439.1558533, -540.0748291, 459.2518311
2: -148.0071106, 277.0126953, -195.7950592, 376.4314880, -524.4385986, 472.8077393
3: -85.1052017, 347.0308228, -113.4945755, 467.3587036, -552.4638062, 460.5253906
4: -135.4534760, 242.7190399, -179.7091980, 329.4500427, -464.9035034, 422.4282227

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088862, upper bound: 175.1058411
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1101041, upper bound: 175.1073906
time: 1.36 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -38.1325684, 146.2725220, -50.1261063, 191.7990570, -229.9316101, 196.3986206
1: -103.4257050, 333.3331299, -134.8771210, 440.0725098, -543.4981689, 468.2102661
2: -151.6874390, 284.7179871, -196.4867859, 377.1958923, -528.8833008, 481.2047119
3: -87.2283630, 356.2762756, -113.7814026, 468.4150391, -555.6432495, 470.0576782
4: -138.8651886, 249.4823303, -180.2232056, 330.1417847, -469.0069580, 429.7054749

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1083043, upper bound: 175.1069933
time: 0.92 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1083043, upper bound: 175.1079208
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -43.7814865, 168.0317688, -49.5331764, 189.5534515, -233.3349304, 217.5649261
1: -117.8903275, 384.4804993, -133.2194672, 435.0702820, -552.9605713, 517.6998901
2: -172.8881378, 330.6353455, -193.5777283, 373.0771484, -545.9652710, 524.2130127
3: -99.5192871, 409.2031860, -112.3526382, 462.9354553, -562.4547119, 521.5558472
4: -158.6732178, 289.4013672, -177.7906342, 326.4732971, -485.1464539, 467.1920166

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1098995, upper bound: 175.1072136
time: 0.97 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1104177, upper bound: 175.1085736
time: 0.97 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -43.7814865, 168.0317688, -54.6163330, 209.3759460, -253.1574249, 222.6481018
1: -117.8903275, 384.4804993, -146.1183472, 481.7488403, -599.6391602, 530.5986938
2: -172.8881378, 330.6353455, -211.9063110, 415.3855286, -588.2736816, 542.5416260
3: -99.5192871, 409.2031860, -123.3006287, 510.7327271, -610.2520142, 532.5037231
4: -158.6732178, 289.4013672, -195.0705261, 363.2395630, -521.9127808, 484.4718933

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1098995, upper bound: 175.1073377
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1104177, upper bound: 175.1086064
time: 1.09 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -42.0362015, 161.2504883, -49.4877510, 189.3600616, -231.3962708, 210.7382050
1: -113.9261780, 368.5407715, -133.0959473, 434.7307129, -548.6568604, 501.6366577
2: -166.6977081, 312.9549561, -193.6129608, 372.6583557, -539.3560791, 506.5679321
3: -96.0392990, 393.7713013, -112.2892151, 462.6279907, -558.6672974, 506.0605164
4: -152.6450806, 274.2651672, -177.7465515, 326.1584473, -478.8035278, 452.0116577

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1068769
time: 0.89 seconds

## Relational analysis of NS_B2_A1_A2_B1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A2_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1077689
time: 1.34 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -42.5559044, 163.3253479, -49.5620232, 189.6665192, -232.2224274, 212.8873291
1: -115.3303528, 373.3948364, -133.3395844, 435.3222046, -550.6523438, 506.7344055
2: -168.7579803, 317.4428406, -194.1568756, 373.1517639, -541.9097290, 511.5997314
3: -97.2007523, 399.0090942, -112.4922867, 463.3413391, -560.5421143, 511.5013733
4: -154.5298309, 278.2167664, -178.1272430, 326.6115112, -481.1412964, 456.3439941

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1069462
time: 0.79 seconds

## Relational analysis of NS_B2_A1_A2_B1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1078776
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.0940170, 184.5474548, -48.9917030, 187.5122375, -235.6062622, 233.5391541
1: -129.5681915, 423.3705444, -131.7478027, 430.5056763, -560.0737915, 555.1183472
2: -189.6818848, 362.3751221, -191.3944092, 369.1946411, -558.8765259, 553.7695312
3: -109.3105011, 450.9009399, -111.1235428, 458.0423279, -567.3528442, 562.0243530
4: -174.0693054, 317.2913818, -175.8094025, 323.0913086, -497.1606140, 493.1007385

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1098010, upper bound: 175.1078815
time: 0.88 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1098995, upper bound: 175.1074764
time: 0.86 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1103085, upper bound: 175.1082428
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.0940170, 184.5474548, -54.0062332, 207.0627747, -255.1567841, 238.5536804
1: -129.5681915, 423.3705444, -144.4248505, 476.6517334, -606.2199097, 567.7954102
2: -189.6818848, 362.3751221, -209.3488007, 411.0442810, -600.7260132, 571.7238159
3: -109.3105011, 450.9009399, -121.8874130, 505.1653137, -614.4757080, 572.7883301
4: -174.0693054, 317.2913818, -192.7769928, 359.4469910, -533.5162964, 510.0683594

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1100980, upper bound: 175.1074777
time: 1.14 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1103084, upper bound: 175.1082482
time: 0.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.05 seconds
NS_B1_A2_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1058412, upper bound: 175.1088862
NS_B1_A2_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1073906, upper bound: 175.1101041
NS_B1_A2_B1_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1069933, upper bound: 175.1083043
NS_B1_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1069933, upper bound: 175.1102329
NS_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1072130, upper bound: 175.1098995
NS_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1079179, upper bound: 175.1104177
NS_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1072130, upper bound: 175.1100508
NS_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1079179, upper bound: 175.1102847
NS_B1_A2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1068576, upper bound: 175.1081714
NS_B1_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102264
NS_B1_A2_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1069462, upper bound: 175.1081714
NS_B1_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1069462, upper bound: 175.1102520
NS_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1074764, upper bound: 175.1100979
NS_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1078739, upper bound: 175.1103085
NS_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1074764, upper bound: 175.1101595
NS_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1078739, upper bound: 175.1102877
NS_B2_A1_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1088862, upper bound: 175.1058411
NS_B2_A1_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1101041, upper bound: 175.1073906
NS_B2_A1_A1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1083043, upper bound: 175.1069933
NS_B2_A1_A1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1083043, upper bound: 175.1079208
NS_B2_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1098995, upper bound: 175.1072136
NS_B2_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1104177, upper bound: 175.1085736
NS_B2_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1098995, upper bound: 175.1073377
NS_B2_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1104177, upper bound: 175.1086064
NS_B2_A1_A2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1068769
NS_B2_A1_A2_B1_A1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1077689
NS_B2_A1_A2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1069462
NS_B2_A1_A2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1081714, upper bound: 175.1078776
NS_B2_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1098995, upper bound: 175.1074764
NS_B2_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1103085, upper bound: 175.1082428
NS_B2_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1100980, upper bound: 175.1074777
NS_B2_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.05
Output dim: 0, lower bound: -175.1103084, upper bound: 175.1082482

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -49.7269707, 190.2378387, -36.1159592, 138.4226074, -188.1495667, 226.3537903
1: -133.7391968, 436.6047974, -97.9803314, 315.0917969, -448.8309937, 534.5851440
2: -194.6496887, 374.3348999, -143.8920135, 269.0180054, -463.6676941, 518.2269287
3: -112.8294601, 464.5947876, -82.6443558, 336.7953796, -449.6248474, 547.2391357
4: -178.6676331, 327.5993652, -131.6683655, 235.6667786, -414.3343811, 459.2677307

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1052286, upper bound: 175.1083229
time: 0.88 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1072905
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1088862
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -49.5115051, 189.4355621, -40.7340813, 156.2126312, -205.7241211, 230.1696472
1: -133.1523590, 434.7861328, -110.1893539, 356.3854980, -489.5377808, 544.9754639
2: -193.7329102, 372.7848511, -160.8235474, 305.0747070, -498.8076172, 533.6083984
3: -112.3275452, 462.6322327, -92.8257828, 380.4505310, -492.7780762, 555.4580078
4: -177.8342743, 326.2420654, -147.4972229, 267.2577515, -445.0919800, 473.7392883

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1052286, upper bound: 175.1096866
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1087777
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1101041
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -49.5573425, 189.6374054, -38.1325684, 146.2725220, -195.8298340, 227.7699738
1: -133.3006744, 435.1870728, -103.4257050, 333.3331299, -466.6337891, 538.6127930
2: -194.0816803, 373.1218872, -151.6874390, 284.7179871, -478.7996826, 524.8093262
3: -112.4463806, 463.1473694, -87.2283630, 356.2762756, -468.7226562, 550.3756104
4: -178.0104675, 326.5596008, -138.8651886, 249.4823303, -427.4927368, 465.4247742

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1077738
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1102187
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -49.0122299, 187.5791168, -41.7009163, 160.0432129, -209.0554352, 229.2800293
1: -131.7638550, 430.6477356, -112.2089539, 366.3145752, -498.0784302, 542.8566895
2: -191.3056641, 369.3864746, -164.0546265, 315.0545044, -506.3601074, 533.4411011
3: -111.1257095, 458.0870667, -94.7165222, 389.7403870, -500.8660889, 552.8035889
4: -175.7514954, 323.2257690, -150.6836090, 275.6957397, -451.4472046, 473.9093628

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1064301, upper bound: 175.1075424
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1098995
time: 0.75 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -49.0870514, 187.8440094, -42.7521591, 164.1196442, -213.2066956, 230.5961609
1: -131.9641266, 431.2652283, -115.0152435, 375.6656799, -507.6298218, 546.2804565
2: -191.5814819, 369.9004822, -168.6048431, 323.1923523, -514.7738037, 538.5053101
3: -111.2900848, 458.7853088, -97.1039658, 399.7120667, -511.0021362, 555.8892822
4: -176.0059509, 323.6730957, -154.7863007, 282.8616028, -458.8675537, 478.4593506

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1069727, upper bound: 175.1076165
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1074965, upper bound: 175.1104177
time: 0.82 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -54.0785713, 207.3355255, -41.7009163, 160.0432129, -214.1217804, 249.0364380
1: -144.6102753, 477.2092590, -112.2089539, 366.3145752, -510.9247437, 589.4182129
2: -209.4584503, 411.6256714, -164.0546265, 315.0545044, -524.5129395, 575.6802979
3: -122.0200729, 505.6874390, -94.7165222, 389.7403870, -511.7604370, 600.4039307
4: -192.8936768, 359.9203186, -150.6836090, 275.6957397, -468.5894165, 510.6038818

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047919, upper bound: 175.1085709
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1072594, upper bound: 175.1100257
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -54.1895218, 207.7527924, -42.7521591, 164.1196442, -218.3091736, 250.5049438
1: -144.9303131, 478.1148071, -115.0152435, 375.6656799, -520.5959473, 593.1300659
2: -210.0805664, 412.3309937, -168.6048431, 323.1923523, -533.2729492, 580.9358521
3: -122.2969818, 506.7732239, -97.1039658, 399.7120667, -522.0089111, 603.8771973
4: -193.4308777, 360.5487671, -154.7863007, 282.8616028, -476.2924805, 515.3350830

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1069933, upper bound: 175.1083043
time: 1.11 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1071710, upper bound: 175.1102847
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -48.9763718, 187.4447479, -42.0362015, 161.2504883, -210.2268677, 229.4809570
1: -131.7162018, 430.2969666, -113.9261780, 368.5407715, -500.2569580, 544.2231445
2: -191.6829529, 368.9616089, -166.6977081, 312.9549561, -504.6378784, 535.6593018
3: -111.1180115, 457.9277954, -96.0392990, 393.7713013, -504.8893127, 553.9670410
4: -175.8527985, 322.9277039, -152.6450806, 274.2651672, -450.1178589, 475.5727844

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102264
time: 0.96 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102264
time: 1.05 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -48.9763718, 187.4447479, -42.5559044, 163.3253479, -212.3016815, 230.0006561
1: -131.7162018, 430.2969666, -115.3303528, 373.3948364, -505.1110229, 545.6271973
2: -191.6829529, 368.9616089, -168.7579803, 317.4428406, -509.1257935, 537.7196045
3: -111.1180115, 457.9277954, -97.2007523, 399.0090942, -510.1271057, 555.1285400
4: -175.8527985, 322.9277039, -154.5298309, 278.2167664, -454.0695496, 477.4575195

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102520
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068576, upper bound: 175.1102520
time: 0.75 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -48.4950600, 185.6190033, -46.4891052, 178.3593750, -226.8544312, 232.1080780
1: -130.3518066, 426.2726440, -125.1002045, 409.4687195, -539.8205566, 551.3728638
2: -189.2047272, 365.6657410, -182.6056671, 350.2776489, -539.4822998, 548.2714233
3: -109.9480133, 453.3977661, -105.5380554, 435.8653259, -545.8133545, 558.9357910
4: -173.8462372, 319.9848938, -167.7167664, 306.6437378, -480.4899292, 487.7015686

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1076032
time: 1.21 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1100980
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -48.5343857, 185.7612000, -47.1172562, 180.8151855, -229.3495789, 232.8784485
1: -130.4618378, 426.6069336, -126.8641815, 414.9409180, -545.4027710, 553.4711304
2: -189.3522339, 365.9397583, -185.6753998, 355.2592468, -544.6114502, 551.6150513
3: -110.0352631, 453.7955017, -107.0362778, 441.8708496, -551.9061279, 560.8317261
4: -173.9837341, 320.2224731, -170.4226685, 311.0518188, -485.0355225, 490.6451416

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1072568, upper bound: 175.1076735
time: 0.74 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1069257, upper bound: 175.1103084
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.4837685, 205.0808716, -46.4891052, 178.3593750, -231.8431244, 251.5699768
1: -142.9631348, 472.2385254, -125.1002045, 409.4687195, -552.4317627, 597.3387451
2: -206.9743958, 407.3968811, -182.6056671, 350.2776489, -557.2520142, 590.0025635
3: -120.6450882, 500.2585144, -105.5380554, 435.8653259, -556.5104370, 605.7965698
4: -190.6638794, 356.2254944, -167.7167664, 306.6437378, -497.3075562, 523.9421997

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1066876, upper bound: 175.1084002
time: 0.89 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1101595
time: 1.00 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.5630455, 205.3767395, -47.1172562, 180.8151855, -234.3782349, 252.4939880
1: -143.1877441, 472.8808289, -126.8641815, 414.9409180, -558.1286621, 599.7449951
2: -207.4463348, 407.8713379, -185.6753998, 355.2592468, -562.7055664, 593.5466919
3: -120.8432236, 501.0561523, -107.0362778, 441.8708496, -562.7140503, 608.0924072
4: -191.0698547, 356.6525879, -170.4226685, 311.0518188, -502.1216431, 527.0752563

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1069462, upper bound: 175.1084002
time: 0.92 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1070939, upper bound: 175.1102877
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -36.1159592, 138.4226074, -49.7269707, 190.2378387, -226.3537903, 188.1495667
1: -97.9803314, 315.0917969, -133.7391968, 436.6047974, -534.5851440, 448.8309937
2: -143.8920135, 269.0180054, -194.6496887, 374.3348999, -518.2269287, 463.6676941
3: -82.6443558, 336.7953796, -112.8294601, 464.5947876, -547.2391357, 449.6248474
4: -131.6683655, 235.6667786, -178.6676331, 327.5993652, -459.2677307, 414.3343811

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1083229, upper bound: 175.1052286
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1047533
time: 0.79 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1058411
time: 0.91 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -40.7340813, 156.2126312, -49.5115051, 189.4355621, -230.1696472, 205.7241211
1: -110.1893539, 356.3854980, -133.1523590, 434.7861328, -544.9754639, 489.5377808
2: -160.8235474, 305.0747070, -193.7329102, 372.7848511, -533.6083984, 498.8076172
3: -92.8257828, 380.4505310, -112.3275452, 462.6322327, -555.4580078, 492.7780762
4: -147.4972229, 267.2577515, -177.8342743, 326.2420654, -473.7392883, 445.0919800

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1083229, upper bound: 175.1070939
time: 1.06 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1065159
time: 1.13 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1087777, upper bound: 175.1073906
time: 1.33 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -41.7009163, 160.0432129, -49.0122299, 187.5791168, -229.2800293, 209.0554352
1: -112.2089539, 366.3145752, -131.7638550, 430.6477356, -542.8566895, 498.0783997
2: -164.0546265, 315.0545044, -191.3056641, 369.3864746, -533.4411011, 506.3601685
3: -94.7165222, 389.7403870, -111.1257095, 458.0870667, -552.8035889, 500.8660889
4: -150.6836090, 275.6957397, -175.7514954, 323.2257690, -473.9093628, 451.4472046

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1064301
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1072136
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -42.7521591, 164.1196442, -49.0870514, 187.8440094, -230.5961609, 213.2066956
1: -115.0152435, 375.6656799, -131.9641266, 431.2652283, -546.2803955, 507.6298218
2: -168.6048431, 323.1923523, -191.5814819, 369.9004822, -538.5053101, 514.7738037
3: -97.1039658, 399.7120667, -111.2900848, 458.7853088, -555.8892822, 511.0021362
4: -154.7863007, 282.8616028, -176.0059509, 323.6730957, -478.4593506, 458.8675537

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1074965
time: 0.79 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1085736
time: 1.12 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -41.7009163, 160.0432129, -54.0785713, 207.3355255, -249.0364380, 214.1217804
1: -112.2089539, 366.3145752, -144.6102753, 477.2092590, -589.4182129, 510.9248047
2: -164.0546265, 315.0545044, -209.4584503, 411.6256714, -575.6802979, 524.5129395
3: -94.7165222, 389.7403870, -122.0200729, 505.6874390, -600.4039307, 511.7604370
4: -150.6836090, 275.6957397, -192.8936768, 359.9203186, -510.6039124, 468.5894165

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077967, upper bound: 175.1047416
time: 0.94 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1100134, upper bound: 175.1072613
time: 1.27 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.7521591, 164.1196442, -54.1895218, 207.7527924, -250.5049438, 218.3091736
1: -115.0152435, 375.6656799, -144.9303131, 478.1148071, -593.1300659, 520.5960083
2: -168.6048431, 323.1923523, -210.0805664, 412.3309937, -580.9358521, 533.2729492
3: -97.1039658, 399.7120667, -122.2969818, 506.7732239, -603.8771973, 522.0089722
4: -154.7863007, 282.8616028, -193.4308777, 360.5487671, -515.3350220, 476.2924805

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1092398, upper bound: 175.1075189
time: 0.95 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076165, upper bound: 175.1086064
time: 0.94 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -46.4891052, 178.3593750, -48.4950600, 185.6190033, -232.1080780, 226.8544312
1: -125.1002045, 409.4687195, -130.3518066, 426.2726440, -551.3728638, 539.8205566
2: -182.6056671, 350.2776489, -189.2047272, 365.6657410, -548.2714233, 539.4822998
3: -105.5380554, 435.8653259, -109.9480133, 453.3977661, -558.9357910, 545.8133545
4: -167.7167664, 306.6437378, -173.8462372, 319.9848938, -487.7015686, 480.4899597

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1065630
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1074772
time: 0.82 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -47.1172562, 180.8151855, -48.5343857, 185.7612000, -232.8784485, 229.3495789
1: -126.8641815, 414.9409180, -130.4618378, 426.6069336, -553.4711304, 545.4027710
2: -185.6753998, 355.2592468, -189.3522339, 365.9397583, -551.6150513, 544.6114502
3: -107.0362778, 441.8708496, -110.0352631, 453.7955017, -560.8317261, 551.9061279
4: -170.4226685, 311.0518188, -173.9837341, 320.2224731, -490.6451416, 485.0355530

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1076735, upper bound: 175.1072568
time: 0.75 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1076735, upper bound: 175.1082495
time: 0.73 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -46.4891052, 178.3593750, -53.4837685, 205.0808716, -251.5699768, 231.8431244
1: -125.1002045, 409.4687195, -142.9631348, 472.2385254, -597.3387451, 552.4317017
2: -182.6056671, 350.2776489, -206.9743958, 407.3968811, -590.0025635, 557.2520752
3: -105.5380554, 435.8653259, -120.6450882, 500.2585144, -605.7965088, 556.5104370
4: -167.7167664, 306.6437378, -190.6638794, 356.2254944, -523.9421997, 497.3075256

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1067980
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1076032, upper bound: 175.1074777
time: 1.24 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -47.1172562, 180.8151855, -53.5630455, 205.3767395, -252.4939880, 234.3782349
1: -126.8641815, 414.9409180, -143.1877441, 472.8808289, -599.7449951, 558.1286621
2: -185.6753998, 355.2592468, -207.4463348, 407.8713379, -593.5466919, 562.7055664
3: -107.0362778, 441.8708496, -120.8432236, 501.0561523, -608.0924072, 562.7140503
4: -170.4226685, 311.0518188, -191.0698547, 356.6525879, -527.0752563, 502.1216431

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1072775
time: 0.84 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1082482
time: 1.02 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.88 seconds
NS_B1_A2_B1_A1_B1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1072905
NS_B1_A2_B1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1088862
NS_B1_A2_B1_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1087777
NS_B1_A2_B1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1101041
NS_B1_A2_B1_A1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1077738
NS_B1_A2_B1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1102187
NS_B1_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1064301, upper bound: 175.1075424
NS_B1_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1098995
NS_B1_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1069727, upper bound: 175.1076165
NS_B1_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1074965, upper bound: 175.1104177
NS_B1_A2_B1_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1047919, upper bound: 175.1085709
NS_B1_A2_B1_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1072594, upper bound: 175.1100257
NS_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1069933, upper bound: 175.1083043
NS_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1071710, upper bound: 175.1102847
NS_B1_A2_B2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102264
NS_B1_A2_B2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102264
NS_B1_A2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1068769, upper bound: 175.1102520
NS_B1_A2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1068576, upper bound: 175.1102520
NS_B1_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1076032
NS_B1_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1100980
NS_B1_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1072568, upper bound: 175.1076735
NS_B1_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1069257, upper bound: 175.1103084
NS_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1066876, upper bound: 175.1084002
NS_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1101595
NS_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1069462, upper bound: 175.1084002
NS_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1070939, upper bound: 175.1102877
NS_B2_A1_A1_B1_A1_A1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1047533
NS_B2_A1_A1_B1_A1_A1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1058411
NS_B2_A1_A1_B1_A1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1065159
NS_B2_A1_A1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1087777, upper bound: 175.1073906
NS_B2_A1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1064301
NS_B2_A1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1072136
NS_B2_A1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1074965
NS_B2_A1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1085736
NS_B2_A1_A1_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1077967, upper bound: 175.1047416
NS_B2_A1_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1100134, upper bound: 175.1072613
NS_B2_A1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1092398, upper bound: 175.1075189
NS_B2_A1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1076165, upper bound: 175.1086064
NS_B2_A1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1065630
NS_B2_A1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1075424, upper bound: 175.1074772
NS_B2_A1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1076735, upper bound: 175.1072568
NS_B2_A1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1076735, upper bound: 175.1082495
NS_B2_A1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1067980
NS_B2_A1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1076032, upper bound: 175.1074777
NS_B2_A1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1072775
NS_B2_A1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.88
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1082482

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -49.2713585, 188.5193024, -36.1159592, 138.4226074, -187.6939697, 224.6352539
1: -132.5078888, 432.6349792, -97.9803314, 315.0917969, -447.5996704, 530.6152954
2: -192.9384308, 371.0262146, -143.8920135, 269.0180054, -461.9564209, 514.9182129
3: -111.7829208, 460.3720093, -82.6443558, 336.7953796, -448.5783081, 543.0163574
4: -176.9704590, 324.7064209, -131.6683655, 235.6667786, -412.6372070, 456.3747864

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1088862
time: 1.07 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1088862
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -48.3467102, 184.9049072, -40.7340813, 156.2126312, -204.5593109, 225.6389923
1: -129.9639282, 424.4339294, -110.1893539, 356.3854980, -486.3494263, 534.6232910
2: -188.9842682, 363.6730652, -160.8235474, 305.0747070, -494.0589600, 524.4965820
3: -109.6410294, 451.5211487, -92.8257828, 380.4505310, -490.0915527, 544.3469238
4: -173.5286255, 318.3197937, -147.4972229, 267.2577515, -440.7863770, 465.8170166

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1087777
time: 0.76 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1087777
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -49.0569458, 187.7222595, -40.7340813, 156.2126312, -205.2695618, 228.4563446
1: -131.9272461, 430.8273010, -110.1893539, 356.3854980, -488.3127441, 541.0166626
2: -192.0337067, 369.4820251, -160.8235474, 305.0747070, -497.1083679, 530.3055420
3: -111.2858200, 458.4247742, -92.8257828, 380.4505310, -491.7363281, 551.2505493
4: -176.1468658, 323.3547668, -147.4972229, 267.2577515, -443.4046021, 470.8519897

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1101041
time: 0.93 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1101041
time: 1.17 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -49.0569458, 187.7222595, -41.4938354, 159.1243439, -208.1812744, 229.2160950
1: -131.9272461, 430.8273010, -112.1910400, 363.2141418, -495.1413879, 543.0183105
2: -192.0337067, 369.4820251, -163.8167725, 311.0861206, -503.1198120, 533.2988281
3: -111.2858200, 458.4247742, -94.5246201, 387.7407532, -499.0265198, 552.9494019
4: -176.1468658, 323.3547668, -150.2441101, 272.5192566, -448.6661377, 473.5988770

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1075630, upper bound: 175.1100212
time: 0.81 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076624, upper bound: 175.1102187
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076624, upper bound: 175.1102187
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -48.4964447, 185.5859375, -41.7009163, 160.0432129, -208.5396423, 227.2868500
1: -130.3096466, 426.2197571, -112.2089539, 366.3145752, -496.6242065, 538.4287109
2: -188.9532166, 365.6941528, -164.0546265, 315.0545044, -504.0077209, 529.7487793
3: -109.8894348, 453.2915039, -94.7165222, 389.7403870, -499.6297913, 548.0078735
4: -173.6568451, 319.9700623, -150.6836090, 275.6957397, -449.3526001, 470.6536865

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1098995
time: 1.57 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1098995
time: 0.86 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -48.4964447, 185.5859375, -42.7521591, 164.1196442, -212.6160736, 228.3380890
1: -130.3096466, 426.2197571, -115.0152435, 375.6656799, -505.9753418, 541.2349854
2: -188.9532166, 365.6941528, -168.6048431, 323.1923523, -512.1455078, 534.2990112
3: -109.8894348, 453.2915039, -97.1039658, 399.7120667, -509.6014709, 550.3954468
4: -173.6568451, 319.9700623, -154.7863007, 282.8616028, -456.5184326, 474.7563477

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1064301, upper bound: 175.1104178
time: 1.03 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1104178
time: 0.98 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -53.7833138, 206.1924133, -40.5854340, 155.7677002, -209.5510101, 246.7778473
1: -143.7907867, 474.6146545, -109.1076279, 356.4072266, -500.1979980, 583.7222900
2: -208.2653046, 409.4764099, -159.6264496, 306.8702393, -515.1355591, 569.1028442
3: -121.3320084, 502.8240051, -92.1414871, 379.1696167, -500.5016174, 594.9655151
4: -191.8020477, 358.0222168, -146.6385345, 268.4741211, -460.2761841, 504.6607666

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1037067, upper bound: 175.1065723
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1074490
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1085709
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -53.6221466, 205.5824738, -45.4009857, 174.3787231, -228.0008698, 250.9834595
1: -143.3622284, 473.2405701, -121.8792267, 399.5579224, -542.9201660, 595.1197510
2: -207.5952911, 408.2840576, -177.0930328, 344.5025024, -552.0976562, 585.3770752
3: -120.9628372, 501.3704224, -102.7702942, 424.6116028, -545.5744629, 604.1407471
4: -191.1930695, 356.9821777, -162.9669495, 301.4320679, -492.6250916, 519.9490967

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1037067, upper bound: 175.1096561
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1062693, upper bound: 175.1090020
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1100257
time: 0.90 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -53.6333466, 205.6430817, -42.7521591, 164.1196442, -217.7529907, 248.3952332
1: -143.3855286, 473.3684692, -115.0152435, 375.6656799, -519.0511475, 588.3836670
2: -207.7257233, 408.3446350, -168.6048431, 323.1923523, -530.9180298, 576.9494629
3: -120.9927979, 501.6055298, -97.1039658, 399.7120667, -520.7047729, 598.7094727
4: -191.3149872, 357.0385437, -154.7863007, 282.8616028, -474.1765747, 511.8248291

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1067637
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1102847
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.9763718, 187.4447479, -41.9563560, 160.9398499, -209.9161987, 229.4010925
1: -131.7162018, 430.2969666, -113.7067642, 367.8437805, -499.5599365, 544.0037231
2: -191.6829529, 368.9616089, -166.3439484, 312.3717346, -504.0546875, 535.3055420
3: -111.1180115, 457.9277954, -95.8541183, 393.0187378, -504.1367493, 553.7817993
4: -175.8527985, 322.9277039, -152.3316345, 273.7511292, -449.6038818, 475.2593384

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076981, upper bound: 175.1102264
time: 1.09 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076981, upper bound: 175.1101564
time: 1.58 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.9763718, 187.4447479, -42.5141945, 163.4579468, -212.4342957, 229.9589386
1: -131.7162018, 430.2969666, -115.3497620, 374.4250183, -506.1412354, 545.6467285
2: -191.6829529, 368.9616089, -168.1880951, 317.1526184, -508.8355103, 537.1497192
3: -111.1180115, 457.9277954, -97.1457825, 399.3579102, -510.4759216, 555.0736084
4: -175.8527985, 322.9277039, -154.1748505, 277.7739258, -453.6267090, 477.1025391

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1071965, upper bound: 175.1099123
time: 0.89 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1071907, upper bound: 175.1098997
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.9763718, 187.4447479, -42.4795380, 163.0301208, -212.0064697, 229.9242859
1: -131.7162018, 430.2969666, -115.1161880, 372.7310791, -504.4472656, 545.4131470
2: -191.6829529, 368.9616089, -168.4167175, 316.8856506, -508.5685425, 537.3782959
3: -111.1180115, 457.9277954, -97.0210266, 398.2921753, -509.4101868, 554.9487915
4: -175.8527985, 322.9277039, -154.2266693, 277.7265015, -453.5792847, 477.1543579

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1060968, upper bound: 175.1102520
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1078776, upper bound: 175.1102331
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -48.9763718, 187.4447479, -42.8379288, 164.7342834, -213.7106628, 230.2826843
1: -131.7162018, 430.2969666, -116.2269821, 377.5905151, -509.3067017, 546.5238647
2: -191.6829529, 368.9616089, -169.5067902, 320.3963623, -512.0792847, 538.4683838
3: -111.1180115, 457.9277954, -97.8678894, 402.6925964, -513.8106079, 555.7955933
4: -175.8527985, 322.9277039, -155.3929901, 280.6516113, -456.5043640, 478.3206787

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1061885, upper bound: 175.1099055
time: 1.01 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1071907, upper bound: 175.1099200
time: 1.24 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -47.9257584, 183.4290771, -46.4891052, 178.3593750, -226.2851257, 229.9181824
1: -128.7508545, 421.4145813, -125.1002045, 409.4687195, -538.2196045, 546.5147705
2: -186.6306915, 361.6015320, -182.6056671, 350.2776489, -536.9083252, 544.2072144
3: -108.5875320, 448.1360168, -105.5380554, 435.8653259, -544.4528809, 553.6739502
4: -171.5543671, 316.4028320, -167.7167664, 306.6437378, -478.1981201, 484.1195984

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1064301, upper bound: 175.1100979
time: 0.99 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1100979
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -47.9257584, 183.4290771, -47.1172562, 180.8151855, -228.7409363, 230.5463257
1: -128.7508545, 421.4145813, -126.8641815, 414.9409180, -543.6917725, 548.2787476
2: -186.6306915, 361.6015320, -185.6753998, 355.2592468, -541.8899536, 547.2768555
3: -108.5875320, 448.1360168, -107.0362778, 441.8708496, -550.4583130, 555.1722412
4: -171.5543671, 316.4028320, -170.4226685, 311.0518188, -482.6062012, 486.8255005

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1103084
time: 0.90 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1103085
time: 1.28 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -52.9971504, 203.2317047, -46.4891052, 178.3593750, -231.3565216, 249.7207947
1: -141.6149750, 468.0567932, -125.1002045, 409.4687195, -551.0836792, 593.1569214
2: -205.0487823, 403.8166809, -182.6056671, 350.2776489, -555.3263550, 586.4223633
3: -119.5156403, 495.8032227, -105.5380554, 435.8653259, -555.3809814, 601.3412476
4: -188.9145203, 353.0823364, -167.7167664, 306.6437378, -495.5582581, 520.7990112

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1067456, upper bound: 175.1101595
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1101593
time: 1.02 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -52.9971504, 203.2317047, -47.1172562, 180.8151855, -233.8123322, 250.3489380
1: -141.6149750, 468.0567932, -126.8641815, 414.9409180, -556.5559082, 594.9208984
2: -205.0487823, 403.8166809, -185.6753998, 355.2592468, -560.3080444, 589.4920044
3: -119.5156403, 495.8032227, -107.0362778, 441.8708496, -561.3864746, 602.8394775
4: -188.9145203, 353.0823364, -170.4226685, 311.0518188, -499.9663391, 523.5050049

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1067456, upper bound: 175.1102877
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1067456, upper bound: 175.1102877
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -40.7340813, 156.2126312, -49.0569458, 187.7222595, -228.4563446, 205.2695618
1: -110.1893539, 356.3854980, -131.9272461, 430.8273010, -541.0166626, 488.3127441
2: -160.8235474, 305.0747070, -192.0337067, 369.4820251, -530.3055420, 497.1083679
3: -92.8257828, 380.4505310, -111.2858200, 458.4247742, -551.2505493, 491.7363586
4: -147.4972229, 267.2577515, -176.1468658, 323.3547668, -470.8519897, 443.4046021

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1073906
time: 0.88 seconds

## Relational analysis of NS_B2_A1_A1_B1_A1_A1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A1_B1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1087777, upper bound: 175.1073906
time: 1.23 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -42.7521591, 164.1196442, -48.4964447, 185.5859375, -228.3380890, 212.6160736
1: -115.0152435, 375.6656799, -130.3096466, 426.2197571, -541.2349854, 505.9753418
2: -168.6048431, 323.1923523, -188.9532166, 365.6941528, -534.2990112, 512.1455688
3: -97.1039658, 399.7120667, -109.8894348, 453.2915039, -550.3954468, 509.6014709
4: -154.7863007, 282.8616028, -173.6568451, 319.9700623, -474.7563477, 456.5184326

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1085723
time: 0.95 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1085723
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -45.4009857, 174.3787231, -53.6221466, 205.5824738, -250.9834595, 228.0008698
1: -121.8792267, 399.5579224, -143.3622284, 473.2405701, -595.1197510, 542.9201660
2: -177.0930328, 344.5025024, -207.5952911, 408.2840576, -585.3770752, 552.0976562
3: -102.7702942, 424.6116028, -120.9628372, 501.3704224, -604.1407471, 545.5744629
4: -162.9669495, 301.4320679, -191.1930695, 356.9821777, -519.9490967, 492.6250916

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1092951, upper bound: 175.1068082
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1100134, upper bound: 175.1072613
time: 1.06 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077967, upper bound: 175.1047416
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -42.7521591, 164.1196442, -53.0550041, 203.3331909, -246.0853577, 217.1746521
1: -115.0152435, 375.6656799, -141.7972565, 468.0505066, -583.0656738, 517.4628906
2: -168.6048431, 323.1923523, -205.2408600, 403.5415955, -572.1464233, 528.4331665
3: -97.1039658, 399.7120667, -119.6506424, 495.9442749, -593.0482178, 519.3626099
4: -154.7863007, 282.8616028, -189.0360718, 352.8395386, -507.6258545, 471.8976746

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1076165, upper bound: 175.1075189
time: 0.83 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1076165, upper bound: 175.1075189
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -42.7521591, 164.1196442, -53.6333466, 205.6430817, -248.3952332, 217.7529907
1: -115.0152435, 375.6656799, -143.3855286, 473.3684692, -588.3837280, 519.0511475
2: -168.6048431, 323.1923523, -207.7257233, 408.3446350, -576.9494629, 530.9180298
3: -97.1039658, 399.7120667, -120.9927979, 501.6055298, -598.7094727, 520.7047729
4: -154.7863007, 282.8616028, -191.3149872, 357.0385437, -511.8248291, 474.1765747

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1092398, upper bound: 175.1086058
time: 0.85 seconds

## Relational analysis of NS_B2_A1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1092398, upper bound: 175.1086058
time: 0.94 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -46.4891052, 178.3593750, -52.4008980, 200.8531952, -247.3423004, 230.7602692
1: -125.1002045, 409.4687195, -139.9884491, 462.5732727, -587.6734009, 549.4570923
2: -182.6056671, 350.2776489, -202.4906311, 398.9004822, -581.5061646, 552.7682495
3: -105.5380554, 435.8653259, -118.1378098, 490.0299072, -595.5679321, 554.0031128
4: -167.7167664, 306.6437378, -186.5703888, 348.7810974, -516.4978027, 493.2141113

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1091458, upper bound: 175.1067980
time: 0.97 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1091458, upper bound: 175.1067980
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -47.1172562, 180.8151855, -52.4008980, 200.8531952, -247.9704590, 233.2160797
1: -126.8641815, 414.9409180, -139.9884491, 462.5732727, -589.4373779, 554.9293823
2: -185.6753998, 355.2592468, -202.4906311, 398.9004822, -584.5758057, 557.7498779
3: -107.0362778, 441.8708496, -118.1378098, 490.0299072, -597.0661621, 560.0085449
4: -170.4226685, 311.0518188, -186.5703888, 348.7810974, -519.2037354, 497.6221924

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1072775
time: 0.82 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1072775
time: 0.80 seconds

## BFS NS instance: NS_B2_A1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -47.1172562, 180.8151855, -52.9971504, 203.2317047, -250.3489380, 233.8123322
1: -126.8641815, 414.9409180, -141.6149750, 468.0567932, -594.9208984, 556.5559082
2: -185.6753998, 355.2592468, -205.0487823, 403.8166809, -589.4920044, 560.3080444
3: -107.0362778, 441.8708496, -119.5156403, 495.8032227, -602.8394775, 561.3864746
4: -170.4226685, 311.0518188, -188.9145203, 353.0823364, -523.5049438, 499.9663391

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1077787
time: 0.96 seconds

## Relational analysis of NS_B2_A1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1077787
time: 0.96 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.32 seconds
NS_B1_A2_B1_A1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1088862
NS_B1_A2_B1_A1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1088862
NS_B1_A2_B1_A1_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1087777
NS_B1_A2_B1_A1_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1087777
NS_B1_A2_B1_A1_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1065159, upper bound: 175.1101041
NS_B1_A2_B1_A1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1047533, upper bound: 175.1101041
NS_B1_A2_B1_A1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076624, upper bound: 175.1102187
NS_B1_A2_B1_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076624, upper bound: 175.1102187
NS_B1_A2_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1098995
NS_B1_A2_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1098995
NS_B1_A2_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1064301, upper bound: 175.1104178
NS_B1_A2_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1104178
NS_B1_A2_B1_A1_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1074490
NS_B1_A2_B1_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1085709
NS_B1_A2_B1_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1062693, upper bound: 175.1090020
NS_B1_A2_B1_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1100257
NS_B1_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1067637
NS_B1_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1102847
NS_B1_A2_B2_A1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076981, upper bound: 175.1102264
NS_B1_A2_B2_A1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076981, upper bound: 175.1101564
NS_B1_A2_B2_A1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1071965, upper bound: 175.1099123
NS_B1_A2_B2_A1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1071907, upper bound: 175.1098997
NS_B1_A2_B2_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1060968, upper bound: 175.1102520
NS_B1_A2_B2_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1078776, upper bound: 175.1102331
NS_B1_A2_B2_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1061885, upper bound: 175.1099055
NS_B1_A2_B2_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1071907, upper bound: 175.1099200
NS_B1_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1064301, upper bound: 175.1100979
NS_B1_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1100979
NS_B1_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1103084
NS_B1_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1065630, upper bound: 175.1103085
NS_B1_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1067456, upper bound: 175.1101595
NS_B1_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1063638, upper bound: 175.1101593
NS_B1_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1067456, upper bound: 175.1102877
NS_B1_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1067456, upper bound: 175.1102877
NS_B2_A1_A1_B1_A1_A1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1072905, upper bound: 175.1073906
NS_B2_A1_A1_B1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1087777, upper bound: 175.1073906
NS_B2_A1_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1085723
NS_B2_A1_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076809, upper bound: 175.1085723
NS_B2_A1_A1_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1100134, upper bound: 175.1072613
NS_B2_A1_A1_B1_A2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1077967, upper bound: 175.1047416
NS_B2_A1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076165, upper bound: 175.1075189
NS_B2_A1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1076165, upper bound: 175.1075189
NS_B2_A1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1092398, upper bound: 175.1086058
NS_B2_A1_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1092398, upper bound: 175.1086058
NS_B2_A1_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1091458, upper bound: 175.1067980
NS_B2_A1_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1091458, upper bound: 175.1067980
NS_B2_A1_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1072775
NS_B2_A1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1072775
NS_B2_A1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1077787
NS_B2_A1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.32
Output dim: 0, lower bound: -175.1088869, upper bound: 175.1077787

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -49.2713585, 188.5193024, -36.0404816, 138.1300507, -187.4014130, 224.5597687
1: -132.5078888, 432.6349792, -97.7722015, 314.4334717, -446.9413452, 530.4071655
2: -192.9384308, 371.0262146, -143.5588379, 268.4643250, -461.4027710, 514.5850830
3: -111.7829208, 460.3720093, -82.4684830, 336.0852966, -447.8681946, 542.8405151
4: -176.9704590, 324.7064209, -131.3724976, 235.1797028, -412.1501465, 456.0789185

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1055865, upper bound: 175.1085522
time: 0.86 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1055865, upper bound: 175.1088862
time: 1.33 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -49.2713585, 188.5193024, -36.8520775, 141.5761414, -190.8474884, 225.3713684
1: -132.5078888, 432.6349792, -100.1317444, 323.2568359, -455.7646790, 532.7667236
2: -192.9384308, 371.0262146, -146.3668823, 275.1226501, -468.0610962, 517.3930664
3: -111.7829208, 460.3720093, -84.3361740, 345.0667114, -456.8496399, 544.7081909
4: -176.9704590, 324.7064209, -134.0566864, 240.9837799, -417.9542236, 458.7631226

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1051973, upper bound: 175.1084386
time: 0.93 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1041292, upper bound: 175.1076232
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -48.3467102, 184.9049072, -40.6669273, 155.9575500, -204.3042297, 225.5718384
1: -129.9639282, 424.4339294, -110.0067673, 355.8016357, -485.7655640, 534.4406738
2: -188.9842682, 363.6730652, -160.5420837, 304.5820007, -493.5662842, 524.2151489
3: -109.6410294, 451.5211487, -92.6721497, 379.8233643, -489.4643555, 544.1932373
4: -173.5286255, 318.3197937, -147.2462311, 266.8246155, -440.3532104, 465.5659790

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1047463, upper bound: 175.1071438
time: 0.80 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1047463, upper bound: 175.1087777
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -48.3467102, 184.9049072, -41.1342278, 158.0476532, -206.3943481, 226.0391083
1: -129.9639282, 424.4339294, -111.3675690, 361.4914246, -491.4553528, 535.8015137
2: -188.9842682, 363.6730652, -161.8961792, 309.2681274, -498.2523804, 525.5692139
3: -109.6410294, 451.5211487, -93.7394028, 385.2932129, -494.9342041, 545.2605591
4: -173.5286255, 318.3197937, -148.6649323, 270.8312073, -444.3598328, 466.9847412

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1040720, upper bound: 175.1082832
time: 0.91 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1033520, upper bound: 175.1083192
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -49.0569458, 187.7222595, -40.6669273, 155.9575500, -205.0144806, 228.3891907
1: -131.9272461, 430.8273010, -110.0067673, 355.8016357, -487.7288513, 540.8340454
2: -192.0337067, 369.4820251, -160.5420837, 304.5820007, -496.6157227, 530.0241089
3: -111.2858200, 458.4247742, -92.6721497, 379.8233643, -491.1091614, 551.0968628
4: -176.1468658, 323.3547668, -147.2462311, 266.8246155, -442.9714661, 470.6009521

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1073645, upper bound: 175.1101040
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1073645, upper bound: 175.1101041
time: 1.25 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -49.0569458, 187.7222595, -41.1342278, 158.0476532, -207.1045990, 228.8564606
1: -131.9272461, 430.8273010, -111.3675690, 361.4914246, -493.4186707, 542.1948853
2: -192.0337067, 369.4820251, -161.8961792, 309.2681274, -501.3018188, 531.3781738
3: -111.2858200, 458.4247742, -93.7394028, 385.2932129, -496.5789795, 552.1641235
4: -176.1468658, 323.3547668, -148.6649323, 270.8312073, -446.9780884, 472.0197144

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1067801, upper bound: 175.1097731
time: 0.79 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1041292, upper bound: 175.1097616
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -49.0569458, 187.7222595, -41.4221840, 158.8485260, -207.9054718, 229.1444397
1: -131.9272461, 430.8273010, -111.9940109, 362.5901489, -494.5173645, 542.8212891
2: -192.0337067, 369.4820251, -163.5008850, 310.5657043, -502.5994263, 532.9829102
3: -111.2858200, 458.4247742, -94.3583374, 387.0707397, -498.3565369, 552.7830811
4: -176.1468658, 323.3547668, -149.9640045, 272.0599060, -448.2067871, 473.3187866

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076624, upper bound: 175.1102187
time: 0.94 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1076624, upper bound: 175.1102187
time: 0.91 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -49.0569458, 187.7222595, -41.8279152, 160.7582092, -209.8151550, 229.5501709
1: -131.9272461, 430.8273010, -113.2510300, 367.8738708, -499.8010864, 544.0783081
2: -192.0337067, 369.4820251, -164.7188721, 315.0371399, -507.0708313, 534.2009277
3: -111.2858200, 458.4247742, -95.3315353, 392.1143494, -503.4001770, 553.7562866
4: -176.1468658, 323.3547668, -151.2644196, 275.8960571, -452.0429077, 474.6191711

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1070195, upper bound: 175.1098781
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1070144, upper bound: 175.1098659
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -48.4964447, 185.5859375, -41.6167221, 159.7222137, -208.2186279, 227.2026367
1: -130.3096466, 426.2197571, -111.9751358, 365.5847473, -495.8944092, 538.1948242
2: -188.9532166, 365.6941528, -163.6916351, 314.4394531, -503.3926392, 529.3857422
3: -109.8894348, 453.2915039, -94.5206604, 388.9538879, -498.8432922, 547.8121338
4: -173.6568451, 319.9700623, -150.3603363, 275.1552429, -448.8120728, 470.3303833

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1062980, upper bound: 175.1090340
time: 0.84 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065937, upper bound: 175.1095005
time: 0.92 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -48.4964447, 185.5859375, -42.4335709, 163.3559418, -211.8523712, 228.0194855
1: -130.3096466, 426.2197571, -114.2980499, 374.7763672, -505.0859985, 540.5178223
2: -188.9532166, 365.6941528, -166.5272522, 321.5397034, -510.4928894, 532.2214355
3: -109.8894348, 453.2915039, -96.4119492, 398.1408081, -508.0301819, 549.7034302
4: -173.6568451, 319.9700623, -153.1114655, 281.2770691, -454.9338989, 473.0815430

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1062980, upper bound: 175.1090340
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065937, upper bound: 175.1095005
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -48.4964447, 185.5859375, -42.6710625, 163.8090668, -212.3054962, 228.2570038
1: -130.3096466, 426.2197571, -114.7889175, 374.9601135, -505.2697754, 541.0086670
2: -188.9532166, 365.6941528, -168.2391052, 322.6051941, -511.5583496, 533.9332275
3: -109.8894348, 453.2915039, -96.9142075, 398.9512024, -508.8406067, 550.2056885
4: -173.6568451, 319.9700623, -154.4624634, 282.3435364, -456.0003662, 474.4325256

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065218, upper bound: 175.1092060
time: 1.33 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068218, upper bound: 175.1096542
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -48.4964447, 185.5859375, -43.2902336, 166.7141266, -215.2105560, 228.8761597
1: -130.3096466, 426.2197571, -116.5713806, 382.5718689, -512.8815308, 542.7911377
2: -188.9532166, 365.6941528, -170.2562408, 328.5741882, -517.5274048, 535.9503174
3: -109.8894348, 453.2915039, -98.3667908, 406.3963318, -516.2857056, 551.6582642
4: -173.6568451, 319.9700623, -156.4771729, 287.4977112, -461.1545410, 476.4472351

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065218, upper bound: 175.1092060
time: 0.89 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068218, upper bound: 175.1096542
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -53.3354340, 204.4983521, -40.5854340, 155.7677002, -209.1031036, 245.0837860
1: -142.5657959, 470.7482910, -109.1076279, 356.4072266, -498.9729614, 579.8558960
2: -206.5099182, 406.1978455, -159.6264496, 306.8702393, -513.3801270, 565.8242798
3: -120.3028946, 498.7206726, -92.1414871, 379.1696167, -499.4725037, 590.8621826
4: -190.2138367, 355.1447144, -146.6385345, 268.4741211, -458.6878357, 501.7832642

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1085709
time: 0.85 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1045772, upper bound: 175.1085709
time: 0.92 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -52.5812950, 201.5162048, -45.4009857, 174.3787231, -226.9599762, 246.9171906
1: -140.4993134, 463.9378662, -121.8792267, 399.5579224, -540.0572510, 585.8170776
2: -203.3000183, 400.0732117, -177.0930328, 344.5025024, -547.8024902, 577.1662598
3: -118.5518799, 491.4995728, -102.7702942, 424.6116028, -543.1634521, 594.2698975
4: -187.2646942, 349.7889404, -162.9669495, 301.4320679, -488.6967163, 512.7558594

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1090020
time: 1.44 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1090020
time: 1.56 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -53.1751671, 203.8949432, -45.4009857, 174.3787231, -227.5538940, 249.2959137
1: -142.1416473, 469.3835449, -121.8792267, 399.5579224, -541.6995850, 591.2627563
2: -205.8575592, 405.0082092, -177.0930328, 344.5025024, -550.3599854, 582.1012573
3: -119.9381332, 497.2791748, -102.7702942, 424.6116028, -544.5496826, 600.0494385
4: -189.6180573, 354.1068726, -162.9669495, 301.4320679, -491.0501099, 517.0738525

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1100257
time: 1.05 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1065938, upper bound: 175.1100257
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.1751671, 203.8949432, -46.3211479, 177.9058380, -231.0810089, 250.2160950
1: -142.1416473, 469.3835449, -124.3440170, 407.6103821, -549.7519531, 593.7275391
2: -205.8575592, 405.0082092, -181.2606354, 351.4640808, -557.3216553, 586.2688599
3: -119.9381332, 497.2791748, -104.8915787, 433.3132629, -553.2513428, 602.1707764
4: -189.6180573, 354.1068726, -166.6935883, 307.5066833, -497.1247559, 520.8004761

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1082515, upper bound: 175.1101569
time: 0.97 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1083432, upper bound: 175.1102847
time: 0.95 seconds

## Relational analysis of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_B1_A2_B1_A1_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1083432, upper bound: 175.1102847
time: 1.04 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -49.0001259, 187.4663086, -41.9563560, 160.9398499, -209.9399719, 229.4226532
1: -131.7982025, 430.1886902, -113.7067642, 367.8437805, -499.6419373, 543.8954468
2: -191.9714966, 368.9599609, -166.3439484, 312.3717346, -504.3432312, 535.3038940
3: -111.1870270, 457.8632202, -95.8541183, 393.0187378, -504.2057495, 553.7173462
4: -176.0626221, 322.9079895, -152.3316345, 273.7511292, -449.8137207, 475.2395935

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1132624, upper bound: 175.1131299
time: 1.05 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1128123, upper bound: 175.1130897
time: 1.08 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -52.0192337, 199.0249023, -41.9563560, 160.9398499, -212.9590454, 240.9812164
1: -139.8026276, 457.9273987, -113.7067642, 367.8437805, -507.6463928, 571.6340942
2: -203.0936737, 391.5033875, -166.3439484, 312.3717346, -515.4653931, 557.8472900
3: -117.9098816, 487.3460693, -95.8541183, 393.0187378, -510.9285278, 583.2000732
4: -186.4018555, 342.7937927, -152.3316345, 273.7511292, -460.1529236, 495.1254272

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1123749, upper bound: 175.1125935
time: 0.98 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1119420, upper bound: 175.1121748
time: 0.99 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -48.3855057, 185.1768188, -41.8525505, 160.9096832, -209.2951965, 227.0293732
1: -129.9740143, 425.2348022, -113.5445328, 368.5521240, -498.5261230, 538.7793579
2: -189.1387634, 364.9731445, -165.5307007, 312.3477173, -501.4864502, 530.5038452
3: -109.6749802, 452.0707397, -95.6253433, 393.0816956, -502.7566833, 547.6961060
4: -173.5770416, 319.3744507, -151.7606201, 273.5581055, -447.1351318, 471.1350098

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1071119, upper bound: 175.1099123
time: 0.98 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1071119, upper bound: 175.1098317
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -48.8309174, 186.8906097, -42.4702492, 163.2917480, -212.1226501, 229.3608551
1: -131.3127441, 429.0158997, -115.2338943, 374.0394897, -505.3521118, 544.2498169
2: -191.0670624, 367.9229126, -168.0064545, 316.8397217, -507.9067993, 535.9293823
3: -110.7762756, 456.5121460, -97.0461884, 398.9441833, -509.7204590, 553.5582275
4: -175.2981262, 322.0047913, -154.0108337, 277.4969482, -452.7950134, 476.0156250

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1063167, upper bound: 175.1098054
time: 0.87 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1068912, upper bound: 175.1098995
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_B2_A1_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -49.0001259, 187.4663086, -42.4795380, 163.0301208, -212.0302429, 229.9458466
1: -131.7982025, 430.1886902, -115.1161880, 372.7310791, -504.5292664, 545.3048706
2: -191.9714966, 368.9599609, -168.4167175, 316.8856506, -508.8571472, 537.3767090
3: -111.1870270, 457.8632202, -97.0210266, 398.2921753, -509.4791870, 554.8842773
4: -176.0626221, 322.9079895, -154.2266693, 277.7265015, -453.7891235, 477.1345825

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1131771, upper bound: 175.1130816
time: 0.98 seconds

## Relational analysis of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B2_A1_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1129396, upper bound: 175.1127831
time: 1.14 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.09 + 417.07 = 420.16 seconds
