## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.857701161


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1155033, 0.8715090, -0.1155033, 0.8715090, -0.9870124, 0.9870124)
1: (-0.1845481, 0.4175723, -0.1845481, 0.4175723, -0.6021204, 0.6021204)
2: (-0.0360203, 0.5334461, -0.0360203, 0.5334461, -0.5694664, 0.5694664)
3: (-0.0873858, 0.2886980, -0.0873858, 0.2886980, -0.3760837, 0.3760837)
4: (-0.0405117, 0.4983537, -0.0405117, 0.4983537, -0.5388654, 0.5388654)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.09 + 0.96 = 3.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.8707626, upper bound: 0.8707626

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8705079, upper bound: 0.8683264
time: 0.29 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8704759, upper bound: 0.8704819
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.84 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.8705079, upper bound: 0.8683264
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.8704759, upper bound: 0.8704819

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.1145118, 0.8573566, -0.1155033, 0.8715090, -0.9860209, 0.9728600
1: -0.1744569, 0.4028882, -0.1845481, 0.4175723, -0.5920292, 0.5874363
2: -0.0377549, 0.5117826, -0.0360203, 0.5334461, -0.5712010, 0.5478030
3: -0.0889104, 0.2788861, -0.0873858, 0.2886980, -0.3776083, 0.3662719
4: -0.0438386, 0.4759026, -0.0405117, 0.4983537, -0.5421923, 0.5164143

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8683264
time: 0.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8683264
time: 0.33 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.1155033, 0.8715090, -0.9567007, 0.8266885
1: -0.1628710, 0.3947579, -0.1845481, 0.4175723, -0.5804433, 0.5793059
2: -0.0253338, 0.5040015, -0.0360203, 0.5334461, -0.5587799, 0.5400218
3: -0.0742436, 0.2651140, -0.0873858, 0.2886980, -0.3629416, 0.3524998
4: -0.0275370, 0.4722455, -0.0405117, 0.4983537, -0.5258908, 0.5127572

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8704759
time: 0.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8704819
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.69 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8683264
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8683264
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8704759
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.69
Output dim: 0, lower bound: -0.8683264, upper bound: 0.8704819

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.1145118, 0.8573566, -0.1145118, 0.8573566, -0.9718685, 0.9718685
1: -0.1744569, 0.4028882, -0.1744569, 0.4028882, -0.5773451, 0.5773451
2: -0.0377549, 0.5117826, -0.0377549, 0.5117826, -0.5495375, 0.5495375
3: -0.0889104, 0.2788861, -0.0889104, 0.2788861, -0.3677965, 0.3677965
4: -0.0438386, 0.4759026, -0.0438386, 0.4759026, -0.5197411, 0.5197411

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8683463, upper bound: 0.8649469
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8683740, upper bound: 0.8683188
time: 0.32 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.1145118, 0.8573566, -0.0851917, 0.7111852, -0.8256971, 0.9425483
1: -0.1744569, 0.4028882, -0.1628710, 0.3947579, -0.5692148, 0.5657592
2: -0.0377549, 0.5117826, -0.0253338, 0.5040015, -0.5417564, 0.5371165
3: -0.0889104, 0.2788861, -0.0742436, 0.2651140, -0.3540244, 0.3531297
4: -0.0438386, 0.4759026, -0.0275370, 0.4722455, -0.5160840, 0.5034396

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8637426, upper bound: 0.8657958
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8641949, upper bound: 0.8655936
time: 0.30 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.1145118, 0.8573566, -0.9425483, 0.8256971
1: -0.1628710, 0.3947579, -0.1744569, 0.4028882, -0.5657592, 0.5692148
2: -0.0253338, 0.5040015, -0.0377549, 0.5117826, -0.5371165, 0.5417564
3: -0.0742436, 0.2651140, -0.0889104, 0.2788861, -0.3531297, 0.3540244
4: -0.0275370, 0.4722455, -0.0438386, 0.4759026, -0.5034396, 0.5160840

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8657958, upper bound: 0.8637426
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8640182
time: 0.31 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.0851917, 0.7111852, -0.7963769, 0.7963769
1: -0.1628710, 0.3947579, -0.1628710, 0.3947579, -0.5576289, 0.5576289
2: -0.0253338, 0.5040015, -0.0253338, 0.5040015, -0.5293353, 0.5293353
3: -0.0742436, 0.2651140, -0.0742436, 0.2651140, -0.3393576, 0.3393576
4: -0.0275370, 0.4722455, -0.0275370, 0.4722455, -0.4997825, 0.4997825

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8639606, upper bound: 0.8681911
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8640182
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.72 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8683463, upper bound: 0.8649469
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8683740, upper bound: 0.8683188
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8637426, upper bound: 0.8657958
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8641949, upper bound: 0.8655936
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8657958, upper bound: 0.8637426
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8640182
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8639606, upper bound: 0.8681911
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.72
Output dim: 0, lower bound: -0.8640182, upper bound: 0.8640182

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1207678, 0.8857167, -0.1145118, 0.8573566, -0.9781244, 1.0002285
1: -0.1729224, 0.4098290, -0.1744569, 0.4028882, -0.5758106, 0.5842859
2: -0.0372349, 0.5202177, -0.0377549, 0.5117826, -0.5490175, 0.5579726
3: -0.0882239, 0.2833865, -0.0889104, 0.2788861, -0.3671101, 0.3722969
4: -0.0434497, 0.4837382, -0.0438386, 0.4759026, -0.5193523, 0.5275767

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8676731, upper bound: 0.8641346
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8662462, upper bound: 0.8645469
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0975043, 0.7434810, -0.1145118, 0.8573566, -0.9548609, 0.8579929
1: -0.1561596, 0.3772267, -0.1744569, 0.4028882, -0.5590478, 0.5516837
2: -0.0281714, 0.4833381, -0.0377549, 0.5117826, -0.5399540, 0.5210930
3: -0.0783249, 0.2558819, -0.0889104, 0.2788861, -0.3572111, 0.3447922
4: -0.0323700, 0.4505141, -0.0438386, 0.4759026, -0.5082725, 0.4943527

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8677028, upper bound: 0.8658413
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8662462, upper bound: 0.8662537
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0614868, 0.3963944, -0.0851917, 0.7111852, -0.7726720, 0.4815861
1: -0.1208844, 0.2997481, -0.1628710, 0.3947579, -0.5156423, 0.4626191
2: -0.0152366, 0.3800658, -0.0253338, 0.5040015, -0.5192381, 0.4053996
3: -0.0598759, 0.1976640, -0.0742436, 0.2651140, -0.3249899, 0.2719076
4: -0.0158319, 0.3537315, -0.0275370, 0.4722455, -0.4880773, 0.3812685

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8570421, upper bound: 0.8502103
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0615426, 0.4474975, -0.0851917, 0.7111852, -0.7727278, 0.5326892
1: -0.1281735, 0.3514168, -0.1628710, 0.3947579, -0.5229314, 0.5142878
2: -0.0151964, 0.4399416, -0.0253338, 0.5040015, -0.5191979, 0.4652755
3: -0.0616855, 0.2241370, -0.0742436, 0.2651140, -0.3267995, 0.2983806
4: -0.0149642, 0.4113535, -0.0275370, 0.4722455, -0.4872096, 0.4388905

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8575454, upper bound: 0.8646592
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8578919, upper bound: 0.8610279
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.0614868, 0.3963944, -0.4815861, 0.7726720
1: -0.1628710, 0.3947579, -0.1208844, 0.2997481, -0.4626191, 0.5156423
2: -0.0253338, 0.5040015, -0.0152366, 0.3800658, -0.4053996, 0.5192381
3: -0.0742436, 0.2651140, -0.0598759, 0.1976640, -0.2719076, 0.3249899
4: -0.0275370, 0.4722455, -0.0158319, 0.3537315, -0.3812685, 0.4880773

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8573052
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8502103, upper bound: 0.8570421
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.0615426, 0.4474975, -0.5326892, 0.7727278
1: -0.1628710, 0.3947579, -0.1281735, 0.3514168, -0.5142878, 0.5229314
2: -0.0253338, 0.5040015, -0.0151964, 0.4399416, -0.4652755, 0.5191979
3: -0.0742436, 0.2651140, -0.0616855, 0.2241370, -0.2983806, 0.3267995
4: -0.0275370, 0.4722455, -0.0149642, 0.4113535, -0.4388905, 0.4872096

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8646592, upper bound: 0.8575454
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8502103, upper bound: 0.8578919
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0347286, 0.2570418, -0.0851917, 0.7111852, -0.7459139, 0.3422335
1: -0.1069368, 0.2860545, -0.1628710, 0.3947579, -0.5016946, 0.4489255
2: -0.0029963, 0.3672378, -0.0253338, 0.5040015, -0.5069978, 0.3925716
3: -0.0436054, 0.1814913, -0.0742436, 0.2651140, -0.3087194, 0.2557349
4: 0.0015518, 0.3457845, -0.0275370, 0.4722455, -0.4706937, 0.3733216

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8614269, upper bound: 0.8630337
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8639606, upper bound: 0.8681911
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0436796, 0.4099228, -0.0851917, 0.7111852, -0.7548648, 0.4951145
1: -0.1306434, 0.3747293, -0.1628710, 0.3947579, -0.5254012, 0.5376003
2: -0.0098941, 0.4705982, -0.0253338, 0.5040015, -0.5138956, 0.4959320
3: -0.0578548, 0.2335650, -0.0742436, 0.2651140, -0.3229688, 0.3078086
4: -0.0080748, 0.4444586, -0.0275370, 0.4722455, -0.4803202, 0.4719956

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573767, upper bound: 0.8625160
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8576761
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.17 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8676731, upper bound: 0.8641346
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8662462, upper bound: 0.8645469
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8677028, upper bound: 0.8658413
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8662462, upper bound: 0.8662537
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8573052, upper bound: 0.8650098
NS_A1_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8570421, upper bound: 0.8502103
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8575454, upper bound: 0.8646592
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8578919, upper bound: 0.8610279
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8573052
NS_A2_B1_B1_B2, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8502103, upper bound: 0.8570421
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8646592, upper bound: 0.8575454
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8502103, upper bound: 0.8578919
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8614269, upper bound: 0.8630337
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8639606, upper bound: 0.8681911
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8573767, upper bound: 0.8625160
NS_A2_B2_A2_A2, status: Status.VERIFIED, split count: 4, time: 3.17
Output dim: 0, lower bound: -0.8576761, upper bound: 0.8576761

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1207678, 0.8857167, -0.0883952, 0.7070733, -0.8278412, 0.9741119
1: -0.1729224, 0.4098290, -0.1392297, 0.3271160, -0.5000383, 0.5490587
2: -0.0372349, 0.5202177, -0.0250158, 0.4209932, -0.4582281, 0.5452335
3: -0.0882239, 0.2833865, -0.0755656, 0.2226389, -0.3108628, 0.3589521
4: -0.0434497, 0.4837382, -0.0321491, 0.3906648, -0.4341145, 0.5158873

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8641877, upper bound: 0.8469981
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8645848, upper bound: 0.8518717
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1065316, 0.7643710, 0.0007711, 0.0833408, -0.1898724, 0.7635999
1: -0.1561574, 0.3832210, -0.0562705, 0.1840587, -0.3402161, 0.4394915
2: -0.0299327, 0.4863305, 0.0136056, 0.2451653, -0.2750980, 0.4727249
3: -0.0786513, 0.2610233, -0.0289737, 0.1052291, -0.1838804, 0.2899970
4: -0.0346600, 0.4524661, 0.0148088, 0.2314044, -0.2660644, 0.4376573

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8603643, upper bound: 0.8473447
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8611061, upper bound: 0.8522182
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0975043, 0.7434810, -0.0883952, 0.7070733, -0.8045776, 0.8318763
1: -0.1561596, 0.3772267, -0.1392297, 0.3271160, -0.4832755, 0.5164564
2: -0.0281714, 0.4833381, -0.0250158, 0.4209932, -0.4491646, 0.5083539
3: -0.0783249, 0.2558819, -0.0755656, 0.2226389, -0.3009638, 0.3314475
4: -0.0323700, 0.4505141, -0.0321491, 0.3906648, -0.4230347, 0.4826632

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8658413, upper bound: 0.8658413
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8658413, upper bound: 0.8658413
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0828425, 0.6276863, 0.0007711, 0.0833408, -0.1661833, 0.6269152
1: -0.1399760, 0.3515594, -0.0562705, 0.1840587, -0.3240346, 0.4078299
2: -0.0210675, 0.4502259, 0.0136056, 0.2451653, -0.2662328, 0.4366203
3: -0.0687888, 0.2341166, -0.0289737, 0.1052291, -0.1740179, 0.2630903
4: -0.0236114, 0.4198722, 0.0148088, 0.2314044, -0.2550158, 0.4050634

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8604160, upper bound: 0.8504260
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8612659, upper bound: 0.8612658
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0368437, 0.2857300, -0.0851917, 0.7111852, -0.7480289, 0.3709217
1: -0.0898776, 0.2260367, -0.1628710, 0.3947579, -0.4846354, 0.3889077
2: -0.0036663, 0.2918608, -0.0253338, 0.5040015, -0.5076678, 0.3171946
3: -0.0473603, 0.1438715, -0.0742436, 0.2651140, -0.3124743, 0.2181150
4: -0.0049514, 0.2711427, -0.0275370, 0.4722455, -0.4771968, 0.2986797

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8568435, upper bound: 0.8464656
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0374762, 0.2723671, -0.0851917, 0.7111852, -0.7486615, 0.3575588
1: -0.0918354, 0.2752437, -0.1628710, 0.3947579, -0.4865933, 0.4381147
2: -0.0027008, 0.3517230, -0.0253338, 0.5040015, -0.5067023, 0.3770568
3: -0.0466951, 0.1686855, -0.0742436, 0.2651140, -0.3118091, 0.2429291
4: -0.0022046, 0.3293097, -0.0275370, 0.4722455, -0.4744501, 0.3568467

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8575447, upper bound: 0.8635817
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8568408, upper bound: 0.8445319
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8575113, upper bound: 0.8646592
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0719207, -0.0697203, 0.5904577, -0.5724968, 0.1416410
1: -0.0531569, 0.1776938, -0.1460007, 0.3685124, -0.4216692, 0.3236945
2: 0.0167318, 0.2380853, -0.0181351, 0.4699813, -0.4532495, 0.2562204
3: -0.0254501, 0.0984449, -0.0648767, 0.2430300, -0.2684801, 0.1633216
4: 0.0184201, 0.2243663, -0.0189566, 0.4409263, -0.4225062, 0.2433229

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8559496, upper bound: 0.8609131
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567994, upper bound: 0.8610279
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.0368437, 0.2857300, -0.3709217, 0.7480289
1: -0.1628710, 0.3947579, -0.0898776, 0.2260367, -0.3889077, 0.4846354
2: -0.0253338, 0.5040015, -0.0036663, 0.2918608, -0.3171946, 0.5076678
3: -0.0742436, 0.2651140, -0.0473603, 0.1438715, -0.2181150, 0.3124743
4: -0.0275370, 0.4722455, -0.0049514, 0.2711427, -0.2986797, 0.4771968

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8464656, upper bound: 0.8568435
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8572711
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.0374762, 0.2723671, -0.3575588, 0.7486615
1: -0.1628710, 0.3947579, -0.0918354, 0.2752437, -0.4381147, 0.4865933
2: -0.0253338, 0.5040015, -0.0027008, 0.3517230, -0.3770568, 0.5067023
3: -0.0742436, 0.2651140, -0.0466951, 0.1686855, -0.2429291, 0.3118091
4: -0.0275370, 0.4722455, -0.0022046, 0.3293097, -0.3568467, 0.4744501

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8635817, upper bound: 0.8575447
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_B1

### Relational analysis result of NS_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8445319, upper bound: 0.8568408
time: 0.40 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2

### Relational analysis result of NS_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8646592, upper bound: 0.8575113
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0697203, 0.5904577, 0.0179609, 0.0719207, -0.1416410, 0.5724968
1: -0.1460007, 0.3685124, -0.0531569, 0.1776938, -0.3236945, 0.4216692
2: -0.0181351, 0.4699813, 0.0167318, 0.2380853, -0.2562204, 0.4532495
3: -0.0648767, 0.2430300, -0.0254501, 0.0984449, -0.1633216, 0.2684801
4: -0.0189566, 0.4409263, 0.0184201, 0.2243663, -0.2433229, 0.4225062

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8500733, upper bound: 0.8567994
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8500733, upper bound: 0.8573757
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.1669477, 1.2970560, -0.0846647, 0.7083977, -0.8753454, 1.3817207
1: -0.3157059, 0.6196167, -0.1624534, 0.3941249, -0.7098308, 0.7820701
2: -0.0773737, 0.7601380, -0.0251481, 0.5032004, -0.5805742, 0.7852861
3: -0.1250427, 0.4522166, -0.0740379, 0.2645617, -0.3896044, 0.5262545
4: -0.0830982, 0.7039683, -0.0273293, 0.4715089, -0.5546070, 0.7312976

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8614269, upper bound: 0.8630337
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8614269, upper bound: 0.8630337
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0338743, 0.2509746, -0.0851917, 0.7111852, -0.7450595, 0.3361663
1: -0.1062411, 0.2851566, -0.1628710, 0.3947579, -0.5009990, 0.4480276
2: -0.0027181, 0.3661370, -0.0253338, 0.5040015, -0.5067196, 0.3914708
3: -0.0432705, 0.1805658, -0.0742436, 0.2651140, -0.3083845, 0.2548094
4: 0.0019501, 0.3448365, -0.0275370, 0.4722455, -0.4702953, 0.3723736

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8573651, upper bound: 0.8668169
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8575187, upper bound: 0.8565627
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0851917, 0.7111852, -0.7236235, 0.2692752
1: -0.0840206, 0.2807598, -0.1628710, 0.3947579, -0.4787785, 0.4436308
2: 0.0052519, 0.3607540, -0.0253338, 0.5040015, -0.4987496, 0.3860878
3: -0.0381413, 0.1672755, -0.0742436, 0.2651140, -0.3032553, 0.2415191
4: 0.0080186, 0.3421003, -0.0275370, 0.4722455, -0.4642269, 0.3696373

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8573757
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8576761
time: 0.37 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.85 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8641877, upper bound: 0.8469981
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8645848, upper bound: 0.8518717
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8603643, upper bound: 0.8473447
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8611061, upper bound: 0.8522182
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8658413, upper bound: 0.8658413
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8658413, upper bound: 0.8658413
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8604160, upper bound: 0.8504260
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8612659, upper bound: 0.8612658
NS_A1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8568435, upper bound: 0.8464656
NS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
NS_A1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8568408, upper bound: 0.8445319
NS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8575113, upper bound: 0.8646592
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8559496, upper bound: 0.8609131
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8567994, upper bound: 0.8610279
NS_A2_B1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8464656, upper bound: 0.8568435
NS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8650098, upper bound: 0.8572711
NS_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8445319, upper bound: 0.8568408
NS_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8646592, upper bound: 0.8575113
NS_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8500733, upper bound: 0.8567994
NS_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8500733, upper bound: 0.8573757
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8614269, upper bound: 0.8630337
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8614269, upper bound: 0.8630337
NS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8573651, upper bound: 0.8668169
NS_A2_B2_A1_A2_A2, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8575187, upper bound: 0.8565627
NS_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8573757
NS_A2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 2.85
Output dim: 0, lower bound: -0.8573757, upper bound: 0.8576761

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0664635, 0.4026110, -0.0883952, 0.7070733, -0.7735369, 0.4910062
1: -0.1186190, 0.3075463, -0.1392297, 0.3271160, -0.4457350, 0.4467760
2: -0.0139998, 0.3883494, -0.0250158, 0.4209932, -0.4349930, 0.4133652
3: -0.0582728, 0.2021195, -0.0755656, 0.2226389, -0.2809117, 0.2776851
4: -0.0144429, 0.3612418, -0.0321491, 0.3906648, -0.4051076, 0.3933909

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8641877, upper bound: 0.8467579
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8641877, upper bound: 0.8469981
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0802144, 0.5413153, -0.0883952, 0.7070733, -0.7872878, 0.6297106
1: -0.1369947, 0.3763355, -0.1392297, 0.3271160, -0.4641106, 0.5155652
2: -0.0193379, 0.4696474, -0.0250158, 0.4209932, -0.4403311, 0.4946632
3: -0.0660914, 0.2424133, -0.0755656, 0.2226389, -0.2887303, 0.3179789
4: -0.0200559, 0.4387749, -0.0321491, 0.3906648, -0.4107206, 0.4709240

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8645848, upper bound: 0.8516315
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8645848, upper bound: 0.8518717
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0527263, 0.2914938, 0.0007711, 0.0833408, -0.1360671, 0.2907227
1: -0.1016844, 0.2815768, -0.0562705, 0.1840587, -0.2857431, 0.3378473
2: -0.0069585, 0.3570874, 0.0136056, 0.2451653, -0.2521238, 0.3434818
3: -0.0482249, 0.1813136, -0.0289737, 0.1052291, -0.1534540, 0.2102873
4: -0.0054734, 0.3329027, 0.0148088, 0.2314044, -0.2368778, 0.3180940

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8495245, upper bound: 0.8464949
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8495245, upper bound: 0.8473447
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0671817, 0.4388015, 0.0007711, 0.0833408, -0.1505226, 0.4380305
1: -0.1220098, 0.3546470, -0.0562705, 0.1840587, -0.3060685, 0.4109176
2: -0.0125384, 0.4435782, 0.0136056, 0.2451653, -0.2577036, 0.4299726
3: -0.0584559, 0.2235658, -0.0289737, 0.1052291, -0.1636850, 0.2525395
4: -0.0125319, 0.4153917, 0.0148088, 0.2314044, -0.2439363, 0.4005829

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8502846, upper bound: 0.8513684
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8502846, upper bound: 0.8522182
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0727152, 0.5994924, -0.0883952, 0.7070733, -0.7797886, 0.6878877
1: -0.1220066, 0.3026006, -0.1392297, 0.3271160, -0.4491225, 0.4418303
2: -0.0156414, 0.3942251, -0.0250158, 0.4209932, -0.4366346, 0.4192409
3: -0.0653569, 0.2009754, -0.0755656, 0.2226389, -0.2879958, 0.2765410
4: -0.0209812, 0.3668004, -0.0321491, 0.3906648, -0.4116459, 0.3989495

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653675, upper bound: 0.8658405
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8642738
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0122321, 0.0727732, -0.0883952, 0.7070733, -0.6948412, 0.1611684
1: -0.0464777, 0.1708565, -0.1392297, 0.3271160, -0.3735937, 0.3100862
2: 0.0183295, 0.2326307, -0.0250158, 0.4209932, -0.4026638, 0.2576465
3: -0.0257124, 0.0932948, -0.0755656, 0.2226389, -0.2483513, 0.1688604
4: 0.0187735, 0.2205515, -0.0321491, 0.3906648, -0.3718913, 0.2527006

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8650901, upper bound: 0.8606791
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8647563, upper bound: 0.8609193
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0330088, 0.1895605, 0.0007711, 0.0833408, -0.1163496, 0.1887894
1: -0.0885417, 0.2514557, -0.0562705, 0.1840587, -0.2726004, 0.3077262
2: -0.0003076, 0.3248876, 0.0136056, 0.2451653, -0.2454729, 0.3112820
3: -0.0399428, 0.1567183, -0.0289737, 0.1052291, -0.1451719, 0.1856920
4: 0.0032163, 0.3046087, 0.0148088, 0.2314044, -0.2281881, 0.2897999

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8495762, upper bound: 0.8495762
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8495762, upper bound: 0.8504260
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0348861, 0.2577052, 0.0007711, 0.0833408, -0.1182269, 0.2569341
1: -0.0997334, 0.3116314, -0.0562705, 0.1840587, -0.2837921, 0.3679019
2: -0.0021665, 0.3960302, 0.0136056, 0.2451653, -0.2473318, 0.3824246
3: -0.0461900, 0.1883021, -0.0289737, 0.1052291, -0.1514190, 0.2172758
4: 0.0003834, 0.3733864, 0.0148088, 0.2314044, -0.2310210, 0.3585776

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8504260, upper bound: 0.8604160
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8502846, upper bound: 0.8612658
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0851917, 0.7111852, -0.7347702, 0.2869540
1: -0.0761097, 0.2042865, -0.1628710, 0.3947579, -0.4708675, 0.3671575
2: 0.0041061, 0.2713708, -0.0253338, 0.5040015, -0.4998954, 0.2967046
3: -0.0387292, 0.1246481, -0.0742436, 0.2651140, -0.3038432, 0.1988917
4: 0.0045154, 0.2540526, -0.0275370, 0.4722455, -0.4677300, 0.2815897

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8649217
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0851917, 0.7111852, -0.7362229, 0.2805617
1: -0.0789018, 0.2572541, -0.1628710, 0.3947579, -0.4736597, 0.4201251
2: 0.0034661, 0.3338276, -0.0253338, 0.5040015, -0.5005354, 0.3591614
3: -0.0388725, 0.1523266, -0.0742436, 0.2651140, -0.3039865, 0.2265701
4: 0.0053548, 0.3143979, -0.0275370, 0.4722455, -0.4668907, 0.3419349

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646102
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646263
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0719207, -0.0225223, 0.1781922, -0.1602314, 0.0944430
1: -0.0531569, 0.1776938, -0.0937605, 0.2620107, -0.3151675, 0.2714543
2: 0.0167318, 0.2380853, 0.0019959, 0.3386969, -0.3219650, 0.2360893
3: -0.0254501, 0.0984449, -0.0374944, 0.1636432, -0.1890934, 0.1359394
4: 0.0184201, 0.2243663, 0.0078057, 0.3198919, -0.3014718, 0.2165606

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8517374
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8607851
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0179609, 0.0719207, -0.0331524, 0.3250504, -0.3070895, 0.1050731
1: -0.0531569, 0.1776938, -0.1187833, 0.3545915, -0.4077483, 0.2964771
2: 0.0167318, 0.2380853, -0.0054786, 0.4459014, -0.4291695, 0.2435638
3: -0.0254501, 0.0984449, -0.0511791, 0.2176074, -0.2430575, 0.1496240
4: 0.0184201, 0.2243663, -0.0022886, 0.4222544, -0.4038344, 0.2266549

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8519152
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8609629
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.0235850, 0.2017623, -0.2869540, 0.7347702
1: -0.1628710, 0.3947579, -0.0761097, 0.2042865, -0.3671575, 0.4708675
2: -0.0253338, 0.5040015, 0.0041061, 0.2713708, -0.2967046, 0.4998954
3: -0.0742436, 0.2651140, -0.0387292, 0.1246481, -0.1988917, 0.3038432
4: -0.0275370, 0.4722455, 0.0045154, 0.2540526, -0.2815897, 0.4677300

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649217, upper bound: 0.8572711
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649217, upper bound: 0.8572711
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0851917, 0.7111852, -0.0250376, 0.1953700, -0.2805617, 0.7362229
1: -0.1628710, 0.3947579, -0.0789018, 0.2572541, -0.4201251, 0.4736597
2: -0.0253338, 0.5040015, 0.0034661, 0.3338276, -0.3591614, 0.5005354
3: -0.0742436, 0.2651140, -0.0388725, 0.1523266, -0.2265701, 0.3039865
4: -0.0275370, 0.4722455, 0.0053548, 0.3143979, -0.3419349, 0.4668907

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8646102, upper bound: 0.8564161
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8443540, upper bound: 0.8573475
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1669477, 1.2970560, -0.0344447, 0.2553055, -0.4222533, 1.3315008
1: -0.3157059, 0.6196167, -0.1066431, 0.2854775, -0.6011834, 0.7262599
2: -0.0773737, 0.7601380, -0.0028965, 0.3665537, -0.4439274, 0.7630345
3: -0.1250427, 0.4522166, -0.0434975, 0.1810370, -0.3060797, 0.4957141
4: -0.0830982, 0.7039683, 0.0016746, 0.3451585, -0.4282566, 0.7022937

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1669477, 1.2970560, -0.0433818, 0.4079795, -0.5749272, 1.3404378
1: -0.3157059, 0.6196167, -0.1303592, 0.3742354, -0.6899413, 0.7499759
2: -0.0773737, 0.7601380, -0.0097980, 0.4700043, -0.5473781, 0.7699360
3: -0.1250427, 0.4522166, -0.0577090, 0.2331523, -0.3581950, 0.5099257
4: -0.0830982, 0.7039683, -0.0079560, 0.4439234, -0.5270215, 0.7119243

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0123150, 0.1449477, -0.0851917, 0.7111852, -0.7235003, 0.2301394
1: -0.0770296, 0.2152297, -0.1628710, 0.3947579, -0.4717875, 0.3781007
2: 0.0070565, 0.2852433, -0.0253338, 0.5040015, -0.4969451, 0.3105771
3: -0.0344040, 0.1318887, -0.0742436, 0.2651140, -0.2995180, 0.2061323
4: 0.0107862, 0.2691944, -0.0275370, 0.4722455, -0.4614592, 0.2967314

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573517, upper bound: 0.8562623
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8573517, upper bound: 0.8565627
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.89 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8641877, upper bound: 0.8467579
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8641877, upper bound: 0.8469981
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8645848, upper bound: 0.8516315
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8645848, upper bound: 0.8518717
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8495245, upper bound: 0.8464949
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8495245, upper bound: 0.8473447
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8502846, upper bound: 0.8513684
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8502846, upper bound: 0.8522182
NS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8653675, upper bound: 0.8658405
NS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8642738
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8650901, upper bound: 0.8606791
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8647563, upper bound: 0.8609193
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8495762, upper bound: 0.8495762
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8495762, upper bound: 0.8504260
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8504260, upper bound: 0.8604160
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8502846, upper bound: 0.8612658
NS_A1_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8649217
NS_A1_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8572711, upper bound: 0.8650098
NS_A1_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646102
NS_A1_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646263
NS_A1_B2_A2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8517374
NS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8607851
NS_A1_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8566212, upper bound: 0.8519152
NS_A1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8609629
NS_A2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8649217, upper bound: 0.8572711
NS_A2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8649217, upper bound: 0.8572711
NS_A2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8646102, upper bound: 0.8564161
NS_A2_B1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8443540, upper bound: 0.8573475
NS_A2_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8573517, upper bound: 0.8562623
NS_A2_B2_A1_A2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.89
Output dim: 0, lower bound: -0.8573517, upper bound: 0.8565627

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0664635, 0.4026110, -0.0368437, 0.2857300, -0.3521936, 0.4394547
1: -0.1186190, 0.3075463, -0.0898776, 0.2260367, -0.3446557, 0.3974239
2: -0.0139998, 0.3883494, -0.0036663, 0.2918608, -0.3058606, 0.3920156
3: -0.0582728, 0.2021195, -0.0473603, 0.1438715, -0.2021442, 0.2494799
4: -0.0144429, 0.3612418, -0.0049514, 0.2711427, -0.2855856, 0.3661932

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8460954
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8467579
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0664635, 0.4026110, -0.0365385, 0.2705888, -0.3370523, 0.4391495
1: -0.1186190, 0.3075463, -0.0904422, 0.2742519, -0.3928710, 0.3979885
2: -0.0139998, 0.3883494, -0.0018313, 0.3511297, -0.3651295, 0.3901807
3: -0.0582728, 0.2021195, -0.0465108, 0.1675767, -0.2258495, 0.2486303
4: -0.0144429, 0.3612418, -0.0017787, 0.3288013, -0.3432442, 0.3630205

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8463356
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8469981
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0802144, 0.5413153, -0.0368437, 0.2857300, -0.3659444, 0.5781590
1: -0.1369947, 0.3763355, -0.0898776, 0.2260367, -0.3630314, 0.4662131
2: -0.0193379, 0.4696474, -0.0036663, 0.2918608, -0.3111987, 0.4733137
3: -0.0660914, 0.2424133, -0.0473603, 0.1438715, -0.2099629, 0.2897736
4: -0.0200559, 0.4387749, -0.0049514, 0.2711427, -0.2911986, 0.4437263

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0802144, 0.5413153, -0.0365385, 0.2705888, -0.3508033, 0.5778539
1: -0.1369947, 0.3763355, -0.0904422, 0.2742519, -0.4112466, 0.4667777
2: -0.0193379, 0.4696474, -0.0018313, 0.3511297, -0.3704676, 0.4714787
3: -0.0660914, 0.2424133, -0.0465108, 0.1675767, -0.2336681, 0.2889241
4: -0.0200559, 0.4387749, -0.0017787, 0.3288013, -0.3488572, 0.4405536

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0659518, 0.5462339, -0.0883952, 0.7070733, -0.7730252, 0.6346291
1: -0.1063366, 0.2925426, -0.1392297, 0.3271160, -0.4334525, 0.4317723
2: -0.0103340, 0.3819265, -0.0250158, 0.4209932, -0.4313272, 0.4069423
3: -0.0601309, 0.1888874, -0.0755656, 0.2226389, -0.2827697, 0.2644529
4: -0.0162994, 0.3559344, -0.0321491, 0.3906648, -0.4069642, 0.3880835

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0434720, 0.4874846, -0.0883952, 0.7070733, -0.7505453, 0.5758798
1: -0.0875649, 0.2564760, -0.1392297, 0.3271160, -0.4146808, 0.3957057
2: 0.0021218, 0.3422225, -0.0250158, 0.4209932, -0.4188714, 0.3672383
3: -0.0488474, 0.1618268, -0.0755656, 0.2226389, -0.2714863, 0.2373924
4: -0.0019805, 0.3208379, -0.0321491, 0.3906648, -0.3926452, 0.3529870

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0122321, 0.0727732, -0.0368437, 0.2857300, -0.2734979, 0.1096169
1: -0.0464777, 0.1708565, -0.0898776, 0.2260367, -0.2725144, 0.2607341
2: 0.0183295, 0.2326307, -0.0036663, 0.2918608, -0.2735314, 0.2362970
3: -0.0257124, 0.0932948, -0.0473603, 0.1438715, -0.1695839, 0.1406551
4: 0.0187735, 0.2205515, -0.0049514, 0.2711427, -0.2523692, 0.2255028

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8498393
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8606791
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0122321, 0.0727732, -0.0365385, 0.2705888, -0.2583567, 0.1093117
1: -0.0464777, 0.1708565, -0.0904422, 0.2742519, -0.3207297, 0.2612987
2: 0.0183295, 0.2326307, -0.0018313, 0.3511297, -0.3328003, 0.2344621
3: -0.0257124, 0.0932948, -0.0465108, 0.1675767, -0.1932892, 0.1398056
4: 0.0187735, 0.2205515, -0.0017787, 0.3288013, -0.3100279, 0.2223302

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8500795
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8609193
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0348861, 0.2577052, 0.0268568, 0.0574614, -0.0923475, 0.2308483
1: -0.0997334, 0.3116314, -0.0368407, 0.0713444, -0.1710778, 0.3484721
2: -0.0021665, 0.3960302, 0.0235914, 0.1126523, -0.1148188, 0.3724387
3: -0.0461900, 0.1883021, -0.0227639, 0.0373147, -0.0835046, 0.2110660
4: 0.0003834, 0.3733864, 0.0225648, 0.1031445, -0.1027611, 0.3508216

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8603643
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8604160
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0348861, 0.2577052, 0.0179609, 0.0719207, -0.1068068, 0.2397443
1: -0.0997334, 0.3116314, -0.0531569, 0.1776938, -0.2774272, 0.3647882
2: -0.0021665, 0.3960302, 0.0167318, 0.2380853, -0.2402517, 0.3792983
3: -0.0461900, 0.1883021, -0.0254501, 0.0984449, -0.1446349, 0.2137522
4: 0.0003834, 0.3733864, 0.0184201, 0.2243663, -0.2239829, 0.3549663

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8611060
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8612658
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0628903, 0.5418991, -0.5654841, 0.2646526
1: -0.0761097, 0.2042865, -0.1266434, 0.3224042, -0.3985139, 0.3309298
2: 0.0041061, 0.2713708, -0.0132233, 0.4170020, -0.4128959, 0.2845941
3: -0.0387292, 0.1246481, -0.0598199, 0.2123892, -0.2511184, 0.1844680
4: 0.0045154, 0.2540526, -0.0154952, 0.3904074, -0.3858920, 0.2695478

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8649135
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8649217
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, 0.0121128, 0.0784250, -0.1020100, 0.1896496
1: -0.0761097, 0.2042865, -0.0574983, 0.1989385, -0.2750482, 0.2617847
2: 0.0041061, 0.2713708, 0.0152508, 0.2656995, -0.2615934, 0.2561200
3: -0.0387292, 0.1246481, -0.0259293, 0.1142975, -0.1530267, 0.1505774
4: 0.0045154, 0.2540526, 0.0184830, 0.2521051, -0.2475897, 0.2355696

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8649513
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8650098
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0347286, 0.2570418, -0.2820795, 0.2300986
1: -0.0789018, 0.2572541, -0.1069368, 0.2860545, -0.3649563, 0.3641908
2: 0.0034661, 0.3338276, -0.0029963, 0.3672378, -0.3637717, 0.3368239
3: -0.0388725, 0.1523266, -0.0436054, 0.1814913, -0.2203638, 0.1959319
4: 0.0053548, 0.3143979, 0.0015518, 0.3457845, -0.3404298, 0.3128461

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8645797
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646102
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0436796, 0.4099228, -0.4349605, 0.2390495
1: -0.0789018, 0.2572541, -0.1306434, 0.3747293, -0.4536311, 0.3878974
2: 0.0034661, 0.3338276, -0.0098941, 0.4705982, -0.4671321, 0.3437217
3: -0.0388725, 0.1523266, -0.0578548, 0.2335650, -0.2724375, 0.2101814
4: 0.0053548, 0.3143979, -0.0080748, 0.4444586, -0.4391038, 0.3224726

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646109
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646263
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0225223, 0.1781922, -0.1515194, 0.0859717
1: -0.0436111, 0.1676113, -0.0937605, 0.2620107, -0.3056218, 0.2613718
2: 0.0208580, 0.2285865, 0.0019959, 0.3386969, -0.3178389, 0.2265906
3: -0.0223003, 0.0900128, -0.0374944, 0.1636432, -0.1859435, 0.1275072
4: 0.0212943, 0.2158635, 0.0078057, 0.3198919, -0.2985975, 0.2080578

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8491698, upper bound: 0.8597675
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8607460
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8567627, upper bound: 0.8607460
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0331524, 0.3250504, -0.2983775, 0.0966017
1: -0.0436111, 0.1676113, -0.1187833, 0.3545915, -0.3982026, 0.2863946
2: 0.0208580, 0.2285865, -0.0054786, 0.4459014, -0.4250434, 0.2340651
3: -0.0223003, 0.0900128, -0.0511791, 0.2176074, -0.2399077, 0.1411919
4: 0.0212943, 0.2158635, -0.0022886, 0.4222544, -0.4009601, 0.2181521

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8503216, upper bound: 0.8599048
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8578578, upper bound: 0.8607497
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8578578, upper bound: 0.8607497
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0628903, 0.5418991, -0.0235850, 0.2017623, -0.2646526, 0.5654841
1: -0.1266434, 0.3224042, -0.0761097, 0.2042865, -0.3309298, 0.3985139
2: -0.0132233, 0.4170020, 0.0041061, 0.2713708, -0.2845941, 0.4128959
3: -0.0598199, 0.2123892, -0.0387292, 0.1246481, -0.1844680, 0.2511184
4: -0.0154952, 0.3904074, 0.0045154, 0.2540526, -0.2695478, 0.3858920

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8561759
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8572711
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0121128, 0.0784250, -0.0235850, 0.2017623, -0.1896496, 0.1020100
1: -0.0574983, 0.1989385, -0.0761097, 0.2042865, -0.2617847, 0.2750482
2: 0.0152508, 0.2656995, 0.0041061, 0.2713708, -0.2561200, 0.2615934
3: -0.0259293, 0.1142975, -0.0387292, 0.1246481, -0.1505774, 0.1530267
4: 0.0184830, 0.2521051, 0.0045154, 0.2540526, -0.2355696, 0.2475897

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8561759
time: 0.33 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8572711
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0347286, 0.2570418, -0.0250376, 0.1953700, -0.2300986, 0.2820795
1: -0.1069368, 0.2860545, -0.0789018, 0.2572541, -0.3641908, 0.3649563
2: -0.0029963, 0.3672378, 0.0034661, 0.3338276, -0.3368239, 0.3637717
3: -0.0436054, 0.1814913, -0.0388725, 0.1523266, -0.1959319, 0.2203638
4: 0.0015518, 0.3457845, 0.0053548, 0.3143979, -0.3128461, 0.3404298

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8645797, upper bound: 0.8564161
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8645797, upper bound: 0.8564161
time: 0.33 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.39 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8460954
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8467579
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8463356
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640898, upper bound: 0.8469981
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8640871, upper bound: 0.8442480
NS_A1_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
NS_A1_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
NS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
NS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8653667, upper bound: 0.8653667
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8498393
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8606791
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8500795
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8642543, upper bound: 0.8609193
NS_A1_B1_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8603643
NS_A1_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8604160
NS_A1_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8611060
NS_A1_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8473446, upper bound: 0.8612658
NS_A1_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8649135
NS_A1_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8649217
NS_A1_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8649513
NS_A1_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8561759, upper bound: 0.8650098
NS_A1_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8645797
NS_A1_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8564161, upper bound: 0.8646102
NS_A1_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646109
NS_A1_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8550188, upper bound: 0.8646263
NS_A1_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8567626, upper bound: 0.8607460
NS_A1_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8567627, upper bound: 0.8607460
NS_A1_B2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8578578, upper bound: 0.8607497
NS_A1_B2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8578578, upper bound: 0.8607497
NS_A2_B1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8561759
NS_A2_B1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8572711
NS_A2_B1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8561759
NS_A2_B1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8649135, upper bound: 0.8572711
NS_A2_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8645797, upper bound: 0.8564161
NS_A2_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.39
Output dim: 0, lower bound: -0.8645797, upper bound: 0.8564161

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0418009, 0.3040377, -0.0368437, 0.2857300, -0.3275310, 0.3408815
1: -0.0888689, 0.2340408, -0.0898776, 0.2260367, -0.3149055, 0.3239184
2: -0.0027575, 0.3023143, -0.0036663, 0.2918608, -0.2946183, 0.3059805
3: -0.0470561, 0.1489143, -0.0473603, 0.1438715, -0.1909275, 0.1962746
4: -0.0043157, 0.2810166, -0.0049514, 0.2711427, -0.2754584, 0.2859680

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8456678
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8460954
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0218512, 0.0558277, -0.0368437, 0.2857300, -0.2638789, 0.0926714
1: -0.0355664, 0.0747686, -0.0898776, 0.2260367, -0.2616031, 0.1646462
2: 0.0237875, 0.1164641, -0.0036663, 0.2918608, -0.2680733, 0.1201303
3: -0.0220385, 0.0388933, -0.0473603, 0.1438715, -0.1659099, 0.0862536
4: 0.0227601, 0.1067764, -0.0049514, 0.2711427, -0.2483826, 0.1117278

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8463304
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8467579
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0418009, 0.3040377, -0.0365385, 0.2705888, -0.3123897, 0.3405762
1: -0.0888689, 0.2340408, -0.0904422, 0.2742519, -0.3631208, 0.3244829
2: -0.0027575, 0.3023143, -0.0018313, 0.3511297, -0.3538872, 0.3041456
3: -0.0470561, 0.1489143, -0.0465108, 0.1675767, -0.2146328, 0.1954251
4: -0.0043157, 0.2810166, -0.0017787, 0.3288013, -0.3331171, 0.2827953

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8628725, upper bound: 0.8463349
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8456651
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8463356
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0218512, 0.0558277, -0.0365385, 0.2705888, -0.2487377, 0.0923662
1: -0.0355664, 0.0747686, -0.0904422, 0.2742519, -0.3098184, 0.1652108
2: 0.0237875, 0.1164641, -0.0018313, 0.3511297, -0.3273422, 0.1182954
3: -0.0220385, 0.0388933, -0.0465108, 0.1675767, -0.1896152, 0.0854041
4: 0.0227601, 0.1067764, -0.0017787, 0.3288013, -0.3060412, 0.1085551

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8628725, upper bound: 0.8469974
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8463276
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8469981
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0489363, 0.3154880, -0.0368437, 0.2857300, -0.3346663, 0.3523318
1: -0.0938130, 0.2896308, -0.0898776, 0.2260367, -0.3198497, 0.3795084
2: -0.0032472, 0.3688352, -0.0036663, 0.2918608, -0.2951081, 0.3725015
3: -0.0476089, 0.1782496, -0.0473603, 0.1438715, -0.1914804, 0.2256099
4: -0.0038001, 0.3450474, -0.0049514, 0.2711427, -0.2749428, 0.3499987

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8438204
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8442480
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0147378, 0.0727523, -0.0368437, 0.2857300, -0.2709922, 0.1095961
1: -0.0532238, 0.1832044, -0.0898776, 0.2260367, -0.2792605, 0.2730820
2: 0.0166814, 0.2444295, -0.0036663, 0.2918608, -0.2751794, 0.2480957
3: -0.0249866, 0.1018386, -0.0473603, 0.1438715, -0.1688580, 0.1491989
4: 0.0184212, 0.2303347, -0.0049514, 0.2711427, -0.2527215, 0.2352861

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8438204
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8442480
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0489363, 0.3154880, -0.0365385, 0.2705888, -0.3195251, 0.3520266
1: -0.0938130, 0.2896308, -0.0904422, 0.2742519, -0.3680650, 0.3800730
2: -0.0032472, 0.3688352, -0.0018313, 0.3511297, -0.3543769, 0.3706665
3: -0.0476089, 0.1782496, -0.0465108, 0.1675767, -0.2151856, 0.2247604
4: -0.0038001, 0.3450474, -0.0017787, 0.3288013, -0.3326014, 0.3468260

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8438177
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8442480
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0147378, 0.0727523, -0.0365385, 0.2705888, -0.2558510, 0.1092908
1: -0.0532238, 0.1832044, -0.0904422, 0.2742519, -0.3274758, 0.2736465
2: 0.0166814, 0.2444295, -0.0018313, 0.3511297, -0.3344483, 0.2462608
3: -0.0249866, 0.1018386, -0.0465108, 0.1675767, -0.1925633, 0.1483494
4: 0.0184212, 0.2303347, -0.0017787, 0.3288013, -0.3103802, 0.2321134

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8438177
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8442480
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0659518, 0.5462339, -0.0818840, 0.6507307, -0.7166825, 0.6281179
1: -0.1063366, 0.2925426, -0.1236651, 0.3159684, -0.4223050, 0.4162077
2: -0.0103340, 0.3819265, -0.0199593, 0.4078017, -0.4181357, 0.4018857
3: -0.0601309, 0.1888874, -0.0702814, 0.2103072, -0.2704381, 0.2591688
4: -0.0162994, 0.3559344, -0.0274717, 0.3789678, -0.3952672, 0.3834061

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0659518, 0.5462339, -0.0582210, 0.5778359, -0.6437877, 0.6044549
1: -0.1063366, 0.2925426, -0.1017476, 0.2747470, -0.3810835, 0.3942902
2: -0.0103340, 0.3819265, -0.0050463, 0.3631158, -0.3734498, 0.3869728
3: -0.0601309, 0.1888874, -0.0557160, 0.1784895, -0.2386204, 0.2446034
4: -0.0162994, 0.3559344, -0.0111473, 0.3393029, -0.3556024, 0.3670817

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0434720, 0.4874846, -0.0818840, 0.6507307, -0.6942026, 0.5693686
1: -0.0875649, 0.2564760, -0.1236651, 0.3159684, -0.4035333, 0.3801411
2: 0.0021218, 0.3422225, -0.0199593, 0.4078017, -0.4056799, 0.3621817
3: -0.0488474, 0.1618268, -0.0702814, 0.2103072, -0.2591546, 0.2321082
4: -0.0019805, 0.3208379, -0.0274717, 0.3789678, -0.3809482, 0.3483096

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0434720, 0.4874846, -0.0582210, 0.5778359, -0.6213078, 0.5457056
1: -0.0875649, 0.2564760, -0.1017476, 0.2747470, -0.3623118, 0.3582236
2: 0.0021218, 0.3422225, -0.0050463, 0.3631158, -0.3609940, 0.3472687
3: -0.0488474, 0.1618268, -0.0557160, 0.1784895, -0.2273369, 0.2175428
4: -0.0019805, 0.3208379, -0.0111473, 0.3393029, -0.3412834, 0.3319851

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0314113, 0.0514877, -0.0368437, 0.2857300, -0.2543187, 0.0883315
1: -0.0278739, 0.0631317, -0.0898776, 0.2260367, -0.2539106, 0.1530093
2: 0.0275082, 0.1049329, -0.0036663, 0.2918608, -0.2643526, 0.1085991
3: -0.0198437, 0.0305073, -0.0473603, 0.1438715, -0.1637151, 0.0778676
4: 0.0252898, 0.0961480, -0.0049514, 0.2711427, -0.2458528, 0.1010994

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8494117
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8495744
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0368437, 0.2857300, -0.2590572, 0.1002931
1: -0.0436111, 0.1676113, -0.0898776, 0.2260367, -0.2696477, 0.2574889
2: 0.0208580, 0.2285865, -0.0036663, 0.2918608, -0.2710028, 0.2322527
3: -0.0223003, 0.0900128, -0.0473603, 0.1438715, -0.1661718, 0.1373731
4: 0.0212943, 0.2158635, -0.0049514, 0.2711427, -0.2498483, 0.2208149

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8602515
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8606791
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0314113, 0.0514877, -0.0365385, 0.2705888, -0.2391775, 0.0880263
1: -0.0278739, 0.0631317, -0.0904422, 0.2742519, -0.3021259, 0.1535739
2: 0.0275082, 0.1049329, -0.0018313, 0.3511297, -0.3236215, 0.1067642
3: -0.0198437, 0.0305073, -0.0465108, 0.1675767, -0.1874204, 0.0770181
4: 0.0252898, 0.0961480, -0.0017787, 0.3288013, -0.3035115, 0.0979267

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8494090
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8496185
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0365385, 0.2705888, -0.2439159, 0.0999879
1: -0.0436111, 0.1676113, -0.0904422, 0.2742519, -0.3178630, 0.2580535
2: 0.0208580, 0.2285865, -0.0018313, 0.3511297, -0.3302717, 0.2304178
3: -0.0223003, 0.0900128, -0.0465108, 0.1675767, -0.1898770, 0.1365236
4: 0.0212943, 0.2158635, -0.0017787, 0.3288013, -0.3075070, 0.2176422

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8602488
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8609186
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0348861, 0.2577052, 0.0239718, 0.0556262, -0.0905123, 0.2337334
1: -0.0997334, 0.3116314, -0.0353261, 0.0739518, -0.1736852, 0.3469574
2: -0.0021665, 0.3960302, 0.0241863, 0.1155859, -0.1177524, 0.3718439
3: -0.0461900, 0.1883021, -0.0220048, 0.0384768, -0.0846667, 0.2103069
4: 0.0003834, 0.3733864, 0.0230348, 0.1060086, -0.1056252, 0.3503515

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8473397, upper bound: 0.8603643
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8603643
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8602515
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0348861, 0.2577052, 0.0312448, 0.0515863, -0.0864724, 0.2264604
1: -0.0997334, 0.3116314, -0.0284957, 0.0633462, -0.1630797, 0.3401271
2: -0.0021665, 0.3960302, 0.0272910, 0.1051541, -0.1073205, 0.3687392
3: -0.0461900, 0.1883021, -0.0199306, 0.0308322, -0.0770221, 0.2082327
4: 0.0003834, 0.3733864, 0.0251836, 0.0963524, -0.0959689, 0.3482028

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8473397, upper bound: 0.8604160
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8604160
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8604160
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0348861, 0.2577052, 0.0152065, 0.0724378, -0.1073239, 0.2424986
1: -0.0997334, 0.3116314, -0.0526925, 0.1826618, -0.2823952, 0.3643239
2: -0.0021665, 0.3960302, 0.0169543, 0.2441368, -0.2463033, 0.3790759
3: -0.0461900, 0.1883021, -0.0247751, 0.1015067, -0.1476966, 0.2130772
4: 0.0003834, 0.3733864, 0.0185942, 0.2301430, -0.2297596, 0.3547922

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8522131, upper bound: 0.8611060
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8602488
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8602488
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0348861, 0.2577052, 0.0266729, 0.0634494, -0.0983355, 0.2310323
1: -0.0997334, 0.3116314, -0.0436111, 0.1676113, -0.2673447, 0.3552425
2: -0.0021665, 0.3960302, 0.0208580, 0.2285865, -0.2307530, 0.3751722
3: -0.0461900, 0.1883021, -0.0223003, 0.0900128, -0.1362028, 0.2106024
4: 0.0003834, 0.3733864, 0.0212943, 0.2158635, -0.2154801, 0.3520920

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8522131, upper bound: 0.8612658
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8612658
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8609180
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0131314, 0.1497056, -0.1732906, 0.2148938
1: -0.0761097, 0.2042865, -0.0775794, 0.2161378, -0.2922475, 0.2818659
2: 0.0041061, 0.2713708, 0.0068115, 0.2863497, -0.2822436, 0.2645593
3: -0.0387292, 0.1246481, -0.0347106, 0.1325780, -0.1713072, 0.1593587
4: 0.0045154, 0.2540526, 0.0104636, 0.2701671, -0.2656517, 0.2435890

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8648318
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8649199
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0124383, 0.1840835, -0.2076685, 0.2142006
1: -0.0761097, 0.2042865, -0.0840206, 0.2807598, -0.3568695, 0.2883071
2: 0.0041061, 0.2713708, 0.0052519, 0.3607540, -0.3566479, 0.2661189
3: -0.0387292, 0.1246481, -0.0381413, 0.1672755, -0.2060048, 0.1627894
4: 0.0045154, 0.2540526, 0.0080186, 0.3421003, -0.3375849, 0.2460340

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8648318
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8649227
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, 0.0295579, 0.0532944, -0.0768794, 0.1722044
1: -0.0761097, 0.2042865, -0.0386547, 0.0809701, -0.1570798, 0.2429411
2: 0.0041061, 0.2713708, 0.0241651, 0.1271675, -0.1230614, 0.2472057
3: -0.0387292, 0.1246481, -0.0201722, 0.0447866, -0.0835158, 0.1448203
4: 0.0045154, 0.2540526, 0.0245350, 0.1182854, -0.1137700, 0.2295177

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649513
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649513
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, 0.0238402, 0.0717446, -0.0953296, 0.1779221
1: -0.0761097, 0.2042865, -0.0526383, 0.2019492, -0.2780589, 0.2569247
2: 0.0041061, 0.2713708, 0.0178157, 0.2683529, -0.2642468, 0.2535551
3: -0.0387292, 0.1246481, -0.0227460, 0.1140152, -0.1527444, 0.1473941
4: 0.0045154, 0.2540526, 0.0202869, 0.2539533, -0.2494379, 0.2337658

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649663
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8650098
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0131314, 0.1497056, -0.1747433, 0.2085014
1: -0.0789018, 0.2572541, -0.0775794, 0.2161378, -0.2950396, 0.3348335
2: 0.0034661, 0.3338276, 0.0068115, 0.2863497, -0.2828836, 0.3270161
3: -0.0388725, 0.1523266, -0.0347106, 0.1325780, -0.1714505, 0.1870371
4: 0.0053548, 0.3143979, 0.0104636, 0.2701671, -0.2648123, 0.3039342

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8645843
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8645843
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, 0.0297398, 0.0531700, -0.0782077, 0.1656302
1: -0.0789018, 0.2572541, -0.0377935, 0.0805572, -0.1594590, 0.2950476
2: 0.0034661, 0.3338276, 0.0244489, 0.1267945, -0.1233284, 0.3093787
3: -0.0388725, 0.1523266, -0.0201008, 0.0442783, -0.0831508, 0.1724274
4: 0.0053548, 0.3143979, 0.0246537, 0.1179379, -0.1125831, 0.2897442

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8646081
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8646102
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0730147, 0.5947001, -0.6197377, 0.2683846
1: -0.0789018, 0.2572541, -0.1552190, 0.4315584, -0.5104601, 0.4124730
2: 0.0034661, 0.3338276, -0.0181923, 0.5395672, -0.5361011, 0.3520199
3: -0.0388725, 0.1523266, -0.0700658, 0.2765815, -0.3154540, 0.2223924
4: 0.0053548, 0.3143979, -0.0182653, 0.5082310, -0.5028762, 0.3326632

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8644980
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8646109
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0316553, 0.3352351, -0.3602727, 0.2270253
1: -0.0789018, 0.2572541, -0.1189680, 0.3589399, -0.4378417, 0.3762221
2: 0.0034661, 0.3338276, -0.0048860, 0.4521151, -0.4486490, 0.3387136
3: -0.0388725, 0.1523266, -0.0515406, 0.2190310, -0.2579035, 0.2038672
4: 0.0053548, 0.3143979, -0.0017406, 0.4287957, -0.4234409, 0.3161385

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8645797
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8646263
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0131314, 0.1497056, -0.1230327, 0.0765808
1: -0.0436111, 0.1676113, -0.0775794, 0.2161378, -0.2597489, 0.2451907
2: 0.0208580, 0.2285865, 0.0068115, 0.2863497, -0.2654917, 0.2217750
3: -0.0223003, 0.0900128, -0.0347106, 0.1325780, -0.1548783, 0.1247234
4: 0.0212943, 0.2158635, 0.0104636, 0.2701671, -0.2488728, 0.2053999

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0297398, 0.0531700, -0.0264971, 0.0337096
1: -0.0436111, 0.1676113, -0.0377935, 0.0805572, -0.1241682, 0.2054048
2: 0.0208580, 0.2285865, 0.0244489, 0.1267945, -0.1059365, 0.2041375
3: -0.0223003, 0.0900128, -0.0201008, 0.0442783, -0.0665786, 0.1101136
4: 0.0212943, 0.2158635, 0.0246537, 0.1179379, -0.0966435, 0.1912098

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0124383, 0.1840835, -0.1574106, 0.0758877
1: -0.0436111, 0.1676113, -0.0840206, 0.2807598, -0.3243709, 0.2516319
2: 0.0208580, 0.2285865, 0.0052519, 0.3607540, -0.3398960, 0.2233346
3: -0.0223003, 0.0900128, -0.0381413, 0.1672755, -0.1895759, 0.1281541
4: 0.0212943, 0.2158635, 0.0080186, 0.3421003, -0.3208060, 0.2078449

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8606597
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8607497
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0238402, 0.0717446, -0.0450717, 0.0396092
1: -0.0436111, 0.1676113, -0.0526383, 0.2019492, -0.2455603, 0.2202496
2: 0.0208580, 0.2285865, 0.0178157, 0.2683529, -0.2474949, 0.2107708
3: -0.0223003, 0.0900128, -0.0227460, 0.1140152, -0.1363155, 0.1127588
4: 0.0212943, 0.2158635, 0.0202869, 0.2539533, -0.2326589, 0.1955767

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8606597
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8607497
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, -0.0235850, 0.2017623, -0.2148938, 0.1732906
1: -0.0775794, 0.2161378, -0.0761097, 0.2042865, -0.2818659, 0.2922475
2: 0.0068115, 0.2863497, 0.0041061, 0.2713708, -0.2645593, 0.2822436
3: -0.0347106, 0.1325780, -0.0387292, 0.1246481, -0.1593587, 0.1713072
4: 0.0104636, 0.2701671, 0.0045154, 0.2540526, -0.2435890, 0.2656517

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8521999
time: 0.35 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8621458
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0124383, 0.1840835, -0.0235850, 0.2017623, -0.2142006, 0.2076685
1: -0.0840206, 0.2807598, -0.0761097, 0.2042865, -0.2883071, 0.3568695
2: 0.0052519, 0.3607540, 0.0041061, 0.2713708, -0.2661189, 0.3566479
3: -0.0381413, 0.1672755, -0.0387292, 0.1246481, -0.1627894, 0.2060048
4: 0.0080186, 0.3421003, 0.0045154, 0.2540526, -0.2460340, 0.3375849

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8521999
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8621458
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.0295579, 0.0532944, -0.0235850, 0.2017623, -0.1722044, 0.0768794
1: -0.0386547, 0.0809701, -0.0761097, 0.2042865, -0.2429411, 0.1570798
2: 0.0241651, 0.1271675, 0.0041061, 0.2713708, -0.2472057, 0.1230614
3: -0.0201722, 0.0447866, -0.0387292, 0.1246481, -0.1448203, 0.0835158
4: 0.0245350, 0.1182854, 0.0045154, 0.2540526, -0.2295177, 0.1137700

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8547786
time: 0.38 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8559130
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.0238402, 0.0717446, -0.0235850, 0.2017623, -0.1779221, 0.0953296
1: -0.0526383, 0.2019492, -0.0761097, 0.2042865, -0.2569247, 0.2780589
2: 0.0178157, 0.2683529, 0.0041061, 0.2713708, -0.2535551, 0.2642468
3: -0.0227460, 0.1140152, -0.0387292, 0.1246481, -0.1473941, 0.1527444
4: 0.0202869, 0.2539533, 0.0045154, 0.2540526, -0.2337658, 0.2494379

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8547786
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_A2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8571449
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_B2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0131314, 0.1497056, -0.0250376, 0.1953700, -0.2085014, 0.1747433
1: -0.0775794, 0.2161378, -0.0789018, 0.2572541, -0.3348335, 0.2950396
2: 0.0068115, 0.2863497, 0.0034661, 0.3338276, -0.3270161, 0.2828836
3: -0.0347106, 0.1325780, -0.0388725, 0.1523266, -0.1870371, 0.1714505
4: 0.0104636, 0.2701671, 0.0053548, 0.3143979, -0.3039342, 0.2648123

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8563340
time: 0.38 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8559589
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.0297398, 0.0531700, -0.0250376, 0.1953700, -0.1656302, 0.0782077
1: -0.0377935, 0.0805572, -0.0789018, 0.2572541, -0.2950476, 0.1594590
2: 0.0244489, 0.1267945, 0.0034661, 0.3338276, -0.3093787, 0.1233284
3: -0.0201008, 0.0442783, -0.0388725, 0.1523266, -0.1724274, 0.0831508
4: 0.0246537, 0.1179379, 0.0053548, 0.3143979, -0.2897442, 0.1125831

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8563340
time: 0.34 seconds

## Relational analysis of NS_A2_B1_B2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8559589
time: 0.34 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.08 seconds
NS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8456678
NS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8460954
NS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8463304
NS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457542, upper bound: 0.8467579
NS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8456651
NS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8463356
NS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8463276
NS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438204, upper bound: 0.8469981
NS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8438204
NS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8442480
NS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8438204
NS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8457515, upper bound: 0.8442480
NS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8438177
NS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8442480
NS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8438177
NS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8438177, upper bound: 0.8442480
NS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8494117
NS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8495744
NS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8602515
NS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8459187, upper bound: 0.8606791
NS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8494090
NS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8496185
NS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8602488
NS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8439850, upper bound: 0.8609186
NS_A1_B1_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8603643
NS_A1_B1_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8602515
NS_A1_B1_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8604160
NS_A1_B1_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8463343, upper bound: 0.8604160
NS_A1_B1_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8602488
NS_A1_B1_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8602488
NS_A1_B1_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8612658
NS_A1_B1_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8444870, upper bound: 0.8609180
NS_A1_B2_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8648318
NS_A1_B2_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8649199
NS_A1_B2_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8648318
NS_A1_B2_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8521999, upper bound: 0.8649227
NS_A1_B2_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649513
NS_A1_B2_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649513
NS_A1_B2_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8649663
NS_A1_B2_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8547786, upper bound: 0.8650098
NS_A1_B2_A2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8645843
NS_A1_B2_A2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8645843
NS_A1_B2_A2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8646081
NS_A1_B2_A2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8563340, upper bound: 0.8646102
NS_A1_B2_A2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8644980
NS_A1_B2_A2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8646109
NS_A1_B2_A2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8645797
NS_A1_B2_A2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8524401, upper bound: 0.8646263
NS_A1_B2_A2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
NS_A1_B2_A2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
NS_A1_B2_A2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
NS_A1_B2_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8566806, upper bound: 0.8607460
NS_A1_B2_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8606597
NS_A1_B2_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8607497
NS_A1_B2_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8606597
NS_A1_B2_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8527866, upper bound: 0.8607497
NS_A2_B1_B1_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8521999
NS_A2_B1_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8621458
NS_A2_B1_B1_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8521999
NS_A2_B1_B1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8461624, upper bound: 0.8621458
NS_A2_B1_B1_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8547786
NS_A2_B1_B1_B1_B2_A2_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8559130
NS_A2_B1_B1_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8547786
NS_A2_B1_B1_B1_B2_A2_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8462878, upper bound: 0.8571449
NS_A2_B1_B2_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8563340
NS_A2_B1_B2_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8559589
NS_A2_B1_B2_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8563340
NS_A2_B1_B2_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 0, lower bound: -0.8443150, upper bound: 0.8559589

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0417780, 0.3037931, -0.2771203, 0.1052274
1: -0.0436111, 0.1676113, -0.0888374, 0.2339897, -0.2776007, 0.2564487
2: 0.0208580, 0.2285865, -0.0027433, 0.3022506, -0.2813926, 0.2313298
3: -0.0223003, 0.0900128, -0.0470393, 0.1488737, -0.1711740, 0.1370521
4: 0.0212943, 0.2158635, -0.0042998, 0.2809594, -0.2596650, 0.2201633

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0235850, 0.2017623, -0.1750894, 0.0870344
1: -0.0436111, 0.1676113, -0.0761097, 0.2042865, -0.2478975, 0.2437210
2: 0.0208580, 0.2285865, 0.0041061, 0.2713708, -0.2505128, 0.2244804
3: -0.0223003, 0.0900128, -0.0387292, 0.1246481, -0.1469484, 0.1287420
4: 0.0212943, 0.2158635, 0.0045154, 0.2540526, -0.2327583, 0.2113481

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0465425, 0.3043410, -0.2776681, 0.1099918
1: -0.0436111, 0.1676113, -0.0909543, 0.2877573, -0.3313684, 0.2585656
2: 0.0208580, 0.2285865, -0.0016528, 0.3672616, -0.3464036, 0.2302393
3: -0.0223003, 0.0900128, -0.0465476, 0.1760883, -0.1983886, 0.1365604
4: 0.0212943, 0.2158635, -0.0024272, 0.3437493, -0.3224549, 0.2182907

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0243365, 0.1945607, -0.1678878, 0.0877858
1: -0.0436111, 0.1676113, -0.0778679, 0.2565908, -0.3002019, 0.2454792
2: 0.0208580, 0.2285865, 0.0040324, 0.3334116, -0.3125536, 0.2245540
3: -0.0223003, 0.0900128, -0.0386840, 0.1515918, -0.1738921, 0.1286968
4: 0.0212943, 0.2158635, 0.0056554, 0.3140403, -0.2927459, 0.2102081

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0243365, 0.1945607, 0.0239718, 0.0556262, -0.0799626, 0.1705889
1: -0.0778679, 0.2565908, -0.0353261, 0.0739518, -0.1518196, 0.2919169
2: 0.0040324, 0.3334116, 0.0241863, 0.1155859, -0.1115535, 0.3092253
3: -0.0386840, 0.1515918, -0.0220048, 0.0384768, -0.0771607, 0.1735967
4: 0.0056554, 0.3140403, 0.0230348, 0.1060086, -0.1003532, 0.2910055

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0239718, 0.0556262, -0.0289533, 0.0394776
1: -0.0436111, 0.1676113, -0.0353261, 0.0739518, -0.1175628, 0.2029374
2: 0.0208580, 0.2285865, 0.0241863, 0.1155859, -0.0947279, 0.2044002
3: -0.0223003, 0.0900128, -0.0220048, 0.0384768, -0.0607771, 0.1120177
4: 0.0212943, 0.2158635, 0.0230348, 0.1060086, -0.0847143, 0.1928287

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0243365, 0.1945607, 0.0312448, 0.0515863, -0.0759228, 0.1633159
1: -0.0778679, 0.2565908, -0.0284957, 0.0633462, -0.1412141, 0.2850866
2: 0.0040324, 0.3334116, 0.0272910, 0.1051541, -0.1011216, 0.3061205
3: -0.0386840, 0.1515918, -0.0199306, 0.0308322, -0.0695161, 0.1715224
4: 0.0056554, 0.3140403, 0.0251836, 0.0963524, -0.0906969, 0.2888567

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0312448, 0.0515863, -0.0249134, 0.0322046
1: -0.0436111, 0.1676113, -0.0284957, 0.0633462, -0.1069573, 0.1961070
2: 0.0208580, 0.2285865, 0.0272910, 0.1051541, -0.0842961, 0.2012955
3: -0.0223003, 0.0900128, -0.0199306, 0.0308322, -0.0531325, 0.1099434
4: 0.0212943, 0.2158635, 0.0251836, 0.0963524, -0.0750580, 0.1906800

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0243365, 0.1945607, 0.0152065, 0.0724378, -0.0967742, 0.1793541
1: -0.0778679, 0.2565908, -0.0526925, 0.1826618, -0.2605297, 0.3092833
2: 0.0040324, 0.3334116, 0.0169543, 0.2441368, -0.2401044, 0.3164573
3: -0.0386840, 0.1515918, -0.0247751, 0.1015067, -0.1401906, 0.1763670
4: 0.0056554, 0.3140403, 0.0185942, 0.2301430, -0.2244876, 0.2954461

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0152065, 0.0724378, -0.0457649, 0.0482429
1: -0.0436111, 0.1676113, -0.0526925, 0.1826618, -0.2262729, 0.2203038
2: 0.0208580, 0.2285865, 0.0169543, 0.2441368, -0.2232788, 0.2116322
3: -0.0223003, 0.0900128, -0.0247751, 0.1015067, -0.1238070, 0.1147880
4: 0.0212943, 0.2158635, 0.0185942, 0.2301430, -0.2088487, 0.1972693

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0243365, 0.1945607, 0.0266729, 0.0634494, -0.0877858, 0.1678878
1: -0.0778679, 0.2565908, -0.0436111, 0.1676113, -0.2454792, 0.3002019
2: 0.0040324, 0.3334116, 0.0208580, 0.2285865, -0.2245540, 0.3125536
3: -0.0386840, 0.1515918, -0.0223003, 0.0900128, -0.1286968, 0.1738921
4: 0.0056554, 0.3140403, 0.0212943, 0.2158635, -0.2102081, 0.2927459

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0266729, 0.0634494, -0.0367765, 0.0367765
1: -0.0436111, 0.1676113, -0.0436111, 0.1676113, -0.2112224, 0.2112224
2: 0.0208580, 0.2285865, 0.0208580, 0.2285865, -0.2077285, 0.2077285
3: -0.0223003, 0.0900128, -0.0223003, 0.0900128, -0.1123131, 0.1123131
4: 0.0212943, 0.2158635, 0.0212943, 0.2158635, -0.1945692, 0.1945692

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0289529, 0.2032966, -0.2268816, 0.2307152
1: -0.0761097, 0.2042865, -0.0891756, 0.2573025, -0.3334122, 0.2934621
2: 0.0041061, 0.2713708, 0.0028422, 0.3348821, -0.3307761, 0.2685286
3: -0.0387292, 0.1246481, -0.0384304, 0.1591823, -0.1979115, 0.1630785
4: 0.0045154, 0.2540526, 0.0068889, 0.3157227, -0.3112073, 0.2471637

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0031947, 0.0998261, -0.1234111, 0.2049571
1: -0.0761097, 0.2042865, -0.0681015, 0.2029338, -0.2790435, 0.2723879
2: 0.0041061, 0.2713708, 0.0107180, 0.2719691, -0.2678631, 0.2606528
3: -0.0387292, 0.1246481, -0.0302884, 0.1220847, -0.1608139, 0.1549365
4: 0.0045154, 0.2540526, 0.0149963, 0.2578355, -0.2533201, 0.2390563

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0349724, 0.2740249, -0.2976099, 0.2367347
1: -0.0761097, 0.2042865, -0.0981120, 0.3256595, -0.4017692, 0.3023984
2: 0.0041061, 0.2713708, 0.0005621, 0.4139100, -0.4098040, 0.2708088
3: -0.0387292, 0.1246481, -0.0445257, 0.1959200, -0.2346492, 0.1691738
4: 0.0045154, 0.2540526, 0.0021093, 0.3914595, -0.3869441, 0.2519434

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, -0.0030553, 0.1526119, -0.1761969, 0.2048176
1: -0.0761097, 0.2042865, -0.0757312, 0.2676867, -0.3437964, 0.2800176
2: 0.0041061, 0.2713708, 0.0086105, 0.3466829, -0.3425768, 0.2627603
3: -0.0387292, 0.1246481, -0.0342591, 0.1571698, -0.1958990, 0.1589072
4: 0.0045154, 0.2540526, 0.0115876, 0.3295513, -0.3250359, 0.2424650

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, 0.0277790, 0.0565974, -0.0801824, 0.1739833
1: -0.0761097, 0.2042865, -0.0464513, 0.1138701, -0.1899798, 0.2507378
2: 0.0041061, 0.2713708, 0.0208591, 0.1657905, -0.1616844, 0.2505117
3: -0.0387292, 0.1246481, -0.0204322, 0.0649709, -0.1037001, 0.1450803
4: 0.0045154, 0.2540526, 0.0229671, 0.1554066, -0.1508912, 0.2310855

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, 0.0315820, 0.0504956, -0.0740806, 0.1701804
1: -0.0761097, 0.2042865, -0.0324080, 0.0746275, -0.1507372, 0.2366945
2: 0.0041061, 0.2713708, 0.0270800, 0.1210596, -0.1169535, 0.2442909
3: -0.0387292, 0.1246481, -0.0187778, 0.0390126, -0.0777418, 0.1434259
4: 0.0045154, 0.2540526, 0.0263120, 0.1127636, -0.1082482, 0.2277406

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, 0.0130985, 0.0806730, -0.1042580, 0.1886639
1: -0.0761097, 0.2042865, -0.0605808, 0.2375910, -0.3137006, 0.2648672
2: 0.0041061, 0.2713708, 0.0148670, 0.3093624, -0.3052563, 0.2565038
3: -0.0387292, 0.1246481, -0.0233690, 0.1373056, -0.1760349, 0.1480171
4: 0.0045154, 0.2540526, 0.0186814, 0.2925131, -0.2879977, 0.2353712

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0235850, 0.2017623, 0.0274709, 0.0656589, -0.0892439, 0.1742914
1: -0.0761097, 0.2042865, -0.0460902, 0.1935771, -0.2696868, 0.2503766
2: 0.0041061, 0.2713708, 0.0207678, 0.2596045, -0.2554984, 0.2506030
3: -0.0387292, 0.1246481, -0.0209610, 0.1065546, -0.1452838, 0.1456091
4: 0.0045154, 0.2540526, 0.0222067, 0.2458720, -0.2413566, 0.2318459

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0289529, 0.2032966, -0.2283342, 0.2243228
1: -0.0789018, 0.2572541, -0.0891756, 0.2573025, -0.3362043, 0.3464297
2: 0.0034661, 0.3338276, 0.0028422, 0.3348821, -0.3314160, 0.3309854
3: -0.0388725, 0.1523266, -0.0384304, 0.1591823, -0.1980548, 0.1907570
4: 0.0053548, 0.3143979, 0.0068889, 0.3157227, -0.3103679, 0.3075090

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A2_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0031947, 0.0998261, -0.1248638, 0.1985647
1: -0.0789018, 0.2572541, -0.0681015, 0.2029338, -0.2818356, 0.3253555
2: 0.0034661, 0.3338276, 0.0107180, 0.2719691, -0.2685030, 0.3231096
3: -0.0388725, 0.1523266, -0.0302884, 0.1220847, -0.1609572, 0.1826150
4: 0.0053548, 0.3143979, 0.0149963, 0.2578355, -0.2524807, 0.2994016

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, 0.0280041, 0.0562890, -0.0813266, 0.1673658
1: -0.0789018, 0.2572541, -0.0453445, 0.1125276, -0.1914294, 0.3025986
2: 0.0034661, 0.3338276, 0.0212393, 0.1652312, -0.1617651, 0.3125884
3: -0.0388725, 0.1523266, -0.0202954, 0.0641751, -0.1030476, 0.1726219
4: 0.0053548, 0.3143979, 0.0231947, 0.1549279, -0.1495731, 0.2912032

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, 0.0316044, 0.0504820, -0.0755197, 0.1637655
1: -0.0789018, 0.2572541, -0.0323029, 0.0746059, -0.1535076, 0.2895569
2: 0.0034661, 0.3338276, 0.0271148, 0.1210361, -0.1175700, 0.3067128
3: -0.0388725, 0.1523266, -0.0187701, 0.0389615, -0.0778340, 0.1710967
4: 0.0053548, 0.3143979, 0.0263256, 0.1127424, -0.1073876, 0.2880723

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0350123, 0.2865622, -0.3115999, 0.2303823
1: -0.0789018, 0.2572541, -0.0988005, 0.3257504, -0.4046522, 0.3560545
2: 0.0034661, 0.3338276, 0.0004058, 0.4140271, -0.4105610, 0.3334218
3: -0.0388725, 0.1523266, -0.0446595, 0.1962665, -0.2351390, 0.1969860
4: 0.0053548, 0.3143979, 0.0019919, 0.3916755, -0.3863207, 0.3124059

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A2_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, 0.0055131, 0.1005383, -0.1255760, 0.1898569
1: -0.0789018, 0.2572541, -0.0665895, 0.2608808, -0.3397826, 0.3238435
2: 0.0034661, 0.3338276, 0.0126805, 0.3364245, -0.3329584, 0.3211471
3: -0.0388725, 0.1523266, -0.0255392, 0.1519263, -0.1907988, 0.1778657
4: 0.0053548, 0.3143979, 0.0168311, 0.3184561, -0.3131013, 0.2975668

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, -0.0030553, 0.1526119, -0.1776495, 0.1984252
1: -0.0789018, 0.2572541, -0.0757312, 0.2676867, -0.3465885, 0.3329853
2: 0.0034661, 0.3338276, 0.0086105, 0.3466829, -0.3432168, 0.3252171
3: -0.0388725, 0.1523266, -0.0342591, 0.1571698, -0.1960423, 0.1865857
4: 0.0053548, 0.3143979, 0.0115876, 0.3295513, -0.3241965, 0.3028103

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0250376, 0.1953700, 0.0274709, 0.0656589, -0.0906965, 0.1678990
1: -0.0789018, 0.2572541, -0.0460902, 0.1935771, -0.2724789, 0.3033442
2: 0.0034661, 0.3338276, 0.0207678, 0.2596045, -0.2561384, 0.3130598
3: -0.0388725, 0.1523266, -0.0209610, 0.1065546, -0.1454271, 0.1732875
4: 0.0053548, 0.3143979, 0.0222067, 0.2458720, -0.2405172, 0.2921911

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0289529, 0.2032966, -0.1766237, 0.0924022
1: -0.0436111, 0.1676113, -0.0891756, 0.2573025, -0.3009136, 0.2567869
2: 0.0208580, 0.2285865, 0.0028422, 0.3348821, -0.3140242, 0.2257443
3: -0.0223003, 0.0900128, -0.0384304, 0.1591823, -0.1814826, 0.1284432
4: 0.0212943, 0.2158635, 0.0068889, 0.3157227, -0.2944283, 0.2089746

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0031947, 0.0998261, -0.0731532, 0.0666441
1: -0.0436111, 0.1676113, -0.0681015, 0.2029338, -0.2465449, 0.2357128
2: 0.0208580, 0.2285865, 0.0107180, 0.2719691, -0.2511111, 0.2178685
3: -0.0223003, 0.0900128, -0.0302884, 0.1220847, -0.1443850, 0.1203013
4: 0.0212943, 0.2158635, 0.0149963, 0.2578355, -0.2365412, 0.2008672

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0280041, 0.0562890, -0.0296161, 0.0354453
1: -0.0436111, 0.1676113, -0.0453445, 0.1125276, -0.1561387, 0.2129558
2: 0.0208580, 0.2285865, 0.0212393, 0.1652312, -0.1443732, 0.2073472
3: -0.0223003, 0.0900128, -0.0202954, 0.0641751, -0.0864754, 0.1103082
4: 0.0212943, 0.2158635, 0.0231947, 0.1549279, -0.1336335, 0.1926689

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0316044, 0.0504820, -0.0238091, 0.0318450
1: -0.0436111, 0.1676113, -0.0323029, 0.0746059, -0.1182169, 0.1999142
2: 0.0208580, 0.2285865, 0.0271148, 0.1210361, -0.1001781, 0.2014717
3: -0.0223003, 0.0900128, -0.0187701, 0.0389615, -0.0612618, 0.1087830
4: 0.0212943, 0.2158635, 0.0263256, 0.1127424, -0.0914481, 0.1895379

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0349724, 0.2740249, -0.2473520, 0.0984217
1: -0.0436111, 0.1676113, -0.0981120, 0.3256595, -0.3692706, 0.2657233
2: 0.0208580, 0.2285865, 0.0005621, 0.4139100, -0.3930520, 0.2280244
3: -0.0223003, 0.0900128, -0.0445257, 0.1959200, -0.2182203, 0.1345385
4: 0.0212943, 0.2158635, 0.0021093, 0.3914595, -0.3701651, 0.2137543

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, -0.0030553, 0.1526119, -0.1259390, 0.0665047
1: -0.0436111, 0.1676113, -0.0757312, 0.2676867, -0.3112978, 0.2433425
2: 0.0208580, 0.2285865, 0.0086105, 0.3466829, -0.3258249, 0.2199759
3: -0.0223003, 0.0900128, -0.0342591, 0.1571698, -0.1794701, 0.1242720
4: 0.0212943, 0.2158635, 0.0115876, 0.3295513, -0.3082570, 0.2042759

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0130985, 0.0806730, -0.0540002, 0.0503509
1: -0.0436111, 0.1676113, -0.0605808, 0.2375910, -0.2812020, 0.2281921
2: 0.0208580, 0.2285865, 0.0148670, 0.3093624, -0.2885044, 0.2137195
3: -0.0223003, 0.0900128, -0.0233690, 0.1373056, -0.1596060, 0.1133819
4: 0.0212943, 0.2158635, 0.0186814, 0.2925131, -0.2712188, 0.1971821

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0266729, 0.0634494, 0.0274709, 0.0656589, -0.0389860, 0.0359785
1: -0.0436111, 0.1676113, -0.0460902, 0.1935771, -0.2371882, 0.2137015
2: 0.0208580, 0.2285865, 0.0207678, 0.2596045, -0.2387465, 0.2078187
3: -0.0223003, 0.0900128, -0.0209610, 0.1065546, -0.1288549, 0.1109738
4: 0.0212943, 0.2158635, 0.0222067, 0.2458720, -0.2245777, 0.1936568

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A2_B1_B1_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0031947, 0.0998261, -0.0235850, 0.2017623, -0.2049571, 0.1234111
1: -0.0681015, 0.2029338, -0.0761097, 0.2042865, -0.2723879, 0.2790435
2: 0.0107180, 0.2719691, 0.0041061, 0.2713708, -0.2606528, 0.2678631
3: -0.0302884, 0.1220847, -0.0387292, 0.1246481, -0.1549365, 0.1608139
4: 0.0149963, 0.2578355, 0.0045154, 0.2540526, -0.2390563, 0.2533201

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A2_B1_B1_B1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0030553, 0.1526119, -0.0235850, 0.2017623, -0.2048176, 0.1761969
1: -0.0757312, 0.2676867, -0.0761097, 0.2042865, -0.2800176, 0.3437964
2: 0.0086105, 0.3466829, 0.0041061, 0.2713708, -0.2627603, 0.3425768
3: -0.0342591, 0.1571698, -0.0387292, 0.1246481, -0.1589072, 0.1958990
4: 0.0115876, 0.3295513, 0.0045154, 0.2540526, -0.2424650, 0.3250359

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 31

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.05 + 415.28 = 418.33 seconds
