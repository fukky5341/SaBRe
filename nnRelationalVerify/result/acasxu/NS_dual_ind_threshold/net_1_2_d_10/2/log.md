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
execution time: IAR + RelationalAnalysis = 1.48 + 1.02 = 2.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
time: 0.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.4334733, 2.4970913, -0.5095432, 2.7755899, -3.2090633, 3.0066345
1: -0.5019389, 3.4698267, -0.5611423, 3.8165379, -4.3184767, 4.0309691
2: -1.2600799, 2.3818066, -1.3674926, 2.7016737, -3.9617536, 3.7492990
3: -1.0162834, 2.9394152, -1.1338987, 3.3577619, -4.3740454, 4.0733137
4: -1.5453744, 3.1090739, -1.7360522, 3.4053361, -4.9507103, 4.8451262

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.36 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.32 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.4753543, 2.6776235, -0.5095432, 2.7755899, -3.2509441, 3.1871667
1: -0.5395899, 3.6900327, -0.5611423, 3.8165379, -4.3561277, 4.2511749
2: -1.3205540, 2.6000621, -1.3674926, 2.7016737, -4.0222278, 3.9675546
3: -1.0897965, 3.1838365, -1.1338987, 3.3577619, -4.4475584, 4.3177352
4: -1.6415637, 3.3140776, -1.7360522, 3.4053361, -5.0468998, 5.0501299

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.33 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.13 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.4334733, 2.4970913, -0.4334733, 2.4970913, -2.9305646, 2.9305646
1: -0.5019389, 3.4698267, -0.5019389, 3.4698267, -3.9717655, 3.9717655
2: -1.2600799, 2.3818066, -1.2600799, 2.3818066, -3.6418865, 3.6418865
3: -1.0162834, 2.9394152, -1.0162834, 2.9394152, -3.9556985, 3.9556985
4: -1.5453744, 3.1090739, -1.5453744, 3.1090739, -4.6544485, 4.6544485

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
time: 0.33 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.4334733, 2.4970913, -0.4753543, 2.6776235, -3.1110969, 2.9724455
1: -0.5019389, 3.4698267, -0.5395899, 3.6900327, -4.1919718, 4.0094166
2: -1.2600799, 2.3818066, -1.3205540, 2.6000621, -3.8601420, 3.7023606
3: -1.0162834, 2.9394152, -1.0897965, 3.1838365, -4.2001200, 4.0292120
4: -1.5453744, 3.1090739, -1.6415637, 3.3140776, -4.8594522, 4.7506375

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
time: 0.34 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.4753543, 2.6776235, -0.4334733, 2.4970913, -2.9724455, 3.1110969
1: -0.5395899, 3.6900327, -0.5019389, 3.4698267, -4.0094166, 4.1919718
2: -1.3205540, 2.6000621, -1.2600799, 2.3818066, -3.7023606, 3.8601420
3: -1.0897965, 3.1838365, -1.0162834, 2.9394152, -4.0292120, 4.2001200
4: -1.6415637, 3.3140776, -1.5453744, 3.1090739, -4.7506375, 4.8594522

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.34 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.4753543, 2.6776235, -0.4753543, 2.6776235, -3.1529779, 3.1529779
1: -0.5395899, 3.6900327, -0.5395899, 3.6900327, -4.2296228, 4.2296228
2: -1.3205540, 2.6000621, -1.3205540, 2.6000621, -3.9206161, 3.9206161
3: -1.0897965, 3.1838365, -1.0897965, 3.1838365, -4.2736330, 4.2736330
4: -1.6415637, 3.3140776, -1.6415637, 3.3140776, -4.9556413, 4.9556413

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.15 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7799989
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7800703, upper bound: 2.7801382
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7793012
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.15
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7797239

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4018040, 2.4201565, -0.4334733, 2.4970913, -2.8988953, 2.8536298
1: -0.4905685, 3.3785930, -0.5019389, 3.4698267, -3.9603953, 3.8805318
2: -1.2406874, 2.2999053, -1.2600799, 2.3818066, -3.6224940, 3.5599852
3: -0.9905678, 2.8093987, -1.0162834, 2.9394152, -3.9299831, 3.8256822
4: -1.4657205, 3.0825684, -1.5453744, 3.1090739, -4.5747943, 4.6279430

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3751003, 2.2736418, -0.4334733, 2.4970913, -2.8721917, 2.7071152
1: -0.4616931, 3.1732640, -0.5019389, 3.4698267, -3.9315197, 3.6752028
2: -1.1646090, 2.1681395, -1.2600799, 2.3818066, -3.5464156, 3.4282193
3: -0.9341383, 2.6221490, -1.0162834, 2.9394152, -3.8735535, 3.6384325
4: -1.3658996, 2.9074883, -1.5453744, 3.1090739, -4.4749737, 4.4528627

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4018040, 2.4201565, -0.4753543, 2.6776235, -3.0794275, 2.8955107
1: -0.4905685, 3.3785930, -0.5395899, 3.6900327, -4.1806011, 3.9181828
2: -1.2406874, 2.2999053, -1.3205540, 2.6000621, -3.8407495, 3.6204593
3: -0.9905678, 2.8093987, -1.0897965, 3.1838365, -4.1744041, 3.8991952
4: -1.4657205, 3.0825684, -1.6415637, 3.3140776, -4.7797980, 4.7241321

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7799989
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7799989
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3751003, 2.2736418, -0.4753543, 2.6776235, -3.0527239, 2.7489963
1: -0.4616931, 3.1732640, -0.5395899, 3.6900327, -4.1517258, 3.7128539
2: -1.1646090, 2.1681395, -1.3205540, 2.6000621, -3.7646711, 3.4886935
3: -0.9341383, 2.6221490, -1.0897965, 3.1838365, -4.1179748, 3.7119455
4: -1.3658996, 2.9074883, -1.6415637, 3.3140776, -4.6799774, 4.5490522

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4174619, 2.5599651, -0.4334733, 2.4970913, -2.9145532, 2.9934385
1: -0.5153302, 3.5426629, -0.5019389, 3.4698267, -3.9851570, 4.0446019
2: -1.2751217, 2.4697607, -1.2600799, 2.3818066, -3.6569283, 3.7298405
3: -1.0362854, 2.9608207, -1.0162834, 2.9394152, -3.9757006, 3.9771042
4: -1.5216012, 3.2471507, -1.5453744, 3.1090739, -4.6306753, 4.7925253

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3987406, 2.4483228, -0.4334733, 2.4970913, -2.8958321, 2.8817961
1: -0.4923947, 3.3845534, -0.5019389, 3.4698267, -3.9622214, 3.8864923
2: -1.2141361, 2.3641450, -1.2600799, 2.3818066, -3.5959427, 3.6242249
3: -0.9919410, 2.8182275, -1.0162834, 2.9394152, -3.9313562, 3.8345108
4: -1.4456424, 3.1076982, -1.5453744, 3.1090739, -4.5547161, 4.6530724

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
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

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4174619, 2.5599651, -0.4753543, 2.6776235, -3.0950854, 3.0353193
1: -0.5153302, 3.5426629, -0.5395899, 3.6900327, -4.2053628, 4.0822530
2: -1.2751217, 2.4697607, -1.3205540, 2.6000621, -3.8751838, 3.7903147
3: -1.0362854, 2.9608207, -1.0897965, 3.1838365, -4.2201219, 4.0506172
4: -1.5216012, 3.2471507, -1.6415637, 3.3140776, -4.8356791, 4.8887143

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7793012
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7793012
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3987406, 2.4483228, -0.4753543, 2.6776235, -3.0763640, 2.9236770
1: -0.4923947, 3.3845534, -0.5395899, 3.6900327, -4.1824274, 3.9241433
2: -1.2141361, 2.3641450, -1.3205540, 2.6000621, -3.8141983, 3.6846991
3: -0.9919410, 2.8182275, -1.0897965, 3.1838365, -4.1757774, 3.9080241
4: -1.4456424, 3.1076982, -1.6415637, 3.3140776, -4.7597198, 4.7492619

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.18 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7803453
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7799989
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7799989
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7796476, upper bound: 2.7801382
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7796476
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7799989, upper bound: 2.7800703
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7793012
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7793012
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.18
Output dim: 0, lower bound: -2.7793012, upper bound: 2.7797239

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4018040, 2.4201565, -0.4018040, 2.4201565, -2.8219604, 2.8219604
1: -0.4905685, 3.3785930, -0.4905685, 3.3785930, -3.8691616, 3.8691616
2: -1.2406874, 2.2999053, -1.2406874, 2.2999053, -3.5405927, 3.5405927
3: -0.9905678, 2.8093987, -0.9905678, 2.8093987, -3.7999663, 3.7999663
4: -1.4657205, 3.0825684, -1.4657205, 3.0825684, -4.5482888, 4.5482888

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4018040, 2.4201565, -0.3751003, 2.2736418, -2.6754458, 2.7952569
1: -0.4905685, 3.3785930, -0.4616931, 3.1732640, -3.6638327, 3.8402860
2: -1.2406874, 2.2999053, -1.1646090, 2.1681395, -3.4088268, 3.4645143
3: -0.9905678, 2.8093987, -0.9341383, 2.6221490, -3.6127167, 3.7435369
4: -1.4657205, 3.0825684, -1.3658996, 2.9074883, -4.3732090, 4.4484682

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3751003, 2.2736418, -0.4018040, 2.4201565, -2.7952569, 2.6754458
1: -0.4616931, 3.1732640, -0.4905685, 3.3785930, -3.8402860, 3.6638327
2: -1.1646090, 2.1681395, -1.2406874, 2.2999053, -3.4645143, 3.4088268
3: -0.9341383, 2.6221490, -0.9905678, 2.8093987, -3.7435369, 3.6127167
4: -1.3658996, 2.9074883, -1.4657205, 3.0825684, -4.4484682, 4.3732090

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3751003, 2.2736418, -0.3751003, 2.2736418, -2.6487422, 2.6487422
1: -0.4616931, 3.1732640, -0.4616931, 3.1732640, -3.6349571, 3.6349571
2: -1.1646090, 2.1681395, -1.1646090, 2.1681395, -3.3327484, 3.3327484
3: -0.9341383, 2.6221490, -0.9341383, 2.6221490, -3.5562873, 3.5562873
4: -1.3658996, 2.9074883, -1.3658996, 2.9074883, -4.2733879, 4.2733879

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4018040, 2.4201565, -0.4174619, 2.5599651, -2.9617691, 2.8376184
1: -0.4905685, 3.3785930, -0.5153302, 3.5426629, -4.0332313, 3.8939233
2: -1.2406874, 2.2999053, -1.2751217, 2.4697607, -3.7104480, 3.5750270
3: -0.9905678, 2.8093987, -1.0362854, 2.9608207, -3.9513884, 3.8456841
4: -1.4657205, 3.0825684, -1.5216012, 3.2471507, -4.7128711, 4.6041698

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7799989
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788218
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4018040, 2.4201565, -0.3987406, 2.4483228, -2.8501267, 2.8188972
1: -0.4905685, 3.3785930, -0.4923947, 3.3845534, -3.8751221, 3.8709877
2: -1.2406874, 2.2999053, -1.2141361, 2.3641450, -3.6048324, 3.5140414
3: -0.9905678, 2.8093987, -0.9919410, 2.8182275, -3.8087955, 3.8013396
4: -1.4657205, 3.0825684, -1.4456424, 3.1076982, -4.5734186, 4.5282106

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7799989
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3751003, 2.2736418, -0.4174619, 2.5599651, -2.9350655, 2.6911037
1: -0.4616931, 3.1732640, -0.5153302, 3.5426629, -4.0043559, 3.6885943
2: -1.1646090, 2.1681395, -1.2751217, 2.4697607, -3.6343696, 3.4432611
3: -0.9341383, 2.6221490, -1.0362854, 2.9608207, -3.8949590, 3.6584344
4: -1.3658996, 2.9074883, -1.5216012, 3.2471507, -4.6130505, 4.4290895

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7788109
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3751003, 2.2736418, -0.3987406, 2.4483228, -2.8234231, 2.6723824
1: -0.4616931, 3.1732640, -0.4923947, 3.3845534, -3.8462465, 3.6656587
2: -1.1646090, 2.1681395, -1.2141361, 2.3641450, -3.5287540, 3.3822756
3: -0.9341383, 2.6221490, -0.9919410, 2.8182275, -3.7523658, 3.6140900
4: -1.3658996, 2.9074883, -1.4456424, 3.1076982, -4.4735975, 4.3531308

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4174619, 2.5599651, -0.4018040, 2.4201565, -2.8376184, 2.9617691
1: -0.5153302, 3.5426629, -0.4905685, 3.3785930, -3.8939233, 4.0332313
2: -1.2751217, 2.4697607, -1.2406874, 2.2999053, -3.5750270, 3.7104480
3: -1.0362854, 2.9608207, -0.9905678, 2.8093987, -3.8456841, 3.9513884
4: -1.5216012, 3.2471507, -1.4657205, 3.0825684, -4.6041698, 4.7128711

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4174619, 2.5599651, -0.3751003, 2.2736418, -2.6911037, 2.9350655
1: -0.5153302, 3.5426629, -0.4616931, 3.1732640, -3.6885943, 4.0043559
2: -1.2751217, 2.4697607, -1.1646090, 2.1681395, -3.4432611, 3.6343696
3: -1.0362854, 2.9608207, -0.9341383, 2.6221490, -3.6584344, 3.8949590
4: -1.5216012, 3.2471507, -1.3658996, 2.9074883, -4.4290895, 4.6130505

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7771481
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3987406, 2.4483228, -0.4018040, 2.4201565, -2.8188972, 2.8501267
1: -0.4923947, 3.3845534, -0.4905685, 3.3785930, -3.8709877, 3.8751221
2: -1.2141361, 2.3641450, -1.2406874, 2.2999053, -3.5140414, 3.6048324
3: -0.9919410, 2.8182275, -0.9905678, 2.8093987, -3.8013396, 3.8087955
4: -1.4456424, 3.1076982, -1.4657205, 3.0825684, -4.5282106, 4.5734186

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
time: 0.37 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3987406, 2.4483228, -0.3751003, 2.2736418, -2.6723824, 2.8234231
1: -0.4923947, 3.3845534, -0.4616931, 3.1732640, -3.6656587, 3.8462465
2: -1.2141361, 2.3641450, -1.1646090, 2.1681395, -3.3822756, 3.5287540
3: -0.9919410, 2.8182275, -0.9341383, 2.6221490, -3.6140900, 3.7523658
4: -1.4456424, 3.1076982, -1.3658996, 2.9074883, -4.3531308, 4.4735975

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
time: 0.41 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4174619, 2.5599651, -0.4174619, 2.5599651, -2.9774270, 2.9774270
1: -0.5153302, 3.5426629, -0.5153302, 3.5426629, -4.0579929, 4.0579929
2: -1.2751217, 2.4697607, -1.2751217, 2.4697607, -3.7448823, 3.7448823
3: -1.0362854, 2.9608207, -1.0362854, 2.9608207, -3.9971061, 3.9971061
4: -1.5216012, 3.2471507, -1.5216012, 3.2471507, -4.7687521, 4.7687521

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7793012
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
time: 0.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4174619, 2.5599651, -0.3987406, 2.4483228, -2.8657846, 2.9587059
1: -0.5153302, 3.5426629, -0.4923947, 3.3845534, -3.8998837, 4.0350575
2: -1.2751217, 2.4697607, -1.2141361, 2.3641450, -3.6392667, 3.6838968
3: -1.0362854, 2.9608207, -0.9919410, 2.8182275, -3.8545129, 3.9527617
4: -1.5216012, 3.2471507, -1.4456424, 3.1076982, -4.6292992, 4.6927929

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7793012
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3987406, 2.4483228, -0.4174619, 2.5599651, -2.9587059, 2.8657846
1: -0.4923947, 3.3845534, -0.5153302, 3.5426629, -4.0350575, 3.8998837
2: -1.2141361, 2.3641450, -1.2751217, 2.4697607, -3.6838968, 3.6392667
3: -0.9919410, 2.8182275, -1.0362854, 2.9608207, -3.9527617, 3.8545129
4: -1.4456424, 3.1076982, -1.5216012, 3.2471507, -4.6927929, 4.6292992

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3987406, 2.4483228, -0.3987406, 2.4483228, -2.8470635, 2.8470635
1: -0.4923947, 3.3845534, -0.4923947, 3.3845534, -3.8769481, 3.8769481
2: -1.2141361, 2.3641450, -1.2141361, 2.3641450, -3.5782812, 3.5782812
3: -0.9919410, 2.8182275, -0.9919410, 2.8182275, -3.8101685, 3.8101685
4: -1.4456424, 3.1076982, -1.4456424, 3.1076982, -4.5533404, 4.5533404

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7768629
time: 0.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.33 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7803453
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7778488, upper bound: 2.7804846
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7799989
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788218
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7799989
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7788109
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7801382
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7796476
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7771481
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7793012
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7793012
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7797239
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7768629

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.4018040, 2.4201565, -2.8097630, 2.7723832
1: -0.4801092, 3.3123057, -0.4905685, 3.3785930, -3.8587022, 3.8028741
2: -1.2162256, 2.2505226, -1.2406874, 2.2999053, -3.5161309, 3.4912100
3: -0.9699728, 2.7406662, -0.9905678, 2.8093987, -3.7793715, 3.7312341
4: -1.4223459, 3.0282202, -1.4657205, 3.0825684, -4.5049143, 4.4939408

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7773000
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7791682
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.4018040, 2.4201565, -2.8175697, 2.7951903
1: -0.4861144, 3.3476477, -0.4905685, 3.3785930, -3.8647075, 3.8382163
2: -1.2313797, 2.2667561, -1.2406874, 2.2999053, -3.5312850, 3.5074434
3: -0.9823158, 2.7700262, -0.9905678, 2.8093987, -3.7917144, 3.7605939
4: -1.4410114, 3.0509758, -1.4657205, 3.0825684, -4.5235796, 4.5166965

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7773000
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791682
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3751003, 2.2736418, -2.6632483, 2.7456796
1: -0.4801092, 3.3123057, -0.4616931, 3.1732640, -3.6533732, 3.7739987
2: -1.2162256, 2.2505226, -1.1646090, 2.1681395, -3.3843651, 3.4151316
3: -0.9699728, 2.7406662, -0.9341383, 2.6221490, -3.5921218, 3.6748044
4: -1.4223459, 3.0282202, -1.3658996, 2.9074883, -4.3298340, 4.3941197

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7778597
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3751003, 2.2736418, -2.6710553, 2.7684867
1: -0.4861144, 3.3476477, -0.4616931, 3.1732640, -3.6593785, 3.8093407
2: -1.2313797, 2.2667561, -1.1646090, 2.1681395, -3.3995192, 3.4313650
3: -0.9823158, 2.7700262, -0.9341383, 2.6221490, -3.6044648, 3.7041645
4: -1.4410114, 3.0509758, -1.3658996, 2.9074883, -4.3484998, 4.4168754

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7778597
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.4018040, 2.4201565, -2.7832375, 2.6270947
1: -0.4512913, 3.1082866, -0.4905685, 3.3785930, -3.8298843, 3.5988550
2: -1.1403358, 2.1198447, -1.2406874, 2.2999053, -3.4402411, 3.3605320
3: -0.9133595, 2.5548997, -0.9905678, 2.8093987, -3.7227583, 3.5454674
4: -1.3233099, 2.8539722, -1.4657205, 3.0825684, -4.4058781, 4.3196926

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7772892
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7791573
time: 0.33 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.4018040, 2.4201565, -2.7911177, 2.6521373
1: -0.4575120, 3.1468184, -0.4905685, 3.3785930, -3.8361049, 3.6373868
2: -1.1562662, 2.1376271, -1.2406874, 2.2999053, -3.4561715, 3.3783145
3: -0.9258730, 2.5864434, -0.9905678, 2.8093987, -3.7352717, 3.5770111
4: -1.3418519, 2.8791902, -1.4657205, 3.0825684, -4.4244204, 4.3449106

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7772892
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791573
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3751003, 2.2736418, -2.6367228, 2.6003911
1: -0.4512913, 3.1082866, -0.4616931, 3.1732640, -3.6245553, 3.5699797
2: -1.1403358, 2.1198447, -1.1646090, 2.1681395, -3.3084753, 3.2844536
3: -0.9133595, 2.5548997, -0.9341383, 2.6221490, -3.5355086, 3.4890380
4: -1.3233099, 2.8539722, -1.3658996, 2.9074883, -4.2307982, 4.2198715

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7772892
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791573
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3751003, 2.2736418, -2.6446030, 2.6254337
1: -0.4575120, 3.1468184, -0.4616931, 3.1732640, -3.6307759, 3.6085114
2: -1.1562662, 2.1376271, -1.1646090, 2.1681395, -3.3244057, 3.3022361
3: -0.9258730, 2.5864434, -0.9341383, 2.6221490, -3.5480220, 3.5205817
4: -1.3418519, 2.8791902, -1.3658996, 2.9074883, -4.2493401, 4.2450895

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7772892
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.4174619, 2.5599651, -2.9495716, 2.7880411
1: -0.4801092, 3.3123057, -0.5153302, 3.5426629, -4.0227718, 3.8276358
2: -1.2162256, 2.2505226, -1.2751217, 2.4697607, -3.6859863, 3.5256443
3: -0.9699728, 2.7406662, -1.0362854, 2.9608207, -3.9307935, 3.7769516
4: -1.4223459, 3.0282202, -1.5216012, 3.2471507, -4.6694965, 4.5498214

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7773018
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.4174619, 2.5599651, -2.9573784, 2.8108482
1: -0.4861144, 3.3476477, -0.5153302, 3.5426629, -4.0287771, 3.8629780
2: -1.2313797, 2.2667561, -1.2751217, 2.4697607, -3.7011404, 3.5418777
3: -0.9823158, 2.7700262, -1.0362854, 2.9608207, -3.9431365, 3.8063116
4: -1.4410114, 3.0509758, -1.5216012, 3.2471507, -4.6881618, 4.5725770

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7773018
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788218
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3987406, 2.4483228, -2.8379292, 2.7693200
1: -0.4801092, 3.3123057, -0.4923947, 3.3845534, -3.8646626, 3.8047004
2: -1.2162256, 2.2505226, -1.2141361, 2.3641450, -3.5803707, 3.4646587
3: -0.9699728, 2.7406662, -0.9919410, 2.8182275, -3.7882004, 3.7326071
4: -1.4223459, 3.0282202, -1.4456424, 3.1076982, -4.5300441, 4.4738626

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7778969
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7788218
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3987406, 2.4483228, -2.8457360, 2.7921271
1: -0.4861144, 3.3476477, -0.4923947, 3.3845534, -3.8706679, 3.8400424
2: -1.2313797, 2.2667561, -1.2141361, 2.3641450, -3.5955248, 3.4808922
3: -0.9823158, 2.7700262, -0.9919410, 2.8182275, -3.8005433, 3.7619672
4: -1.4410114, 3.0509758, -1.4456424, 3.1076982, -4.5487099, 4.4966183

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772093, upper bound: 2.7778969
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772093, upper bound: 2.7788218
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.4174619, 2.5599651, -2.9230461, 2.6427526
1: -0.4512913, 3.1082866, -0.5153302, 3.5426629, -3.9939542, 3.6236167
2: -1.1403358, 2.1198447, -1.2751217, 2.4697607, -3.6100965, 3.3949664
3: -0.9133595, 2.5548997, -1.0362854, 2.9608207, -3.8741803, 3.5911851
4: -1.3233099, 2.8539722, -1.5216012, 3.2471507, -4.5704603, 4.3755732

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7772910
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7788109
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.4174619, 2.5599651, -2.9309263, 2.6677952
1: -0.4575120, 3.1468184, -0.5153302, 3.5426629, -4.0001750, 3.6621485
2: -1.1562662, 2.1376271, -1.2751217, 2.4697607, -3.6260269, 3.4127488
3: -0.9258730, 2.5864434, -1.0362854, 2.9608207, -3.8866937, 3.6227288
4: -1.3418519, 2.8791902, -1.5216012, 3.2471507, -4.5890026, 4.4007912

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3987406, 2.4483228, -2.8114038, 2.6240315
1: -0.4512913, 3.1082866, -0.4923947, 3.3845534, -3.8358448, 3.6006813
2: -1.1403358, 2.1198447, -1.2141361, 2.3641450, -3.5044808, 3.3339808
3: -0.9133595, 2.5548997, -0.9919410, 2.8182275, -3.7315869, 3.5468407
4: -1.3233099, 2.8539722, -1.4456424, 3.1076982, -4.4310083, 4.2996144

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759008, upper bound: 2.7772910
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759008, upper bound: 2.7788109
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3987406, 2.4483228, -2.8192840, 2.6490741
1: -0.4575120, 3.1468184, -0.4923947, 3.3845534, -3.8420653, 3.6392131
2: -1.1562662, 2.1376271, -1.2141361, 2.3641450, -3.5204113, 3.3517632
3: -0.9258730, 2.5864434, -0.9919410, 2.8182275, -3.7441006, 3.5783844
4: -1.3418519, 2.8791902, -1.4456424, 3.1076982, -4.4495502, 4.3248324

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.4018040, 2.4201565, -2.8257761, 2.9158249
1: -0.5055367, 3.4809954, -0.4905685, 3.3785930, -3.8841295, 3.9715638
2: -1.2521493, 2.4233136, -1.2406874, 2.2999053, -3.5520546, 3.6640010
3: -1.0172695, 2.8950694, -0.9905678, 2.8093987, -3.8266683, 3.8856373
4: -1.4811087, 3.1972914, -1.4657205, 3.0825684, -4.5636768, 4.6630120

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7752799
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7771481
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.4018040, 2.4201565, -2.8321061, 2.9269733
1: -0.5089593, 3.5015557, -0.4905685, 3.3785930, -3.8875523, 3.9921241
2: -1.2644246, 2.4245837, -1.2406874, 2.2999053, -3.5643299, 3.6652710
3: -1.0243788, 2.9105000, -0.9905678, 2.8093987, -3.8337774, 3.9010677
4: -1.4927979, 3.2066495, -1.4657205, 3.0825684, -4.5753660, 4.6723700

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7752799
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7771481
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3751003, 2.2736418, -2.6792614, 2.8891213
1: -0.5055367, 3.4809954, -0.4616931, 3.1732640, -3.6788006, 3.9426885
2: -1.2521493, 2.4233136, -1.1646090, 2.1681395, -3.4202888, 3.5879226
3: -1.0172695, 2.8950694, -0.9341383, 2.6221490, -3.6394186, 3.8292077
4: -1.4811087, 3.1972914, -1.3658996, 2.9074883, -4.3885970, 4.5631909

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7758395
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7771481
time: 0.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3751003, 2.2736418, -2.6855912, 2.9002697
1: -0.5089593, 3.5015557, -0.4616931, 3.1732640, -3.6822233, 3.9632487
2: -1.2644246, 2.4245837, -1.1646090, 2.1681395, -3.4325640, 3.5891926
3: -1.0243788, 2.9105000, -0.9341383, 2.6221490, -3.6465278, 3.8446383
4: -1.4927979, 3.2066495, -1.3658996, 2.9074883, -4.4002862, 4.5725489

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7758395
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.4018040, 2.4201565, -2.8059001, 2.8036869
1: -0.4820945, 3.3220072, -0.4905685, 3.3785930, -3.8606875, 3.8125758
2: -1.1904538, 2.3187947, -1.2406874, 2.2999053, -3.4903591, 3.5594821
3: -0.9715070, 2.7528343, -0.9905678, 2.8093987, -3.7809057, 3.7434020
4: -1.4044091, 3.0569649, -1.4657205, 3.0825684, -4.4869776, 4.5226855

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7753411
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7772093
time: 0.39 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.4018040, 2.4201565, -2.8081899, 2.8200445
1: -0.4848027, 3.3492410, -0.4905685, 3.3785930, -3.8633957, 3.8398094
2: -1.2020583, 2.3271937, -1.2406874, 2.2999053, -3.5019636, 3.5678811
3: -0.9766790, 2.7750435, -0.9905678, 2.8093987, -3.7860775, 3.7656112
4: -1.4156674, 3.0728226, -1.4657205, 3.0825684, -4.4982357, 4.5385432

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7753411
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7772093
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3751003, 2.2736418, -2.6593854, 2.7769833
1: -0.4820945, 3.3220072, -0.4616931, 3.1732640, -3.6553586, 3.7837002
2: -1.1904538, 2.3187947, -1.1646090, 2.1681395, -3.3585932, 3.4834037
3: -0.9715070, 2.7528343, -0.9341383, 2.6221490, -3.5936561, 3.6869726
4: -1.4044091, 3.0569649, -1.3658996, 2.9074883, -4.3118973, 4.4228644

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7757081
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7772093
time: 0.39 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3751003, 2.2736418, -2.6616752, 2.7933409
1: -0.4848027, 3.3492410, -0.4616931, 3.1732640, -3.6580667, 3.8109341
2: -1.2020583, 2.3271937, -1.1646090, 2.1681395, -3.3701978, 3.4918027
3: -0.9766790, 2.7750435, -0.9341383, 2.6221490, -3.5988278, 3.7091818
4: -1.4156674, 3.0728226, -1.3658996, 2.9074883, -4.3231559, 4.4387221

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
time: 0.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.4174619, 2.5599651, -2.9655848, 2.9314828
1: -0.5055367, 3.4809954, -0.5153302, 3.5426629, -4.0481997, 3.9963255
2: -1.2521493, 2.4233136, -1.2751217, 2.4697607, -3.7219100, 3.6984353
3: -1.0172695, 2.8950694, -1.0362854, 2.9608207, -3.9780903, 3.9313548
4: -1.4811087, 3.1972914, -1.5216012, 3.2471507, -4.7282591, 4.7188926

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7752799
time: 0.40 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.4174619, 2.5599651, -2.9719148, 2.9426312
1: -0.5089593, 3.5015557, -0.5153302, 3.5426629, -4.0516224, 4.0168858
2: -1.2644246, 2.4245837, -1.2751217, 2.4697607, -3.7341852, 3.6997054
3: -1.0243788, 2.9105000, -1.0362854, 2.9608207, -3.9851995, 3.9467854
4: -1.4927979, 3.2066495, -1.5216012, 3.2471507, -4.7399483, 4.7282505

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7752799
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768017
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3987406, 2.4483228, -2.8539424, 2.9127617
1: -0.5055367, 3.4809954, -0.4923947, 3.3845534, -3.8900900, 3.9733901
2: -1.2521493, 2.4233136, -1.2141361, 2.3641450, -3.6162944, 3.6374497
3: -1.0172695, 2.8950694, -0.9919410, 2.8182275, -3.8354969, 3.8870103
4: -1.4811087, 3.1972914, -1.4456424, 3.1076982, -4.5888071, 4.6429338

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7758395
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7768017
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3987406, 2.4483228, -2.8602724, 2.9239101
1: -0.5089593, 3.5015557, -0.4923947, 3.3845534, -3.8935127, 3.9939504
2: -1.2644246, 2.4245837, -1.2141361, 2.3641450, -3.6285696, 3.6387198
3: -1.0243788, 2.9105000, -0.9919410, 2.8182275, -3.8426063, 3.9024410
4: -1.4927979, 3.2066495, -1.4456424, 3.1076982, -4.6004963, 4.6522918

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7758395
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7768017
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.4174619, 2.5599651, -2.9457088, 2.8193448
1: -0.4820945, 3.3220072, -0.5153302, 3.5426629, -4.0247574, 3.8373375
2: -1.1904538, 2.3187947, -1.2751217, 2.4697607, -3.6602144, 3.5939164
3: -0.9715070, 2.7528343, -1.0362854, 2.9608207, -3.9323277, 3.7891197
4: -1.4044091, 3.0569649, -1.5216012, 3.2471507, -4.6515598, 4.5785661

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7753411
time: 0.40 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.4174619, 2.5599651, -2.9479985, 2.8357024
1: -0.4848027, 3.3492410, -0.5153302, 3.5426629, -4.0274653, 3.8645711
2: -1.2020583, 2.3271937, -1.2751217, 2.4697607, -3.6718190, 3.6023154
3: -0.9766790, 2.7750435, -1.0362854, 2.9608207, -3.9374995, 3.8113289
4: -1.4156674, 3.0728226, -1.5216012, 3.2471507, -4.6628180, 4.5944238

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7753411
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
time: 0.42 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3987406, 2.4483228, -2.8340664, 2.8006234
1: -0.4820945, 3.3220072, -0.4923947, 3.3845534, -3.8666480, 3.8144019
2: -1.1904538, 2.3187947, -1.2141361, 2.3641450, -3.5545988, 3.5329309
3: -0.9715070, 2.7528343, -0.9919410, 2.8182275, -3.7897344, 3.7447753
4: -1.4044091, 3.0569649, -1.4456424, 3.1076982, -4.5121074, 4.5026073

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759380, upper bound: 2.7757081
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759380, upper bound: 2.7768629
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3987406, 2.4483228, -2.8363562, 2.8169813
1: -0.4848027, 3.3492410, -0.4923947, 3.3845534, -3.8693562, 3.8416357
2: -1.2020583, 2.3271937, -1.2141361, 2.3641450, -3.5662034, 3.5413299
3: -0.9766790, 2.7750435, -0.9919410, 2.8182275, -3.7949066, 3.7669845
4: -1.4156674, 3.0728226, -1.4456424, 3.1076982, -4.5233655, 4.5184650

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7757081
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
time: 0.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.40 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7773000
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7773000, upper bound: 2.7791682
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7773000
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791682
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7778597
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791682
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7778597
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791682
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7772892
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7778597, upper bound: 2.7791573
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7772892
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791682, upper bound: 2.7791573
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7772892
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772892, upper bound: 2.7791573
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7772892
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7791573, upper bound: 2.7791573
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7773018
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7752799, upper bound: 2.7788218
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7773018
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788218
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7778969
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7753411, upper bound: 2.7788218
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772093, upper bound: 2.7778969
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772093, upper bound: 2.7788218
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7772910
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7758395, upper bound: 2.7788109
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7759008, upper bound: 2.7772910
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7759008, upper bound: 2.7788109
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7772910
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7771481, upper bound: 2.7788109
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7752799
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7773018, upper bound: 2.7771481
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7752799
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7771481
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7758395
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7772910, upper bound: 2.7771481
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7758395
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7771481
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7753411
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7778969, upper bound: 2.7772093
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7753411
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788218, upper bound: 2.7772093
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7757081
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7772093
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7772093
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7752799
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7752817, upper bound: 2.7768017
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7752799
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768017
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7758395
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7753429, upper bound: 2.7768017
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7758395
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7768017
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7753411
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7758768, upper bound: 2.7768629
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7753411
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7759380, upper bound: 2.7757081
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7759380, upper bound: 2.7768629
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7757081
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.40
Output dim: 0, lower bound: -2.7768017, upper bound: 2.7768629

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3896064, 2.3705792, -2.7601857, 2.7601857
1: -0.4801092, 3.3123057, -0.4801092, 3.3123057, -3.7924149, 3.7924149
2: -1.2162256, 2.2505226, -1.2162256, 2.2505226, -3.4667482, 3.4667482
3: -0.9699728, 2.7406662, -0.9699728, 2.7406662, -3.7106390, 3.7106390
4: -1.4223459, 3.0282202, -1.4223459, 3.0282202, -4.4505663, 4.4505663

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7765046
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7742238
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3974134, 2.3933864, -2.7829928, 2.7679925
1: -0.4801092, 3.3123057, -0.4861144, 3.3476477, -3.8277569, 3.7984202
2: -1.2162256, 2.2505226, -1.2313797, 2.2667561, -3.4829817, 3.4819024
3: -0.9699728, 2.7406662, -0.9823158, 2.7700262, -3.7399991, 3.7229819
4: -1.4223459, 3.0282202, -1.4410114, 3.0509758, -4.4733219, 4.4692316

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7777688
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3896064, 2.3705792, -2.7679925, 2.7829928
1: -0.4861144, 3.3476477, -0.4801092, 3.3123057, -3.7984202, 3.8277569
2: -1.2313797, 2.2667561, -1.2162256, 2.2505226, -3.4819024, 3.4829817
3: -0.9823158, 2.7700262, -0.9699728, 2.7406662, -3.7229819, 3.7399991
4: -1.4410114, 3.0509758, -1.4223459, 3.0282202, -4.4692316, 4.4733219

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7747670
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7726478
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3974134, 2.3933864, -2.7907996, 2.7907996
1: -0.4861144, 3.3476477, -0.4861144, 3.3476477, -3.8337622, 3.8337622
2: -1.2313797, 2.2667561, -1.2313797, 2.2667561, -3.4981358, 3.4981358
3: -0.9823158, 2.7700262, -0.9823158, 2.7700262, -3.7523420, 3.7523420
4: -1.4410114, 3.0509758, -1.4410114, 3.0509758, -4.4919872, 4.4919872

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7750910
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3630810, 2.2252908, -2.6148973, 2.7336602
1: -0.4801092, 3.3123057, -0.4512913, 3.1082866, -3.5883958, 3.7635970
2: -1.2162256, 2.2505226, -1.1403358, 2.1198447, -3.3360703, 3.3908584
3: -0.9699728, 2.7406662, -0.9133595, 2.5548997, -3.5248725, 3.6540256
4: -1.4223459, 3.0282202, -1.3233099, 2.8539722, -4.2763181, 4.3515301

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7789359
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3709612, 2.2503333, -2.6399398, 2.7415404
1: -0.4801092, 3.3123057, -0.4575120, 3.1468184, -3.6269276, 3.7698176
2: -1.2162256, 2.2505226, -1.1562662, 2.1376271, -3.3538527, 3.4067888
3: -0.9699728, 2.7406662, -0.9258730, 2.5864434, -3.5564163, 3.6665392
4: -1.4223459, 3.0282202, -1.3418519, 2.8791902, -4.3015361, 4.3700724

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7800277
time: 0.34 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3630810, 2.2252908, -2.6227040, 2.7564673
1: -0.4861144, 3.3476477, -0.4512913, 3.1082866, -3.5944011, 3.7989390
2: -1.2313797, 2.2667561, -1.1403358, 2.1198447, -3.3512244, 3.4070919
3: -0.9823158, 2.7700262, -0.9133595, 2.5548997, -3.5372155, 3.6833858
4: -1.4410114, 3.0509758, -1.3233099, 2.8539722, -4.2949839, 4.3742857

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786091, upper bound: 2.7777588
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3709612, 2.2503333, -2.6477466, 2.7643476
1: -0.4861144, 3.3476477, -0.4575120, 3.1468184, -3.6329329, 3.8051596
2: -1.2313797, 2.2667561, -1.1562662, 2.1376271, -3.3690069, 3.4230223
3: -0.9823158, 2.7700262, -0.9258730, 2.5864434, -3.5687592, 3.6958992
4: -1.4410114, 3.0509758, -1.3418519, 2.8791902, -4.3202019, 4.3928280

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786091, upper bound: 2.7779473
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3896064, 2.3705792, -2.7336602, 2.6148973
1: -0.4512913, 3.1082866, -0.4801092, 3.3123057, -3.7635970, 3.5883958
2: -1.1403358, 2.1198447, -1.2162256, 2.2505226, -3.3908584, 3.3360703
3: -0.9133595, 2.5548997, -0.9699728, 2.7406662, -3.6540256, 3.5248725
4: -1.3233099, 2.8539722, -1.4223459, 3.0282202, -4.3515301, 4.2763181

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745329, upper bound: 2.7766235
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3974134, 2.3933864, -2.7564673, 2.6227040
1: -0.4512913, 3.1082866, -0.4861144, 3.3476477, -3.7989390, 3.5944011
2: -1.1403358, 2.1198447, -1.2313797, 2.2667561, -3.4070919, 3.3512244
3: -0.9133595, 2.5548997, -0.9823158, 2.7700262, -3.6833858, 3.5372155
4: -1.3233099, 2.8539722, -1.4410114, 3.0509758, -4.3742857, 4.2949839

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7745329, upper bound: 2.7778879
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3896064, 2.3705792, -2.7415404, 2.6399398
1: -0.4575120, 3.1468184, -0.4801092, 3.3123057, -3.7698176, 3.6269276
2: -1.1562662, 2.1376271, -1.2162256, 2.2505226, -3.4067888, 3.3538527
3: -0.9258730, 2.5864434, -0.9699728, 2.7406662, -3.6665392, 3.5564163
4: -1.3418519, 2.8791902, -1.4223459, 3.0282202, -4.3700724, 4.3015361

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7747084
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3974134, 2.3933864, -2.7643476, 2.6477466
1: -0.4575120, 3.1468184, -0.4861144, 3.3476477, -3.8051596, 3.6329329
2: -1.1562662, 2.1376271, -1.2313797, 2.2667561, -3.4230223, 3.3690069
3: -0.9258730, 2.5864434, -0.9823158, 2.7700262, -3.6958992, 3.5687592
4: -1.3418519, 2.8791902, -1.4410114, 3.0509758, -4.3928280, 4.3202019

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7750266
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3630810, 2.2252908, -2.5883718, 2.5883718
1: -0.4512913, 3.1082866, -0.4512913, 3.1082866, -3.5595779, 3.5595779
2: -1.1403358, 2.1198447, -1.1403358, 2.1198447, -3.2601805, 3.2601805
3: -0.9133595, 2.5548997, -0.9133595, 2.5548997, -3.4682593, 3.4682593
4: -1.3233099, 2.8539722, -1.3233099, 2.8539722, -4.1772823, 4.1772823

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7786973
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7787504
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3709612, 2.2503333, -2.6134143, 2.5962520
1: -0.4512913, 3.1082866, -0.4575120, 3.1468184, -3.5981097, 3.5657985
2: -1.1403358, 2.1198447, -1.1562662, 2.1376271, -3.2779629, 3.2761109
3: -0.9133595, 2.5548997, -0.9258730, 2.5864434, -3.4998031, 3.4807727
4: -1.3233099, 2.8539722, -1.3418519, 2.8791902, -4.2025003, 4.1958241

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7801671
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7801671
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3630810, 2.2252908, -2.5962520, 2.6134143
1: -0.4575120, 3.1468184, -0.4512913, 3.1082866, -3.5657985, 3.5981097
2: -1.1562662, 2.1376271, -1.1403358, 2.1198447, -3.2761109, 3.2779629
3: -0.9258730, 2.5864434, -0.9133595, 2.5548997, -3.4807727, 3.4998031
4: -1.3418519, 2.8791902, -1.3233099, 2.8539722, -4.1958241, 4.2025003

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7770552
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7769808
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3709612, 2.2503333, -2.6212945, 2.6212945
1: -0.4575120, 3.1468184, -0.4575120, 3.1468184, -3.6043303, 3.6043303
2: -1.1562662, 2.1376271, -1.1562662, 2.1376271, -3.2938933, 3.2938933
3: -0.9258730, 2.5864434, -0.9258730, 2.5864434, -3.5123165, 3.5123165
4: -1.3418519, 2.8791902, -1.3418519, 2.8791902, -4.2210422, 4.2210422

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7778307
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7777063
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.4056197, 2.5140209, -2.9036274, 2.7761989
1: -0.4801092, 3.3123057, -0.5055367, 3.4809954, -3.9611046, 3.8178425
2: -1.2162256, 2.2505226, -1.2521493, 2.4233136, -3.6395392, 3.5026720
3: -0.9699728, 2.7406662, -1.0172695, 2.8950694, -3.8650422, 3.7579355
4: -1.4223459, 3.0282202, -1.4811087, 3.1972914, -4.6196375, 4.5093288

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741680, upper bound: 2.7766659
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7744278
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.4119495, 2.5251694, -2.9147758, 2.7825289
1: -0.4801092, 3.3123057, -0.5089593, 3.5015557, -3.9816649, 3.8212650
2: -1.2162256, 2.2505226, -1.2644246, 2.4245837, -3.6408093, 3.5149472
3: -0.9699728, 2.7406662, -1.0243788, 2.9105000, -3.8804729, 3.7650449
4: -1.4223459, 3.0282202, -1.4927979, 3.2066495, -4.6289954, 4.5210180

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741680, upper bound: 2.7792876
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7770054
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.4056197, 2.5140209, -2.9114342, 2.7990060
1: -0.4861144, 3.3476477, -0.5055367, 3.4809954, -3.9671099, 3.8531842
2: -1.2313797, 2.2667561, -1.2521493, 2.4233136, -3.6546934, 3.5189054
3: -0.9823158, 2.7700262, -1.0172695, 2.8950694, -3.8773851, 3.7872958
4: -1.4410114, 3.0509758, -1.4811087, 3.1972914, -4.6383028, 4.5320845

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741680, upper bound: 2.7748819
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7728518
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.4119495, 2.5251694, -2.9225826, 2.8053360
1: -0.4861144, 3.3476477, -0.5089593, 3.5015557, -3.9876702, 3.8566070
2: -1.2313797, 2.2667561, -1.2644246, 2.4245837, -3.6559634, 3.5311806
3: -0.9823158, 2.7700262, -1.0243788, 2.9105000, -3.8928158, 3.7944050
4: -1.4410114, 3.0509758, -1.4927979, 3.2066495, -4.6476612, 4.5437737

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759808, upper bound: 2.7761403
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7740142
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3857437, 2.4018829, -2.7914894, 2.7563229
1: -0.4801092, 3.3123057, -0.4820945, 3.3220072, -3.8021164, 3.7944002
2: -1.2162256, 2.2505226, -1.1904538, 2.3187947, -3.5350204, 3.4409764
3: -0.9699728, 2.7406662, -0.9715070, 2.7528343, -3.7228072, 3.7121730
4: -1.4223459, 3.0282202, -1.4044091, 3.0569649, -4.4793110, 4.4326291

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3896064, 2.3705792, -0.3880334, 2.4182405, -2.8078470, 2.7586126
1: -0.4801092, 3.3123057, -0.4848027, 3.3492410, -3.8293502, 3.7971084
2: -1.2162256, 2.2505226, -1.2020583, 2.3271937, -3.5434194, 3.4525809
3: -0.9699728, 2.7406662, -0.9766790, 2.7750435, -3.7450163, 3.7173452
4: -1.4223459, 3.0282202, -1.4156674, 3.0728226, -4.4951687, 4.4438877

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3857437, 2.4018829, -2.7992964, 2.7791300
1: -0.4861144, 3.3476477, -0.4820945, 3.3220072, -3.8081217, 3.8297422
2: -1.2313797, 2.2667561, -1.1904538, 2.3187947, -3.5501745, 3.4572098
3: -0.9823158, 2.7700262, -0.9715070, 2.7528343, -3.7351501, 3.7415333
4: -1.4410114, 3.0509758, -1.4044091, 3.0569649, -4.4979763, 4.4553847

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7647012, upper bound: 2.7647692
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769698, upper bound: 2.7777932
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3974134, 2.3933864, -0.3880334, 2.4182405, -2.8156538, 2.7814198
1: -0.4861144, 3.3476477, -0.4848027, 3.3492410, -3.8353555, 3.8324504
2: -1.2313797, 2.2667561, -1.2020583, 2.3271937, -3.5585735, 3.4688144
3: -0.9823158, 2.7700262, -0.9766790, 2.7750435, -3.7573593, 3.7467051
4: -1.4410114, 3.0509758, -1.4156674, 3.0728226, -4.5138340, 4.4666433

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7647012, upper bound: 2.7647692
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769698, upper bound: 2.7779244
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.4056197, 2.5140209, -2.8771019, 2.6309104
1: -0.4512913, 3.1082866, -0.5055367, 3.4809954, -3.9322867, 3.6138234
2: -1.1403358, 2.1198447, -1.2521493, 2.4233136, -3.5636494, 3.3719940
3: -0.9133595, 2.5548997, -1.0172695, 2.8950694, -3.8084288, 3.5721693
4: -1.3233099, 2.8539722, -1.4811087, 3.1972914, -4.5206013, 4.3350811

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741680, upper bound: 2.7767873
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.4119495, 2.5251694, -2.8882504, 2.6372404
1: -0.4512913, 3.1082866, -0.5089593, 3.5015557, -3.9528470, 3.6172459
2: -1.1403358, 2.1198447, -1.2644246, 2.4245837, -3.5649195, 3.3842692
3: -0.9133595, 2.5548997, -1.0243788, 2.9105000, -3.8238597, 3.5792785
4: -1.3233099, 2.8539722, -1.4927979, 3.2066495, -4.5299597, 4.3467703

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743753, upper bound: 2.7794065
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
time: 0.44 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.4056197, 2.5140209, -2.8849821, 2.6559529
1: -0.4575120, 3.1468184, -0.5055367, 3.4809954, -3.9385073, 3.6523552
2: -1.1562662, 2.1376271, -1.2521493, 2.4233136, -3.5795798, 3.3897765
3: -0.9258730, 2.5864434, -1.0172695, 2.8950694, -3.8209424, 3.6037130
4: -1.3418519, 2.8791902, -1.4811087, 3.1972914, -4.5391436, 4.3602991

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759765, upper bound: 2.7748129
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.4119495, 2.5251694, -2.8961306, 2.6622829
1: -0.4575120, 3.1468184, -0.5089593, 3.5015557, -3.9590676, 3.6557777
2: -1.1562662, 2.1376271, -1.2644246, 2.4245837, -3.5808499, 3.4020517
3: -0.9258730, 2.5864434, -1.0243788, 2.9105000, -3.8363731, 3.6108222
4: -1.3418519, 2.8791902, -1.4927979, 3.2066495, -4.5485015, 4.3719883

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743753, upper bound: 2.7760452
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3857437, 2.4018829, -2.7649639, 2.6110344
1: -0.4512913, 3.1082866, -0.4820945, 3.3220072, -3.7732985, 3.5903811
2: -1.1403358, 2.1198447, -1.1904538, 2.3187947, -3.4591305, 3.3102984
3: -0.9133595, 2.5548997, -0.9715070, 2.7528343, -3.6661940, 3.5264068
4: -1.3233099, 2.8539722, -1.4044091, 3.0569649, -4.3802748, 4.2583814

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608479, upper bound: 2.7670756
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7786973
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7787504
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3630810, 2.2252908, -0.3880334, 2.4182405, -2.7813215, 2.6133242
1: -0.4512913, 3.1082866, -0.4848027, 3.3492410, -3.8005323, 3.5930893
2: -1.1403358, 2.1198447, -1.2020583, 2.3271937, -3.4675295, 3.3219030
3: -0.9133595, 2.5548997, -0.9766790, 2.7750435, -3.6884031, 3.5315785
4: -1.3233099, 2.8539722, -1.4156674, 3.0728226, -4.3961325, 4.2696395

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608479, upper bound: 2.7670756
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7799106
time: 0.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7799106
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3857437, 2.4018829, -2.7728441, 2.6360769
1: -0.4575120, 3.1468184, -0.4820945, 3.3220072, -3.7795191, 3.6289129
2: -1.1562662, 2.1376271, -1.1904538, 2.3187947, -3.4750609, 3.3280809
3: -0.9258730, 2.5864434, -0.9715070, 2.7528343, -3.6787074, 3.5579505
4: -1.3418519, 2.8791902, -1.4044091, 3.0569649, -4.3988171, 4.2835994

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608755, upper bound: 2.7644214
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7770552
time: 0.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7769808
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3709612, 2.2503333, -0.3880334, 2.4182405, -2.7892017, 2.6383667
1: -0.4575120, 3.1468184, -0.4848027, 3.3492410, -3.8067529, 3.6316211
2: -1.1562662, 2.1376271, -1.2020583, 2.3271937, -3.4834599, 3.3396854
3: -0.9258730, 2.5864434, -0.9766790, 2.7750435, -3.7009165, 3.5631223
4: -1.3418519, 2.8791902, -1.4156674, 3.0728226, -4.4146748, 4.2948575

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7608755, upper bound: 2.7644214
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7778236
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7776769
time: 0.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3896064, 2.3705792, -2.7761989, 2.9036274
1: -0.5055367, 3.4809954, -0.4801092, 3.3123057, -3.8178425, 3.9611046
2: -1.2521493, 2.4233136, -1.2162256, 2.2505226, -3.5026720, 3.6395392
3: -1.0172695, 2.8950694, -0.9699728, 2.7406662, -3.7579355, 3.8650422
4: -1.4811087, 3.1972914, -1.4223459, 3.0282202, -4.5093288, 4.6196375

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7760001
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7742530
time: 0.40 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3974134, 2.3933864, -2.7990060, 2.9114342
1: -0.5055367, 3.4809954, -0.4861144, 3.3476477, -3.8531842, 3.9671099
2: -1.2521493, 2.4233136, -1.2313797, 2.2667561, -3.5189054, 3.6546934
3: -1.0172695, 2.8950694, -0.9823158, 2.7700262, -3.7872958, 3.8773851
4: -1.4811087, 3.1972914, -1.4410114, 3.0509758, -4.5320845, 4.6383028

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7772661
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7755190
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3896064, 2.3705792, -2.7825289, 2.9147758
1: -0.5089593, 3.5015557, -0.4801092, 3.3123057, -3.8212650, 3.9816649
2: -1.2644246, 2.4245837, -1.2162256, 2.2505226, -3.5149472, 3.6408093
3: -1.0243788, 2.9105000, -0.9699728, 2.7406662, -3.7650449, 3.8804729
4: -1.4927979, 3.2066495, -1.4223459, 3.0282202, -4.5210180, 4.6289954

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7735495
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7725126
time: 0.44 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3974134, 2.3933864, -2.8053360, 2.9225826
1: -0.5089593, 3.5015557, -0.4861144, 3.3476477, -3.8566070, 3.9876702
2: -1.2644246, 2.4245837, -1.2313797, 2.2667561, -3.5311806, 3.6559634
3: -1.0243788, 2.9105000, -0.9823158, 2.7700262, -3.7944050, 3.8928158
4: -1.4927979, 3.2066495, -1.4410114, 3.0509758, -4.5437737, 4.6476612

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7738449
time: 0.35 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7728425
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3630810, 2.2252908, -2.6309104, 2.8771019
1: -0.5055367, 3.4809954, -0.4512913, 3.1082866, -3.6138234, 3.9322867
2: -1.2521493, 2.4233136, -1.1403358, 2.1198447, -3.3719940, 3.5636494
3: -1.0172695, 2.8950694, -0.9133595, 2.5548997, -3.5721693, 3.8084288
4: -1.4811087, 3.1972914, -1.3233099, 2.8539722, -4.3350811, 4.5206013

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7782382
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7699404
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3709612, 2.2503333, -2.6559529, 2.8849821
1: -0.5055367, 3.4809954, -0.4575120, 3.1468184, -3.6523552, 3.9385073
2: -1.2521493, 2.4233136, -1.1562662, 2.1376271, -3.3897765, 3.5795798
3: -1.0172695, 2.8950694, -0.9258730, 2.5864434, -3.6037130, 3.8209424
4: -1.4811087, 3.1972914, -1.3418519, 2.8791902, -4.3602991, 4.5391436

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7793301
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7711735
time: 0.39 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3630810, 2.2252908, -2.6372404, 2.8882504
1: -0.5089593, 3.5015557, -0.4512913, 3.1082866, -3.6172459, 3.9528470
2: -1.2644246, 2.4245837, -1.1403358, 2.1198447, -3.3842692, 3.5649195
3: -1.0243788, 2.9105000, -0.9133595, 2.5548997, -3.5792785, 3.8238597
4: -1.4927979, 3.2066495, -1.3233099, 2.8539722, -4.3467703, 4.5299597

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783385, upper bound: 2.7757387
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752083, upper bound: 2.7698186
time: 0.40 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3709612, 2.2503333, -2.6622829, 2.8961306
1: -0.5089593, 3.5015557, -0.4575120, 3.1468184, -3.6557777, 3.9590676
2: -1.2644246, 2.4245837, -1.1562662, 2.1376271, -3.4020517, 3.5808499
3: -1.0243788, 2.9105000, -0.9258730, 2.5864434, -3.6108222, 3.8363731
4: -1.4927979, 3.2066495, -1.3418519, 2.8791902, -4.3719883, 4.5485015

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783385, upper bound: 2.7758748
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752083, upper bound: 2.7703720
time: 0.39 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3896064, 2.3705792, -2.7563229, 2.7914894
1: -0.4820945, 3.3220072, -0.4801092, 3.3123057, -3.7944002, 3.8021164
2: -1.1904538, 2.3187947, -1.2162256, 2.2505226, -3.4409764, 3.5350204
3: -0.9715070, 2.7528343, -0.9699728, 2.7406662, -3.7121730, 3.7228072
4: -1.4044091, 3.0569649, -1.4223459, 3.0282202, -4.4326291, 4.4793110

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7763857
time: 0.39 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7698002
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3974134, 2.3933864, -2.7791300, 2.7992964
1: -0.4820945, 3.3220072, -0.4861144, 3.3476477, -3.8297422, 3.8081217
2: -1.1904538, 2.3187947, -1.2313797, 2.2667561, -3.4572098, 3.5501745
3: -0.9715070, 2.7528343, -0.9823158, 2.7700262, -3.7415333, 3.7351501
4: -1.4044091, 3.0569649, -1.4410114, 3.0509758, -4.4553847, 4.4979763

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7776517
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7710662
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3896064, 2.3705792, -2.7586126, 2.8078470
1: -0.4848027, 3.3492410, -0.4801092, 3.3123057, -3.7971084, 3.8293502
2: -1.2020583, 2.3271937, -1.2162256, 2.2505226, -3.4525809, 3.5434194
3: -0.9766790, 2.7750435, -0.9699728, 2.7406662, -3.7173452, 3.7450163
4: -1.4156674, 3.0728226, -1.4223459, 3.0282202, -4.4438877, 4.4951687

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7736132
time: 0.39 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7745035
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3974134, 2.3933864, -2.7814198, 2.8156538
1: -0.4848027, 3.3492410, -0.4861144, 3.3476477, -3.8324504, 3.8353555
2: -1.2020583, 2.3271937, -1.2313797, 2.2667561, -3.4688144, 3.5585735
3: -0.9766790, 2.7750435, -0.9823158, 2.7700262, -3.7467051, 3.7573593
4: -1.4156674, 3.0728226, -1.4410114, 3.0509758, -4.4666433, 4.5138340

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7738952
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
time: 0.41 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7751405
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3630810, 2.2252908, -2.6110344, 2.7649639
1: -0.4820945, 3.3220072, -0.4512913, 3.1082866, -3.5903811, 3.7732985
2: -1.1904538, 2.3187947, -1.1403358, 2.1198447, -3.3102984, 3.4591305
3: -0.9715070, 2.7528343, -0.9133595, 2.5548997, -3.5264068, 3.6661940
4: -1.4044091, 3.0569649, -1.3233099, 2.8539722, -4.2583814, 4.3802748

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7757595
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7786459
time: 0.38 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3709612, 2.2503333, -2.6360769, 2.7728441
1: -0.4820945, 3.3220072, -0.4575120, 3.1468184, -3.6289129, 3.7795191
2: -1.1904538, 2.3187947, -1.1562662, 2.1376271, -3.3280809, 3.4750609
3: -0.9715070, 2.7528343, -0.9258730, 2.5864434, -3.5579505, 3.6787074
4: -1.4044091, 3.0569649, -1.3418519, 2.8791902, -4.2835994, 4.3988171

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7762219
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
time: 0.40 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3630810, 2.2252908, -2.6133242, 2.7813215
1: -0.4848027, 3.3492410, -0.4512913, 3.1082866, -3.5930893, 3.8005323
2: -1.2020583, 2.3271937, -1.1403358, 2.1198447, -3.3219030, 3.4675295
3: -0.9766790, 2.7750435, -0.9133595, 2.5548997, -3.5315785, 3.6884031
4: -1.4156674, 3.0728226, -1.3233099, 2.8539722, -4.2696395, 4.3961325

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3709612, 2.2503333, -2.6383667, 2.7892017
1: -0.4848027, 3.3492410, -0.4575120, 3.1468184, -3.6316211, 3.8067529
2: -1.2020583, 2.3271937, -1.1562662, 2.1376271, -3.3396854, 3.4834599
3: -0.9766790, 2.7750435, -0.9258730, 2.5864434, -3.5631223, 3.7009165
4: -1.4156674, 3.0728226, -1.3418519, 2.8791902, -4.2948575, 4.4146748

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662126
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7760039
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.4056197, 2.5140209, -2.9196405, 2.9196405
1: -0.5055367, 3.4809954, -0.5055367, 3.4809954, -3.9865322, 3.9865322
2: -1.2521493, 2.4233136, -1.2521493, 2.4233136, -3.6754630, 3.6754630
3: -1.0172695, 2.8950694, -1.0172695, 2.8950694, -3.9123387, 3.9123387
4: -1.4811087, 3.1972914, -1.4811087, 3.1972914, -4.6784000, 4.6784000

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741718, upper bound: 2.7760039
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727166, upper bound: 2.7744465
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.4119495, 2.5251694, -2.9307890, 2.9259706
1: -0.5055367, 3.4809954, -0.5089593, 3.5015557, -4.0070925, 3.9899547
2: -1.2521493, 2.4233136, -1.2644246, 2.4245837, -3.6767330, 3.6877382
3: -1.0172695, 2.8950694, -1.0243788, 2.9105000, -3.9277697, 3.9194481
4: -1.4811087, 3.1972914, -1.4927979, 3.2066495, -4.6877584, 4.6900892

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7741718, upper bound: 2.7786643
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7727166, upper bound: 2.7770346
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.4056197, 2.5140209, -2.9259706, 2.9307890
1: -0.5089593, 3.5015557, -0.5055367, 3.4809954, -3.9899547, 4.0070925
2: -1.2644246, 2.4245837, -1.2521493, 2.4233136, -3.6877382, 3.6767330
3: -1.0243788, 2.9105000, -1.0172695, 2.8950694, -3.9194481, 3.9277697
4: -1.4927979, 3.2066495, -1.4811087, 3.1972914, -4.6900892, 4.6877584

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7735495
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7725157
time: 0.43 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.4119495, 2.5251694, -2.9371190, 2.9371190
1: -0.5089593, 3.5015557, -0.5089593, 3.5015557, -4.0105152, 4.0105152
2: -1.2644246, 2.4245837, -1.2644246, 2.4245837, -3.6890082, 3.6890082
3: -1.0243788, 2.9105000, -1.0243788, 2.9105000, -3.9348788, 3.9348788
4: -1.4927979, 3.2066495, -1.4927979, 3.2066495, -4.6994476, 4.6994476

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7741585
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7736922
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3857437, 2.4018829, -2.8075025, 2.8997645
1: -0.5055367, 3.4809954, -0.4820945, 3.3220072, -3.8275437, 3.9630899
2: -1.2521493, 2.4233136, -1.1904538, 2.3187947, -3.5709441, 3.6137674
3: -1.0172695, 2.8950694, -0.9715070, 2.7528343, -3.7701039, 3.8665762
4: -1.4811087, 3.1972914, -1.4044091, 3.0569649, -4.5380735, 4.6017003

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
time: 0.40 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4056197, 2.5140209, -0.3880334, 2.4182405, -2.8238602, 2.9020543
1: -0.5055367, 3.4809954, -0.4848027, 3.3492410, -3.8547778, 3.9657981
2: -1.2521493, 2.4233136, -1.2020583, 2.3271937, -3.5793431, 3.6253719
3: -1.0172695, 2.8950694, -0.9766790, 2.7750435, -3.7923131, 3.8717484
4: -1.4811087, 3.1972914, -1.4156674, 3.0728226, -4.5539312, 4.6129589

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
time: 0.40 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
time: 0.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3857437, 2.4018829, -2.8138323, 2.9109130
1: -0.5089593, 3.5015557, -0.4820945, 3.3220072, -3.8309665, 3.9836502
2: -1.2644246, 2.4245837, -1.1904538, 2.3187947, -3.5832193, 3.6150374
3: -1.0243788, 2.9105000, -0.9715070, 2.7528343, -3.7772131, 3.8820071
4: -1.4927979, 3.2066495, -1.4044091, 3.0569649, -4.5497627, 4.6110587

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
time: 0.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
time: 0.40 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.4119495, 2.5251694, -0.3880334, 2.4182405, -2.8301902, 2.9132028
1: -0.5089593, 3.5015557, -0.4848027, 3.3492410, -3.8582003, 3.9863584
2: -1.2644246, 2.4245837, -1.2020583, 2.3271937, -3.5916183, 3.6266420
3: -1.0243788, 2.9105000, -0.9766790, 2.7750435, -3.7994223, 3.8871789
4: -1.4927979, 3.2066495, -1.4156674, 3.0728226, -4.5656204, 4.6223168

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.4056197, 2.5140209, -2.8997645, 2.8075025
1: -0.4820945, 3.3220072, -0.5055367, 3.4809954, -3.9630899, 3.8275437
2: -1.1904538, 2.3187947, -1.2521493, 2.4233136, -3.6137674, 3.5709441
3: -0.9715070, 2.7528343, -1.0172695, 2.8950694, -3.8665762, 3.7701039
4: -1.4044091, 3.0569649, -1.4811087, 3.1972914, -4.6017003, 4.5380735

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7763899
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7700042
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.4119495, 2.5251694, -2.9109130, 2.8138323
1: -0.4820945, 3.3220072, -0.5089593, 3.5015557, -3.9836502, 3.8309665
2: -1.1904538, 2.3187947, -1.2644246, 2.4245837, -3.6150374, 3.5832193
3: -0.9715070, 2.7528343, -1.0243788, 2.9105000, -3.8820071, 3.7772131
4: -1.4044091, 3.0569649, -1.4927979, 3.2066495, -4.6110587, 4.5497627

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7790485
time: 0.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7725818
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.4056197, 2.5140209, -2.9020543, 2.8238602
1: -0.4848027, 3.3492410, -0.5055367, 3.4809954, -3.9657981, 3.8547778
2: -1.2020583, 2.3271937, -1.2521493, 2.4233136, -3.6253719, 3.5793431
3: -0.9766790, 2.7750435, -1.0172695, 2.8950694, -3.8717484, 3.7923131
4: -1.4156674, 3.0728226, -1.4811087, 3.1972914, -4.6129589, 4.5539312

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7736132
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.4119495, 2.5251694, -2.9132028, 2.8301902
1: -0.4848027, 3.3492410, -0.5089593, 3.5015557, -3.9863584, 3.8582003
2: -1.2020583, 2.3271937, -1.2644246, 2.4245837, -3.6266420, 3.5916183
3: -0.9766790, 2.7750435, -1.0243788, 2.9105000, -3.8871789, 3.7994223
4: -1.4156674, 3.0728226, -1.4927979, 3.2066495, -4.6223168, 4.5656204

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7741815
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
time: 0.40 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3857437, 2.4018829, -2.7876265, 2.7876265
1: -0.4820945, 3.3220072, -0.4820945, 3.3220072, -3.8041017, 3.8041017
2: -1.1904538, 2.3187947, -1.1904538, 2.3187947, -3.5092485, 3.5092485
3: -0.9715070, 2.7528343, -0.9715070, 2.7528343, -3.7243414, 3.7243414
4: -1.4044091, 3.0569649, -1.4044091, 3.0569649, -4.4613738, 4.4613738

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
time: 0.38 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.3857437, 2.4018829, -0.3880334, 2.4182405, -2.8039842, 2.7899163
1: -0.4820945, 3.3220072, -0.4848027, 3.3492410, -3.8313355, 3.8068099
2: -1.1904538, 2.3187947, -1.2020583, 2.3271937, -3.5176475, 3.5208530
3: -0.9715070, 2.7528343, -0.9766790, 2.7750435, -3.7465506, 3.7295132
4: -1.4044091, 3.0569649, -1.4156674, 3.0728226, -4.4772315, 4.4726324

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
time: 0.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3857437, 2.4018829, -2.7899163, 2.8039842
1: -0.4848027, 3.3492410, -0.4820945, 3.3220072, -3.8068099, 3.8313355
2: -1.2020583, 2.3271937, -1.1904538, 2.3187947, -3.5208530, 3.5176475
3: -0.9766790, 2.7750435, -0.9715070, 2.7528343, -3.7295132, 3.7465506
4: -1.4156674, 3.0728226, -1.4044091, 3.0569649, -4.4726324, 4.4772315

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
time: 0.41 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.39 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.3880334, 2.4182405, -0.3880334, 2.4182405, -2.8062739, 2.8062739
1: -0.4848027, 3.3492410, -0.4848027, 3.3492410, -3.8340437, 3.8340437
2: -1.2020583, 2.3271937, -1.2020583, 2.3271937, -3.5292521, 3.5292521
3: -0.9766790, 2.7750435, -0.9766790, 2.7750435, -3.7517223, 3.7517223
4: -1.4156674, 3.0728226, -1.4156674, 3.0728226, -4.4884901, 4.4884901

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
time: 0.40 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.60 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7765046
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7742238
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7777688
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7761210, upper bound: 2.7747670
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7726478
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7743032, upper bound: 2.7750910
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7786973
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7787504
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7801671
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7801671
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7770552
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7769808
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7778307
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7777063
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7741680, upper bound: 2.7766659
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7744278
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7741680, upper bound: 2.7792876
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7770054
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7741680, upper bound: 2.7748819
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7728518
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7759808, upper bound: 2.7761403
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7740142
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7628797, upper bound: 2.7669565
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7694030, upper bound: 2.7770164
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7711345, upper bound: 2.7754593
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7710565, upper bound: 2.7752083
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7786973
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7787504
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7750905, upper bound: 2.7799106
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7757999, upper bound: 2.7799106
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7770552
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7769808
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7767031, upper bound: 2.7778236
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7768772, upper bound: 2.7776769
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7760001
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7742530
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7772661
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7728518, upper bound: 2.7755190
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7758298, upper bound: 2.7735495
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7725126
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7743070, upper bound: 2.7738449
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7728425
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7782382
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7699404
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7768216, upper bound: 2.7793301
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7755150, upper bound: 2.7711735
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7783385, upper bound: 2.7757387
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7752083, upper bound: 2.7698186
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7783385, upper bound: 2.7758748
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7752083, upper bound: 2.7703720
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7763857
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7698002
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7746036, upper bound: 2.7776517
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7663166, upper bound: 2.7710662
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7745035
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7611295, upper bound: 2.7601122
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7787443, upper bound: 2.7751405
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7757595
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7786459
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7713889, upper bound: 2.7762219
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7778860, upper bound: 2.7800703
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662127
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7757081
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7644246, upper bound: 2.7662126
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7788109, upper bound: 2.7760039
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7741718, upper bound: 2.7760039
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7727166, upper bound: 2.7744465
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7741718, upper bound: 2.7786643
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7727166, upper bound: 2.7770346
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7735495
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7725157
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7741585
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7752942, upper bound: 2.7736922
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7551075, upper bound: 2.7541058
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7554752, upper bound: 2.7551539
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524367
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7526843, upper bound: 2.7524708
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7763899
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7700042
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7790485
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7667442, upper bound: 2.7725818
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7756946, upper bound: 2.7736132
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7744455, upper bound: 2.7741815
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7629832, upper bound: 2.7616153
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7583522
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7554825, upper bound: 2.7563737
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7524105
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.60
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3739614, 2.2898805, -0.3896064, 2.3705792, -2.7445407, 2.6794870
1: -0.4637825, 3.2020445, -0.4801092, 3.3123057, -3.7760882, 3.6821537
2: -1.1733108, 2.1740575, -1.2162256, 2.2505226, -3.4238334, 3.3902831
3: -0.9386251, 2.6226339, -0.9699728, 2.7406662, -3.6792912, 3.5926068
4: -1.3490784, 2.9333854, -1.4223459, 3.0282202, -4.3772984, 4.3557310

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4101669, 2.4480877, -0.3896064, 2.3705792, -2.7807460, 2.8376942
1: -0.4886297, 3.3980122, -0.4801092, 3.3123057, -3.8009353, 3.8781214
2: -1.2252474, 2.3572233, -1.2162256, 2.2505226, -3.4757700, 3.5734489
3: -0.9874058, 2.8292980, -0.9699728, 2.7406662, -3.7280719, 3.7992709
4: -1.4586418, 3.0615449, -1.4223459, 3.0282202, -4.4868622, 4.4838905

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7742238
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3739614, 2.2898805, -0.3974134, 2.3933864, -2.7673478, 2.6872940
1: -0.4637825, 3.2020445, -0.4861144, 3.3476477, -3.8114302, 3.6881590
2: -1.1733108, 2.1740575, -1.2313797, 2.2667561, -3.4400668, 3.4054372
3: -0.9386251, 2.6226339, -0.9823158, 2.7700262, -3.7086513, 3.6049497
4: -1.3490784, 2.9333854, -1.4410114, 3.0509758, -4.4000540, 4.3743968

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.34 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4101669, 2.4480877, -0.3974134, 2.3933864, -2.8035531, 2.8455009
1: -0.4886297, 3.3980122, -0.4861144, 3.3476477, -3.8362775, 3.8841267
2: -1.2252474, 2.3572233, -1.2313797, 2.2667561, -3.4920034, 3.5886030
3: -0.9874058, 2.8292980, -0.9823158, 2.7700262, -3.7574320, 3.8116138
4: -1.4586418, 3.0615449, -1.4410114, 3.0509758, -4.5096178, 4.5025563

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7726478, upper bound: 2.7754898
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3809908, 2.3117228, -0.3896064, 2.3705792, -2.7515700, 2.7013292
1: -0.4694015, 3.2361622, -0.4801092, 3.3123057, -3.7817073, 3.7162714
2: -1.1875644, 2.1893144, -1.2162256, 2.2505226, -3.4380870, 3.4055400
3: -0.9499736, 2.6506827, -0.9699728, 2.7406662, -3.6906397, 3.6206555
4: -1.3663981, 2.9551375, -1.4223459, 3.0282202, -4.3946180, 4.3774834

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754898, upper bound: 2.7726478
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4191682, 2.4729218, -0.3896064, 2.3705792, -2.7897475, 2.8625283
1: -0.4953873, 3.4356165, -0.4801092, 3.3123057, -3.8076930, 3.9157257
2: -1.2422593, 2.3750792, -1.2162256, 2.2505226, -3.4927819, 3.5913048
3: -1.0012507, 2.8649900, -0.9699728, 2.7406662, -3.7419169, 3.8349628
4: -1.4800587, 3.0869579, -1.4223459, 3.0282202, -4.5082788, 4.5093040

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7726478
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742238, upper bound: 2.7726478
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3809908, 2.3117228, -0.3974134, 2.3933864, -2.7743771, 2.7091360
1: -0.4694015, 3.2361622, -0.4861144, 3.3476477, -3.8170493, 3.7222767
2: -1.1875644, 2.1893144, -1.2313797, 2.2667561, -3.4543204, 3.4206941
3: -0.9499736, 2.6506827, -0.9823158, 2.7700262, -3.7199998, 3.6329985
4: -1.3663981, 2.9551375, -1.4410114, 3.0509758, -4.4173737, 4.3961487

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4191682, 2.4729218, -0.3974134, 2.3933864, -2.8125546, 2.8703351
1: -0.4953873, 3.4356165, -0.4861144, 3.3476477, -3.8430350, 3.9217310
2: -1.2422593, 2.3750792, -1.2313797, 2.2667561, -3.5090153, 3.6064589
3: -1.0012507, 2.8649900, -0.9823158, 2.7700262, -3.7712770, 3.8473058
4: -1.4800587, 3.0869579, -1.4410114, 3.0509758, -4.5310345, 4.5279694

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7739138, upper bound: 2.7729641
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2664788, 1.7967149, -0.3630810, 2.2252908, -2.4917696, 2.1597958
1: -0.3620793, 2.5064969, -0.4512913, 3.1082866, -3.4703660, 2.9577882
2: -0.8944936, 1.7365243, -1.1403358, 2.1198447, -3.0143383, 2.8768601
3: -0.7412283, 1.9995631, -0.9133595, 2.5548997, -3.2961280, 2.9129226
4: -0.9644823, 2.3460627, -1.3233099, 2.8539722, -3.8184545, 3.6693726

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757383, upper bound: 2.7785136
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757383, upper bound: 2.7786973
time: 0.38 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3106845, 2.0026269, -0.3630810, 2.2252908, -2.5359752, 2.3657079
1: -0.4048957, 2.8021774, -0.4512913, 3.1082866, -3.5131824, 3.2534688
2: -1.0137308, 1.9121870, -1.1403358, 2.1198447, -3.1335754, 3.0525227
3: -0.8244714, 2.2569513, -0.9133595, 2.5548997, -3.3793712, 3.1703110
4: -1.1290723, 2.5932925, -1.3233099, 2.8539722, -3.9830446, 3.9166024

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790752, upper bound: 2.7785136
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790752, upper bound: 2.7787504
time: 0.37 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2664788, 1.7967149, -0.3709612, 2.2503333, -2.5168121, 2.1676760
1: -0.3620793, 2.5064969, -0.4575120, 3.1468184, -3.5088978, 2.9640088
2: -0.8944936, 1.7365243, -1.1562662, 2.1376271, -3.0321207, 2.8927906
3: -0.7412283, 1.9995631, -0.9258730, 2.5864434, -3.3276718, 2.9254360
4: -0.9644823, 2.3460627, -1.3418519, 2.8791902, -3.8436725, 3.6879146

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7801671
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757169, upper bound: 2.7801671
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3106845, 2.0026269, -0.3709612, 2.2503333, -2.5610178, 2.3735881
1: -0.4048957, 2.8021774, -0.4575120, 3.1468184, -3.5517142, 3.2596893
2: -1.0137308, 1.9121870, -1.1562662, 2.1376271, -3.1513579, 3.0684533
3: -0.8244714, 2.2569513, -0.9258730, 2.5864434, -3.4109149, 3.1828244
4: -1.1290723, 2.5932925, -1.3418519, 2.8791902, -4.0082626, 3.9351444

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7801671
time: 0.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7774392, upper bound: 2.7801671
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2821565, 1.8435556, -0.3630810, 2.2252908, -2.5074472, 2.2066364
1: -0.3734180, 2.5757110, -0.4512913, 3.1082866, -3.4817047, 3.0270023
2: -0.9245632, 1.7732220, -1.1403358, 2.1198447, -3.0444078, 2.9135578
3: -0.7641129, 2.0653176, -0.9133595, 2.5548997, -3.3190126, 2.9786773
4: -1.0053809, 2.3925400, -1.3233099, 2.8539722, -3.8593531, 3.7158499

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7768776
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7769808
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.3181820, 2.0252068, -0.3630810, 2.2252908, -2.5434728, 2.3882878
1: -0.4108701, 2.8373315, -0.4512913, 3.1082866, -3.5191567, 3.2886229
2: -1.0286827, 1.9279337, -1.1403358, 2.1198447, -3.1485274, 3.0682695
3: -0.8367355, 2.2843473, -0.9133595, 2.5548997, -3.3916352, 3.1977067
4: -1.1456609, 2.6164696, -1.3233099, 2.8539722, -3.9996331, 3.9397795

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800387, upper bound: 2.7768776
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800387, upper bound: 2.7769808
time: 0.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.2821565, 1.8435556, -0.3709612, 2.2503333, -2.5324898, 2.2145166
1: -0.3734180, 2.5757110, -0.4575120, 3.1468184, -3.5202365, 3.0332229
2: -0.9245632, 1.7732220, -1.1562662, 2.1376271, -3.0621903, 2.9294882
3: -0.7641129, 2.0653176, -0.9258730, 2.5864434, -3.3505564, 2.9911907
4: -1.0053809, 2.3925400, -1.3418519, 2.8791902, -3.8845711, 3.7343919

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7777000
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7777062
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.3181820, 2.0252068, -0.3709612, 2.2503333, -2.5685153, 2.3961680
1: -0.4108701, 2.8373315, -0.4575120, 3.1468184, -3.5576885, 3.2948434
2: -1.0286827, 1.9279337, -1.1562662, 2.1376271, -3.1663098, 3.0841999
3: -0.8367355, 2.2843473, -0.9258730, 2.5864434, -3.4231789, 3.2102203
4: -1.1456609, 2.6164696, -1.3418519, 2.8791902, -4.0248508, 3.9583216

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7776999
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7784632, upper bound: 2.7777062
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3739614, 2.2898805, -0.4056197, 2.5140209, -2.8879824, 2.6955001
1: -0.4637825, 3.2020445, -0.5055367, 3.4809954, -3.9447780, 3.7075810
2: -1.1733108, 2.1740575, -1.2521493, 2.4233136, -3.5966244, 3.4262068
3: -0.9386251, 2.6226339, -1.0172695, 2.8950694, -3.8336945, 3.6399035
4: -1.3490784, 2.9333854, -1.4811087, 3.1972914, -4.5463696, 4.4144940

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742530, upper bound: 2.7744278
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742530, upper bound: 2.7744278
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4101669, 2.4480877, -0.4056197, 2.5140209, -2.9241877, 2.8537073
1: -0.4886297, 3.3980122, -0.5055367, 3.4809954, -3.9696250, 3.9035487
2: -1.2252474, 2.3572233, -1.2521493, 2.4233136, -3.6485610, 3.6093726
3: -0.9874058, 2.8292980, -1.0172695, 2.8950694, -3.8824751, 3.8465676
4: -1.4586418, 3.0615449, -1.4811087, 3.1972914, -4.6559334, 4.5426536

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742530, upper bound: 2.7744278
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7742530, upper bound: 2.7744278
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3739614, 2.2898805, -0.4119495, 2.5251694, -2.8991308, 2.7018299
1: -0.4637825, 3.2020445, -0.5089593, 3.5015557, -3.9653382, 3.7110038
2: -1.1733108, 2.1740575, -1.2644246, 2.4245837, -3.5978944, 3.4384820
3: -0.9386251, 2.6226339, -1.0243788, 2.9105000, -3.8491251, 3.6470127
4: -1.3490784, 2.9333854, -1.4927979, 3.2066495, -4.5557280, 4.4261832

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7770054
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7770054
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4101669, 2.4480877, -0.4119495, 2.5251694, -2.9353361, 2.8600373
1: -0.4886297, 3.3980122, -0.5089593, 3.5015557, -3.9901853, 3.9069715
2: -1.2252474, 2.3572233, -1.2644246, 2.4245837, -3.6498311, 3.6216478
3: -0.9874058, 2.8292980, -1.0243788, 2.9105000, -3.8979058, 3.8536768
4: -1.4586418, 3.0615449, -1.4927979, 3.2066495, -4.6652913, 4.5543427

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7770054
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7725126, upper bound: 2.7770054
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.3809908, 2.3117228, -0.4056197, 2.5140209, -2.8950117, 2.7173424
1: -0.4694015, 3.2361622, -0.5055367, 3.4809954, -3.9503970, 3.7416987
2: -1.1875644, 2.1893144, -1.2521493, 2.4233136, -3.6108780, 3.4414637
3: -0.9499736, 2.6506827, -1.0172695, 2.8950694, -3.8450429, 3.6679521
4: -1.3663981, 2.9551375, -1.4811087, 3.1972914, -4.5636892, 4.4362459

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755190, upper bound: 2.7728518
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755190, upper bound: 2.7728518
time: 0.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4191682, 2.4729218, -0.4056197, 2.5140209, -2.9331892, 2.8785415
1: -0.4953873, 3.4356165, -0.5055367, 3.4809954, -3.9763827, 3.9411530
2: -1.2422593, 2.3750792, -1.2521493, 2.4233136, -3.6655729, 3.6272285
3: -1.0012507, 2.8649900, -1.0172695, 2.8950694, -3.8963201, 3.8822594
4: -1.4800587, 3.0869579, -1.4811087, 3.1972914, -4.6773500, 4.5680666

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755190, upper bound: 2.7728518
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755190, upper bound: 2.7728518
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.3809908, 2.3117228, -0.4119495, 2.5251694, -2.9061601, 2.7236724
1: -0.4694015, 3.2361622, -0.5089593, 3.5015557, -3.9709573, 3.7451215
2: -1.1875644, 2.1893144, -1.2644246, 2.4245837, -3.6121480, 3.4537389
3: -0.9499736, 2.6506827, -1.0243788, 2.9105000, -3.8604736, 3.6750615
4: -1.3663981, 2.9551375, -1.4927979, 3.2066495, -4.5730476, 4.4479351

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7740140
time: 0.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7740142
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4191682, 2.4729218, -0.4119495, 2.5251694, -2.9443376, 2.8848715
1: -0.4953873, 3.4356165, -0.5089593, 3.5015557, -3.9969430, 3.9445758
2: -1.2422593, 2.3750792, -1.2644246, 2.4245837, -3.6668429, 3.6395037
3: -1.0012507, 2.8649900, -1.0243788, 2.9105000, -3.9117508, 3.8893688
4: -1.4800587, 3.0869579, -1.4927979, 3.2066495, -4.6867085, 4.5797558

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7740142
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7737786, upper bound: 2.7740142
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3739614, 2.2898805, -0.3857437, 2.4018829, -2.7758443, 2.6756241
1: -0.4637825, 3.2020445, -0.4820945, 3.3220072, -3.7857897, 3.6841390
2: -1.1733108, 2.1740575, -1.1904538, 2.3187947, -3.4921055, 3.3645113
3: -0.9386251, 2.6226339, -0.9715070, 2.7528343, -3.6914594, 3.5941410
4: -1.3490784, 2.9333854, -1.4044091, 3.0569649, -4.4060431, 4.3377943

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.4101669, 2.4480877, -0.3857437, 2.4018829, -2.8120499, 2.8338313
1: -0.4886297, 3.3980122, -0.4820945, 3.3220072, -3.8106370, 3.8801067
2: -1.2252474, 2.3572233, -1.1904538, 2.3187947, -3.5440421, 3.5476770
3: -0.9874058, 2.8292980, -0.9715070, 2.7528343, -3.7402401, 3.8008051
4: -1.4586418, 3.0615449, -1.4044091, 3.0569649, -4.5156069, 4.4659538

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698002, upper bound: 2.7685836
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3739614, 2.2898805, -0.3880334, 2.4182405, -2.7922020, 2.6779139
1: -0.4637825, 3.2020445, -0.4848027, 3.3492410, -3.8130236, 3.6868472
2: -1.1733108, 2.1740575, -1.2020583, 2.3271937, -3.5005045, 3.3761158
3: -0.9386251, 2.6226339, -0.9766790, 2.7750435, -3.7136686, 3.5993128
4: -1.3490784, 2.9333854, -1.4156674, 3.0728226, -4.4219007, 4.3490529

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.4101669, 2.4480877, -0.3880334, 2.4182405, -2.8284073, 2.8361211
1: -0.4886297, 3.3980122, -0.4848027, 3.3492410, -3.8378706, 3.8828149
2: -1.2252474, 2.3572233, -1.2020583, 2.3271937, -3.5524411, 3.5592816
3: -0.9874058, 2.8292980, -0.9766790, 2.7750435, -3.7624493, 3.8059769
4: -1.4586418, 3.0615449, -1.4156674, 3.0728226, -4.5314646, 4.4772124

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7551952, upper bound: 2.7646612
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.2664788, 1.7967149, -0.4056197, 2.5140209, -2.7804997, 2.2023346
1: -0.3620793, 2.5064969, -0.5055367, 3.4809954, -3.8430748, 3.0120335
2: -0.8944936, 1.7365243, -1.2521493, 2.4233136, -3.3178072, 2.9886737
3: -0.7412283, 1.9995631, -1.0172695, 2.8950694, -3.6362977, 3.0168326
4: -0.9644823, 2.3460627, -1.4811087, 3.1972914, -4.1617737, 3.8271713

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7695200, upper bound: 2.7773175
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7695200, upper bound: 2.7773175
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3106845, 2.0026269, -0.4056197, 2.5140209, -2.8247054, 2.4082465
1: -0.4048957, 2.8021774, -0.5055367, 3.4809954, -3.8858912, 3.3077140
2: -1.0137308, 1.9121870, -1.2521493, 2.4233136, -3.4370444, 3.1643362
3: -0.8244714, 2.2569513, -1.0172695, 2.8950694, -3.7195406, 3.2742209
4: -1.1290723, 2.5932925, -1.4811087, 3.1972914, -4.3263636, 4.0744009

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7699404, upper bound: 2.7773175
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7699404, upper bound: 2.7773175
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.2664788, 1.7967149, -0.4119495, 2.5251694, -2.7916481, 2.2086644
1: -0.3620793, 2.5064969, -0.5089593, 3.5015557, -3.8636351, 3.0154562
2: -0.8944936, 1.7365243, -1.2644246, 2.4245837, -3.3190773, 3.0009489
3: -0.7412283, 1.9995631, -1.0243788, 2.9105000, -3.6517284, 3.0239420
4: -0.9644823, 2.3460627, -1.4927979, 3.2066495, -4.1711321, 3.8388605

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3106845, 2.0026269, -0.4119495, 2.5251694, -2.8358538, 2.4145765
1: -0.4048957, 2.8021774, -0.5089593, 3.5015557, -3.9064515, 3.3111367
2: -1.0137308, 1.9121870, -1.2644246, 2.4245837, -3.4383144, 3.1766114
3: -0.8244714, 2.2569513, -1.0243788, 2.9105000, -3.7349715, 3.2813301
4: -1.1290723, 2.5932925, -1.4927979, 3.2066495, -4.3357220, 4.0860901

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7698186, upper bound: 2.7770164
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.2821565, 1.8435556, -0.4056197, 2.5140209, -2.7961774, 2.2491753
1: -0.3734180, 2.5757110, -0.5055367, 3.4809954, -3.8544135, 3.0812478
2: -0.9245632, 1.7732220, -1.2521493, 2.4233136, -3.3478768, 3.0253713
3: -0.7641129, 2.0653176, -1.0172695, 2.8950694, -3.6591823, 3.0825872
4: -1.0053809, 2.3925400, -1.4811087, 3.1972914, -4.2026720, 3.8736486

Time for backsubstitution: 1.64 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.50 + 418.29 = 420.78 seconds
