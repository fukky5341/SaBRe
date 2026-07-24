## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 27.5202488034


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.8926516, 16.6545238, -4.8926516, 16.6545238, -21.5471706, 21.5471706)
1: (-6.9941020, 17.1100998, -6.9941020, 17.1100998, -24.1041965, 24.1041965)
2: (-5.9263554, 19.1902447, -5.9263554, 19.1902447, -25.1166000, 25.1166000)
3: (-7.0618343, 24.5106659, -7.0618343, 24.5106659, -31.5725002, 31.5725002)
4: (-5.7611084, 22.7441978, -5.7611084, 22.7441978, -28.5053062, 28.5053062)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 1.66 = 2.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -27.5477966, upper bound: 27.5477966

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5300102, upper bound: 27.5386763
time: 0.60 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5457634, upper bound: 27.5457635
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.48 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 3, lower bound: -27.5300102, upper bound: 27.5386763
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.48
Output dim: 3, lower bound: -27.5457634, upper bound: 27.5457635

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.4779983, 15.6032467, -4.8926516, 16.6545238, -21.1325207, 20.4958973
1: -6.4072104, 15.9771137, -6.9941020, 17.1100998, -23.5173073, 22.9712143
2: -5.4055552, 17.9303551, -5.9263554, 19.1902447, -24.5958004, 23.8567104
3: -6.4871049, 22.9893456, -7.0618343, 24.5106659, -30.9977703, 30.0511799
4: -5.3025470, 21.2900162, -5.7611084, 22.7441978, -28.0467453, 27.0511246

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5229229
time: 0.55 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5386763
time: 0.66 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.8002801, 16.4875641, -4.8926516, 16.6545238, -21.4548035, 21.3802128
1: -6.8583674, 16.9091358, -6.9941020, 17.1100998, -23.9684658, 23.9032364
2: -5.7989798, 18.9445438, -5.9263554, 19.1902447, -24.9892235, 24.8708992
3: -6.9374380, 24.2584801, -7.0618343, 24.5106659, -31.4481030, 31.3203144
4: -5.6505985, 22.4603367, -5.7611084, 22.7441978, -28.3947964, 28.2214451

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5300102
time: 0.68 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5457634
time: 0.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.01 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5229229
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 3, lower bound: -27.5229229, upper bound: 27.5386763
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5300102
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.01
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5457634

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.4779983, 15.6032467, -4.4779983, 15.6032467, -20.0812454, 20.0812454
1: -6.4072104, 15.9771137, -6.4072104, 15.9771137, -22.3843231, 22.3843231
2: -5.4055552, 17.9303551, -5.4055552, 17.9303551, -23.3359108, 23.3359108
3: -6.4871049, 22.9893456, -6.4871049, 22.9893456, -29.4764500, 29.4764500
4: -5.3025470, 21.2900162, -5.3025470, 21.2900162, -26.5925636, 26.5925636

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
time: 0.70 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.4779983, 15.6032467, -4.8002801, 16.4875641, -20.9655628, 20.4035263
1: -6.4072104, 15.9771137, -6.8583674, 16.9091358, -23.3163452, 22.8354797
2: -5.4055552, 17.9303551, -5.7989798, 18.9445438, -24.3500996, 23.7293358
3: -6.4871049, 22.9893456, -6.9374380, 24.2584801, -30.7455845, 29.9267845
4: -5.3025470, 21.2900162, -5.6505985, 22.4603367, -27.7628841, 26.9406128

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5386763
time: 0.66 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.8002801, 16.4875641, -4.4779983, 15.6032467, -20.4035263, 20.9655628
1: -6.8583674, 16.9091358, -6.4072104, 15.9771137, -22.8354816, 23.3163452
2: -5.7989798, 18.9445438, -5.4055552, 17.9303551, -23.7293358, 24.3500996
3: -6.9374380, 24.2584801, -6.4871049, 22.9893456, -29.9267845, 30.7455845
4: -5.6505985, 22.4603367, -5.3025470, 21.2900162, -26.9406128, 27.7628841

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378310, upper bound: 27.5262572
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386761, upper bound: 27.5299076
time: 0.68 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.8002801, 16.4875641, -4.8002801, 16.4875641, -21.2878418, 21.2878418
1: -6.8583674, 16.9091358, -6.8583674, 16.9091358, -23.7675037, 23.7675018
2: -5.7989798, 18.9445438, -5.7989798, 18.9445438, -24.7435226, 24.7435226
3: -6.9374380, 24.2584801, -6.9374380, 24.2584801, -31.1959171, 31.1959171
4: -5.6505985, 22.4603367, -5.6505985, 22.4603367, -28.1109352, 28.1109352

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378311, upper bound: 27.5420105
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5456609
time: 0.50 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.76 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5193727
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5218199, upper bound: 27.5386763
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5378310, upper bound: 27.5262572
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5386761, upper bound: 27.5299076
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5378311, upper bound: 27.5420105
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.76
Output dim: 3, lower bound: -27.5386763, upper bound: 27.5456609

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.8096104, 13.5872917, -4.4779983, 15.6032467, -19.4128571, 18.0652905
1: -5.5293055, 13.9026499, -6.4072104, 15.9771137, -21.5064201, 20.3098602
2: -4.6612096, 15.6515274, -5.4055552, 17.9303551, -22.5915642, 21.0570812
3: -5.5544548, 20.0211716, -6.4871049, 22.9893456, -28.5438004, 26.5082760
4: -4.5881572, 18.5381012, -5.3025470, 21.2900162, -25.8781719, 23.8406487

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5182697
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5193727
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.3976188, 15.3468742, -4.4779983, 15.6032467, -20.0008659, 19.8248711
1: -6.2940626, 15.7216339, -6.4072104, 15.9771137, -22.2711735, 22.1288433
2: -5.3095002, 17.6520042, -5.4055552, 17.9303551, -23.2398548, 23.0575581
3: -6.3730483, 22.6263657, -6.4871049, 22.9893456, -29.3623943, 29.1134701
4: -5.2157865, 20.9694633, -5.3025470, 21.2900162, -26.5058022, 26.2720089

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5218199
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5229229
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8096104, 13.5872917, -4.8002801, 16.4875641, -20.2971725, 18.3875713
1: -5.5293055, 13.9026499, -6.8583674, 16.9091358, -22.4384422, 20.7610168
2: -4.6612096, 15.6515274, -5.7989798, 18.9445438, -23.6057529, 21.4505081
3: -5.5544548, 20.0211716, -6.9374380, 24.2584801, -29.8129349, 26.9586105
4: -4.5881572, 18.5381012, -5.6505985, 22.4603367, -27.0484943, 24.1886978

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5251542, upper bound: 27.5342809
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5251542, upper bound: 27.5351259
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.3976188, 15.3468742, -4.8002801, 16.4875641, -20.8851814, 20.1471519
1: -6.2940626, 15.7216339, -6.8583674, 16.9091358, -23.2031956, 22.5799999
2: -5.3095002, 17.6520042, -5.7989798, 18.9445438, -24.2540436, 23.4509850
3: -6.3730483, 22.6263657, -6.9374380, 24.2584801, -30.6315289, 29.5638027
4: -5.2157865, 20.9694633, -5.6505985, 22.4603367, -27.6761227, 26.6200619

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5378311
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5386761
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.0815225, 14.3603363, -4.4779983, 15.6032467, -19.6847687, 18.8383350
1: -5.9201736, 14.7163811, -6.4072104, 15.9771137, -21.8972874, 21.1235886
2: -5.0029421, 16.5449162, -5.4055552, 17.9303551, -22.9332962, 21.9504719
3: -5.9411888, 21.1416397, -6.4871049, 22.9893456, -28.9305344, 27.6287441
4: -4.8949690, 19.5565891, -5.3025470, 21.2900162, -26.1849861, 24.8591347

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5251542
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5262572
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.4779983, 15.6032467, -20.3238564, 20.7103252
1: -6.7455416, 16.6543579, -6.4072104, 15.9771137, -22.7226543, 23.0615692
2: -5.7037611, 18.6685505, -5.4055552, 17.9303551, -23.6341133, 24.0741062
3: -6.8226271, 23.8968391, -6.4871049, 22.9893456, -29.8119736, 30.3839436
4: -5.5644245, 22.1408634, -5.3025470, 21.2900162, -26.8544388, 27.4434109

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5288045
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5351259, upper bound: 27.5299076
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.0815225, 14.3603363, -4.8002801, 16.4875641, -20.5690842, 19.1606159
1: -5.9201736, 14.7163811, -6.8583674, 16.9091358, -22.8293095, 21.5747452
2: -5.0029421, 16.5449162, -5.7989798, 18.9445438, -23.9474831, 22.3438950
3: -5.9411888, 21.1416397, -6.9374380, 24.2584801, -30.1996670, 28.0790768
4: -4.8949690, 19.5565891, -5.6505985, 22.4603367, -27.3553047, 25.2071838

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5411624
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5420104
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.8002801, 16.4875641, -21.2081718, 21.0326080
1: -6.7455416, 16.6543579, -6.8583674, 16.9091358, -23.6546783, 23.5127258
2: -5.7037611, 18.6685505, -5.7989798, 18.9445438, -24.6483021, 24.4675293
3: -6.8226271, 23.8968391, -6.9374380, 24.2584801, -31.0811062, 30.8342743
4: -5.5644245, 22.1408634, -5.6505985, 22.4603367, -28.0247612, 27.7914619

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5448052
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5456608
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.24 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5182697
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5193727
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5218199
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5229229
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5251542, upper bound: 27.5342809
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5251542, upper bound: 27.5351259
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5378311
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5262572, upper bound: 27.5386761
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5342809, upper bound: 27.5251542
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5182697, upper bound: 27.5262572
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5193727, upper bound: 27.5288045
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5351259, upper bound: 27.5299076
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5411624
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5411653, upper bound: 27.5420104
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5448052
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.24
Output dim: 3, lower bound: -27.5419888, upper bound: 27.5456608

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.3976188, 15.3468742, -3.8096104, 13.5872917, -17.9849110, 19.1564846
1: -6.2940626, 15.7216339, -5.5293055, 13.9026499, -20.1967125, 21.2509384
2: -5.3095002, 17.6520042, -4.6612096, 15.6515274, -20.9610271, 22.3132133
3: -6.3730483, 22.6263657, -5.5544548, 20.0211716, -26.3942204, 28.1808205
4: -5.2157865, 20.9694633, -4.5881572, 18.5381012, -23.7538872, 25.5576191

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4984862
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4828583, upper bound: 27.4851750
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.3976188, 15.3468742, -4.3976188, 15.3468742, -19.7444916, 19.7444916
1: -6.2940626, 15.7216339, -6.2940626, 15.7216339, -22.0156937, 22.0156937
2: -5.3095002, 17.6520042, -5.3095002, 17.6520042, -22.9615021, 22.9615021
3: -6.3730483, 22.6263657, -6.3730483, 22.6263657, -28.9994144, 28.9994144
4: -5.2157865, 20.9694633, -5.2157865, 20.9694633, -26.1852493, 26.1852493

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4984862
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4828583, upper bound: 27.4853801
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.8096104, 13.5872917, -4.0815225, 14.3603363, -18.1699467, 17.6688137
1: -5.5293055, 13.9026499, -5.9201736, 14.7163811, -20.2456837, 19.8228226
2: -4.6612096, 15.6515274, -5.0029421, 16.5449162, -21.2061253, 20.6544666
3: -5.5544548, 20.0211716, -5.9411888, 21.1416397, -26.6960945, 25.9623604
4: -4.5881572, 18.5381012, -4.8949690, 19.5565891, -24.1447430, 23.4330711

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5280439
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4826532
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.8096104, 13.5872917, -4.7206116, 16.2323303, -20.0419388, 18.3079014
1: -5.5293055, 13.9026499, -6.7455416, 16.6543579, -22.1836624, 20.6481915
2: -4.6612096, 15.6515274, -5.7037611, 18.6685505, -23.3297596, 21.3552837
3: -5.5544548, 20.0211716, -6.8226271, 23.8968391, -29.4512939, 26.8437996
4: -4.5881572, 18.5381012, -5.5644245, 22.1408634, -26.7290192, 24.1025238

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5286491
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5045526, upper bound: 27.5184922
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.3976188, 15.3468742, -4.0815225, 14.3603363, -18.7579556, 19.4283924
1: -6.2940626, 15.7216339, -5.9201736, 14.7163811, -21.0104370, 21.6418076
2: -5.3095002, 17.6520042, -5.0029421, 16.5449162, -21.8544159, 22.6549435
3: -6.3730483, 22.6263657, -5.9411888, 21.1416397, -27.5146885, 28.5675526
4: -5.2157865, 20.9694633, -4.8949690, 19.5565891, -24.7723751, 25.8644333

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5177716, upper bound: 27.5337203
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5047577, upper bound: 27.5204090
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3976188, 15.3468742, -4.7206116, 16.2323303, -20.6299438, 20.0674801
1: -6.2940626, 15.7216339, -6.7455416, 16.6543579, -22.9484215, 22.4671745
2: -5.3095002, 17.6520042, -5.7037611, 18.6685505, -23.9780502, 23.3557606
3: -6.3730483, 22.6263657, -6.8226271, 23.8968391, -30.2698879, 29.4489918
4: -5.2157865, 20.9694633, -5.5644245, 22.1408634, -27.3566494, 26.5338860

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5177716, upper bound: 27.5337203
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5047577, upper bound: 27.5210142
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.0815225, 14.3603363, -3.8096104, 13.5872917, -17.6688137, 18.1699467
1: -5.9201736, 14.7163811, -5.5293055, 13.9026499, -19.8228226, 20.2456818
2: -5.0029421, 16.5449162, -4.6612096, 15.6515274, -20.6544685, 21.2061253
3: -5.9411888, 21.1416397, -5.5544548, 20.0211716, -25.9623604, 26.6960945
4: -4.8949690, 19.5565891, -4.5881572, 18.5381012, -23.4330711, 24.1447430

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064619, upper bound: 27.4995373
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5178870, upper bound: 27.5045526
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.0815225, 14.3603363, -4.3976188, 15.3468742, -19.4283924, 18.7579556
1: -5.9201736, 14.7163811, -6.2940626, 15.7216339, -21.6418076, 21.0104370
2: -5.0029421, 16.5449162, -5.3095002, 17.6520042, -22.6549435, 21.8544159
3: -5.9411888, 21.1416397, -6.3730483, 22.6263657, -28.5675526, 27.5146885
4: -4.8949690, 19.5565891, -5.2157865, 20.9694633, -25.8644333, 24.7723751

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4925684, upper bound: 27.4997424
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4828583
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -3.8096104, 13.5872917, -18.3079014, 20.0419369
1: -6.7455416, 16.6543579, -5.5293055, 13.9026499, -20.6481915, 22.1836624
2: -5.7037611, 18.6685505, -4.6612096, 15.6515274, -21.3552837, 23.3297596
3: -6.8226271, 23.8968391, -5.5544548, 20.0211716, -26.8437996, 29.4512939
4: -5.5644245, 22.1408634, -4.5881572, 18.5381012, -24.1025238, 26.7290192

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5072746, upper bound: 27.5064950
time: 0.48 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.3976188, 15.3468742, -20.0674801, 20.6299458
1: -6.7455416, 16.6543579, -6.2940626, 15.7216339, -22.4671745, 22.9484215
2: -5.7037611, 18.6685505, -5.3095002, 17.6520042, -23.3557606, 23.9780502
3: -6.8226271, 23.8968391, -6.3730483, 22.6263657, -29.4489918, 30.2698879
4: -5.5644245, 22.1408634, -5.2157865, 20.9694633, -26.5338860, 27.3566494

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5072746, upper bound: 27.5067001
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.0815225, 14.3603363, -4.0815225, 14.3603363, -18.4418564, 18.4418564
1: -5.9201736, 14.7163811, -5.9201736, 14.7163811, -20.6365509, 20.6365528
2: -5.0029421, 16.5449162, -5.0029421, 16.5449162, -21.5478573, 21.5478573
3: -5.9411888, 21.1416397, -5.9411888, 21.1416397, -27.0828247, 27.0828266
4: -4.8949690, 19.5565891, -4.8949690, 19.5565891, -24.4515572, 24.4515572

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5257201, upper bound: 27.5267461
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5397819
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.0815225, 14.3603363, -4.7206116, 16.2323303, -20.3138466, 19.0809460
1: -5.9201736, 14.7163811, -6.7455416, 16.6543579, -22.5745316, 21.4619217
2: -5.0029421, 16.5449162, -5.7037611, 18.6685505, -23.6714935, 22.2486744
3: -5.9411888, 21.1416397, -6.8226271, 23.8968391, -29.8380241, 27.9642639
4: -4.8949690, 19.5565891, -5.5644245, 22.1408634, -27.0358315, 25.1210098

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5257201, upper bound: 27.5276985
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5397819
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.0815225, 14.3603363, -19.0809479, 20.3138466
1: -6.7455416, 16.6543579, -5.9201736, 14.7163811, -21.4619198, 22.5745316
2: -5.7037611, 18.6685505, -5.0029421, 16.5449162, -22.2486744, 23.6714935
3: -6.8226271, 23.8968391, -5.9411888, 21.1416397, -27.9642639, 29.8380260
4: -5.5644245, 22.1408634, -4.8949690, 19.5565891, -25.1210098, 27.0358315

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5286731, upper bound: 27.5403694
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5433140
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7206116, 16.2323303, -4.7206116, 16.2323303, -20.9529343, 20.9529324
1: -6.7455416, 16.6543579, -6.7455416, 16.6543579, -23.3998985, 23.3998985
2: -5.7037611, 18.6685505, -5.7037611, 18.6685505, -24.3723106, 24.3723106
3: -6.8226271, 23.8968391, -6.8226271, 23.8968391, -30.7194633, 30.7194653
4: -5.5644245, 22.1408634, -5.5644245, 22.1408634, -27.7052860, 27.7052860

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5286731, upper bound: 27.5414444
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5439897
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.26 seconds
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4984862
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.4828583, upper bound: 27.4851750
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.4958722, upper bound: 27.4984862
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.4828583, upper bound: 27.4853801
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5280439
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4826532
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5144679, upper bound: 27.5286491
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5045526, upper bound: 27.5184922
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5177716, upper bound: 27.5337203
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5047577, upper bound: 27.5204090
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5177716, upper bound: 27.5337203
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5047577, upper bound: 27.5210142
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5064619, upper bound: 27.4995373
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5178870, upper bound: 27.5045526
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.4925684, upper bound: 27.4997424
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.4826532, upper bound: 27.4828583
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5072746, upper bound: 27.5064950
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5072746, upper bound: 27.5067001
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5184920, upper bound: 27.5081514
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5257201, upper bound: 27.5267461
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5397819
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5257201, upper bound: 27.5276985
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5397864, upper bound: 27.5397819
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5286731, upper bound: 27.5403694
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5433140
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5286731, upper bound: 27.5414444
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.26
Output dim: 3, lower bound: -27.5403756, upper bound: 27.5439897

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.5618644, 12.9171572, -4.0815225, 14.3603363, -17.9221992, 16.9986801
1: -5.1606417, 13.1869688, -5.9201736, 14.7163811, -19.8770161, 19.1071415
2: -4.3412910, 14.8576298, -5.0029421, 16.5449162, -20.8862076, 19.8605728
3: -5.1961222, 19.0373878, -5.9411888, 21.1416397, -26.3377590, 24.9785748
4: -4.2972727, 17.5959225, -4.8949690, 19.5565891, -23.8538628, 22.4908905

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4995373, upper bound: 27.5064619
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4995373, upper bound: 27.5064620
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.5618644, 12.9171572, -4.7206116, 16.2323303, -19.7941895, 17.6377659
1: -5.1606417, 13.1869688, -6.7455416, 16.6543579, -21.8149986, 19.9325104
2: -4.3412910, 14.8576298, -5.7037611, 18.6685505, -23.0098419, 20.5613899
3: -5.1961222, 19.0373878, -6.8226271, 23.8968391, -29.0929604, 25.8600121
4: -4.2972727, 17.5959225, -5.5644245, 22.1408634, -26.4381371, 23.1603451

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064950, upper bound: 27.5072745
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5064950, upper bound: 27.5072745
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1293182, 14.6288443, -4.0815225, 14.3603363, -18.4896545, 18.7103653
1: -5.9055848, 14.9554243, -5.9201736, 14.7163811, -20.6219616, 20.8755989
2: -4.9716229, 16.8073635, -5.0029421, 16.5449162, -21.5165367, 21.8103046
3: -5.9947290, 21.5765266, -5.9411888, 21.1416397, -27.1363678, 27.5177155
4: -4.9104395, 19.9661102, -4.8949690, 19.5565891, -24.4670258, 24.8610802

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089838
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089839
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.5115433, 15.8061476, -4.0815225, 14.3603363, -18.8718796, 19.8876686
1: -6.4535074, 16.1446762, -5.9201736, 14.7163811, -21.1698856, 22.0648479
2: -5.4166827, 18.0737495, -5.0029421, 16.5449162, -21.9615993, 23.0766907
3: -6.5231509, 23.1954803, -5.9411888, 21.1416397, -27.6647911, 29.1366673
4: -5.3110218, 21.4500999, -4.8949690, 19.5565891, -24.8676109, 26.3450699

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089838
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5204090
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1293182, 14.6288443, -4.7206116, 16.2323303, -20.3616447, 19.3494530
1: -5.9055848, 14.9554243, -6.7455416, 16.6543579, -22.5599422, 21.7009659
2: -4.9716229, 16.8073635, -5.7037611, 18.6685505, -23.6401730, 22.5111217
3: -5.9947290, 21.5765266, -6.8226271, 23.8968391, -29.8915672, 28.3991547
4: -4.9104395, 19.9661102, -5.5644245, 22.1408634, -27.0513020, 25.5305328

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.5115433, 15.8061476, -4.7206116, 16.2323303, -20.7438698, 20.5267563
1: -6.4535074, 16.1446762, -6.7455416, 16.6543579, -23.1078644, 22.8902149
2: -5.4166827, 18.0737495, -5.7037611, 18.6685505, -24.0852337, 23.7775116
3: -6.5231509, 23.1954803, -6.8226271, 23.8968391, -30.4199905, 30.0181065
4: -5.3110218, 21.4500999, -5.5644245, 22.1408634, -27.4518852, 27.0145226

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097963
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5210141
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.8377326, 13.7042408, -4.0815225, 14.3603363, -18.1980686, 17.7857609
1: -5.5586572, 14.0156584, -5.9201736, 14.7163811, -20.2750340, 19.9358330
2: -4.6897707, 15.7574348, -5.0029421, 16.5449162, -21.2346878, 20.7603760
3: -5.5885458, 20.1776047, -5.9411888, 21.1416397, -26.7301846, 26.1187916
4: -4.6097212, 18.6277122, -4.8949690, 19.5565891, -24.1663094, 23.5226822

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.4826532, upper bound: 27.5233461
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5267461
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1616135, 14.7958717, -4.0815225, 14.3603363, -18.5219498, 18.8773937
1: -6.0217419, 15.1112652, -5.9201736, 14.7163811, -20.7381191, 21.0314388
2: -5.0678825, 16.9106216, -5.0029421, 16.5449162, -21.6127987, 21.9135609
3: -6.0606642, 21.6794033, -5.9411888, 21.1416397, -27.2023048, 27.6205883
4: -4.9605665, 20.0012703, -4.8949690, 19.5565891, -24.5171547, 24.8962402

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5283614
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5397816
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8377326, 13.7042408, -4.7206116, 16.2323303, -20.0700569, 18.4248486
1: -5.5586572, 14.0156584, -6.7455416, 16.6543579, -22.2130146, 20.7612000
2: -4.6897707, 15.7574348, -5.7037611, 18.6685505, -23.3583221, 21.4611931
3: -5.5885458, 20.1776047, -6.8226271, 23.8968391, -29.4853859, 27.0002308
4: -4.6097212, 18.6277122, -5.5644245, 22.1408634, -26.7505836, 24.1921368

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5284407, upper bound: 27.5239205
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5284407, upper bound: 27.5239205
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1616135, 14.7958717, -4.7206116, 16.2323303, -20.3939362, 19.5164795
1: -6.0217419, 15.1112652, -6.7455416, 16.6543579, -22.6760998, 21.8568058
2: -5.0678825, 16.9106216, -5.7037611, 18.6685505, -23.7364330, 22.6143799
3: -6.0606642, 21.6794033, -6.8226271, 23.8968391, -29.9575043, 28.5020275
4: -4.9605665, 20.0012703, -5.5644245, 22.1408634, -27.1014290, 25.5656929

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5291740
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5403917
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.4364772, 15.4927216, -4.0815225, 14.3603363, -18.7968140, 19.5742435
1: -6.3414073, 15.8659678, -5.9201736, 14.7163811, -21.0577812, 21.7861404
2: -5.3552303, 17.7862968, -5.0029421, 16.5449162, -21.9001446, 22.7892380
3: -6.4260044, 22.8168812, -5.9411888, 21.1416397, -27.5676441, 28.7580700
4: -5.2513251, 21.1006165, -4.8949690, 19.5565891, -24.8079109, 25.9955864

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5303038
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5403694
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7756276, 16.5842285, -4.0815225, 14.3603363, -19.1359634, 20.6657486
1: -6.8313746, 16.9627132, -5.9201736, 14.7163811, -21.5477524, 22.8828869
2: -5.7417583, 18.9490433, -5.0029421, 16.5449162, -22.2866707, 23.9519844
3: -6.9129910, 24.3177433, -5.9411888, 21.1416397, -28.0546265, 30.2589302
4: -5.6055784, 22.4766064, -4.8949690, 19.5565891, -25.1621628, 27.3715744

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5319601
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5433136
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.4364772, 15.4927216, -4.7206116, 16.2323303, -20.6688080, 20.2133293
1: -6.3414073, 15.8659678, -6.7455416, 16.6543579, -22.9957657, 22.6115093
2: -5.3552303, 17.7862968, -5.7037611, 18.6685505, -24.0237808, 23.4900551
3: -6.4260044, 22.8168812, -6.8226271, 23.8968391, -30.3228436, 29.6395073
4: -5.2513251, 21.1006165, -5.5644245, 22.1408634, -27.3921871, 26.6650391

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7756276, 16.5842285, -4.7206116, 16.2323303, -21.0079536, 21.3048363
1: -6.8313746, 16.9627132, -6.7455416, 16.6543579, -23.4857330, 23.7082558
2: -5.7417583, 18.9490433, -5.7037611, 18.6685505, -24.4103088, 24.6528015
3: -6.9129910, 24.3177433, -6.8226271, 23.8968391, -30.8098259, 31.1403694
4: -5.6055784, 22.4766064, -5.5644245, 22.1408634, -27.7464390, 28.0410290

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398041, upper bound: 27.5327727
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5398041, upper bound: 27.5439895
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.18 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.4995373, upper bound: 27.5064619
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.4995373, upper bound: 27.5064620
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5064950, upper bound: 27.5072745
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5064950, upper bound: 27.5072745
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089838
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089839
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5089838
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.4997424, upper bound: 27.5204090
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097965
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5097963
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5046431, upper bound: 27.5210141
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.4826532, upper bound: 27.5233461
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5233089, upper bound: 27.5267461
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5283614
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5347714, upper bound: 27.5397816
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5284407, upper bound: 27.5239205
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5284407, upper bound: 27.5239205
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5291740
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5417291, upper bound: 27.5403917
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5303038
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5241587, upper bound: 27.5403694
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5319601
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5353764, upper bound: 27.5433136
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5306785, upper bound: 27.5311163
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5398041, upper bound: 27.5327727
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.18
Output dim: 3, lower bound: -27.5398041, upper bound: 27.5439895

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.5115433, 15.8061476, -4.1616135, 14.7958717, -19.3074150, 19.9677582
1: -6.4535074, 16.1446762, -6.0217419, 15.1112652, -21.5647736, 22.1664143
2: -5.4166827, 18.0737495, -5.0678825, 16.9106216, -22.3273048, 23.1416321
3: -6.5231509, 23.1954803, -6.0606642, 21.6794033, -28.2025547, 29.2561455
4: -5.3110218, 21.4500999, -4.9605665, 20.0012703, -25.3122921, 26.4106674

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4925169, upper bound: 27.5046426
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4964113, upper bound: 27.5066369
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.5115433, 15.8061476, -4.7756276, 16.5842285, -21.0957718, 20.5817738
1: -6.4535074, 16.1446762, -6.8313746, 16.9627132, -23.4162216, 22.9760494
2: -5.4166827, 18.0737495, -5.7417583, 18.9490433, -24.3657265, 23.8155079
3: -6.5231509, 23.1954803, -6.9129910, 24.3177433, -30.8408947, 30.1084671
4: -5.3110218, 21.4500999, -5.6055784, 22.4766064, -27.7876282, 27.0556755

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4941892, upper bound: 27.5054616
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5016666, upper bound: 27.5074979
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.8377326, 13.7042408, -3.8377326, 13.7042408, -17.5419731, 17.5419731
1: -5.5586572, 14.0156584, -5.5586572, 14.0156584, -19.5743160, 19.5743160
2: -4.6897707, 15.7574348, -4.6897707, 15.7574348, -20.4472046, 20.4472046
3: -5.5885458, 20.1776047, -5.5885458, 20.1776047, -25.7661514, 25.7661514
4: -4.6097212, 18.6277122, -4.6097212, 18.6277122, -23.2374344, 23.2374344

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5109197, upper bound: 27.4991147
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4911297
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.8377326, 13.7042408, -4.1616135, 14.7958717, -18.6336040, 17.8658524
1: -5.5586572, 14.0156584, -6.0217419, 15.1112652, -20.6699181, 20.0373993
2: -4.6897707, 15.7574348, -5.0678825, 16.9106216, -21.6003914, 20.8253174
3: -5.5885458, 20.1776047, -6.0606642, 21.6794033, -27.2679482, 26.2382679
4: -4.6097212, 18.6277122, -4.9605665, 20.0012703, -24.6109924, 23.5882797

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5109198, upper bound: 27.5203706
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4780598, upper bound: 27.4982269
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.1616135, 14.7958717, -3.8377326, 13.7042408, -17.8658524, 18.6336040
1: -6.0217419, 15.1112652, -5.5586572, 14.0156584, -20.0373993, 20.6699181
2: -5.0678825, 16.9106216, -4.6897707, 15.7574348, -20.8253174, 21.6003914
3: -6.0606642, 21.6794033, -5.5885458, 20.1776047, -26.2382660, 27.2679482
4: -4.9605665, 20.0012703, -4.6097212, 18.6277122, -23.5882797, 24.6109924

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5226030, upper bound: 27.5055992
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.1616135, 14.7958717, -4.1616135, 14.7958717, -18.9574833, 18.9574833
1: -6.0217419, 15.1112652, -6.0217419, 15.1112652, -21.1330032, 21.1330032
2: -5.0678825, 16.9106216, -5.0678825, 16.9106216, -21.9785042, 21.9785042
3: -6.0606642, 21.6794033, -6.0606642, 21.6794033, -27.7400665, 27.7400665
4: -4.9605665, 20.0012703, -4.9605665, 20.0012703, -24.9618378, 24.9618378

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5226033, upper bound: 27.5055992
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212413, upper bound: 27.5046310
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.8377326, 13.7042408, -4.4364772, 15.4927216, -19.3304539, 18.1407185
1: -5.5586572, 14.0156584, -6.3414073, 15.8659678, -21.4246254, 20.3570652
2: -4.6897707, 15.7574348, -5.3552303, 17.7862968, -22.4760666, 21.1126652
3: -5.5885458, 20.1776047, -6.4260044, 22.8168812, -28.4054260, 26.6036091
4: -4.6097212, 18.6277122, -5.2513251, 21.1006165, -25.7103386, 23.8790379

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258424, upper bound: 27.5190219
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5042530
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.8377326, 13.7042408, -4.7756276, 16.5842285, -20.4219608, 18.4798679
1: -5.5586572, 14.0156584, -6.8313746, 16.9627132, -22.5213699, 20.8470325
2: -4.6897707, 15.7574348, -5.7417583, 18.9490433, -23.6388130, 21.4991913
3: -5.5885458, 20.1776047, -6.9129910, 24.3177433, -29.9062881, 27.0905895
4: -4.6097212, 18.6277122, -5.6055784, 22.4766064, -27.0863266, 24.2332897

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258425, upper bound: 27.5190219
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4896675, upper bound: 27.5042530
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.1616135, 14.7958717, -4.4364772, 15.4927216, -19.6543350, 19.2323494
1: -6.0217419, 15.1112652, -6.3414073, 15.8659678, -21.8877106, 21.4526691
2: -5.0678825, 16.9106216, -5.3552303, 17.7862968, -22.8541794, 22.2658520
3: -6.0606642, 21.6794033, -6.4260044, 22.8168812, -28.8775444, 28.1054077
4: -4.9605665, 20.0012703, -5.2513251, 21.1006165, -26.0611839, 25.2525940

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392073, upper bound: 27.5267365
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378455, upper bound: 27.5257680
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1616135, 14.7958717, -4.7756276, 16.5842285, -20.7458382, 19.5714989
1: -6.0217419, 15.1112652, -6.8313746, 16.9627132, -22.9844551, 21.9426384
2: -5.0678825, 16.9106216, -5.7417583, 18.9490433, -24.0169258, 22.6523781
3: -6.0606642, 21.6794033, -6.9129910, 24.3177433, -30.3784065, 28.5923901
4: -4.9605665, 20.0012703, -5.6055784, 22.4766064, -27.4371719, 25.6068459

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392076, upper bound: 27.5267365
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5378457, upper bound: 27.5257683
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.4364772, 15.4927216, -3.8377326, 13.7042408, -18.1407185, 19.3304539
1: -6.3414073, 15.8659678, -5.5586572, 14.0156584, -20.3570652, 21.4246254
2: -5.3552303, 17.7862968, -4.6897707, 15.7574348, -21.1126652, 22.4760666
3: -6.4260044, 22.8168812, -5.5885458, 20.1776047, -26.6036091, 28.4054260
4: -5.2513251, 21.1006165, -4.6097212, 18.6277122, -23.8790379, 25.7103386

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5116008, upper bound: 27.5046928
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077340
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.4364772, 15.4927216, -4.1616135, 14.7958717, -19.2323494, 19.6543331
1: -6.3414073, 15.8659678, -6.0217419, 15.1112652, -21.4526691, 21.8877106
2: -5.3552303, 17.7862968, -5.0678825, 16.9106216, -22.2658501, 22.8541794
3: -6.4260044, 22.8168812, -6.0606642, 21.6794033, -28.1054077, 28.8775444
4: -5.2513251, 21.1006165, -4.9605665, 20.0012703, -25.2525940, 26.0611839

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5116007, upper bound: 27.5046927
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077341
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.7756276, 16.5842285, -3.8377326, 13.7042408, -18.4798660, 20.4219608
1: -6.8313746, 16.9627132, -5.5586572, 14.0156584, -20.8470325, 22.5213699
2: -5.7417583, 18.9490433, -4.6897707, 15.7574348, -21.4991932, 23.6388130
3: -6.9129910, 24.3177433, -5.5885458, 20.1776047, -27.0905895, 29.9062881
4: -5.6055784, 22.4766064, -4.6097212, 18.6277122, -24.2332897, 27.0863266

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096353
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.7756276, 16.5842285, -4.1616135, 14.7958717, -19.5714989, 20.7458382
1: -6.8313746, 16.9627132, -6.0217419, 15.1112652, -21.9426384, 22.9844551
2: -5.7417583, 18.9490433, -5.0678825, 16.9106216, -22.6523781, 24.0169258
3: -6.9129910, 24.3177433, -6.0606642, 21.6794033, -28.5923882, 30.3784065
4: -5.6055784, 22.4766064, -4.9605665, 20.0012703, -25.6068459, 27.4371719

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227556, upper bound: 27.5329363
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.4364772, 15.4927216, -4.4364772, 15.4927216, -19.9291992, 19.9291992
1: -6.3414073, 15.8659678, -6.3414073, 15.8659678, -22.2073746, 22.2073746
2: -5.3552303, 17.7862968, -5.3552303, 17.7862968, -23.1415253, 23.1415253
3: -6.4260044, 22.8168812, -6.4260044, 22.8168812, -29.2428856, 29.2428856
4: -5.2513251, 21.1006165, -5.2513251, 21.1006165, -26.3519402, 26.3519402

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5261767, upper bound: 27.5251796
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.4364772, 15.4927216, -4.7756276, 16.5842285, -21.0207062, 20.2683487
1: -6.3414073, 15.8659678, -6.8313746, 16.9627132, -23.3041191, 22.6973419
2: -5.3552303, 17.7862968, -5.7417583, 18.9490433, -24.3042717, 23.5280514
3: -6.4260044, 22.8168812, -6.9129910, 24.3177433, -30.7437477, 29.7298679
4: -5.2513251, 21.1006165, -5.6055784, 22.4766064, -27.7279301, 26.7061920

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5261766, upper bound: 27.5251796
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.7756276, 16.5842285, -4.4364772, 15.4927216, -20.2683487, 21.0207062
1: -6.8313746, 16.9627132, -6.3414073, 15.8659678, -22.6973419, 23.3041191
2: -5.7417583, 18.9490433, -5.3552303, 17.7862968, -23.5280533, 24.3042717
3: -6.9129910, 24.3177433, -6.4260044, 22.8168812, -29.7298698, 30.7437477
4: -5.6055784, 22.4766064, -5.2513251, 21.1006165, -26.7061920, 27.7279301

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327025, upper bound: 27.5270950
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380358, upper bound: 27.5056236
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.7756276, 16.5842285, -4.7756276, 16.5842285, -21.3598537, 21.3598537
1: -6.8313746, 16.9627132, -6.8313746, 16.9627132, -23.7940884, 23.7940884
2: -5.7417583, 18.9490433, -5.7417583, 18.9490433, -24.6907997, 24.6907978
3: -6.9129910, 24.3177433, -6.9129910, 24.3177433, -31.2307301, 31.2307301
4: -5.6055784, 22.4766064, -5.6055784, 22.4766064, -28.0821819, 28.0821819

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327025, upper bound: 27.5270950
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5380361, upper bound: 27.5307727
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.46 seconds
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.4925169, upper bound: 27.5046426
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.4964113, upper bound: 27.5066369
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.4941892, upper bound: 27.5054616
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5016666, upper bound: 27.5074979
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5109197, upper bound: 27.4991147
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4911297
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5109198, upper bound: 27.5203706
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.4780598, upper bound: 27.4982269
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5226030, upper bound: 27.5055992
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5226033, upper bound: 27.5055992
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5212413, upper bound: 27.5046310
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5258424, upper bound: 27.5190219
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.4945134, upper bound: 27.5042530
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5258425, upper bound: 27.5190219
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.4896675, upper bound: 27.5042530
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5392073, upper bound: 27.5267365
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5378455, upper bound: 27.5257680
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5392076, upper bound: 27.5267365
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5378457, upper bound: 27.5257683
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5116008, upper bound: 27.5046928
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077340
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5116007, upper bound: 27.5046927
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5122670, upper bound: 27.5077341
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096353
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5227556, upper bound: 27.5329363
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5236297, upper bound: 27.5096354
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5261767, upper bound: 27.5251796
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5261766, upper bound: 27.5251796
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5287497, upper bound: 27.5288713
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5327025, upper bound: 27.5270950
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5380358, upper bound: 27.5056236
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5327025, upper bound: 27.5270950
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -27.5380361, upper bound: 27.5307727

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.3112977, 12.2320118, -4.1616135, 14.7958717, -18.1071701, 16.3936253
1: -4.7633290, 12.4608803, -6.0217419, 15.1112652, -19.8745937, 18.4826221
2: -3.9792581, 14.0205498, -5.0678825, 16.9106216, -20.8898792, 19.0884323
3: -4.8087497, 18.0676193, -6.0606642, 21.6794033, -26.4881516, 24.1282845
4: -3.9650612, 16.5860634, -4.9605665, 20.0012703, -23.9663315, 21.5466309

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4982271
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4811035, upper bound: 27.4982269
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -3.8377326, 13.7042408, -17.3506985, 17.1879234
1: -5.2464833, 13.5860958, -5.5586572, 14.0156584, -19.2621422, 19.1447525
2: -4.3717928, 15.2154961, -4.6897707, 15.7574348, -20.1292267, 19.9052658
3: -5.2993393, 19.6015205, -5.5885458, 20.1776047, -25.4769440, 25.1900673
4: -4.3268204, 18.0051994, -4.6097212, 18.6277122, -22.9545326, 22.6149197

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -3.8377326, 13.7042408, -17.8234024, 18.5062714
1: -5.9215693, 14.9546156, -5.5586572, 14.0156584, -19.9372272, 20.5132732
2: -4.9671373, 16.7323151, -4.6897707, 15.7574348, -20.7245712, 21.4220848
3: -5.9684920, 21.4562817, -5.5885458, 20.1776047, -26.1460953, 27.0448265
4: -4.8678341, 19.7656593, -4.6097212, 18.6277122, -23.4955463, 24.3753815

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -4.1616135, 14.7958717, -18.4423294, 17.5118027
1: -5.2464833, 13.5860958, -6.0217419, 15.1112652, -20.3577461, 19.6078377
2: -4.3717928, 15.2154961, -5.0678825, 16.9106216, -21.2824135, 20.2833786
3: -5.2993393, 19.6015205, -6.0606642, 21.6794033, -26.9787407, 25.6621838
4: -4.3268204, 18.0051994, -4.9605665, 20.0012703, -24.3280907, 22.9657650

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -4.1616135, 14.7958717, -18.9150333, 18.8301506
1: -5.9215693, 14.9546156, -6.0217419, 15.1112652, -21.0328293, 20.9763565
2: -4.9671373, 16.7323151, -5.0678825, 16.9106216, -21.8777580, 21.8001976
3: -5.9684920, 21.4562817, -6.0606642, 21.6794033, -27.6478920, 27.5169430
4: -4.8678341, 19.7656593, -4.9605665, 20.0012703, -24.8691044, 24.7262268

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.3112977, 12.2320118, -4.4364772, 15.4927216, -18.8040199, 16.6684875
1: -4.7633290, 12.4608803, -6.3414073, 15.8659678, -20.6292973, 18.8022881
2: -3.9792581, 14.0205498, -5.3552303, 17.7862968, -21.7655525, 19.3757782
3: -4.8087497, 18.0676193, -6.4260044, 22.8168812, -27.6256313, 24.4936237
4: -3.9650612, 16.5860634, -5.2513251, 21.1006165, -25.0656776, 21.8373890

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5047237
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5057743
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.3112977, 12.2320118, -4.7756276, 16.5842285, -19.8955269, 17.0076370
1: -4.7633290, 12.4608803, -6.8313746, 16.9627132, -21.7260418, 19.2922554
2: -3.9792581, 14.0205498, -5.7417583, 18.9490433, -22.9282990, 19.7623062
3: -4.8087497, 18.0676193, -6.9129910, 24.3177433, -29.1264935, 24.9806061
4: -3.9650612, 16.5860634, -5.6055784, 22.4766064, -26.4416676, 22.1916428

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5164287, upper bound: 27.5156451
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5264372, upper bound: 27.5227532
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -4.4364772, 15.4927216, -19.1391792, 17.7866688
1: -5.2464833, 13.5860958, -6.3414073, 15.8659678, -21.1124516, 19.9275036
2: -4.3717928, 15.2154961, -5.3552303, 17.7862968, -22.1580887, 20.5707245
3: -5.2993393, 19.6015205, -6.4260044, 22.8168812, -28.1162205, 26.0275249
4: -4.3268204, 18.0051994, -5.2513251, 21.1006165, -25.4274368, 23.2565212

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5251020
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5251019
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -4.4364772, 15.4927216, -19.6118832, 19.1050148
1: -5.9215693, 14.9546156, -6.3414073, 15.8659678, -21.7875366, 21.2960224
2: -4.9671373, 16.7323151, -5.3552303, 17.7862968, -22.7534332, 22.0875454
3: -5.9684920, 21.4562817, -6.4260044, 22.8168812, -28.7853737, 27.8822861
4: -4.8678341, 19.7656593, -5.2513251, 21.1006165, -25.9684505, 25.0169830

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5251018
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5257680
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -4.7756276, 16.5842285, -20.2306862, 18.1258183
1: -5.2464833, 13.5860958, -6.8313746, 16.9627132, -22.2091942, 20.4174709
2: -4.3717928, 15.2154961, -5.7417583, 18.9490433, -23.3208351, 20.9572525
3: -5.2993393, 19.6015205, -6.9129910, 24.3177433, -29.6170826, 26.5145073
4: -4.3268204, 18.0051994, -5.6055784, 22.4766064, -26.8034267, 23.6107731

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5345981
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368592, upper bound: 27.5364981
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -4.7756276, 16.5842285, -20.7033901, 19.4441662
1: -5.9215693, 14.9546156, -6.8313746, 16.9627132, -22.8842812, 21.7859898
2: -4.9671373, 16.7323151, -5.7417583, 18.9490433, -23.9161797, 22.4740715
3: -5.9684920, 21.4562817, -6.9129910, 24.3177433, -30.2862320, 28.3692665
4: -4.8678341, 19.7656593, -5.6055784, 22.4766064, -27.3444405, 25.3712349

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5345981
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5364978
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -3.8377326, 13.7042408, -17.9439144, 18.9784660
1: -6.0757089, 15.4544592, -5.5586572, 14.0156584, -20.0913677, 21.0131130
2: -5.0678725, 17.2726898, -4.6897707, 15.7574348, -20.8253078, 21.9624596
3: -6.1715178, 22.2449474, -5.5885458, 20.1776047, -26.3491230, 27.8334923
4: -5.0068088, 20.4940319, -4.6097212, 18.6277122, -23.6345215, 25.1037521

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -3.8377326, 13.7042408, -18.5194244, 20.5504875
1: -6.8487349, 17.0677147, -5.5586572, 14.0156584, -20.8643932, 22.6263695
2: -5.7475977, 19.0672207, -4.6897707, 15.7574348, -21.5050316, 23.7569904
3: -6.9301233, 24.4769001, -5.5885458, 20.1776047, -27.1077271, 30.0654449
4: -5.6081338, 22.6030693, -4.6097212, 18.6277122, -24.2358456, 27.2127914

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096354
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096354
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -4.1616135, 14.7958717, -19.0355473, 19.3023453
1: -6.0757089, 15.4544592, -6.0217419, 15.1112652, -21.1869678, 21.4761982
2: -5.0678725, 17.2726898, -5.0678825, 16.9106216, -21.9784946, 22.3405724
3: -6.1715178, 22.2449474, -6.0606642, 21.6794033, -27.8509216, 28.3056107
4: -5.0068088, 20.4940319, -4.9605665, 20.0012703, -25.0080795, 25.4545975

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329350
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329350
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -4.1616135, 14.7958717, -19.6110573, 20.8743687
1: -6.8487349, 17.0677147, -6.0217419, 15.1112652, -21.9599991, 23.0894547
2: -5.7475977, 19.0672207, -5.0678825, 16.9106216, -22.6582165, 24.1351032
3: -6.9301233, 24.4769001, -6.0606642, 21.6794033, -28.6095276, 30.5375633
4: -5.6081338, 22.6030693, -4.9605665, 20.0012703, -25.6094036, 27.5636349

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.8882565, 14.0178547, -4.4364772, 15.4927216, -19.3809776, 18.4543324
1: -5.5631576, 14.3240805, -6.3414073, 15.8659678, -21.4291248, 20.6654854
2: -4.6534100, 16.0694389, -5.3552303, 17.7862968, -22.4397068, 21.4246655
3: -5.6674542, 20.6948471, -6.4260044, 22.8168812, -28.4843349, 27.1208515
4: -4.6273518, 19.0684605, -5.2513251, 21.1006165, -25.7279682, 24.3197842

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249857, upper bound: 27.5251523
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251796
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.4765372, 15.6225147, -4.4364772, 15.4927216, -19.9692574, 20.0589924
1: -6.3536415, 15.9724932, -6.3414073, 15.8659678, -22.2196083, 22.3139000
2: -5.3602648, 17.9039478, -5.3552303, 17.7862968, -23.1465607, 23.2591763
3: -6.4377475, 22.9774036, -6.4260044, 22.8168812, -29.2546291, 29.4034081
4: -5.2538066, 21.2279301, -5.2513251, 21.1006165, -26.3544216, 26.4792538

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5282051
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5288713
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8882565, 14.0178547, -4.7756276, 16.5842285, -20.4724808, 18.7934818
1: -5.5631576, 14.3240805, -6.8313746, 16.9627132, -22.5258694, 21.1554546
2: -4.6534100, 16.0694389, -5.7417583, 18.9490433, -23.6024532, 21.8111935
3: -5.6674542, 20.6948471, -6.9129910, 24.3177433, -29.9851971, 27.6078339
4: -4.6273518, 19.0684605, -5.6055784, 22.4766064, -27.1039581, 24.6740360

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5260153, upper bound: 27.5334170
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5260152, upper bound: 27.5352874
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.4765372, 15.6225147, -4.7756276, 16.5842285, -21.0607624, 20.3981400
1: -6.3536415, 15.9724932, -6.8313746, 16.9627132, -23.3163548, 22.8038673
2: -5.3602648, 17.9039478, -5.7417583, 18.9490433, -24.3093071, 23.6457043
3: -6.4377475, 22.9774036, -6.9129910, 24.3177433, -30.7554913, 29.8903904
4: -5.2538066, 21.2279301, -5.6055784, 22.4766064, -27.7304115, 26.8335056

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5275651, upper bound: 27.5378473
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5275650, upper bound: 27.5394851
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -4.4364772, 15.4927216, -19.7323952, 19.5772114
1: -6.0757089, 15.4544592, -6.3414073, 15.8659678, -21.9416771, 21.7958641
2: -5.0678725, 17.2726898, -5.3552303, 17.7862968, -22.8541679, 22.6279202
3: -6.1715178, 22.2449474, -6.4260044, 22.8168812, -28.9883995, 28.6709518
4: -5.0068088, 20.4940319, -5.2513251, 21.1006165, -26.1074257, 25.7453556

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327023, upper bound: 27.5270949
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270949
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -4.4364772, 15.4927216, -20.3079071, 21.1492329
1: -6.8487349, 17.0677147, -6.3414073, 15.8659678, -22.7147026, 23.4091225
2: -5.7475977, 19.0672207, -5.3552303, 17.7862968, -23.5338898, 24.4224491
3: -6.9301233, 24.4769001, -6.4260044, 22.8168812, -29.7470055, 30.9029045
4: -5.6081338, 22.6030693, -5.2513251, 21.1006165, -26.7087498, 27.8543911

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371678, upper bound: 27.5301065
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5371678, upper bound: 27.5307727
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -4.7756276, 16.5842285, -20.8239002, 19.9163609
1: -6.0757089, 15.4544592, -6.8313746, 16.9627132, -23.0384197, 22.2858334
2: -5.0678725, 17.2726898, -5.7417583, 18.9490433, -24.0169144, 23.0144482
3: -6.1715178, 22.2449474, -6.9129910, 24.3177433, -30.4892616, 29.1579342
4: -5.0068088, 20.4940319, -5.6055784, 22.4766064, -27.4834156, 26.0996094

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5368323
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5369100
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -4.7756276, 16.5842285, -21.3994122, 21.4883823
1: -6.8487349, 17.0677147, -6.8313746, 16.9627132, -23.8114471, 23.8990898
2: -5.7475977, 19.0672207, -5.7417583, 18.9490433, -24.6966362, 24.8089752
3: -6.9301233, 24.4769001, -6.9129910, 24.3177433, -31.2478676, 31.3898869
4: -5.6081338, 22.6030693, -5.6055784, 22.4766064, -28.0847397, 28.2086430

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392167, upper bound: 27.5377042
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5392168, upper bound: 27.5388739
time: 0.74 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.32 seconds
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.4896675, upper bound: 27.4982271
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.4811035, upper bound: 27.4982269
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
NS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
NS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5212409, upper bound: 27.5046307
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
NS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
NS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5347419, upper bound: 27.5328193
NS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5047237
NS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.4994109, upper bound: 27.5057743
NS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5164287, upper bound: 27.5156451
NS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5264372, upper bound: 27.5227532
NS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5251020
NS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5348042, upper bound: 27.5251019
NS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5251018
NS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5348039, upper bound: 27.5257680
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5345981
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5368592, upper bound: 27.5364981
NS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5368590, upper bound: 27.5345981
NS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5368589, upper bound: 27.5364978
NS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
NS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5227553, upper bound: 27.5070346
NS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096354
NS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5236294, upper bound: 27.5096354
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329350
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5362564, upper bound: 27.5329350
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5371304, upper bound: 27.5334066
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5249857, upper bound: 27.5251523
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5249858, upper bound: 27.5251796
NS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5282051
NS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5258300, upper bound: 27.5288713
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5260153, upper bound: 27.5334170
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5260152, upper bound: 27.5352874
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5275651, upper bound: 27.5378473
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5275650, upper bound: 27.5394851
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5327023, upper bound: 27.5270949
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5327024, upper bound: 27.5270949
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5371678, upper bound: 27.5301065
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5371678, upper bound: 27.5307727
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5368323
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5379823, upper bound: 27.5369100
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5392167, upper bound: 27.5377042
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.32
Output dim: 3, lower bound: -27.5392168, upper bound: 27.5388739

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -3.3112977, 12.2320118, -15.8784695, 16.6614895
1: -5.2464833, 13.5860958, -4.7633290, 12.4608803, -17.7073612, 18.3494244
2: -4.3717928, 15.2154961, -3.9792581, 14.0205498, -18.3923416, 19.1947536
3: -5.2993393, 19.6015205, -4.8087497, 18.0676193, -23.3669586, 24.4102707
4: -4.3268204, 18.0051994, -3.9650612, 16.5860634, -20.9128838, 21.9702606

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087120, upper bound: 27.5053748
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217726, upper bound: 27.5055992
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -3.8090727, 13.6119699, -17.2584229, 17.1592655
1: -5.2464833, 13.5860958, -5.4754810, 13.8948555, -19.1413383, 19.0615768
2: -4.3717928, 15.2154961, -4.6048141, 15.6138811, -19.9856720, 19.8203106
3: -5.2993393, 19.6015205, -5.5135355, 20.0045624, -25.3039017, 25.1150551
4: -4.3268204, 18.0051994, -4.5309124, 18.4355297, -22.7623501, 22.5361118

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087122, upper bound: 27.5053748
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217726, upper bound: 27.5055992
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -3.3112977, 12.2320118, -16.3511734, 17.9798355
1: -5.9215693, 14.9546156, -4.7633290, 12.4608803, -18.3824482, 19.7179451
2: -4.9671373, 16.7323151, -3.9792581, 14.0205498, -18.9876862, 20.7115726
3: -5.9684920, 21.4562817, -4.8087497, 18.0676193, -24.0361099, 26.2650318
4: -4.8678341, 19.7656593, -3.9650612, 16.5860634, -21.4538975, 23.7307205

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5092667, upper bound: 27.5046307
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5203528, upper bound: 27.5043498
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -3.8090727, 13.6119699, -17.7311325, 18.4776115
1: -5.9215693, 14.9546156, -5.4754810, 13.8948555, -19.8164253, 20.4300957
2: -4.9671373, 16.7323151, -4.6048141, 15.6138811, -20.5810165, 21.3371296
3: -5.9684920, 21.4562817, -5.5135355, 20.0045624, -25.9730511, 26.9698162
4: -4.8678341, 19.7656593, -4.5309124, 18.4355297, -23.3033638, 24.2965717

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5092672, upper bound: 27.5046309
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5203525, upper bound: 27.5043496
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -3.6464579, 13.3501921, -16.9966488, 16.9966488
1: -5.2464833, 13.5860958, -5.2464833, 13.5860958, -18.8325787, 18.8325787
2: -4.3717928, 15.2154961, -4.3717928, 15.2154961, -19.5872879, 19.5872879
3: -5.2993393, 19.6015205, -5.2993393, 19.6015205, -24.9008598, 24.9008598
4: -4.3268204, 18.0051994, -4.3268204, 18.0051994, -22.3320179, 22.3320179

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -4.1191621, 14.6685381, -18.3149967, 17.4693546
1: -5.2464833, 13.5860958, -5.9215693, 14.9546156, -20.2010994, 19.5076656
2: -4.3717928, 15.2154961, -4.9671373, 16.7323151, -21.1041069, 20.1826324
3: -5.2993393, 19.6015205, -5.9684920, 21.4562817, -26.7556190, 25.5700092
4: -4.3268204, 18.0051994, -4.8678341, 19.7656593, -24.0924797, 22.8730335

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -3.6464579, 13.3501921, -17.4693546, 18.3149967
1: -5.9215693, 14.9546156, -5.2464833, 13.5860958, -19.5076656, 20.2010994
2: -4.9671373, 16.7323151, -4.3717928, 15.2154961, -20.1826324, 21.1041069
3: -5.9684920, 21.4562817, -5.2993393, 19.6015205, -25.5700092, 26.7556210
4: -4.8678341, 19.7656593, -4.3268204, 18.0051994, -22.8730335, 24.0924797

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340553, upper bound: 27.5323892
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -4.1191621, 14.6685381, -18.7877007, 18.7877007
1: -5.9215693, 14.9546156, -5.9215693, 14.9546156, -20.8761845, 20.8761845
2: -4.9671373, 16.7323151, -4.9671373, 16.7323151, -21.6994514, 21.6994514
3: -5.9684920, 21.4562817, -5.9684920, 21.4562817, -27.4247704, 27.4247723
4: -4.8678341, 19.7656593, -4.8678341, 19.7656593, -24.6334934, 24.6334934

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340553, upper bound: 27.5323892
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.3112977, 12.2320118, -4.7579708, 16.5425053, -19.8538036, 16.9899769
1: -4.7633290, 12.4608803, -6.8075414, 16.9199429, -21.6832714, 19.2684212
2: -3.9792581, 14.0205498, -5.7232504, 18.9006500, -22.8799076, 19.7437992
3: -4.8087497, 18.0676193, -6.8910208, 24.2590141, -29.0677643, 24.9586391
4: -3.9650612, 16.5860634, -5.5892906, 22.4202957, -26.3853569, 22.1753540

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4887731, upper bound: 27.5067612
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5263602, upper bound: 27.5222524
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -3.8882565, 14.0178547, -17.6643124, 17.2384453
1: -5.2464833, 13.5860958, -5.5631576, 14.3240805, -19.5705624, 19.1492538
2: -4.3717928, 15.2154961, -4.6534100, 16.0694389, -20.4412289, 19.8689060
3: -5.2993393, 19.6015205, -5.6674542, 20.6948471, -25.9941864, 25.2689743
4: -4.3268204, 18.0051994, -4.6273518, 19.0684605, -23.3952808, 22.6325512

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5067265, upper bound: 27.5082477
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4924928, upper bound: 27.4758690
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -4.4765372, 15.6225147, -19.2689705, 17.8267288
1: -5.2464833, 13.5860958, -6.3536415, 15.9724932, -21.2189770, 19.9397373
2: -4.3717928, 15.2154961, -5.3602648, 17.9039478, -22.2757397, 20.5757599
3: -5.2993393, 19.6015205, -6.4377475, 22.9774036, -28.2767410, 26.0392685
4: -4.3268204, 18.0051994, -5.2538066, 21.2279301, -25.5547504, 23.2590027

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5067274, upper bound: 27.5082496
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4924929, upper bound: 27.4799054
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -3.8882565, 14.0178547, -18.1370163, 18.5567932
1: -5.9215693, 14.9546156, -5.5631576, 14.3240805, -20.2456493, 20.5177727
2: -4.9671373, 16.7323151, -4.6534100, 16.0694389, -21.0365753, 21.3857250
3: -5.9684920, 21.4562817, -5.6674542, 20.6948471, -26.6633358, 27.1237354
4: -4.8678341, 19.7656593, -4.6273518, 19.0684605, -23.9362946, 24.3930111

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233130, upper bound: 27.5249567
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343988, upper bound: 27.5246757
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -4.4765372, 15.6225147, -19.7416763, 19.1450748
1: -5.9215693, 14.9546156, -6.3536415, 15.9724932, -21.8940620, 21.3082581
2: -4.9671373, 16.7323151, -5.3602648, 17.9039478, -22.8710861, 22.0925789
3: -5.9684920, 21.4562817, -6.4377475, 22.9774036, -28.9458904, 27.8940277
4: -4.8678341, 19.7656593, -5.2538066, 21.2279301, -26.0957642, 25.0194645

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233130, upper bound: 27.5249568
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343984, upper bound: 27.5246755
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -4.2396750, 15.1407347, -18.7871933, 17.5898647
1: -5.2464833, 13.5860958, -6.0757089, 15.4544592, -20.7009392, 19.6618042
2: -4.3717928, 15.2154961, -5.0678725, 17.2726898, -21.6444817, 20.2833691
3: -5.2993393, 19.6015205, -6.1715178, 22.2449474, -27.5442867, 25.7730389
4: -4.3268204, 18.0051994, -5.0068088, 20.4940319, -24.8208523, 23.0120087

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087325, upper bound: 27.5081963
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4827818
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.6464579, 13.3501921, -4.8151884, 16.7127552, -20.3592129, 18.1653767
1: -5.2464833, 13.5860958, -6.8487349, 17.0677147, -22.3141975, 20.4348297
2: -4.3717928, 15.2154961, -5.7475977, 19.0672207, -23.4390125, 20.9630890
3: -5.2993393, 19.6015205, -6.9301233, 24.4769001, -29.7762394, 26.5316429
4: -4.3268204, 18.0051994, -5.6081338, 22.6030693, -26.9298878, 23.6133327

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5087309, upper bound: 27.5081939
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4827818
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -4.2396750, 15.1407347, -19.2598972, 18.9082127
1: -5.9215693, 14.9546156, -6.0757089, 15.4544592, -21.3760262, 21.0303249
2: -4.9671373, 16.7323151, -5.0678725, 17.2726898, -22.2398262, 21.8001881
3: -5.9684920, 21.4562817, -6.1715178, 22.2449474, -28.2134361, 27.6278000
4: -4.8678341, 19.7656593, -5.0068088, 20.4940319, -25.3618660, 24.7724686

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5248834, upper bound: 27.5342407
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5364534, upper bound: 27.5340457
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.1191621, 14.6685381, -4.8151884, 16.7127552, -20.8319168, 19.4837246
1: -5.9215693, 14.9546156, -6.8487349, 17.0677147, -22.9892807, 21.8033504
2: -4.9671373, 16.7323151, -5.7475977, 19.0672207, -24.0343571, 22.4799099
3: -5.9684920, 21.4562817, -6.9301233, 24.4769001, -30.4453888, 28.3864059
4: -4.8678341, 19.7656593, -5.6081338, 22.6030693, -27.4709034, 25.3737926

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5248835, upper bound: 27.5342407
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5364535, upper bound: 27.5340459
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -3.3112977, 12.2320118, -16.4716873, 18.4520321
1: -6.0757089, 15.4544592, -4.7633290, 12.4608803, -18.5365849, 20.2177887
2: -5.0678725, 17.2726898, -3.9792581, 14.0205498, -19.0884228, 21.2519474
3: -6.1715178, 22.2449474, -4.8087497, 18.0676193, -24.2391376, 27.0536976
4: -5.0068088, 20.4940319, -3.9650612, 16.5860634, -21.5928726, 24.4590931

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5090457, upper bound: 27.5059294
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5219007, upper bound: 27.5070346
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -3.8090727, 13.6119699, -17.8516388, 18.9498081
1: -6.0757089, 15.4544592, -5.4754810, 13.8948555, -19.9705620, 20.9299393
2: -5.0678725, 17.2726898, -4.6048141, 15.6138811, -20.6817513, 21.8775043
3: -6.1715178, 22.2449474, -5.5135355, 20.0045624, -26.1760807, 27.7584839
4: -5.0068088, 20.4940319, -4.5309124, 18.4355297, -23.4423389, 25.0249443

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5146069, upper bound: 27.5070346
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5196645, upper bound: 27.5033578
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -3.3112977, 12.2320118, -17.0471973, 20.0240536
1: -6.8487349, 17.0677147, -4.7633290, 12.4608803, -19.3096142, 21.8310432
2: -5.7475977, 19.0672207, -3.9792581, 14.0205498, -19.7681446, 23.0464783
3: -6.9301233, 24.4769001, -4.8087497, 18.0676193, -24.9977417, 29.2856503
4: -5.6081338, 22.6030693, -3.9650612, 16.5860634, -22.1941967, 26.5681305

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136903, upper bound: 27.4967301
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217360, upper bound: 27.5072179
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -3.8090727, 13.6119699, -18.4271488, 20.5218277
1: -6.8487349, 17.0677147, -5.4754810, 13.8948555, -20.7435913, 22.5431957
2: -5.7475977, 19.0672207, -4.6048141, 15.6138811, -21.3614731, 23.6720352
3: -6.9301233, 24.4769001, -5.5135355, 20.0045624, -26.9346848, 29.9904366
4: -5.6081338, 22.6030693, -4.5309124, 18.4355297, -24.0436630, 27.1339817

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5136898, upper bound: 27.4967299
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217359, upper bound: 27.5072179
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -3.6464579, 13.3501921, -17.5898647, 18.7871933
1: -6.0757089, 15.4544592, -5.2464833, 13.5860958, -19.6618042, 20.7009392
2: -5.0678725, 17.2726898, -4.3717928, 15.2154961, -20.2833691, 21.6444817
3: -6.1715178, 22.2449474, -5.2993393, 19.6015205, -25.7730389, 27.5442867
4: -5.0068088, 20.4940319, -4.3268204, 18.0051994, -23.0120087, 24.8208523

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5280571, upper bound: 27.5329329
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5334089, upper bound: 27.5313172
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -4.1191621, 14.6685381, -18.9082127, 19.2598972
1: -6.0757089, 15.4544592, -5.9215693, 14.9546156, -21.0303249, 21.3760242
2: -5.0678725, 17.2726898, -4.9671373, 16.7323151, -21.8001881, 22.2398262
3: -6.1715178, 22.2449474, -5.9684920, 21.4562817, -27.6278000, 28.2134361
4: -5.0068088, 20.4940319, -4.8678341, 19.7656593, -24.7724686, 25.3618660

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5280571, upper bound: 27.5329329
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5334089, upper bound: 27.5313172
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -3.6464579, 13.3501921, -18.1653786, 20.3592129
1: -6.8487349, 17.0677147, -5.2464833, 13.5860958, -20.4348297, 22.3141956
2: -5.7475977, 19.0672207, -4.3717928, 15.2154961, -20.9630909, 23.4390144
3: -6.9301233, 24.4769001, -5.2993393, 19.6015205, -26.5316429, 29.7762394
4: -5.6081338, 22.6030693, -4.3268204, 18.0051994, -23.6133327, 26.9298878

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5272090, upper bound: 27.5247717
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5359641, upper bound: 27.5317517
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -4.1191621, 14.6685381, -19.4837246, 20.8319168
1: -6.8487349, 17.0677147, -5.9215693, 14.9546156, -21.8033504, 22.9892845
2: -5.7475977, 19.0672207, -4.9671373, 16.7323151, -22.4799099, 24.0343590
3: -6.9301233, 24.4769001, -5.9684920, 21.4562817, -28.3864059, 30.4453888
4: -5.6081338, 22.6030693, -4.8678341, 19.7656593, -25.3737926, 27.4709034

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5272090, upper bound: 27.5247717
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5359641, upper bound: 27.5317517
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.8882565, 14.0178547, -3.8882565, 14.0178547, -17.9061089, 17.9061089
1: -5.5631576, 14.3240805, -5.5631576, 14.3240805, -19.8872356, 19.8872356
2: -4.6534100, 16.0694389, -4.6534100, 16.0694389, -20.7228489, 20.7228489
3: -5.6674542, 20.6948471, -5.6674542, 20.6948471, -26.3623009, 26.3623009
4: -4.6273518, 19.0684605, -4.6273518, 19.0684605, -23.6958122, 23.6958122

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5164413, upper bound: 27.5251433
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206792, upper bound: 27.5207235
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.8882565, 14.0178547, -4.4765372, 15.6225147, -19.5107651, 18.4943924
1: -5.5631576, 14.3240805, -6.3536415, 15.9724932, -21.5356503, 20.6777210
2: -4.6534100, 16.0694389, -5.3602648, 17.9039478, -22.5573578, 21.4297009
3: -5.6674542, 20.6948471, -6.4377475, 22.9774036, -28.6448574, 27.1325951
4: -4.6273518, 19.0684605, -5.2538066, 21.2279301, -25.8552818, 24.3222656

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5164413, upper bound: 27.5251702
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206792, upper bound: 27.5208300
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.4765372, 15.6225147, -3.8882565, 14.0178547, -18.4943924, 19.5107651
1: -6.3536415, 15.9724932, -5.5631576, 14.3240805, -20.6777210, 21.5356503
2: -5.3602648, 17.9039478, -4.6534100, 16.0694389, -21.4297009, 22.5573578
3: -6.4377475, 22.9774036, -5.6674542, 20.6948471, -27.1325951, 28.6448574
4: -5.2538066, 21.2279301, -4.6273518, 19.0684605, -24.3222656, 25.8552818

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5175703, upper bound: 27.5271027
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5255464, upper bound: 27.5280599
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.4765372, 15.6225147, -4.4765372, 15.6225147, -20.0990486, 20.0990486
1: -6.3536415, 15.9724932, -6.3536415, 15.9724932, -22.3261337, 22.3261337
2: -5.3602648, 17.9039478, -5.3602648, 17.9039478, -23.2642136, 23.2642136
3: -6.4377475, 22.9774036, -6.4377475, 22.9774036, -29.4151497, 29.4151516
4: -5.2538066, 21.2279301, -5.2538066, 21.2279301, -26.4817352, 26.4817352

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5175704, upper bound: 27.5275258
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5255464, upper bound: 27.5285877
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.8882565, 14.0178547, -4.2396750, 15.1407347, -19.0289860, 18.2575302
1: -5.5631576, 14.3240805, -6.0757089, 15.4544592, -21.0176125, 20.3997860
2: -4.6534100, 16.0694389, -5.0678725, 17.2726898, -21.9260998, 21.1373100
3: -5.6674542, 20.6948471, -6.1715178, 22.2449474, -27.9124012, 26.8663654
4: -4.6273518, 19.0684605, -5.0068088, 20.4940319, -25.1213837, 24.0752697

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5179895, upper bound: 27.5334112
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5223216, upper bound: 27.5282791
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.8882565, 14.0178547, -4.8151884, 16.7127552, -20.6010113, 18.8330402
1: -5.5631576, 14.3240805, -6.8487349, 17.0677147, -22.6308708, 21.1728153
2: -4.6534100, 16.0694389, -5.7475977, 19.0672207, -23.7206306, 21.8170319
3: -5.6674542, 20.6948471, -6.9301233, 24.4769001, -30.1443539, 27.6249695
4: -4.6273518, 19.0684605, -5.6081338, 22.6030693, -27.2304211, 24.6765938

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5179897, upper bound: 27.5334114
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5223217, upper bound: 27.5302325
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.4765372, 15.6225147, -4.2396750, 15.1407347, -19.6172714, 19.8621864
1: -6.3536415, 15.9724932, -6.0757089, 15.4544592, -21.8080997, 22.0482006
2: -5.3602648, 17.9039478, -5.0678725, 17.2726898, -22.6329536, 22.9718208
3: -6.4377475, 22.9774036, -6.1715178, 22.2449474, -28.6826954, 29.1489220
4: -5.2538066, 21.2279301, -5.0068088, 20.4940319, -25.7478371, 26.2347393

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5184113, upper bound: 27.5359805
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5272810, upper bound: 27.5374719
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.4765372, 15.6225147, -4.8151884, 16.7127552, -21.1892929, 20.4376984
1: -6.3536415, 15.9724932, -6.8487349, 17.0677147, -23.4213562, 22.8212280
2: -5.3602648, 17.9039478, -5.7475977, 19.0672207, -24.4274845, 23.6515427
3: -6.4377475, 22.9774036, -6.9301233, 24.4769001, -30.9146481, 29.9075279
4: -5.2538066, 21.2279301, -5.6081338, 22.6030693, -27.8568726, 26.8360634

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5184113, upper bound: 27.5374000
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5272811, upper bound: 27.5390456
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -3.8882565, 14.0178547, -18.2575302, 19.0289860
1: -6.0757089, 15.4544592, -5.5631576, 14.3240805, -20.3997860, 21.0176125
2: -5.0678725, 17.2726898, -4.6534100, 16.0694389, -21.1373100, 21.9260998
3: -6.1715178, 22.2449474, -5.6674542, 20.6948471, -26.8663654, 27.9124012
4: -5.0068088, 20.4940319, -4.6273518, 19.0684605, -24.0752697, 25.1213837

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4932452, upper bound: 27.5032880
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4958982, upper bound: 27.4960266
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -4.4765372, 15.6225147, -19.8621864, 19.6172714
1: -6.0757089, 15.4544592, -6.3536415, 15.9724932, -22.0482006, 21.8080997
2: -5.0678725, 17.2726898, -5.3602648, 17.9039478, -22.9718208, 22.6329536
3: -6.1715178, 22.2449474, -6.4377475, 22.9774036, -29.1489220, 28.6826954
4: -5.0068088, 20.4940319, -5.2538066, 21.2279301, -26.2347393, 25.7478371

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4932454, upper bound: 27.5032881
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4958981, upper bound: 27.4960270
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -3.8882565, 14.0178547, -18.8330402, 20.6010113
1: -6.8487349, 17.0677147, -5.5631576, 14.3240805, -21.1728153, 22.6308708
2: -5.7475977, 19.0672207, -4.6534100, 16.0694389, -21.8170319, 23.7206306
3: -6.9301233, 24.4769001, -5.6674542, 20.6948471, -27.6249695, 30.1443539
4: -5.6081338, 22.6030693, -4.6273518, 19.0684605, -24.6765938, 27.2304211

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5247142, upper bound: 27.5170793
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5366675, upper bound: 27.5298710
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -4.4765372, 15.6225147, -20.4376984, 21.1892929
1: -6.8487349, 17.0677147, -6.3536415, 15.9724932, -22.8212280, 23.4213562
2: -5.7475977, 19.0672207, -5.3602648, 17.9039478, -23.6515427, 24.4274845
3: -6.9301233, 24.4769001, -6.4377475, 22.9774036, -29.9075279, 30.9146481
4: -5.6081338, 22.6030693, -5.2538066, 21.2279301, -26.8360634, 27.8568726

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5247142, upper bound: 27.5170792
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5366675, upper bound: 27.5305212
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -4.2396750, 15.1407347, -19.3804073, 19.3804073
1: -6.0757089, 15.4544592, -6.0757089, 15.4544592, -21.5301647, 21.5301647
2: -5.0678725, 17.2726898, -5.0678725, 17.2726898, -22.3405628, 22.3405628
3: -6.1715178, 22.2449474, -6.1715178, 22.2449474, -28.4164658, 28.4164658
4: -5.0068088, 20.4940319, -5.0068088, 20.4940319, -25.5008411, 25.5008411

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093615
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.2396750, 15.1407347, -4.8151884, 16.7127552, -20.9524307, 19.9559193
1: -6.0757089, 15.4544592, -6.8487349, 17.0677147, -23.1434212, 22.3031940
2: -5.0678725, 17.2726898, -5.7475977, 19.0672207, -24.1350918, 23.0202866
3: -6.1715178, 22.2449474, -6.9301233, 24.4769001, -30.6484184, 29.1750717
4: -5.0068088, 20.4940319, -5.6081338, 22.6030693, -27.6098785, 26.1021652

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093601
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -4.2396750, 15.1407347, -19.9559193, 20.9524307
1: -6.8487349, 17.0677147, -6.0757089, 15.4544592, -22.3031940, 23.1434212
2: -5.7475977, 19.0672207, -5.0678725, 17.2726898, -23.0202866, 24.1350918
3: -6.9301233, 24.4769001, -6.1715178, 22.2449474, -29.1750717, 30.6484184
4: -5.6081338, 22.6030693, -5.0068088, 20.4940319, -26.1021652, 27.6098785

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5283869, upper bound: 27.5254426
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387269, upper bound: 27.5373719
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.8151884, 16.7127552, -4.8151884, 16.7127552, -21.5279427, 21.5279427
1: -6.8487349, 17.0677147, -6.8487349, 17.0677147, -23.9164505, 23.9164486
2: -5.7475977, 19.0672207, -5.7475977, 19.0672207, -24.8148136, 24.8148136
3: -6.9301233, 24.4769001, -6.9301233, 24.4769001, -31.4070244, 31.4070244
4: -5.6081338, 22.6030693, -5.6081338, 22.6030693, -28.2112026, 28.2112026

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5283867, upper bound: 27.5254423
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5387271, upper bound: 27.5384336
time: 0.80 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.44 seconds
NS_A2_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5087120, upper bound: 27.5053748
NS_A2_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5217726, upper bound: 27.5055992
NS_A2_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5087122, upper bound: 27.5053748
NS_A2_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5217726, upper bound: 27.5055992
NS_A2_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5092667, upper bound: 27.5046307
NS_A2_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5203528, upper bound: 27.5043498
NS_A2_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5092672, upper bound: 27.5046309
NS_A2_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5203525, upper bound: 27.5043496
NS_A2_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
NS_A2_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
NS_A2_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324873
NS_A2_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5324614
NS_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
NS_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5340553, upper bound: 27.5323892
NS_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323993
NS_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5340553, upper bound: 27.5323892
NS_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4887731, upper bound: 27.5067612
NS_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5263602, upper bound: 27.5222524
NS_A2_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5067265, upper bound: 27.5082477
NS_A2_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4924928, upper bound: 27.4758690
NS_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5067274, upper bound: 27.5082496
NS_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4924929, upper bound: 27.4799054
NS_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5233130, upper bound: 27.5249567
NS_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5343988, upper bound: 27.5246757
NS_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5233130, upper bound: 27.5249568
NS_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5343984, upper bound: 27.5246755
NS_A2_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5087325, upper bound: 27.5081963
NS_A2_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4827818
NS_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5087309, upper bound: 27.5081939
NS_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4990864, upper bound: 27.4827818
NS_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5248834, upper bound: 27.5342407
NS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5364534, upper bound: 27.5340457
NS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5248835, upper bound: 27.5342407
NS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5364535, upper bound: 27.5340459
NS_A2_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5090457, upper bound: 27.5059294
NS_A2_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5219007, upper bound: 27.5070346
NS_A2_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5146069, upper bound: 27.5070346
NS_A2_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5196645, upper bound: 27.5033578
NS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5136903, upper bound: 27.4967301
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5217360, upper bound: 27.5072179
NS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5136898, upper bound: 27.4967299
NS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5217359, upper bound: 27.5072179
NS_A2_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5280571, upper bound: 27.5329329
NS_A2_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5334089, upper bound: 27.5313172
NS_A2_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5280571, upper bound: 27.5329329
NS_A2_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5334089, upper bound: 27.5313172
NS_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5272090, upper bound: 27.5247717
NS_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5359641, upper bound: 27.5317517
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5272090, upper bound: 27.5247717
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5359641, upper bound: 27.5317517
NS_A2_B2_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5164413, upper bound: 27.5251433
NS_A2_B2_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5206792, upper bound: 27.5207235
NS_A2_B2_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5164413, upper bound: 27.5251702
NS_A2_B2_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5206792, upper bound: 27.5208300
NS_A2_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5175703, upper bound: 27.5271027
NS_A2_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5255464, upper bound: 27.5280599
NS_A2_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5175704, upper bound: 27.5275258
NS_A2_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5255464, upper bound: 27.5285877
NS_A2_B2_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5179895, upper bound: 27.5334112
NS_A2_B2_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5223216, upper bound: 27.5282791
NS_A2_B2_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5179897, upper bound: 27.5334114
NS_A2_B2_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5223217, upper bound: 27.5302325
NS_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5184113, upper bound: 27.5359805
NS_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5272810, upper bound: 27.5374719
NS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5184113, upper bound: 27.5374000
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5272811, upper bound: 27.5390456
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4932452, upper bound: 27.5032880
NS_A2_B2_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4958982, upper bound: 27.4960266
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4932454, upper bound: 27.5032881
NS_A2_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.4958981, upper bound: 27.4960270
NS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5247142, upper bound: 27.5170793
NS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5366675, upper bound: 27.5298710
NS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5247142, upper bound: 27.5170792
NS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5366675, upper bound: 27.5305212
NS_A2_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093615
NS_A2_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
NS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5093601
NS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5062530
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5283869, upper bound: 27.5254426
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5387269, upper bound: 27.5373719
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5283867, upper bound: 27.5254423
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.44
Output dim: 3, lower bound: -27.5387271, upper bound: 27.5384336

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.5840602, 13.1680984, -3.3112977, 12.2320118, -15.8160725, 16.4793968
1: -5.1535125, 13.3938541, -4.7633290, 12.4608803, -17.6143932, 18.1571827
2: -4.2921443, 15.0030842, -3.9792581, 14.0205498, -18.3126945, 18.9823418
3: -5.2045627, 19.3389301, -4.8087497, 18.0676193, -23.2721825, 24.1476803
4: -4.2549839, 17.7536049, -3.9650612, 16.5860634, -20.8410473, 21.7186661

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5250597, upper bound: 27.4962724
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5250597, upper bound: 27.5253347
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.5840602, 13.1680984, -3.8090727, 13.6119699, -17.1960258, 16.9771709
1: -5.1535125, 13.3938541, -5.4754810, 13.8948555, -19.0483665, 18.8693352
2: -4.2921443, 15.0030842, -4.6048141, 15.6138811, -19.9060230, 19.6078987
3: -5.2045627, 19.3389301, -5.5135355, 20.0045624, -25.2091255, 24.8524666
4: -4.2549839, 17.7536049, -4.5309124, 18.4355297, -22.6905136, 22.2845173

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217729, upper bound: 27.4942323
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217726, upper bound: 27.5055992
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -3.3112977, 12.2320118, -16.2860603, 17.7877083
1: -5.8242021, 14.7517223, -4.7633290, 12.4608803, -18.2850819, 19.5150509
2: -4.8843389, 16.5075340, -3.9792581, 14.0205498, -18.9048882, 20.4867897
3: -5.8696055, 21.1784897, -4.8087497, 18.0676193, -23.9372253, 25.9872398
4: -4.7930698, 19.4997158, -3.9650612, 16.5860634, -21.3791332, 23.4647770

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236400, upper bound: 27.4950230
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236397, upper bound: 27.5240851
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -3.8090727, 13.6119699, -17.6660156, 18.2854824
1: -5.8242021, 14.7517223, -5.4754810, 13.8948555, -19.7190571, 20.2272034
2: -4.8843389, 16.5075340, -4.6048141, 15.6138811, -20.4982166, 21.1123486
3: -5.8696055, 21.1784897, -5.5135355, 20.0045624, -25.8741665, 26.6920242
4: -4.7930698, 19.4997158, -4.5309124, 18.4355297, -23.2285995, 24.0306282

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5203528, upper bound: 27.4929829
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5203525, upper bound: 27.5043496
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.6164749, 13.2819958, -3.6464579, 13.3501921, -16.9666634, 16.9284515
1: -5.1846161, 13.4975052, -5.2464833, 13.5860958, -18.7707119, 18.7439861
2: -4.3129835, 15.1071482, -4.3717928, 15.2154961, -19.5284786, 19.4789391
3: -5.2438045, 19.4920826, -5.2993393, 19.6015205, -24.8453255, 24.7914219
4: -4.2717013, 17.8941593, -4.3268204, 18.0051994, -22.2768993, 22.2209797

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233782, upper bound: 27.5233781
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233782, upper bound: 27.5350774
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.5840602, 13.1680984, -3.6464579, 13.3501921, -16.9342518, 16.8145561
1: -5.1535125, 13.3938541, -5.2464833, 13.5860958, -18.7396069, 18.6403370
2: -4.2921443, 15.0030842, -4.3717928, 15.2154961, -19.5076408, 19.3748779
3: -5.2045627, 19.3389301, -5.2993393, 19.6015205, -24.8060837, 24.6382694
4: -4.2549839, 17.7536049, -4.3268204, 18.0051994, -22.2601814, 22.0804234

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5365006, upper bound: 27.5233780
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5365006, upper bound: 27.5354904
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.6164749, 13.2819958, -4.1191621, 14.6685381, -18.2850132, 17.4011574
1: -5.1846161, 13.4975052, -5.9215693, 14.9546156, -20.1392326, 19.4190731
2: -4.3129835, 15.1071482, -4.9671373, 16.7323151, -21.0452995, 20.0742836
3: -5.2438045, 19.4920826, -5.9684920, 21.4562817, -26.7000847, 25.4605713
4: -4.2717013, 17.8941593, -4.8678341, 19.7656593, -24.0373611, 22.7619934

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5228198
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224149, upper bound: 27.5324614
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.5840602, 13.1680984, -4.1191621, 14.6685381, -18.2525978, 17.2872601
1: -5.1535125, 13.3938541, -5.9215693, 14.9546156, -20.1081276, 19.3154240
2: -4.2921443, 15.0030842, -4.9671373, 16.7323151, -21.0244598, 19.9702225
3: -5.2045627, 19.3389301, -5.9684920, 21.4562817, -26.6608429, 25.3074188
4: -4.2549839, 17.7536049, -4.8678341, 19.7656593, -24.0206432, 22.6214390

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5354754, upper bound: 27.5228199
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5138545, upper bound: 27.5324622
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1237140, 14.6792870, -3.6464579, 13.3501921, -17.4739056, 18.3257446
1: -5.9060030, 14.9510040, -5.2464833, 13.5860958, -19.4920998, 20.1974869
2: -4.9519839, 16.7172890, -4.3717928, 15.2154961, -20.1674767, 21.0890808
3: -5.9551606, 21.4694099, -5.2993393, 19.6015205, -25.5566788, 26.7687492
4: -4.8511004, 19.7662888, -4.3268204, 18.0051994, -22.8562984, 24.0931091

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233874, upper bound: 27.5224144
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5233875, upper bound: 27.5335810
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -3.6464579, 13.3501921, -17.4042397, 18.1228676
1: -5.8242021, 14.7517223, -5.2464833, 13.5860958, -19.4102974, 19.9982014
2: -4.8843389, 16.5075340, -4.3717928, 15.2154961, -20.0998344, 20.8793240
3: -5.8696055, 21.1784897, -5.2993393, 19.6015205, -25.4711246, 26.4778290
4: -4.7930698, 19.4997158, -4.3268204, 18.0051994, -22.7982693, 23.8265343

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5350805, upper bound: 27.5224149
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5350806, upper bound: 27.5335807
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1237140, 14.6792870, -4.1191621, 14.6685381, -18.7922516, 18.7984486
1: -5.9060030, 14.9510040, -5.9215693, 14.9546156, -20.8606186, 20.8725739
2: -4.9519839, 16.7172890, -4.9671373, 16.7323151, -21.6842976, 21.6844254
3: -5.9551606, 21.4694099, -5.9684920, 21.4562817, -27.4114380, 27.4378986
4: -4.8511004, 19.7662888, -4.8678341, 19.7656593, -24.6167603, 24.6341228

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5224149
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5224580, upper bound: 27.5323893
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -4.1191621, 14.6685381, -18.7225857, 18.5955715
1: -5.8242021, 14.7517223, -5.9215693, 14.9546156, -20.7788181, 20.6732864
2: -4.8843389, 16.5075340, -4.9671373, 16.7323151, -21.6166534, 21.4746704
3: -5.8696055, 21.1784897, -5.9684920, 21.4562817, -27.3258858, 27.1469784
4: -4.7930698, 19.4997158, -4.8678341, 19.7656593, -24.5587292, 24.3675499

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340554, upper bound: 27.5224149
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5340554, upper bound: 27.5323893
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.2503290, 12.0559664, -4.7579708, 16.5425053, -19.7928314, 16.8139343
1: -4.6728878, 12.2730789, -6.8075414, 16.9199429, -21.5928307, 19.0806198
2: -3.9022455, 13.8133469, -5.7232504, 18.9006500, -22.8028946, 19.5365963
3: -4.7164497, 17.8124847, -6.8910208, 24.2590141, -28.9754639, 24.7035046
4: -3.8956451, 16.3417244, -5.5892906, 22.4202957, -26.3159409, 21.9310112

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5025150, upper bound: 27.4736592
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4879419, upper bound: 27.4630904
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1237140, 14.6792870, -3.8882565, 14.0178547, -18.1415691, 18.5675392
1: -5.9060030, 14.9510040, -5.5631576, 14.3240805, -20.2300835, 20.5141602
2: -4.9519839, 16.7172890, -4.6534100, 16.0694389, -21.0214176, 21.3706989
3: -5.9551606, 21.4694099, -5.6674542, 20.6948471, -26.6500053, 27.1368637
4: -4.8511004, 19.7662888, -4.6273518, 19.0684605, -23.9195614, 24.3936405

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5223844, upper bound: 27.5140000
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5179529, upper bound: 27.5182576
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -3.8882565, 14.0178547, -18.0719032, 18.3646660
1: -5.8242021, 14.7517223, -5.5631576, 14.3240805, -20.1482811, 20.3148766
2: -4.8843389, 16.5075340, -4.6534100, 16.0694389, -20.9537754, 21.1609440
3: -5.8696055, 21.1784897, -5.6674542, 20.6948471, -26.5644512, 26.8459435
4: -4.7930698, 19.4997158, -4.6273518, 19.0684605, -23.8615303, 24.1270676

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5343946, upper bound: 27.5178814
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299637, upper bound: 27.5221388
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1237140, 14.6792870, -4.4765372, 15.6225147, -19.7462234, 19.1558247
1: -5.9060030, 14.9510040, -6.3536415, 15.9724932, -21.8784962, 21.3046455
2: -4.9519839, 16.7172890, -5.3602648, 17.9039478, -22.8559303, 22.0775528
3: -5.9551606, 21.4694099, -6.4377475, 22.9774036, -28.9325600, 27.9071579
4: -4.8511004, 19.7662888, -5.2538066, 21.2279301, -26.0790310, 25.0200939

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5199127, upper bound: 27.5000860
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5199123, upper bound: 27.5246755
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -4.4765372, 15.6225147, -19.6765614, 18.9529476
1: -5.8242021, 14.7517223, -6.3536415, 15.9724932, -21.7966957, 21.1053619
2: -4.8843389, 16.5075340, -5.3602648, 17.9039478, -22.7882862, 21.8677959
3: -5.8696055, 21.1784897, -6.4377475, 22.9774036, -28.8470058, 27.6162376
4: -4.7930698, 19.4997158, -5.2538066, 21.2279301, -26.0209999, 24.7535191

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5277314, upper bound: 27.5000859
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5277314, upper bound: 27.5246756
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1237140, 14.6792870, -4.2396750, 15.1407347, -19.2644463, 18.9189606
1: -5.9060030, 14.9510040, -6.0757089, 15.4544592, -21.3604622, 21.0267124
2: -4.9519839, 16.7172890, -5.0678725, 17.2726898, -22.2246723, 21.7851620
3: -5.9551606, 21.4694099, -6.1715178, 22.2449474, -28.2001057, 27.6409283
4: -4.8511004, 19.7662888, -5.0068088, 20.4940319, -25.3451328, 24.7730980

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236433, upper bound: 27.5199365
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5200067, upper bound: 27.5259810
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -4.2396750, 15.1407347, -19.1947823, 18.7160854
1: -5.8242021, 14.7517223, -6.0757089, 15.4544592, -21.2786598, 20.8274269
2: -4.8843389, 16.5075340, -5.0678725, 17.2726898, -22.1570282, 21.5754051
3: -5.8696055, 21.1784897, -6.1715178, 22.2449474, -28.1145515, 27.3500080
4: -4.7930698, 19.4997158, -5.0068088, 20.4940319, -25.2871017, 24.5065250

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5364495, upper bound: 27.5262485
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5328116, upper bound: 27.5318302
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1237140, 14.6792870, -4.8151884, 16.7127552, -20.8364697, 19.4944725
1: -5.9060030, 14.9510040, -6.8487349, 17.0677147, -22.9737167, 21.7997398
2: -4.9519839, 16.7172890, -5.7475977, 19.0672207, -24.0192013, 22.4648838
3: -5.9551606, 21.4694099, -6.9301233, 24.4769001, -30.4320583, 28.3995323
4: -4.8511004, 19.7662888, -5.6081338, 22.6030693, -27.4541702, 25.3744221

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5133212, upper bound: 27.5133212
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5236427, upper bound: 27.5329253
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.0540481, 14.4764099, -4.8151884, 16.7127552, -20.7668037, 19.2915974
1: -5.8242021, 14.7517223, -6.8487349, 17.0677147, -22.8919163, 21.6004562
2: -4.8843389, 16.5075340, -5.7475977, 19.0672207, -23.9515591, 22.2551250
3: -5.8696055, 21.1784897, -6.9301233, 24.4769001, -30.3465042, 28.1086121
4: -4.7930698, 19.4997158, -5.6081338, 22.6030693, -27.3961391, 25.1078491

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5217399, upper bound: 27.5137414
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5352984, upper bound: 27.5327821
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1643739, 14.9286327, -3.3112977, 12.2320118, -16.3963852, 18.2399311
1: -5.9677429, 15.2310114, -4.7633290, 12.4608803, -18.4286232, 19.9943409
2: -4.9760747, 17.0250912, -3.9792581, 14.0205498, -18.9966240, 21.0043488
3: -6.0638986, 21.9368782, -4.8087497, 18.0676193, -24.1315174, 26.7456284
4: -4.9241924, 20.1996479, -3.9650612, 16.5860634, -21.5102558, 24.1647091

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5251878, upper bound: 27.4977075
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5251878, upper bound: 27.5267701
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7990613, 16.6746864, -3.3112977, 12.2320118, -17.0310726, 19.9859848
1: -6.8271613, 17.0287075, -4.7633290, 12.4608803, -19.2880421, 21.7920361
2: -5.7306623, 19.0230827, -3.9792581, 14.0205498, -19.7512112, 23.0023403
3: -6.9100881, 24.4234161, -4.8087497, 18.0676193, -24.9777050, 29.2321663
4: -5.5933805, 22.5517769, -3.9650612, 16.5860634, -22.1794434, 26.5168381

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5242303, upper bound: 27.4980437
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5302404, upper bound: 27.5290099
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7990613, 16.6746864, -3.8090727, 13.6119699, -18.4110260, 20.4837589
1: -6.8271613, 17.0287075, -5.4754810, 13.8948555, -20.7220173, 22.5041885
2: -5.7306623, 19.0230827, -4.6048141, 15.6138811, -21.3445396, 23.6278973
3: -6.9100881, 24.4234161, -5.5135355, 20.0045624, -26.9146461, 29.9369507
4: -5.5933805, 22.5517769, -4.5309124, 18.4355297, -24.0289097, 27.0826893

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5214207, upper bound: 27.4958465
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5208835, upper bound: 27.5072179
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.1655493, 14.8964396, -3.6464579, 13.3501921, -17.5157394, 18.5428982
1: -5.9687357, 15.2046061, -5.2464833, 13.5860958, -19.5548325, 20.4510880
2: -4.9837847, 16.9894066, -4.3717928, 15.2154961, -20.1992760, 21.3611984
3: -6.0602751, 21.8867779, -5.2993393, 19.6015205, -25.6617966, 27.1861172
4: -4.9282579, 20.1611748, -4.3268204, 18.0051994, -22.9334545, 24.4879951

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5284486, upper bound: 27.5293584
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5284486, upper bound: 27.5336064
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1891284, 15.0038404, -3.6464579, 13.3501921, -17.5393181, 18.6502991
1: -6.0037899, 15.3114023, -5.2464833, 13.5860958, -19.5898857, 20.5578861
2: -5.0072761, 17.1143532, -4.3717928, 15.2154961, -20.2227726, 21.4861450
3: -6.1000462, 22.0504875, -5.2993393, 19.6015205, -25.7015667, 27.3498268
4: -4.9527273, 20.3066311, -4.3268204, 18.0051994, -22.9579258, 24.6334496

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5336149, upper bound: 27.5293584
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5336149, upper bound: 27.5336062
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.1655493, 14.8964396, -4.1191621, 14.6685381, -18.8340874, 19.0156021
1: -5.9687357, 15.2046061, -5.9215693, 14.9546156, -20.9233513, 21.1261730
2: -4.9837847, 16.9894066, -4.9671373, 16.7323151, -21.7160969, 21.9565430
3: -6.0602751, 21.8867779, -5.9684920, 21.4562817, -27.5165558, 27.8552666
4: -4.9282579, 20.1611748, -4.8678341, 19.7656593, -24.6939163, 25.0290089

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5220147, upper bound: 27.5183071
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5275780, upper bound: 27.5324656
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.1891284, 15.0038404, -4.1191621, 14.6685381, -18.8576660, 19.1230030
1: -6.0037899, 15.3114023, -5.9215693, 14.9546156, -20.9584045, 21.2329712
2: -5.0072761, 17.1143532, -4.9671373, 16.7323151, -21.7395916, 22.0814896
3: -6.1000462, 22.0504875, -5.9684920, 21.4562817, -27.5563278, 28.0189781
4: -4.9527273, 20.3066311, -4.8678341, 19.7656593, -24.7183876, 25.1744652

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5283780, upper bound: 27.5176290
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5330407, upper bound: 27.5308991
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.4173279, 12.4798775, -3.6294351, 13.3022995, -16.7196255, 16.1093121
1: -4.8212452, 12.6474104, -5.2222657, 13.5357523, -18.3569984, 17.8696766
2: -4.0527782, 14.1146898, -4.3512917, 15.1592112, -19.2119865, 18.4659805
3: -4.8410463, 18.2321167, -5.2749639, 19.5312481, -24.3722954, 23.5070801
4: -4.0038462, 16.7538681, -4.3080854, 17.9397602, -21.9436073, 21.0619507

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5272358, upper bound: 27.5120705
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5276711, upper bound: 27.5250965
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7990613, 16.6746864, -3.6464579, 13.3501921, -18.1492538, 20.3211441
1: -6.8271613, 17.0287075, -5.2464833, 13.5860958, -20.4132576, 22.2751904
2: -5.7306623, 19.0230827, -4.3717928, 15.2154961, -20.9461575, 23.3948746
3: -6.9100881, 24.4234161, -5.2993393, 19.6015205, -26.5116062, 29.7227554
4: -5.5933805, 22.5517769, -4.3268204, 18.0051994, -23.5985756, 26.8785973

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5225441, upper bound: 27.5172168
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4678744, upper bound: 27.4851068
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4173279, 12.4798775, -4.1014361, 14.6185932, -18.0359173, 16.5813141
1: -4.8212452, 12.6474104, -5.8962598, 14.9020357, -19.7232819, 18.5436707
2: -4.0527782, 14.1146898, -4.9457264, 16.6737061, -20.7264843, 19.0604172
3: -4.8410463, 18.2321167, -5.9427748, 21.3831673, -26.2242126, 24.1748924
4: -4.0038462, 16.7538681, -4.8483124, 19.6974640, -23.7013092, 21.6021805

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5264567, upper bound: 27.5136158
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5261481, upper bound: 27.5238740
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7990613, 16.6746864, -4.1191621, 14.6685381, -19.4675999, 20.7938480
1: -6.8271613, 17.0287075, -5.9215693, 14.9546156, -21.7817764, 22.9502773
2: -5.7306623, 19.0230827, -4.9671373, 16.7323151, -22.4629765, 23.9902191
3: -6.9100881, 24.4234161, -5.9684920, 21.4562817, -28.3663635, 30.3919029
4: -5.5933805, 22.5517769, -4.8678341, 19.7656593, -25.3590374, 27.4196110

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5356489, upper bound: 27.5213480
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5353049, upper bound: 27.5311402
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.8157699, 13.7626266, -3.8882565, 14.0178547, -17.8336239, 17.6508808
1: -5.4549894, 14.0627117, -5.5631576, 14.3240805, -19.7790699, 19.6258640
2: -4.5692644, 15.7729368, -4.6534100, 16.0694389, -20.6386967, 20.4263458
3: -5.5578747, 20.3214417, -5.6674542, 20.6948471, -26.2527218, 25.9888954
4: -4.5480394, 18.7221661, -4.6273518, 19.0684605, -23.6165009, 23.3495178

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5164413, upper bound: 27.5164787
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5164413, upper bound: 27.5207235
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8396118, 13.8858604, -3.8882565, 14.0178547, -17.8574638, 17.7741165
1: -5.4948812, 14.1856594, -5.5631576, 14.3240805, -19.8189602, 19.7488155
2: -4.5944114, 15.9167309, -4.6534100, 16.0694389, -20.6638451, 20.5701408
3: -5.5989089, 20.5076370, -5.6674542, 20.6948471, -26.2937565, 26.1750908
4: -4.5751128, 18.8878918, -4.6273518, 19.0684605, -23.6435738, 23.5152435

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206793, upper bound: 27.5164787
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5206793, upper bound: 27.5207235
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8157699, 13.7626266, -4.4765372, 15.6225147, -19.4382782, 18.2391644
1: -5.4549894, 14.0627117, -6.3536415, 15.9724932, -21.4274826, 20.4163513
2: -4.5692644, 15.7729368, -5.3602648, 17.9039478, -22.4732075, 21.1332016
3: -5.5578747, 20.3214417, -6.4377475, 22.9774036, -28.5352783, 26.7591896
4: -4.5480394, 18.7221661, -5.2538066, 21.2279301, -25.7759705, 23.9759712

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5097044, upper bound: 27.5034479
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5166256, upper bound: 27.5249643
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8396118, 13.8858604, -4.4765372, 15.6225147, -19.4621201, 18.3623981
1: -5.4948812, 14.1856594, -6.3536415, 15.9724932, -21.4673748, 20.5393009
2: -4.5944114, 15.9167309, -5.3602648, 17.9039478, -22.4983597, 21.2769947
3: -5.5989089, 20.5076370, -6.4377475, 22.9774036, -28.5763130, 26.9453850
4: -4.5751128, 18.8878918, -5.2538066, 21.2279301, -25.8030434, 24.1416969

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5153503, upper bound: 27.4992419
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5212075, upper bound: 27.5206288
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.5284038, 15.7637367, -3.8882565, 14.0178547, -18.5462589, 19.6519890
1: -6.4160857, 16.1120987, -5.5631576, 14.3240805, -20.7401657, 21.6752567
2: -5.4068069, 18.0494289, -4.6534100, 16.0694389, -21.4762440, 22.7028389
3: -6.5094790, 23.1657505, -5.6674542, 20.6948471, -27.2043247, 28.8332043
4: -5.2920933, 21.4156857, -4.6273518, 19.0684605, -24.3605518, 26.0430374

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5096839, upper bound: 27.5114133
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5058825, upper bound: 27.5180237
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3922210, 15.3852606, -3.8882565, 14.0178547, -18.4100761, 19.2735119
1: -6.2310572, 15.7225313, -5.5631576, 14.3240805, -20.5551376, 21.2856827
2: -5.2559814, 17.6267986, -4.6534100, 16.0694389, -21.3254185, 22.2802086
3: -6.3180189, 22.6312351, -5.6674542, 20.6948471, -27.0128651, 28.2986889
4: -5.1607614, 20.8988686, -4.6273518, 19.0684605, -24.2292213, 25.5262203

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5255320, upper bound: 27.5193802
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5211009, upper bound: 27.5236375
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.5284038, 15.7637367, -4.4765372, 15.6225147, -20.1509171, 20.2402725
1: -6.4160857, 16.1120987, -6.3536415, 15.9724932, -22.3885784, 22.4657402
2: -5.4068069, 18.0494289, -5.3602648, 17.9039478, -23.3107548, 23.4096928
3: -6.5094790, 23.1657505, -6.4377475, 22.9774036, -29.4868793, 29.6034985
4: -5.2920933, 21.4156857, -5.2538066, 21.2279301, -26.5200214, 26.6694908

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5196820, upper bound: 27.5195234
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5196820, upper bound: 27.5275258
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3922210, 15.3852606, -4.4765372, 15.6225147, -20.0147324, 19.8617954
1: -6.2310572, 15.7225313, -6.3536415, 15.9724932, -22.2035503, 22.0761719
2: -5.2559814, 17.6267986, -5.3602648, 17.9039478, -23.1599293, 22.9870625
3: -6.3180189, 22.6312351, -6.4377475, 22.9774036, -29.2954216, 29.0689831
4: -5.1607614, 20.8988686, -5.2538066, 21.2279301, -26.3886909, 26.1526737

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271438, upper bound: 27.5200929
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5271438, upper bound: 27.5285877
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.8157699, 13.7626266, -4.2396750, 15.1407347, -18.9565010, 18.0023022
1: -5.4549894, 14.0627117, -6.0757089, 15.4544592, -20.9094467, 20.1384163
2: -4.5692644, 15.7729368, -5.0678725, 17.2726898, -21.8419514, 20.8408089
3: -5.5578747, 20.3214417, -6.1715178, 22.2449474, -27.8028221, 26.4929600
4: -4.5480394, 18.7221661, -5.0068088, 20.4940319, -25.0420723, 23.7289753

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4953321, upper bound: 27.4948374
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4771941, upper bound: 27.4832494
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8396118, 13.8858604, -4.2396750, 15.1407347, -18.9803429, 18.1255360
1: -5.4948812, 14.1856594, -6.0757089, 15.4544592, -20.9493370, 20.2613659
2: -4.5944114, 15.9167309, -5.0678725, 17.2726898, -21.8671017, 20.9846039
3: -5.5989089, 20.5076370, -6.1715178, 22.2449474, -27.8438568, 26.6791553
4: -4.5751128, 18.8878918, -5.0068088, 20.4940319, -25.0691452, 23.8947010

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5157991, upper bound: 27.5110717
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.4978549, upper bound: 27.4994919
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8157699, 13.7626266, -4.8151884, 16.7127552, -20.5285244, 18.5778141
1: -5.4549894, 14.0627117, -6.8487349, 17.0677147, -22.5227013, 20.9114456
2: -4.5692644, 15.7729368, -5.7475977, 19.0672207, -23.6364803, 21.5205307
3: -5.5578747, 20.3214417, -6.9301233, 24.4769001, -30.0347748, 27.2515640
4: -4.5480394, 18.7221661, -5.6081338, 22.6030693, -27.1511078, 24.3302994

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5132499, upper bound: 27.5279259
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5197000, upper bound: 27.5345625
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8396118, 13.8858604, -4.8151884, 16.7127552, -20.5523663, 18.7010479
1: -5.4948812, 14.1856594, -6.8487349, 17.0677147, -22.5625954, 21.0343933
2: -4.5944114, 15.9167309, -5.7475977, 19.0672207, -23.6616306, 21.6643238
3: -5.5989089, 20.5076370, -6.9301233, 24.4769001, -30.0758095, 27.4377594
4: -4.5751128, 18.8878918, -5.6081338, 22.6030693, -27.1781826, 24.4960251

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5177418, upper bound: 27.5230709
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5243507, upper bound: 27.5294180
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.5284038, 15.7637367, -4.2396750, 15.1407347, -19.6691360, 20.0034103
1: -6.4160857, 16.1120987, -6.0757089, 15.4544592, -21.8705444, 22.1878071
2: -5.4068069, 18.0494289, -5.0678725, 17.2726898, -22.6794968, 23.1173019
3: -6.5094790, 23.1657505, -6.1715178, 22.2449474, -28.7544250, 29.3372688
4: -5.2920933, 21.4156857, -5.0068088, 20.4940319, -25.7861233, 26.4224949

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5085926, upper bound: 27.5114580
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5059352, upper bound: 27.5196920
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.3922210, 15.3852606, -4.2396750, 15.1407347, -19.5329552, 19.6249332
1: -6.2310572, 15.7225313, -6.0757089, 15.4544592, -21.6855145, 21.7982349
2: -5.2559814, 17.6267986, -5.0678725, 17.2726898, -22.5286713, 22.6946697
3: -6.3180189, 22.6312351, -6.1715178, 22.2449474, -28.5629654, 28.8027534
4: -5.1607614, 20.8988686, -5.0068088, 20.4940319, -25.6547928, 25.9056778

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5232682, upper bound: 27.5241110
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5007452, upper bound: 27.5100725
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.5284038, 15.7637367, -4.8151884, 16.7127552, -21.2411594, 20.5789223
1: -6.4160857, 16.1120987, -6.8487349, 17.0677147, -23.4837990, 22.9608345
2: -5.4068069, 18.0494289, -5.7475977, 19.0672207, -24.4740276, 23.7970219
3: -6.5094790, 23.1657505, -6.9301233, 24.4769001, -30.9863777, 30.0958748
4: -5.2920933, 21.4156857, -5.6081338, 22.6030693, -27.8951588, 27.0238190

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5099129, upper bound: 27.5291611
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5210367, upper bound: 27.5367940
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.3922210, 15.3852606, -4.8151884, 16.7127552, -21.1049767, 20.2004452
1: -6.2310572, 15.7225313, -6.8487349, 17.0677147, -23.2987709, 22.5712643
2: -5.2559814, 17.6267986, -5.7475977, 19.0672207, -24.3232021, 23.3743916
3: -6.3180189, 22.6312351, -6.9301233, 24.4769001, -30.7949181, 29.5613594
4: -5.1607614, 20.8988686, -5.6081338, 22.6030693, -27.7638302, 26.5070019

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5188183, upper bound: 27.5308495
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5299182, upper bound: 27.5383863
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.4173279, 12.4798775, -3.8713453, 13.9685307, -17.3858566, 16.3512192
1: -4.8212452, 12.6474104, -5.5383916, 14.2723341, -19.0935783, 18.1858025
2: -4.0527782, 14.1146898, -4.6325893, 16.0115547, -20.0643330, 18.7472801
3: -4.8410463, 18.2321167, -5.6425877, 20.6229019, -25.4639473, 23.8747044
4: -4.0038462, 16.7538681, -4.6082716, 19.0015182, -23.0053635, 21.3621387

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5249615, upper bound: 27.5119591
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5210746, upper bound: 27.5174855
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7990613, 16.6746864, -3.8882565, 14.0178547, -18.8169155, 20.5629425
1: -6.8271613, 17.0287075, -5.5631576, 14.3240805, -21.1512413, 22.5918655
2: -5.7306623, 19.0230827, -4.6534100, 16.0694389, -21.8000984, 23.6764927
3: -6.9100881, 24.4234161, -5.6674542, 20.6948471, -27.6049309, 30.0908699
4: -5.5933805, 22.5517769, -4.6273518, 19.0684605, -24.6618385, 27.1791286

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5366585, upper bound: 27.5214179
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5322123, upper bound: 27.5257516
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4173279, 12.4798775, -4.4577675, 15.5696583, -18.9869862, 16.9376431
1: -4.8212452, 12.6474104, -6.3263965, 15.9169388, -20.7381840, 18.9738064
2: -4.0527782, 14.1146898, -5.3375654, 17.8418102, -21.8945885, 19.4522552
3: -4.8410463, 18.2321167, -6.4104652, 22.9006729, -27.7417183, 24.6425819
4: -4.0038462, 16.7538681, -5.2332711, 21.1558743, -25.1597214, 21.9871349

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5204482, upper bound: 27.4905766
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5232700, upper bound: 27.5100105
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.7990613, 16.6746864, -4.4765372, 15.6225147, -20.4215736, 21.1512241
1: -6.8271613, 17.0287075, -6.3536415, 15.9724932, -22.7996540, 23.3823490
2: -5.7306623, 19.0230827, -5.3602648, 17.9039478, -23.6346092, 24.3833466
3: -6.9100881, 24.4234161, -6.4377475, 22.9774036, -29.8874874, 30.8611641
4: -5.5933805, 22.5517769, -5.2538066, 21.2279301, -26.8213081, 27.8055820

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5360388, upper bound: 27.5216924
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5372450, upper bound: 27.5302407
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.4173279, 12.4798775, -4.2230864, 15.0922928, -18.5096207, 16.7029629
1: -4.8212452, 12.6474104, -6.0523210, 15.4032078, -20.2244530, 18.6997318
2: -4.0527782, 14.1146898, -5.0479345, 17.2158184, -21.2685966, 19.1626244
3: -4.8410463, 18.2321167, -6.1470199, 22.1739235, -27.0149689, 24.3791332
4: -4.0038462, 16.7538681, -4.9881954, 20.4277515, -24.4315987, 21.7420578

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5284779, upper bound: 27.5196750
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5251887, upper bound: 27.5251617
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.7990613, 16.6746864, -4.2396750, 15.1407347, -19.9397945, 20.9143600
1: -6.8271613, 17.0287075, -6.0757089, 15.4544592, -22.2816200, 23.1044140
2: -5.7306623, 19.0230827, -5.0678725, 17.2726898, -23.0033531, 24.0909557
3: -6.9100881, 24.4234161, -6.1715178, 22.2449474, -29.1550331, 30.5949345
4: -5.5933805, 22.5517769, -5.0068088, 20.4940319, -26.0874119, 27.5585861

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -27.5266116, upper bound: 27.5176336
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -27.5062530, upper bound: 27.5097582
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4173279, 12.4798775, -4.7984142, 16.6628838, -20.0802059, 17.2782917
1: -4.8212452, 12.6474104, -6.8244834, 17.0149822, -21.8362274, 19.4718914
2: -4.0527782, 14.1146898, -5.7268453, 19.0087166, -23.0614948, 19.8415356
3: -4.8410463, 18.2321167, -6.9047499, 24.4037094, -29.2447548, 25.1368675
4: -4.0038462, 16.7538681, -5.5888400, 22.5347195, -26.5385666, 22.3427086

Time for backsubstitution: 1.02 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.43 + 417.98 = 420.41 seconds
