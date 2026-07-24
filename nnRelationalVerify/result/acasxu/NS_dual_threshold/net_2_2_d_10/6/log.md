## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 8224.860104876458


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2726.0979004, 5988.6259766, -2726.0979004, 5988.6259766, -8714.7236328, 8714.7236328)
1: (-2609.0822754, 5295.7792969, -2609.0822754, 5295.7792969, -7904.8613281, 7904.8613281)
2: (-2140.2480469, 5647.6152344, -2140.2480469, 5647.6152344, -7787.8632812, 7787.8632812)
3: (-3858.4916992, 5377.4936523, -3858.4916992, 5377.4936523, -9235.9843750, 9235.9843750)
4: (-2558.7094727, 6049.1518555, -2558.7094727, 6049.1518555, -8607.8613281, 8607.8613281)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 2.38 = 3.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -8224.9423543, upper bound: 8224.9423543

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9418054, upper bound: 8224.9405347
time: 0.95 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9417001, upper bound: 8224.9417001
time: 0.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.92 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 3, lower bound: -8224.9418054, upper bound: 8224.9405347
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 3, lower bound: -8224.9417001, upper bound: 8224.9417001

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2704.7670898, 5941.3129883, -2726.0979004, 5988.6259766, -8693.3925781, 8667.4101562
1: -2588.8239746, 5254.3627930, -2609.0822754, 5295.7792969, -7884.6035156, 7863.4453125
2: -2123.5346680, 5603.3583984, -2140.2480469, 5647.6152344, -7771.1499023, 7743.6064453
3: -3828.7812500, 5335.4995117, -3858.4916992, 5377.4936523, -9206.2753906, 9193.9902344
4: -2538.6975098, 6001.9726562, -2558.7094727, 6049.1518555, -8587.8476562, 8560.6816406

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9405334, upper bound: 8224.9405334
time: 1.03 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9405333, upper bound: 8224.9405334
time: 1.00 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3263.3847656, 7220.7407227, -2719.1560059, 5973.8164062, -9237.1982422, 9939.2099609
1: -3123.1401367, 6396.0961914, -2602.5493164, 5282.6445312, -8405.7832031, 8998.6455078
2: -2556.2697754, 6820.0346680, -2134.8027344, 5633.6899414, -8189.9594727, 8954.5117188
3: -4629.0981445, 6480.8876953, -3848.9887695, 5364.0971680, -9993.1953125, 10329.8769531
4: -3054.7038574, 7301.1337891, -2552.2294922, 6034.3076172, -9089.0117188, 9852.2988281

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9405333, upper bound: 8224.9417001
time: 1.02 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9405334, upper bound: 8224.9417001
time: 0.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 3, lower bound: -8224.9405334, upper bound: 8224.9405334
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 3, lower bound: -8224.9405333, upper bound: 8224.9405334
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 3, lower bound: -8224.9405333, upper bound: 8224.9417001
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.41
Output dim: 3, lower bound: -8224.9405334, upper bound: 8224.9417001

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2704.7670898, 5941.3129883, -2704.7670898, 5941.3129883, -8646.0800781, 8646.0800781
1: -2588.8239746, 5254.3627930, -2588.8239746, 5254.3627930, -7843.1865234, 7843.1865234
2: -2123.5346680, 5603.3583984, -2123.5346680, 5603.3583984, -7726.8930664, 7726.8930664
3: -3828.7812500, 5335.4995117, -3828.7812500, 5335.4995117, -9164.2812500, 9164.2812500
4: -2538.6975098, 6001.9726562, -2538.6975098, 6001.9726562, -8540.6699219, 8540.6699219

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9360270, upper bound: 8224.9378297
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356411
time: 0.99 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2704.7670898, 5941.3129883, -3262.0710449, 7217.8583984, -9921.9316406, 9203.3837891
1: -2588.8239746, 5254.3627930, -3121.9272461, 6393.5161133, -8982.3398438, 8376.2900391
2: -2123.5346680, 5603.3583984, -2555.2612305, 6817.3188477, -8940.5205078, 8158.6191406
3: -3828.7812500, 5335.4995117, -4627.3496094, 6478.2944336, -10307.0742188, 9962.8486328
4: -2538.6975098, 6001.9726562, -3053.4948730, 7298.2583008, -9835.8906250, 9055.4677734

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9360270, upper bound: 8224.9383247
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356573
time: 1.15 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3262.0710449, 7217.8583984, -2704.7670898, 5941.3129883, -9203.3828125, 9921.9316406
1: -3121.9272461, 6393.5161133, -2588.8239746, 5254.3627930, -8376.2900391, 8982.3398438
2: -2555.2612305, 6817.3188477, -2123.5346680, 5603.3583984, -8158.6191406, 8940.5205078
3: -4627.3496094, 6478.2944336, -3828.7812500, 5335.4995117, -9962.8486328, 10307.0742188
4: -3053.4948730, 7298.2583008, -2538.6975098, 6001.9726562, -9055.4677734, 9835.8906250

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9383242, upper bound: 8224.9381029
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9376978
time: 0.96 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3269.0053711, 7233.0786133, -3269.0053711, 7233.0786133, -10498.9062500, 10498.9062500
1: -3128.3374023, 6407.1474609, -3128.3374023, 6407.1474609, -9532.1894531, 9532.1894531
2: -2560.5898438, 6831.6767578, -2560.5898438, 6831.6767578, -9390.3134766, 9390.3134766
3: -4636.5922852, 6491.9858398, -4636.5922852, 6491.9858398, -11124.5449219, 11124.5449219
4: -3059.8762207, 7313.4609375, -3059.8762207, 7313.4609375, -10370.8632812, 10370.8632812

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9383242, upper bound: 8224.9381029
time: 1.02 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356573, upper bound: 8224.9377034
time: 0.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.51 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9360270, upper bound: 8224.9378297
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356411
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9360270, upper bound: 8224.9383247
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356573
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9383242, upper bound: 8224.9381029
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9376978
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9383242, upper bound: 8224.9381029
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -8224.9356573, upper bound: 8224.9377034

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2688.2653809, 5904.9609375, -2704.7670898, 5941.3129883, -8629.5761719, 8609.7285156
1: -2572.7792969, 5222.4223633, -2588.8239746, 5254.3627930, -7827.1420898, 7811.2460938
2: -2110.4904785, 5569.0732422, -2123.5346680, 5603.3583984, -7713.8486328, 7692.6079102
3: -3804.6308594, 5302.9140625, -3828.7812500, 5335.4995117, -9140.1308594, 9131.6953125
4: -2523.1408691, 5964.9345703, -2538.6975098, 6001.9726562, -8525.1132812, 8503.6308594

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356411
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356411
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3108.9772949, 6837.6162109, -2688.0766602, 5904.9052734, -9013.8828125, 9525.6923828
1: -2970.1589355, 6052.8901367, -2572.4729004, 5222.3735352, -8192.5322266, 8625.3632812
2: -2438.7509766, 6449.1220703, -2110.2626953, 5568.9858398, -8007.7363281, 8559.3828125
3: -4381.5820312, 6139.6430664, -3804.0485840, 5302.7504883, -9684.3320312, 9943.6894531
4: -2915.3234863, 6898.4062500, -2522.8957520, 5964.6821289, -8880.0058594, 9421.3017578

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356410
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356410
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2688.2653809, 5904.9609375, -3261.9099121, 7217.5039062, -9905.0283203, 9166.8701172
1: -2572.7792969, 5222.4223633, -3121.7778320, 6393.1987305, -8965.9765625, 8344.2001953
2: -2110.4904785, 5569.0732422, -2555.1367188, 6816.9843750, -8927.0927734, 8124.2099609
3: -3804.6308594, 5302.9140625, -4627.1347656, 6477.9750977, -10282.6054688, 9930.0488281
4: -2523.1408691, 5964.9345703, -3053.3464355, 7297.9057617, -9819.9423828, 9018.2812500

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9382579, upper bound: 8224.9364715
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9382643, upper bound: 8224.9376732
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3108.9772949, 6837.6162109, -3230.5571289, 7150.7509766, -10258.5576172, 10068.1738281
1: -2970.1589355, 6052.8901367, -3091.1821289, 6334.5190430, -9304.6767578, 9144.0722656
2: -2438.7509766, 6449.1220703, -2529.9992676, 6754.0473633, -9192.0839844, 8979.1210938
3: -4381.5820312, 6139.6430664, -4581.2773438, 6417.8388672, -10799.4208984, 10720.9189453
4: -2915.3234863, 6898.4062500, -3023.4567871, 7229.6884766, -10143.6093750, 9921.8632812

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9361145, upper bound: 8224.9257572
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9376978, upper bound: 8224.9353450
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3261.9099121, 7217.5039062, -2688.2653809, 5904.9609375, -9166.8701172, 9905.0283203
1: -3121.7778320, 6393.1987305, -2572.7792969, 5222.4223633, -8344.2001953, 8965.9775391
2: -2555.1367188, 6816.9843750, -2110.4904785, 5569.0732422, -8124.2099609, 8927.0927734
3: -4627.1347656, 6477.9750977, -3804.6308594, 5302.9140625, -9930.0488281, 10282.6054688
4: -3053.3464355, 7297.9057617, -2523.1408691, 5964.9345703, -9018.2812500, 9819.9423828

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9364715, upper bound: 8224.9382580
time: 0.98 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9376732, upper bound: 8224.9382644
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3230.5571289, 7150.7509766, -3108.9772949, 6837.6162109, -10068.1738281, 10258.5576172
1: -3091.1821289, 6334.5190430, -2970.1589355, 6052.8901367, -9144.0722656, 9304.6767578
2: -2529.9992676, 6754.0473633, -2438.7509766, 6449.1220703, -8979.1210938, 9192.0839844
3: -4581.2773438, 6417.8388672, -4381.5820312, 6139.6430664, -10720.9189453, 10799.4208984
4: -3023.4567871, 7229.6884766, -2915.3234863, 6898.4062500, -9921.8632812, 10143.6093750

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9361145
time: 1.05 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9353450, upper bound: 8224.9376978
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3269.0053711, 7233.0786133, -3255.5808105, 7203.2763672, -10468.9570312, 10485.3906250
1: -3128.3374023, 6407.1474609, -3115.3762207, 6380.9965820, -9505.9394531, 9519.0712891
2: -2560.5898438, 6831.6767578, -2550.0085449, 6803.6005859, -9362.1074219, 9379.6162109
3: -4636.5922852, 6491.9858398, -4617.1958008, 6465.3300781, -11097.7812500, 11104.8828125
4: -3059.8762207, 7313.4609375, -3047.2282715, 7283.3076172, -10340.5468750, 10358.1347656

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3235.7497559, 7162.1518555, -3602.9416504, 7984.3793945, -11213.5546875, 10761.5996094
1: -3095.9863281, 6344.7299805, -3442.3225098, 7079.1108398, -10168.9316406, 9782.9755859
2: -2533.9902344, 6764.8085938, -2819.2053223, 7541.7124023, -10070.9033203, 9582.3984375
3: -4588.2109375, 6428.0947266, -5092.3066406, 7166.4702148, -11748.1953125, 11514.1464844
4: -3028.2365723, 7241.0854492, -3369.4929199, 8063.6860352, -11086.2539062, 10608.1279297

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.30 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356411
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356411
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356410
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9356411, upper bound: 8224.9356410
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9382579, upper bound: 8224.9364715
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9382643, upper bound: 8224.9376732
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9361145, upper bound: 8224.9257572
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9376978, upper bound: 8224.9353450
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9364715, upper bound: 8224.9382580
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9376732, upper bound: 8224.9382644
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9361145
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9353450, upper bound: 8224.9376978
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.30
Output dim: 3, lower bound: -8224.9373682, upper bound: 8224.9377034

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2688.2653809, 5904.9609375, -2688.2653809, 5904.9609375, -8593.2246094, 8593.2236328
1: -2572.7792969, 5222.4223633, -2572.7792969, 5222.4223633, -7795.2016602, 7795.2016602
2: -2110.4904785, 5569.0732422, -2110.4904785, 5569.0732422, -7679.5634766, 7679.5634766
3: -3804.6308594, 5302.9140625, -3804.6308594, 5302.9140625, -9107.5449219, 9107.5449219
4: -2523.1408691, 5964.9345703, -2523.1408691, 5964.9345703, -8488.0751953, 8488.0751953

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9358689, upper bound: 8224.9368281
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9359657, upper bound: 8224.9376811
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2688.2653809, 5904.9609375, -3108.9772949, 6837.6162109, -9525.8798828, 9013.9355469
1: -2572.7792969, 5222.4223633, -2970.1589355, 6052.8901367, -8625.6679688, 8192.5810547
2: -2110.4904785, 5569.0732422, -2438.7509766, 6449.1220703, -8559.6123047, 8007.8232422
3: -3804.6308594, 5302.9140625, -4381.5820312, 6139.6430664, -9944.2734375, 9684.4960938
4: -2523.1408691, 5964.9345703, -2915.3234863, 6898.4062500, -9421.5468750, 8880.2568359

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9344322, upper bound: 8224.9356316
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9343850, upper bound: 8224.9344527
time: 1.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3108.9772949, 6837.6162109, -2688.2653809, 5904.9609375, -9013.9355469, 9525.8798828
1: -2970.1589355, 6052.8901367, -2572.7792969, 5222.4223633, -8192.5810547, 8625.6679688
2: -2438.7509766, 6449.1220703, -2110.4904785, 5569.0732422, -8007.8232422, 8559.6123047
3: -4381.5820312, 6139.6430664, -3804.6308594, 5302.9140625, -9684.4960938, 9944.2734375
4: -2915.3234863, 6898.4062500, -2523.1408691, 5964.9345703, -8880.2568359, 9421.5468750

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9295835, upper bound: 8224.9337325
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9340576, upper bound: 8224.9340576
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3108.9772949, 6837.6162109, -3108.9772949, 6837.6162109, -9946.5927734, 9946.5927734
1: -2970.1589355, 6052.8901367, -2970.1589355, 6052.8901367, -9023.0488281, 9023.0488281
2: -2438.7509766, 6449.1220703, -2438.7509766, 6449.1220703, -8887.8730469, 8887.8730469
3: -4381.5820312, 6139.6430664, -4381.5820312, 6139.6430664, -10521.2246094, 10521.2246094
4: -2915.3234863, 6898.4062500, -2915.3234863, 6898.4062500, -9813.7285156, 9813.7285156

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9353673, upper bound: 8224.9355140
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9356035, upper bound: 8224.9356035
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -2664.2185059, 5851.1953125, -3260.4567871, 7214.3012695, -9877.7675781, 9111.6503906
1: -2549.8625488, 5175.0590820, -3120.4235840, 6390.3505859, -8940.2128906, 8295.4824219
2: -2091.6650391, 5518.5791016, -2554.0114746, 6813.9770508, -8905.2509766, 8072.5903320
3: -3770.8688965, 5254.8842773, -4625.1713867, 6475.1074219, -10245.9765625, 9880.0556641
4: -2500.5051270, 5911.0834961, -3051.9985352, 7294.7133789, -9794.1035156, 8963.0820312

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9355003, upper bound: 8224.9345162
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9354998, upper bound: 8224.9329017
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -2975.9682617, 6580.2929688, -3237.4562988, 7164.0585938, -10138.3759766, 9817.7451172
1: -2851.7900391, 5822.9477539, -3098.3991699, 6346.1181641, -9196.9150391, 8921.3466797
2: -2333.5395508, 6210.0786133, -2535.7783203, 6766.7275391, -9099.5585938, 8745.8564453
3: -4227.8481445, 5905.6386719, -4592.9448242, 6429.8320312, -10657.6796875, 10497.8720703
4: -2789.5031738, 6650.3745117, -3030.0368652, 7244.2773438, -10032.2343750, 9680.4111328

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9333577, upper bound: 8224.9333927
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3091.5627441, 6798.8862305, -3229.2622070, 7147.8969727, -10238.2744141, 10028.1484375
1: -2953.5268555, 6018.8945312, -3089.9726562, 6331.9853516, -9285.5117188, 9108.8652344
2: -2425.1264648, 6412.7519531, -2528.9956055, 6751.3686523, -9175.7705078, 8941.7480469
3: -4356.8413086, 6105.0532227, -4579.5239258, 6415.2861328, -10772.1269531, 10684.5771484
4: -2898.9755859, 6859.4921875, -3022.2536621, 7226.8452148, -10124.4023438, 9881.7460938

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9361145, upper bound: 8224.9257572
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9361145, upper bound: 8224.9257572
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3384.1684570, 7491.2138672, -3205.2763672, 7095.4370117, -10477.7207031, 10694.5449219
1: -3236.7431641, 6634.2602539, -3067.0483398, 6285.7412109, -9520.9091797, 9699.3056641
2: -2650.7839355, 7070.0722656, -2510.0080566, 6702.0146484, -9352.1425781, 9579.7197266
3: -4788.7915039, 6722.5493164, -4546.0371094, 6367.9990234, -11154.9941406, 11265.8750000
4: -3168.7065430, 7561.9980469, -2999.3937988, 7174.2075195, -10341.3769531, 10561.0195312

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9376978, upper bound: 8224.9353450
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9376978, upper bound: 8224.9353450
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3260.4567871, 7214.3012695, -2664.2185059, 5851.1953125, -9111.6503906, 9877.7675781
1: -3120.4235840, 6390.3505859, -2549.8625488, 5175.0590820, -8295.4824219, 8940.2128906
2: -2554.0114746, 6813.9770508, -2091.6650391, 5518.5791016, -8072.5908203, 8905.2509766
3: -4625.1713867, 6475.1074219, -3770.8688965, 5254.8842773, -9880.0556641, 10245.9765625
4: -3051.9985352, 7294.7133789, -2500.5051270, 5911.0834961, -8963.0820312, 9794.1044922

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9345162, upper bound: 8224.9355002
time: 0.87 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9329017, upper bound: 8224.9354998
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3237.4562988, 7164.0585938, -2975.9682617, 6580.2929688, -9817.7451172, 10138.3759766
1: -3098.3991699, 6346.1181641, -2851.7900391, 5822.9477539, -8921.3466797, 9196.9150391
2: -2535.7783203, 6766.7275391, -2333.5395508, 6210.0786133, -8745.8564453, 9099.5585938
3: -4592.9448242, 6429.8320312, -4227.8481445, 5905.6386719, -10497.8720703, 10657.6796875
4: -3030.0368652, 7244.2773438, -2789.5031738, 6650.3745117, -9680.4111328, 10032.2343750

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9333927, upper bound: 8224.9333577
time: 0.96 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3229.2622070, 7147.8969727, -3091.5627441, 6798.8862305, -10028.1484375, 10238.2744141
1: -3089.9726562, 6331.9853516, -2953.5268555, 6018.8945312, -9108.8652344, 9285.5117188
2: -2528.9956055, 6751.3686523, -2425.1264648, 6412.7519531, -8941.7480469, 9175.7695312
3: -4579.5239258, 6415.2861328, -4356.8413086, 6105.0532227, -10684.5771484, 10772.1269531
4: -3022.2536621, 7226.8452148, -2898.9755859, 6859.4921875, -9881.7460938, 10124.4023438

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9361145
time: 1.18 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9361145
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3205.2763672, 7095.4370117, -3384.1684570, 7491.2138672, -10694.5449219, 10477.7207031
1: -3067.0483398, 6285.7412109, -3236.7431641, 6634.2602539, -9699.3056641, 9520.9091797
2: -2510.0080566, 6702.0146484, -2650.7839355, 7070.0722656, -9579.7197266, 9352.1435547
3: -4546.0371094, 6367.9990234, -4788.7915039, 6722.5493164, -11265.8750000, 11154.9941406
4: -2999.3937988, 7174.2075195, -3168.7065430, 7561.9980469, -10561.0195312, 10341.3769531

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9353450, upper bound: 8224.9376978
time: 0.97 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9376978
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3255.5808105, 7203.2763672, -3255.5808105, 7203.2763672, -10455.4414062, 10455.4414062
1: -3115.3762207, 6380.9965820, -3115.3762207, 6380.9965820, -9492.8203125, 9492.8203125
2: -2550.0085449, 6803.6005859, -2550.0085449, 6803.6005859, -9351.4101562, 9351.4101562
3: -4617.1958008, 6465.3300781, -4617.1958008, 6465.3300781, -11078.1191406, 11078.1191406
4: -3047.2282715, 7283.3076172, -3047.2282715, 7283.3076172, -10327.8173828, 10327.8173828

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9393539, upper bound: 8224.9380810
time: 0.97 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9387037, upper bound: 8224.9378957
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3602.9416504, 7984.3793945, -3255.5808105, 7203.2763672, -10802.6738281, 11233.0576172
1: -3442.3225098, 7079.1108398, -3115.3762207, 6380.9965820, -9819.1992188, 10187.8154297
2: -2819.2053223, 7541.7124023, -2550.0085449, 6803.6005859, -9621.1367188, 10086.4013672
3: -5092.3066406, 7166.4702148, -4617.1958008, 6465.3300781, -11551.3417969, 11776.5830078
4: -3369.4929199, 8063.6860352, -3047.2282715, 7283.3076172, -10650.2832031, 11104.9042969

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9370773, upper bound: 8224.9352997
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9349813, upper bound: 8224.9352997
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3255.5808105, 7203.2763672, -3602.9416504, 7984.3793945, -11233.0576172, 10802.6738281
1: -3115.3762207, 6380.9965820, -3442.3225098, 7079.1108398, -10187.8154297, 9819.1992188
2: -2550.0085449, 6803.6005859, -2819.2053223, 7541.7124023, -10086.4013672, 9621.1367188
3: -4617.1958008, 6465.3300781, -5092.3066406, 7166.4702148, -11776.5830078, 11551.3417969
4: -3047.2282715, 7283.3076172, -3369.4929199, 8063.6860352, -11104.9042969, 10650.2832031

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9349579, upper bound: 8224.9358388
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9349550, upper bound: 8224.9351364
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3602.9416504, 7984.3793945, -3602.9416504, 7984.3793945, -11580.2890625, 11580.2890625
1: -3442.3225098, 7079.1108398, -3442.3225098, 7079.1108398, -10514.1943359, 10514.1943359
2: -2819.2053223, 7541.7124023, -2819.2053223, 7541.7124023, -10356.1279297, 10356.1279297
3: -5092.3066406, 7166.4702148, -5092.3066406, 7166.4702148, -12249.8046875, 12249.8046875
4: -3369.4929199, 8063.6860352, -3369.4929199, 8063.6860352, -11427.3701172, 11427.3710938

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9372578, upper bound: 8224.9375278
time: 0.98 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9372578, upper bound: 8224.9376759
time: 0.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.44 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9358689, upper bound: 8224.9368281
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9359657, upper bound: 8224.9376811
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9344322, upper bound: 8224.9356316
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9343850, upper bound: 8224.9344527
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9295835, upper bound: 8224.9337325
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9340576, upper bound: 8224.9340576
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9353673, upper bound: 8224.9355140
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9356035, upper bound: 8224.9356035
NS_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9355003, upper bound: 8224.9345162
NS_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9354998, upper bound: 8224.9329017
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9333577, upper bound: 8224.9333927
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9361145, upper bound: 8224.9257572
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9361145, upper bound: 8224.9257572
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9376978, upper bound: 8224.9353450
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9376978, upper bound: 8224.9353450
NS_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9345162, upper bound: 8224.9355002
NS_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9329017, upper bound: 8224.9354998
NS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9333927, upper bound: 8224.9333577
NS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9361145
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9361145
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9353450, upper bound: 8224.9376978
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9257572, upper bound: 8224.9376978
NS_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9393539, upper bound: 8224.9380810
NS_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9387037, upper bound: 8224.9378957
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9370773, upper bound: 8224.9352997
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9349813, upper bound: 8224.9352997
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9349579, upper bound: 8224.9358388
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9349550, upper bound: 8224.9351364
NS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9372578, upper bound: 8224.9375278
NS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.44
Output dim: 3, lower bound: -8224.9372578, upper bound: 8224.9376759

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2671.8115234, 5869.3222656, -2657.1162109, 5837.5673828, -8509.3779297, 8526.4384766
1: -2556.6040039, 5191.1035156, -2542.1381836, 5163.2006836, -7719.8037109, 7733.2416992
2: -2097.3637695, 5535.4765625, -2085.6232910, 5505.5283203, -7602.8920898, 7621.0996094
3: -3780.0588379, 5270.7519531, -3758.0598145, 5242.0830078, -9022.1416016, 9028.8115234
4: -2507.4460449, 5928.3544922, -2493.4133301, 5895.7192383, -8403.1640625, 8421.7675781

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2657.5747070, 5839.8007812, -2805.8913574, 6155.1562500, -8812.7304688, 8645.6904297
1: -2542.0659180, 5165.2988281, -2679.1462402, 5448.7744141, -7990.8398438, 7844.4448242
2: -2085.8010254, 5507.6015625, -2202.5554199, 5805.0957031, -7890.8964844, 7710.1572266
3: -3757.2170410, 5243.5991211, -3948.4985352, 5528.6254883, -9285.8408203, 9192.0976562
4: -2493.5732422, 5897.1499023, -2632.7983398, 6209.9604492, -8703.5332031, 8529.9482422

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2673.2287598, 5872.0693359, -3108.9772949, 6837.6162109, -9510.8447266, 8981.0439453
1: -2558.2563477, 5193.4687500, -2970.1589355, 6052.8901367, -8611.1455078, 8163.6279297
2: -2098.6103516, 5538.1176758, -2438.7509766, 6449.1220703, -8547.7294922, 7976.8676758
3: -3782.9814453, 5273.3549805, -4381.5820312, 6139.6430664, -9922.6230469, 9654.9375000
4: -2508.8974609, 5931.6074219, -2915.3234863, 6898.4062500, -9407.3037109, 8846.9306641

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9340380, upper bound: 8224.9340957
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9340337, upper bound: 8224.9350173
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2997.9763184, 6590.5844727, -3081.8137207, 6778.3398438, -9776.3154297, 9672.3984375
1: -2864.3105469, 5830.3593750, -2944.3298340, 6000.6166992, -8864.9257812, 8774.6894531
2: -2352.2241211, 6214.8535156, -2417.5187988, 6393.4365234, -8745.6582031, 8632.3710938
3: -4226.5307617, 5914.5864258, -4343.6816406, 6086.3237305, -10312.8535156, 10258.2656250
4: -2811.2846680, 6649.4057617, -2889.6538086, 6838.8247070, -9650.1093750, 9539.0585938

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9343034, upper bound: 8224.9344527
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9343102, upper bound: 8224.9339802
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3108.9772949, 6837.6162109, -2673.2287598, 5872.0693359, -8981.0439453, 9510.8447266
1: -2970.1589355, 6052.8901367, -2558.2563477, 5193.4687500, -8163.6279297, 8611.1464844
2: -2438.7509766, 6449.1220703, -2098.6103516, 5538.1176758, -7976.8676758, 8547.7294922
3: -4381.5820312, 6139.6430664, -3782.9814453, 5273.3549805, -9654.9375000, 9922.6230469
4: -2915.3234863, 6898.4062500, -2508.8974609, 5931.6074219, -8846.9306641, 9407.3037109

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9340957, upper bound: 8224.9340380
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9350173, upper bound: 8224.9340337
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3081.8137207, 6778.3398438, -2997.9763184, 6590.5844727, -9672.3984375, 9776.3154297
1: -2944.3298340, 6000.6166992, -2864.3105469, 5830.3593750, -8774.6894531, 8864.9257812
2: -2417.5187988, 6393.4365234, -2352.2241211, 6214.8535156, -8632.3710938, 8745.6572266
3: -4343.6816406, 6086.3237305, -4226.5307617, 5914.5864258, -10258.2656250, 10312.8535156
4: -2889.6538086, 6838.8247070, -2811.2846680, 6649.4057617, -9539.0585938, 9650.1093750

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9344526, upper bound: 8224.9343034
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9339802, upper bound: 8224.9343102
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3077.8027344, 6771.2934570, -3092.9487305, 6803.4985352, -9881.3007812, 9864.2421875
1: -2939.2700195, 5994.8706055, -2954.2712402, 6023.0410156, -8962.3095703, 8949.1396484
2: -2413.6916504, 6386.5634766, -2425.8508301, 6416.9331055, -8830.6250000, 8812.4140625
3: -4334.7470703, 6079.5493164, -4357.4956055, 6108.7270508, -10443.4746094, 10437.0449219
4: -2885.3129883, 6830.0566406, -2899.8764648, 6863.2714844, -9748.5839844, 9729.9335938

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9351777, upper bound: 8224.9351776
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9351777, upper bound: 8224.9355140
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3325.8334961, 7307.5283203, -3079.1313477, 6774.0053711, -10099.8388672, 10386.6601562
1: -3170.6003418, 6475.9086914, -2940.8813477, 5997.2875977, -9167.8876953, 9416.7900391
2: -2608.5979004, 6892.8872070, -2415.0480957, 6389.3754883, -8997.9736328, 9307.9355469
3: -4661.9726562, 6563.8291016, -4337.1860352, 6082.1816406, -10744.1513672, 10901.0156250
4: -3117.6538086, 7363.6215820, -2886.8737793, 6833.2011719, -9950.8525391, 10250.4921875

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9355140, upper bound: 8224.9353673
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9355140, upper bound: 8224.9356035
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -2649.1137695, 5818.1484375, -3260.4531250, 7214.2929688, -9862.5927734, 9078.6015625
1: -2535.2770996, 5145.9692383, -3120.4196777, 6390.3432617, -8925.6201172, 8266.3876953
2: -2079.7299805, 5487.4799805, -2554.0090332, 6813.9682617, -8893.2333984, 8041.4892578
3: -3749.1301270, 5225.1914062, -4625.1669922, 6475.0996094, -10224.2275391, 9850.3583984
4: -2486.1972656, 5877.6064453, -3051.9951172, 7294.7050781, -9779.7373047, 8929.6015625

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9329575, upper bound: 8224.9337378
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_A1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9336022, upper bound: 8224.9333939
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -2977.6518555, 6545.1640625, -3232.0219727, 7154.4599609, -10130.8828125, 9777.1855469
1: -2844.8762207, 5790.4023438, -3092.6586914, 6337.7167969, -9182.5927734, 8883.0605469
2: -2336.2890625, 6172.2416992, -2531.2380371, 6757.6171875, -9093.0703125, 8703.4794922
3: -4197.7250977, 5874.0014648, -4583.4179688, 6420.7124023, -10618.4375000, 10457.4179688
4: -2792.0566406, 6603.8305664, -3024.6655273, 7233.3876953, -10024.0439453, 9628.4960938

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A1_A2_A1

### Relational analysis result of NS_A1_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9344546, upper bound: 8224.9329017
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_A2

### Relational analysis result of NS_A1_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9344546, upper bound: 8224.9325911
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2936.6823730, 6493.2592773, -3174.9748535, 7026.1669922, -9961.2285156, 9668.2343750
1: -2814.3159180, 5746.4018555, -3038.8183594, 6224.8193359, -9038.1630859, 8785.2207031
2: -2303.0380859, 6128.1572266, -2487.2573242, 6636.9033203, -8939.2451172, 8615.4140625
3: -4172.1743164, 5828.0327148, -4504.4204102, 6306.8081055, -10478.9824219, 10331.8808594
4: -2753.0058594, 6562.5644531, -2971.9921875, 7105.0307617, -9856.5078125, 9534.5566406

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2891.1013184, 6392.6147461, -3418.7656250, 7568.1743164, -10452.0214844, 9810.5312500
1: -2770.6791992, 5657.7270508, -3278.3342285, 6705.2636719, -9470.5341797, 8932.9384766
2: -2267.1994629, 6033.6508789, -2677.5400391, 7152.5898438, -9414.2236328, 8711.1914062
3: -4107.7182617, 5737.7602539, -4871.2177734, 6792.7373047, -10896.0312500, 10601.1660156
4: -2709.8520508, 6461.5102539, -3196.0339355, 7663.9375000, -10365.6992188, 9657.5429688

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9282379, upper bound: 8224.9327142
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -3091.5627441, 6798.8862305, -3248.9765625, 7188.7910156, -10279.1201172, 10047.8632812
1: -2953.5268555, 6018.8945312, -3109.2565918, 6368.0415039, -9321.5664062, 9128.1474609
2: -2425.1264648, 6412.7519531, -2544.9169922, 6789.9355469, -9214.2841797, 8957.6689453
3: -4356.8413086, 6105.0532227, -4608.3740234, 6452.3110352, -10809.1523438, 10713.4277344
4: -2898.9755859, 6859.4921875, -3041.1291504, 7268.8349609, -10166.3232422, 9900.6210938

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9290615, upper bound: 8224.9212381
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3091.5627441, 6798.8862305, -3593.7851562, 7964.4379883, -11051.2812500, 10392.6708984
1: -2953.5268555, 6018.8945312, -3433.7868652, 7061.4604492, -10012.6015625, 9452.6806641
2: -2425.1264648, 6412.7519531, -2812.0895996, 7523.0512695, -9944.2783203, 9224.8417969
3: -4356.8413086, 6105.0532227, -5079.9565430, 7148.7114258, -11505.5527344, 11185.0097656
4: -2898.9755859, 6859.4921875, -3360.9667969, 8043.8530273, -10938.0576172, 10220.4580078

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9306925, upper bound: 8224.9196877
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -3384.1684570, 7491.2138672, -3224.0891113, 7134.3725586, -10516.5957031, 10713.0283203
1: -3236.7431641, 6634.2602539, -3085.4758301, 6320.1074219, -9555.2216797, 9717.2246094
2: -2650.7839355, 7070.0722656, -2525.2312012, 6738.7685547, -9388.8369141, 9594.4228516
3: -4788.7915039, 6722.5493164, -4573.5732422, 6403.3081055, -11190.2568359, 11292.8144531
4: -3168.7065430, 7561.9980469, -3017.4287109, 7214.2329102, -10381.3212891, 10578.7128906

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9357852, upper bound: 8224.9336850
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9350840, upper bound: 8224.9336850
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -3384.1684570, 7491.2138672, -3567.3950195, 7906.2504883, -11284.9433594, 11056.1943359
1: -3236.7431641, 6634.2602539, -3408.3666992, 7010.2304688, -10242.1855469, 10039.5341797
2: -2650.7839355, 7070.0722656, -2791.1271973, 7468.3256836, -10115.2285156, 9860.8447266
3: -4788.7915039, 6722.5493164, -5042.8906250, 7096.4599609, -11880.7041016, 11760.2060547
4: -3168.7065430, 7561.9980469, -3335.8266602, 7985.4487305, -11149.2070312, 10897.3046875

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9324938, upper bound: 8224.9286311
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9297632, upper bound: 8224.9278045
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3260.4531250, 7214.2929688, -2649.1137695, 5818.1484375, -9078.6015625, 9862.5927734
1: -3120.4196777, 6390.3432617, -2535.2770996, 5145.9692383, -8266.3876953, 8925.6201172
2: -2554.0090332, 6813.9682617, -2079.7299805, 5487.4799805, -8041.4892578, 8893.2333984
3: -4625.1669922, 6475.0996094, -3749.1301270, 5225.1914062, -9850.3583984, 10224.2275391
4: -3051.9951172, 7294.7050781, -2486.1972656, 5877.6064453, -8929.6015625, 9779.7373047

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9337378, upper bound: 8224.9335158
time: 1.00 seconds

## Relational analysis of NS_A2_B1_B1_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9333939, upper bound: 8224.9336022
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3232.0219727, 7154.4599609, -2977.6518555, 6545.1640625, -9777.1855469, 10130.8837891
1: -3092.6586914, 6337.7167969, -2844.8762207, 5790.4023438, -8883.0605469, 9182.5927734
2: -2531.2380371, 6757.6171875, -2336.2890625, 6172.2416992, -8703.4794922, 9093.0703125
3: -4583.4179688, 6420.7124023, -4197.7250977, 5874.0014648, -10457.4179688, 10618.4375000
4: -3024.6655273, 7233.3876953, -2792.0566406, 6603.8305664, -9628.4960938, 10024.0439453

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1_B1_B2_B1

### Relational analysis result of NS_A2_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9329017, upper bound: 8224.9344546
time: 1.03 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_B2

### Relational analysis result of NS_A2_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9325911, upper bound: 8224.9344546
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3174.9748535, 7026.1669922, -2936.6823730, 6493.2592773, -9668.2343750, 9961.2285156
1: -3038.8183594, 6224.8193359, -2814.3159180, 5746.4018555, -8785.2207031, 9038.1630859
2: -2487.2573242, 6636.9033203, -2303.0380859, 6128.1572266, -8615.4140625, 8939.2451172
3: -4504.4204102, 6306.8081055, -4172.1743164, 5828.0327148, -10331.8798828, 10478.9824219
4: -2971.9921875, 7105.0307617, -2753.0058594, 6562.5644531, -9534.5566406, 9856.5078125

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
time: 0.92 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3418.7656250, 7568.1743164, -2891.1013184, 6392.6147461, -9810.5312500, 10452.0214844
1: -3278.3342285, 6705.2636719, -2770.6791992, 5657.7270508, -8932.9384766, 9470.5341797
2: -2677.5400391, 7152.5898438, -2267.1994629, 6033.6508789, -8711.1914062, 9414.2236328
3: -4871.2177734, 6792.7373047, -4107.7182617, 5737.7602539, -10601.1660156, 10896.0312500
4: -3196.0339355, 7663.9375000, -2709.8520508, 6461.5102539, -9657.5429688, 10365.6992188

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
time: 0.87 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3248.9765625, 7188.7910156, -3091.5627441, 6798.8862305, -10047.8632812, 10279.1201172
1: -3109.2565918, 6368.0415039, -2953.5268555, 6018.8945312, -9128.1474609, 9321.5664062
2: -2544.9169922, 6789.9355469, -2425.1264648, 6412.7519531, -8957.6689453, 9214.2841797
3: -4608.3740234, 6452.3110352, -4356.8413086, 6105.0532227, -10713.4277344, 10809.1523438
4: -3041.1291504, 7268.8349609, -2898.9755859, 6859.4921875, -9900.6210938, 10166.3232422

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9212381, upper bound: 8224.9290615
time: 0.96 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3593.9106445, 7964.7099609, -3091.5627441, 6798.8862305, -10392.7968750, 11051.5537109
1: -3433.9033203, 7061.7031250, -2953.5268555, 6018.8945312, -9452.7978516, 10012.8447266
2: -2812.1867676, 7523.3061523, -2425.1264648, 6412.7519531, -9224.9384766, 9944.5312500
3: -5080.1250000, 7148.9531250, -4356.8413086, 6105.0532227, -11185.1777344, 11505.7949219
4: -3361.0830078, 8044.1240234, -2898.9755859, 6859.4921875, -10220.5742188, 10938.3281250

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9196876, upper bound: 8224.9306925
time: 0.96 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3224.0891113, 7134.3725586, -3384.1684570, 7491.2138672, -10713.0283203, 10516.5957031
1: -3085.4758301, 6320.1074219, -3236.7431641, 6634.2602539, -9717.2246094, 9555.2216797
2: -2525.2312012, 6738.7685547, -2650.7839355, 7070.0722656, -9594.4228516, 9388.8369141
3: -4573.5732422, 6403.3081055, -4788.7915039, 6722.5493164, -11292.8144531, 11190.2568359
4: -3017.4287109, 7214.2329102, -3168.7065430, 7561.9980469, -10578.7128906, 10381.3212891

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9336850, upper bound: 8224.9357853
time: 0.98 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9336850, upper bound: 8224.9350840
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3567.4562988, 7906.3837891, -3384.1684570, 7491.2138672, -11056.2548828, 11285.0771484
1: -3408.4240723, 7010.3496094, -3236.7431641, 6634.2602539, -10039.5898438, 10242.3037109
2: -2791.1748047, 7468.4506836, -2650.7839355, 7070.0722656, -9860.8916016, 10115.3544922
3: -5042.9736328, 7096.5795898, -4788.7915039, 6722.5493164, -11760.2871094, 11880.8242188
4: -3335.8835449, 7985.5825195, -3168.7065430, 7561.9980469, -10897.3603516, 11149.3398438

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9286311, upper bound: 8224.9324938
time: 0.97 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9278045, upper bound: 8224.9297632
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3240.2238770, 7170.1000977, -3226.2919922, 7140.0424805, -10375.9980469, 10392.2656250
1: -3100.3298340, 6351.9287109, -3086.6608887, 6325.5874023, -9421.4960938, 9434.1406250
2: -2537.7697754, 6772.4008789, -2526.6550293, 6744.1152344, -9278.7968750, 9296.0859375
3: -4594.4501953, 6435.4526367, -4573.7587891, 6408.3671875, -10997.2460938, 11003.4521484
4: -3032.5888672, 7249.4091797, -3019.3017578, 7218.6523438, -10247.4736328, 10265.1933594

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3222.8557129, 7134.9287109, -3377.5024414, 7466.7236328, -10685.8759766, 10509.0126953
1: -3082.8098145, 6321.1992188, -3226.3676758, 6619.1162109, -9698.2929688, 9543.8779297
2: -2523.6411133, 6739.2641602, -2645.4055176, 7052.1118164, -9573.3759766, 9382.6503906
3: -4567.4799805, 6403.1049805, -4769.1718750, 6702.6010742, -11265.7500000, 11167.5332031
4: -3015.6740723, 7212.5014648, -3161.0864258, 7541.7163086, -10554.3437500, 10371.0996094

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3602.9416504, 7984.3793945, -3242.6298828, 7174.7456055, -10774.0185547, 11220.0214844
1: -3442.3225098, 7079.1108398, -3102.9155273, 6355.9785156, -9794.0927734, 10175.2089844
2: -2819.2053223, 7541.7124023, -2539.7963867, 6776.8281250, -9594.2500000, 10076.0820312
3: -5092.3066406, 7166.4702148, -4598.6967773, 6439.8178711, -11525.7324219, 11757.8330078
4: -3369.4929199, 8063.6860352, -3034.9916992, 7254.5795898, -10621.3769531, 11092.5947266

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9315752, upper bound: 8224.9305311
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9310652, upper bound: 8224.9275810
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3573.0463867, 7919.9555664, -3550.1335449, 7862.0419922, -11429.6396484, 11462.9013672
1: -3413.7192383, 7022.2788086, -3392.1254883, 6966.0209961, -10373.9287109, 10407.0458984
2: -2795.6486816, 7481.1547852, -2779.1311035, 7424.8403320, -10217.0029297, 10254.8916016
3: -5050.0683594, 7108.3686523, -5018.0639648, 7052.3916016, -12094.5937500, 12117.7421875
4: -3341.0986328, 7998.6923828, -3320.4201660, 7940.6850586, -11277.2587891, 11312.8798828

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9349846, upper bound: 8224.9344472
time: 1.07 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9343133, upper bound: 8224.9344472
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3242.6298828, 7174.7456055, -3602.9416504, 7984.3793945, -11220.0224609, 10774.0185547
1: -3102.9155273, 6355.9785156, -3442.3225098, 7079.1108398, -10175.2089844, 9794.0927734
2: -2539.7963867, 6776.8281250, -2819.2053223, 7541.7124023, -10076.0820312, 9594.2500000
3: -4598.6967773, 6439.8178711, -5092.3066406, 7166.4702148, -11757.8330078, 11525.7324219
4: -3034.9916992, 7254.5795898, -3369.4929199, 8063.6860352, -11092.5947266, 10621.3769531

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9300672, upper bound: 8224.9315745
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9271684, upper bound: 8224.9310315
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3550.1335449, 7862.0419922, -3573.0463867, 7919.9555664, -11462.9013672, 11429.6396484
1: -3392.1254883, 6966.0209961, -3413.7192383, 7022.2788086, -10407.0458984, 10373.9287109
2: -2779.1311035, 7424.8403320, -2795.6486816, 7481.1547852, -10254.8916016, 10217.0029297
3: -5018.0639648, 7052.3916016, -5050.0683594, 7108.3686523, -12117.7421875, 12094.5937500
4: -3320.4201660, 7940.6850586, -3341.0986328, 7998.6923828, -11312.8798828, 11277.2587891

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9343131, upper bound: 8224.9351364
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9343132, upper bound: 8224.9343206
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3587.6674805, 7951.7968750, -3573.2648926, 7920.9741211, -11500.3994141, 11517.1210938
1: -3427.3283691, 7050.4785156, -3413.2243652, 7023.3901367, -10442.3281250, 10455.3691406
2: -2806.9609375, 7510.9858398, -2795.4338379, 7481.9296875, -10282.8701172, 10300.6718750
3: -5069.7436523, 7136.9799805, -5048.6835938, 7109.0903320, -12168.4345703, 12175.0039062
4: -3354.7941895, 8030.2739258, -3340.9497070, 7998.7880859, -11346.3007812, 11364.4345703

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9375278
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9375278
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3572.1247559, 7918.8920898, -3818.4758301, 8453.0224609, -12016.8935547, 11730.1425781
1: -3412.1384277, 7021.7597656, -3642.5478516, 7500.3364258, -10904.0410156, 10656.1816406
2: -2794.7048340, 7480.2236328, -2988.3198242, 7983.8696289, -10772.5527344, 10463.4609375
3: -5046.6425781, 7107.3691406, -5374.3549805, 7589.2177734, -12625.8623047, 12471.2861328
4: -3340.1542969, 7996.5161133, -3571.1904297, 8528.2265625, -11861.0449219, 11561.8544922

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9376759
time: 0.93 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9376759
time: 1.07 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.72 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9384622, upper bound: 8224.9384622
NS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9340380, upper bound: 8224.9340957
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9340337, upper bound: 8224.9350173
NS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9343034, upper bound: 8224.9344527
NS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9343102, upper bound: 8224.9339802
NS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9340957, upper bound: 8224.9340380
NS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9350173, upper bound: 8224.9340337
NS_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9344526, upper bound: 8224.9343034
NS_A1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9339802, upper bound: 8224.9343102
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9351777, upper bound: 8224.9351776
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9351777, upper bound: 8224.9355140
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9355140, upper bound: 8224.9353673
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9355140, upper bound: 8224.9356035
NS_A1_B2_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9329575, upper bound: 8224.9337378
NS_A1_B2_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9336022, upper bound: 8224.9333939
NS_A1_B2_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9344546, upper bound: 8224.9329017
NS_A1_B2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9344546, upper bound: 8224.9325911
NS_A1_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
NS_A1_B2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
NS_A1_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9282379, upper bound: 8224.9327142
NS_A1_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
NS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9290615, upper bound: 8224.9212381
NS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
NS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9306925, upper bound: 8224.9196877
NS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
NS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9357852, upper bound: 8224.9336850
NS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9350840, upper bound: 8224.9336850
NS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9324938, upper bound: 8224.9286311
NS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9297632, upper bound: 8224.9278045
NS_A2_B1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9337378, upper bound: 8224.9335158
NS_A2_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9333939, upper bound: 8224.9336022
NS_A2_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9329017, upper bound: 8224.9344546
NS_A2_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9325911, upper bound: 8224.9344546
NS_A2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
NS_A2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
NS_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
NS_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
NS_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9212381, upper bound: 8224.9290615
NS_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
NS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9196876, upper bound: 8224.9306925
NS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
NS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9336850, upper bound: 8224.9357853
NS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9336850, upper bound: 8224.9350840
NS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9286311, upper bound: 8224.9324938
NS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9278045, upper bound: 8224.9297632
NS_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
NS_A2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
NS_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
NS_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9388336, upper bound: 8224.9394247
NS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9315752, upper bound: 8224.9305311
NS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9310652, upper bound: 8224.9275810
NS_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9349846, upper bound: 8224.9344472
NS_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9343133, upper bound: 8224.9344472
NS_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9300672, upper bound: 8224.9315745
NS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9271684, upper bound: 8224.9310315
NS_A2_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9343131, upper bound: 8224.9351364
NS_A2_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9343132, upper bound: 8224.9343206
NS_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9375278
NS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9375278
NS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9376759
NS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -8224.9372966, upper bound: 8224.9376759

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2657.1162109, 5837.5673828, -2657.1162109, 5837.5673828, -8494.6835938, 8494.6835938
1: -2542.1381836, 5163.2006836, -2542.1381836, 5163.2006836, -7705.3388672, 7705.3388672
2: -2085.6232910, 5505.5283203, -2085.6232910, 5505.5283203, -7591.1513672, 7591.1513672
3: -3758.0598145, 5242.0830078, -3758.0598145, 5242.0830078, -9000.1425781, 9000.1425781
4: -2493.4133301, 5895.7192383, -2493.4133301, 5895.7192383, -8389.1328125, 8389.1328125

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9382459, upper bound: 8224.9379536
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9380682, upper bound: 8224.9379549
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2805.8913574, 6155.1562500, -2657.1162109, 5837.5673828, -8643.4560547, 8812.2724609
1: -2679.1462402, 5448.7744141, -2542.1381836, 5163.2006836, -7842.3466797, 7990.9125977
2: -2202.5554199, 5805.0957031, -2085.6232910, 5505.5283203, -7708.0839844, 7890.7187500
3: -3948.4985352, 5528.6254883, -3758.0598145, 5242.0830078, -9190.5820312, 9286.6835938
4: -2632.7983398, 6209.9604492, -2493.4133301, 5895.7192383, -8528.5166016, 8703.3740234

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9370239, upper bound: 8224.9363711
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9370239, upper bound: 8224.9362242
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2657.1162109, 5837.5673828, -2805.8913574, 6155.1562500, -8812.2724609, 8643.4560547
1: -2542.1381836, 5163.2006836, -2679.1462402, 5448.7744141, -7990.9125977, 7842.3466797
2: -2085.6232910, 5505.5283203, -2202.5554199, 5805.0957031, -7890.7187500, 7708.0839844
3: -3758.0598145, 5242.0830078, -3948.4985352, 5528.6254883, -9286.6835938, 9190.5820312
4: -2493.4133301, 5895.7192383, -2632.7983398, 6209.9604492, -8703.3740234, 8528.5166016

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9363711, upper bound: 8224.9361977
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9358256, upper bound: 8224.9358256
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -2805.8913574, 6155.1562500, -2805.8913574, 6155.1562500, -8961.0458984, 8961.0449219
1: -2679.1462402, 5448.7744141, -2679.1462402, 5448.7744141, -8127.9204102, 8127.9204102
2: -2202.5554199, 5805.0957031, -2202.5554199, 5805.0957031, -8007.6508789, 8007.6508789
3: -3948.4985352, 5528.6254883, -3948.4985352, 5528.6254883, -9477.1230469, 9477.1230469
4: -2632.7983398, 6209.9604492, -2632.7983398, 6209.9604492, -8842.7578125, 8842.7587891

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9379536, upper bound: 8224.9381905
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9379548, upper bound: 8224.9379549
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -2649.1137695, 5818.1484375, -3108.5541992, 6836.6762695, -9485.7890625, 8926.7011719
1: -2535.2770996, 5145.9692383, -2969.7551270, 6052.0644531, -8587.3417969, 8115.7246094
2: -2079.7299805, 5487.4799805, -2438.4201660, 6448.2377930, -8527.9667969, 7925.8999023
3: -3749.1301270, 5225.1914062, -4380.9819336, 6138.8007812, -9887.9296875, 9606.1738281
4: -2486.1972656, 5877.6064453, -2914.9265137, 6897.4609375, -9383.6582031, 8792.5332031

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9289417, upper bound: 8224.9287347
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9262022, upper bound: 8224.9280605
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -2961.8073730, 6549.1972656, -3082.9812012, 6780.1928711, -9742.0000000, 9632.1777344
1: -2838.1613770, 5795.6123047, -2945.2717285, 6002.6254883, -8840.7871094, 8740.8837891
2: -2322.3623047, 6180.8681641, -2418.0507812, 6395.7460938, -8718.1083984, 8598.9189453
3: -4207.6103516, 5877.7426758, -4345.7895508, 6088.5917969, -10296.2011719, 10223.5322266
4: -2776.0825195, 6619.0249023, -2890.4191895, 6841.7573242, -9617.8398438, 9509.4414062

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9288725, upper bound: 8224.9293547
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9261464, upper bound: 8224.9286398
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2957.4621582, 6502.6098633, -3077.9487305, 6770.0307617, -9727.4931641, 9580.5585938
1: -2826.0749512, 5751.9179688, -2940.6850586, 5993.2368164, -8819.3095703, 8692.6015625
2: -2320.2167969, 6132.0644531, -2414.4707031, 6385.6372070, -8705.8525391, 8546.5351562
3: -4171.7846680, 5835.2275391, -4338.4541016, 6078.8540039, -10250.6386719, 10173.6816406
4: -2773.0759277, 6561.6909180, -2886.0214844, 6830.5390625, -9603.6142578, 9447.7099609

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9322265, upper bound: 8224.9336208
time: 6.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9320559, upper bound: 8224.9318985
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3459.7575684, 7648.3627930, -3055.6723633, 6722.4863281, -10182.2441406, 10701.2099609
1: -3317.7458496, 6753.2924805, -2920.0053711, 5950.2646484, -9268.0078125, 9673.2958984
2: -2713.1774902, 7208.3940430, -2397.0195312, 6340.5839844, -9053.7617188, 9603.5810547
3: -4925.4042969, 6855.1660156, -4309.2041016, 6035.5039062, -10960.7285156, 11164.3701172
4: -3244.7053223, 7725.1791992, -2865.1770020, 6783.0170898, -10027.7226562, 10586.8183594

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9337582, upper bound: 8224.9330681
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9328802, upper bound: 8224.9331523
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3108.5541992, 6836.6762695, -2649.1137695, 5818.1484375, -8926.7011719, 9485.7890625
1: -2969.7551270, 6052.0644531, -2535.2770996, 5145.9692383, -8115.7246094, 8587.3417969
2: -2438.4201660, 6448.2377930, -2079.7299805, 5487.4799805, -7925.8999023, 8527.9667969
3: -4380.9819336, 6138.8007812, -3749.1301270, 5225.1914062, -9606.1738281, 9887.9296875
4: -2914.9265137, 6897.4609375, -2486.1972656, 5877.6064453, -8792.5332031, 9383.6582031

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9287347, upper bound: 8224.9289417
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9280605, upper bound: 8224.9262022
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3082.9812012, 6780.1928711, -2961.8073730, 6549.1972656, -9632.1787109, 9742.0000000
1: -2945.2717285, 6002.6254883, -2838.1613770, 5795.6123047, -8740.8837891, 8840.7871094
2: -2418.0507812, 6395.7460938, -2322.3623047, 6180.8681641, -8598.9189453, 8718.1083984
3: -4345.7895508, 6088.5917969, -4207.6103516, 5877.7426758, -10223.5322266, 10296.2011719
4: -2890.4191895, 6841.7573242, -2776.0825195, 6619.0249023, -9509.4414062, 9617.8398438

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9293547, upper bound: 8224.9288725
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9286398, upper bound: 8224.9261464
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3077.9487305, 6770.0307617, -2957.4621582, 6502.6098633, -9580.5585938, 9727.4931641
1: -2940.6850586, 5993.2368164, -2826.0749512, 5751.9179688, -8692.6015625, 8819.3095703
2: -2414.4707031, 6385.6372070, -2320.2167969, 6132.0644531, -8546.5351562, 8705.8535156
3: -4338.4541016, 6078.8540039, -4171.7846680, 5835.2275391, -10173.6816406, 10250.6386719
4: -2886.0214844, 6830.5390625, -2773.0759277, 6561.6909180, -9447.7099609, 9603.6142578

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9336209, upper bound: 8224.9322265
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9318985, upper bound: 8224.9320559
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3055.6723633, 6722.4863281, -3459.7575684, 7648.3627930, -10701.2099609, 10182.2441406
1: -2920.0053711, 5950.2646484, -3317.7458496, 6753.2924805, -9673.2958984, 9268.0078125
2: -2397.0195312, 6340.5839844, -2713.1774902, 7208.3940430, -9603.5810547, 9053.7617188
3: -4309.2041016, 6035.5039062, -4925.4042969, 6855.1660156, -11164.3701172, 10960.7285156
4: -2865.1770020, 6783.0170898, -3244.7053223, 7725.1791992, -10586.8183594, 10027.7226562

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9330681, upper bound: 8224.9337582
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9331523, upper bound: 8224.9335179
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3077.8027344, 6771.2934570, -3077.8027344, 6771.2934570, -9849.0957031, 9849.0957031
1: -2939.2700195, 5994.8706055, -2939.2700195, 5994.8706055, -8934.1406250, 8934.1406250
2: -2413.6916504, 6386.5634766, -2413.6916504, 6386.5634766, -8800.2529297, 8800.2539062
3: -4334.7470703, 6079.5493164, -4334.7470703, 6079.5493164, -10414.2968750, 10414.2968750
4: -2885.3129883, 6830.0566406, -2885.3129883, 6830.0566406, -9715.3691406, 9715.3691406

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9332152, upper bound: 8224.9334052
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9344189, upper bound: 8224.9342663
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3077.8027344, 6771.2934570, -3325.8334961, 7307.5283203, -10385.3310547, 10097.1269531
1: -2939.2700195, 5994.8706055, -3170.6003418, 6475.9086914, -9415.1787109, 9165.4707031
2: -2413.6916504, 6386.5634766, -2608.5979004, 6892.8872070, -9306.5791016, 8995.1611328
3: -4334.7470703, 6079.5493164, -4661.9726562, 6563.8291016, -10898.5761719, 10741.5205078
4: -2885.3129883, 6830.0566406, -3117.6538086, 7363.6215820, -10248.9316406, 9947.7109375

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9351197, upper bound: 8224.9354455
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9351061, upper bound: 8224.9354613
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3325.8334961, 7307.5283203, -3077.8027344, 6771.2934570, -10097.1269531, 10385.3310547
1: -3170.6003418, 6475.9086914, -2939.2700195, 5994.8706055, -9165.4707031, 9415.1787109
2: -2608.5979004, 6892.8872070, -2413.6916504, 6386.5634766, -8995.1611328, 9306.5791016
3: -4661.9726562, 6563.8291016, -4334.7470703, 6079.5493164, -10741.5205078, 10898.5761719
4: -3117.6538086, 7363.6215820, -2885.3129883, 6830.0566406, -9947.7109375, 10248.9316406

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9354455, upper bound: 8224.9353582
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9354613, upper bound: 8224.9353582
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3325.8334961, 7307.5283203, -3325.8334961, 7307.5283203, -10633.3613281, 10633.3613281
1: -3170.6003418, 6475.9086914, -3170.6003418, 6475.9086914, -9646.5087891, 9646.5087891
2: -2608.5979004, 6892.8872070, -2608.5979004, 6892.8872070, -9501.4853516, 9501.4853516
3: -4661.9726562, 6563.8291016, -4661.9726562, 6563.8291016, -11225.8017578, 11225.8017578
4: -3117.6538086, 7363.6215820, -3117.6538086, 7363.6215820, -10481.2734375, 10481.2734375

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9347943, upper bound: 8224.9345232
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9342663, upper bound: 8224.9345311
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -2598.3046875, 5707.3593750, -3223.0895996, 7132.9448242, -9730.0556641, 8930.4453125
1: -2487.2863770, 5048.5620117, -3085.0844727, 6318.7382812, -8806.0244141, 8133.6455078
2: -2039.8798828, 5383.6860352, -2524.6430664, 6737.7543945, -8776.7460938, 7908.3286133
3: -3678.7441406, 5126.0644531, -4573.4848633, 6402.1840820, -10080.9277344, 9699.5488281
4: -2438.0556641, 5766.7446289, -3016.4846191, 7213.3134766, -9649.7236328, 8783.2294922

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9334732, upper bound: 8224.9326219
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_A1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9334732, upper bound: 8224.9333939
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -2689.6623535, 5926.7617188, -3226.7541504, 7140.0576172, -9828.7177734, 9153.5146484
1: -2576.0070801, 5244.7817383, -3088.3220215, 6325.2900391, -8901.2958984, 8333.1035156
2: -2110.9492188, 5591.7143555, -2527.6218262, 6744.2646484, -8854.8251953, 8119.3359375
3: -3812.6948242, 5321.8217773, -4577.4018555, 6408.6030273, -10221.2958984, 9899.2236328
4: -2523.0671387, 5987.4780273, -3020.1462402, 7219.9604492, -9741.8125000, 9007.6220703

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9335906, upper bound: 8224.9326219
time: 2.41 seconds

## Relational analysis of NS_A1_B2_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9335906, upper bound: 8224.9333939
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -2937.1411133, 6457.1582031, -3227.4531250, 7144.5800781, -10080.5996094, 9684.6103516
1: -2806.6442871, 5711.9545898, -3088.3496094, 6328.9243164, -9135.5673828, 8800.3046875
2: -2304.2856445, 6089.4409180, -2527.6345215, 6748.3271484, -9051.8691406, 8617.0742188
3: -4142.9741211, 5794.6376953, -4577.2343750, 6411.8037109, -10554.7763672, 10371.8720703
4: -2753.8535156, 6516.1079102, -3020.3598633, 7223.5297852, -9976.0654297, 9536.4677734

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_A1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9344156, upper bound: 8224.9326876
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9344156, upper bound: 8224.9328518
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -3440.1328125, 7604.4233398, -3220.0642090, 7128.8906250, -10565.2460938, 10819.5195312
1: -3299.0310059, 6714.5864258, -3081.4467773, 6314.8793945, -9609.0185547, 9792.1328125
2: -2697.8039551, 7167.1674805, -2521.8447266, 6733.5102539, -9428.4072266, 9686.0898438
3: -4897.8286133, 6815.8955078, -4567.3779297, 6397.5629883, -11287.6396484, 11378.4208984
4: -3226.1899414, 7681.2348633, -3013.4401855, 7207.7895508, -10430.8330078, 10690.0537109

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_A1_A2_A2_A1

### Relational analysis result of NS_A1_B2_A1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9267079, upper bound: 8224.9268963
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_A1_A2_A2_A2

### Relational analysis result of NS_A1_B2_A1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9256530, upper bound: 8224.9243950
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2914.8850098, 6445.1616211, -3174.9233398, 7026.0532227, -9939.3281250, 9620.0830078
1: -2793.5290527, 5704.0649414, -3038.7702637, 6224.7167969, -9017.2998047, 8742.8349609
2: -2286.1232910, 6082.8447266, -2487.2175293, 6636.7958984, -8922.2246094, 8570.0625000
3: -4141.2792969, 5785.1049805, -4504.3520508, 6306.7055664, -10447.9843750, 10288.8818359
4: -2732.7878418, 6513.9584961, -2971.9450684, 7104.9179688, -9836.1738281, 9485.9023438

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9289678, upper bound: 8224.9297406
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9288349, upper bound: 8224.9274791
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3162.5458984, 6995.9721680, -3175.1994629, 7026.6616211, -10185.7978516, 10165.9433594
1: -3036.8833008, 6192.5639648, -3039.0253906, 6225.2617188, -9257.4531250, 9227.1699219
2: -2479.3664551, 6607.1318359, -2487.4299316, 6637.3691406, -9114.6240234, 9091.1484375
3: -4514.1806641, 6279.2006836, -4504.7187500, 6307.2529297, -10813.9238281, 10778.7011719
4: -2960.4914551, 7082.2211914, -2972.1989746, 7105.5249023, -10063.4130859, 10049.3818359

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9307532, upper bound: 8224.9307863
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9305342, upper bound: 8224.9273606
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2914.8850098, 6445.1616211, -3418.5881348, 7567.7822266, -10475.4775391, 9863.0185547
1: -2793.5290527, 5704.0649414, -3278.1701660, 6704.9150391, -9493.1533203, 8979.2158203
2: -2286.1232910, 6082.8447266, -2677.4033203, 7152.2216797, -9432.7832031, 8760.2480469
3: -4141.2792969, 5785.1049805, -4870.9824219, 6792.3872070, -10929.5341797, 10648.3701172
4: -2732.7878418, 6513.9584961, -3195.8703613, 7663.5488281, -10388.2363281, 9709.8281250

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9303073, upper bound: 8224.9327142
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3162.5458984, 6995.9721680, -3418.8630371, 7568.3881836, -10721.9462891, 10407.9501953
1: -3036.8833008, 6192.5639648, -3278.4243164, 6705.4555664, -9733.3056641, 9463.1865234
2: -2479.3664551, 6607.1318359, -2677.6149902, 7152.7915039, -9625.1777344, 9280.0986328
3: -4514.1806641, 6279.2006836, -4871.3471680, 6792.9311523, -11295.3242188, 11138.1884766
4: -2960.4914551, 7082.2211914, -3196.1237793, 7664.1523438, -10615.4707031, 10272.3457031

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9263210, upper bound: 8224.9290551
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9263813, upper bound: 8224.9265821
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3030.3769531, 6663.7158203, -3208.5258789, 7099.3886719, -10128.5654297, 9872.2421875
1: -2895.1418457, 5900.2783203, -3070.6914062, 6289.4262695, -9184.5683594, 8970.9697266
2: -2377.6005859, 6285.6259766, -2513.5024414, 6705.8076172, -9082.6582031, 8799.1279297
3: -4270.1630859, 5984.6254883, -4551.0976562, 6372.5761719, -10642.7392578, 10535.7216797
4: -2842.0678711, 6723.1992188, -3003.5273438, 7178.6313477, -10019.2119141, 9726.7265625

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9285513, upper bound: 8224.9192492
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9285513, upper bound: 8224.9192492
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3271.9775391, 7200.5834961, -3166.4062500, 7006.2773438, -10275.0888672, 10366.9902344
1: -3132.9624023, 6375.1850586, -3030.2819824, 6207.3955078, -9337.6269531, 9405.3574219
2: -2566.4416504, 6795.8281250, -2480.3530273, 6618.3413086, -9182.4384766, 9276.1816406
3: -4635.0522461, 6465.5712891, -4491.2885742, 6289.0400391, -10921.5781250, 10955.4423828
4: -3064.6372070, 7277.0117188, -2963.5898438, 7085.0268555, -10146.8574219, 10240.6015625

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9285513, upper bound: 8224.9192492
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9285513, upper bound: 8224.9192492
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3052.0690918, 6711.5078125, -3533.2880859, 7831.2421875, -10878.3964844, 10244.7958984
1: -2915.8315430, 5942.2358398, -3376.1706543, 6944.4726562, -9857.7861328, 9318.4062500
2: -2394.4382324, 6330.5854492, -2765.0695801, 7397.7436523, -9788.0908203, 9095.6552734
3: -4300.8779297, 6027.2148438, -4994.5532227, 7029.9628906, -11330.8408203, 11021.7675781
4: -2862.2287598, 6771.4238281, -3304.6203613, 7909.5200195, -10766.8193359, 10076.0439453

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
time: 1.00 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3008.0937500, 6614.7045898, -3770.6071777, 8358.9648438, -11356.8261719, 10385.3105469
1: -2873.5646973, 5856.9575195, -3609.9064941, 7411.3232422, -10278.0654297, 9466.8642578
2: -2359.8000488, 6239.6079102, -2950.3520508, 7899.5922852, -10250.5644531, 9189.9599609
3: -4238.0610352, 5940.3134766, -5353.6396484, 7502.5102539, -11737.9892578, 11287.5517578
4: -2820.5949707, 6673.7646484, -3522.7041016, 8454.4287109, -11263.5009766, 10196.4687500

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9283179, upper bound: 8224.9192093
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3384.1684570, 7491.2138672, -3211.1638184, 7105.9721680, -10488.0732422, 10700.0166016
1: -3236.7431641, 6634.2602539, -3073.0444336, 6295.2041016, -9530.2285156, 9704.6542969
2: -2650.7839355, 7070.0722656, -2515.0283203, 6712.1191406, -9362.0722656, 9584.1162109
3: -4788.7915039, 6722.5493164, -4555.1347656, 6377.9160156, -11164.7714844, 11274.1308594
4: -3168.7065430, 7561.9980469, -3005.2019043, 7185.6333008, -10352.5488281, 10566.4130859

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9347456, upper bound: 8224.9338239
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9353937, upper bound: 8224.9337353
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3356.4438477, 7431.2382812, -3513.0461426, 7780.7426758, -11133.3046875, 10941.7421875
1: -3210.3264160, 6581.4272461, -3356.7871094, 6894.4482422, -10101.4306641, 9935.0312500
2: -2629.1452637, 7013.7416992, -2749.8818359, 7348.5019531, -9974.9443359, 9762.6943359
3: -4749.7456055, 6668.6176758, -4966.4628906, 6979.5703125, -11725.8818359, 11630.1533203
4: -3142.5583496, 7501.5952148, -3285.2968750, 7859.2407227, -10998.1123047, 10785.9824219

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9350839, upper bound: 8224.9337298
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9340549, upper bound: 8224.9337296
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3345.6171875, 7406.1479492, -3506.8432617, 7772.9389648, -11112.8691406, 10910.4697266
1: -3199.9543457, 6559.5512695, -3350.7102051, 6893.1210938, -10088.1318359, 9907.0634766
2: -2620.7675781, 6990.0371094, -2744.0646973, 7342.8940430, -9959.5810547, 9733.6855469
3: -4734.2426758, 6646.6938477, -4957.4296875, 6977.5966797, -11707.0527344, 11598.7734375
4: -3132.7685547, 7476.1455078, -3279.4294434, 7851.0190430, -10978.6562500, 10755.0146484

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9297632, upper bound: 8224.9278045
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9297632, upper bound: 8224.9278045
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3302.4838867, 7310.8564453, -3743.9204102, 8300.4306641, -11591.7949219, 11050.3964844
1: -3158.5742188, 6475.4736328, -3584.3083496, 7359.7685547, -10508.9238281, 10052.9160156
2: -2586.7548828, 6900.4570312, -2929.1926270, 7844.5498047, -10422.4492188, 9827.7939453
3: -4673.0629883, 6561.1342773, -5316.4096680, 7449.9155273, -12113.6630859, 11864.5898438
4: -3091.8854980, 7380.3115234, -3497.2727051, 8395.7421875, -11475.8154297, 10875.6894531

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9297632, upper bound: 8224.9278045
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9297632, upper bound: 8224.9278045
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3223.0895996, 7132.9448242, -2598.3046875, 5707.3593750, -8930.4453125, 9730.0556641
1: -3085.0844727, 6318.7382812, -2487.2863770, 5048.5620117, -8133.6455078, 8806.0244141
2: -2524.6430664, 6737.7543945, -2039.8798828, 5383.6860352, -7908.3286133, 8776.7470703
3: -4573.4848633, 6402.1840820, -3678.7441406, 5126.0644531, -9699.5488281, 10080.9277344
4: -3016.4846191, 7213.3134766, -2438.0556641, 5766.7446289, -8783.2285156, 9649.7236328

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9326219, upper bound: 8224.9334732
time: 0.97 seconds

## Relational analysis of NS_A2_B1_B1_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9326219, upper bound: 8224.9335158
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3226.7541504, 7140.0576172, -2689.6623535, 5926.7617188, -9153.5146484, 9828.7177734
1: -3088.3220215, 6325.2900391, -2576.0070801, 5244.7817383, -8333.1035156, 8901.2949219
2: -2527.6218262, 6744.2646484, -2110.9492188, 5591.7143555, -8119.3359375, 8854.8251953
3: -4577.4018555, 6408.6030273, -3812.6948242, 5321.8217773, -9899.2236328, 10221.2958984
4: -3020.1462402, 7219.9604492, -2523.0671387, 5987.4780273, -9007.6220703, 9741.8125000

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9326219, upper bound: 8224.9335907
time: 1.17 seconds

## Relational analysis of NS_A2_B1_B1_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9326219, upper bound: 8224.9336022
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3227.4531250, 7144.5800781, -2937.1411133, 6457.1582031, -9684.6103516, 10080.5996094
1: -3088.3496094, 6328.9243164, -2806.6442871, 5711.9545898, -8800.3046875, 9135.5673828
2: -2527.6345215, 6748.3271484, -2304.2856445, 6089.4409180, -8617.0751953, 9051.8691406
3: -4577.2343750, 6411.8037109, -4142.9741211, 5794.6376953, -10371.8720703, 10554.7763672
4: -3020.3598633, 7223.5297852, -2753.8535156, 6516.1079102, -9536.4677734, 9976.0654297

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9326876, upper bound: 8224.9344156
time: 0.98 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9328517, upper bound: 8224.9344156
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_B1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3220.0642090, 7128.8906250, -3440.1328125, 7604.4233398, -10819.5195312, 10565.2460938
1: -3081.4467773, 6314.8793945, -3299.0310059, 6714.5864258, -9792.1328125, 9609.0185547
2: -2521.8447266, 6733.5102539, -2697.8039551, 7167.1674805, -9686.0898438, 9428.4072266
3: -4567.3779297, 6397.5629883, -4897.8286133, 6815.8955078, -11378.4208984, 11287.6396484
4: -3013.4401855, 7207.7895508, -3226.1899414, 7681.2348633, -10690.0537109, 10430.8330078

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B1_B1_B2_B2_B1

### Relational analysis result of NS_A2_B1_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9268963, upper bound: 8224.9267079
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2_B2_B2

### Relational analysis result of NS_A2_B1_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9243950, upper bound: 8224.9260648
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3174.9233398, 7026.0532227, -2914.8850098, 6445.1616211, -9620.0830078, 9939.3281250
1: -3038.7702637, 6224.7167969, -2793.5290527, 5704.0649414, -8742.8349609, 9017.2998047
2: -2487.2175293, 6636.7958984, -2286.1232910, 6082.8447266, -8570.0625000, 8922.2246094
3: -4504.3520508, 6306.7055664, -4141.2792969, 5785.1049805, -10288.8818359, 10447.9843750
4: -2971.9450684, 7104.9179688, -2732.7878418, 6513.9584961, -9485.9023438, 9836.1738281

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9294041, upper bound: 8224.9289678
time: 1.02 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9274791, upper bound: 8224.9288349
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3175.1994629, 7026.6616211, -3162.5458984, 6995.9721680, -10165.9433594, 10185.7978516
1: -3039.0253906, 6225.2617188, -3036.8833008, 6192.5639648, -9227.1699219, 9257.4531250
2: -2487.4299316, 6637.3691406, -2479.3664551, 6607.1318359, -9091.1484375, 9114.6240234
3: -4504.7187500, 6307.2529297, -4514.1806641, 6279.2006836, -10778.7011719, 10813.9238281
4: -2972.1989746, 7105.5249023, -2960.4914551, 7082.2211914, -10049.3818359, 10063.4130859

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 22

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9307863, upper bound: 8224.9307532
time: 0.82 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9273606, upper bound: 8224.9305342
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3418.5881348, 7567.7822266, -2914.8850098, 6445.1616211, -9863.0185547, 10475.4775391
1: -3278.1701660, 6704.9150391, -2793.5290527, 5704.0649414, -8979.2158203, 9493.1533203
2: -2677.4033203, 7152.2216797, -2286.1232910, 6082.8447266, -8760.2480469, 9432.7832031
3: -4870.9824219, 6792.3872070, -4141.2792969, 5785.1049805, -10648.3701172, 10929.5341797
4: -3195.8703613, 7663.5488281, -2732.7878418, 6513.9584961, -9709.8281250, 10388.2363281

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
time: 2.86 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9327142, upper bound: 8224.9303073
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3418.8630371, 7568.3881836, -3162.5458984, 6995.9721680, -10407.9501953, 10721.9462891
1: -3278.4243164, 6705.4555664, -3036.8833008, 6192.5639648, -9463.1865234, 9733.3056641
2: -2677.6149902, 7152.7915039, -2479.3664551, 6607.1318359, -9280.0986328, 9625.1777344
3: -4871.3471680, 6792.9311523, -4514.1806641, 6279.2006836, -11138.1884766, 11295.3242188
4: -3196.1237793, 7664.1523438, -2960.4914551, 7082.2211914, -10272.3457031, 10615.4707031

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9290551, upper bound: 8224.9263210
time: 0.90 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9265821, upper bound: 8224.9263813
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3208.5258789, 7099.3886719, -3030.3769531, 6663.7158203, -9872.2421875, 10128.5654297
1: -3070.6914062, 6289.4262695, -2895.1418457, 5900.2783203, -8970.9697266, 9184.5683594
2: -2513.5024414, 6705.8076172, -2377.6005859, 6285.6259766, -8799.1279297, 9082.6582031
3: -4551.0976562, 6372.5761719, -4270.1630859, 5984.6254883, -10535.7216797, 10642.7392578
4: -3003.5273438, 7178.6313477, -2842.0678711, 6723.1992188, -9726.7265625, 10019.2119141

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192492, upper bound: 8224.9285513
time: 0.94 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192492, upper bound: 8224.9285513
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3166.4062500, 7006.2773438, -3271.9775391, 7200.5834961, -10366.9902344, 10275.0888672
1: -3030.2819824, 6207.3955078, -3132.9624023, 6375.1850586, -9405.3574219, 9337.6269531
2: -2480.3530273, 6618.3413086, -2566.4416504, 6795.8281250, -9276.1816406, 9182.4384766
3: -4491.2885742, 6289.0400391, -4635.0522461, 6465.5712891, -10955.4423828, 10921.5781250
4: -2963.5898438, 7085.0268555, -3064.6372070, 7277.0117188, -10240.6015625, 10146.8574219

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192492, upper bound: 8224.9285513
time: 0.94 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192492, upper bound: 8224.9285513
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -3533.4123535, 7831.5122070, -3052.0690918, 6711.5078125, -10244.9199219, 10878.6660156
1: -3376.2863770, 6944.7104492, -2915.8315430, 5942.2358398, -9318.5224609, 9858.0253906
2: -2765.1657715, 7397.9960938, -2394.4382324, 6330.5854492, -9095.7509766, 9788.3427734
3: -4994.7197266, 7030.2021484, -4300.8779297, 6027.2148438, -11021.9345703, 11331.0800781
4: -3304.7353516, 7909.7875977, -2862.2287598, 6771.4238281, -10076.1591797, 10767.0869141

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
time: 1.07 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3770.7224121, 8359.2128906, -3008.0937500, 6614.7045898, -10385.4267578, 11357.0742188
1: -3610.0131836, 7411.5434570, -2873.5646973, 5856.9575195, -9466.9707031, 10278.2871094
2: -2950.4411621, 7899.8247070, -2359.8000488, 6239.6079102, -9190.0488281, 10250.7978516
3: -5353.7929688, 7502.7324219, -4238.0610352, 5940.3134766, -11287.7060547, 11738.2109375
4: -3522.8107910, 8454.6757812, -2820.5949707, 6673.7646484, -10196.5751953, 11263.7460938

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
time: 1.22 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9192093, upper bound: 8224.9283179
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3211.1638184, 7105.9721680, -3384.1684570, 7491.2138672, -10700.0166016, 10488.0732422
1: -3073.0444336, 6295.2041016, -3236.7431641, 6634.2602539, -9704.6542969, 9530.2285156
2: -2515.0283203, 6712.1191406, -2650.7839355, 7070.0722656, -9584.1162109, 9362.0732422
3: -4555.1347656, 6377.9160156, -4788.7915039, 6722.5493164, -11274.1308594, 11164.7705078
4: -3005.2019043, 7185.6333008, -3168.7065430, 7561.9980469, -10566.4130859, 10352.5488281

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9338239, upper bound: 8224.9347456
time: 0.98 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9337353, upper bound: 8224.9353938
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3513.0461426, 7780.7426758, -3356.4438477, 7431.2382812, -10941.7421875, 11133.3046875
1: -3356.7871094, 6894.4482422, -3210.3264160, 6581.4272461, -9935.0312500, 10101.4306641
2: -2749.8818359, 7348.5019531, -2629.1452637, 7013.7416992, -9762.6943359, 9974.9443359
3: -4966.4628906, 6979.5703125, -4749.7456055, 6668.6176758, -11630.1533203, 11725.8808594
4: -3285.2968750, 7859.2407227, -3142.5583496, 7501.5952148, -10785.9824219, 10998.1123047

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 38

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9337298, upper bound: 8224.9350840
time: 0.96 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9337296, upper bound: 8224.9340549
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3506.9035645, 7773.0703125, -3345.6171875, 7406.1479492, -10910.5312500, 11113.0009766
1: -3350.7666016, 6893.2377930, -3199.9543457, 6559.5512695, -9907.1210938, 10088.2470703
2: -2744.1115723, 7343.0166016, -2620.7675781, 6990.0371094, -9733.7324219, 9959.7041016
3: -4957.5112305, 6977.7133789, -4734.2426758, 6646.6938477, -11598.8535156, 11707.1689453
4: -3279.4855957, 7851.1489258, -3132.7685547, 7476.1455078, -10755.0712891, 10978.7851562

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9278045, upper bound: 8224.9297632
time: 1.69 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9278045, upper bound: 8224.9297632
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3743.9780273, 8300.5537109, -3302.4838867, 7310.8564453, -11050.4531250, 11591.9169922
1: -3584.3615723, 7359.8784180, -3158.5742188, 6475.4736328, -10052.9697266, 10509.0332031
2: -2929.2373047, 7844.6650391, -2586.7548828, 6900.4570312, -9827.8369141, 10422.5644531
3: -5316.4858398, 7450.0253906, -4673.0629883, 6561.1342773, -11864.6660156, 12113.7734375
4: -3497.3256836, 8395.8642578, -3091.8854980, 7380.3115234, -10875.7421875, 11475.9365234

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9278045, upper bound: 8224.9297632
time: 1.05 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9278045, upper bound: 8224.9297632
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3226.2919922, 7140.0424805, -3226.2919922, 7140.0424805, -10361.9423828, 10361.9423828
1: -3086.6608887, 6325.5874023, -3086.6608887, 6325.5874023, -9407.5917969, 9407.5917969
2: -2526.6550293, 6744.1152344, -2526.6550293, 6744.1152344, -9267.5390625, 9267.5390625
3: -4573.7587891, 6408.3671875, -4573.7587891, 6408.3671875, -10976.1464844, 10976.1464844
4: -3019.3017578, 7218.6523438, -3019.3017578, 7218.6523438, -10234.0830078, 10234.0830078

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9388380, upper bound: 8224.9391171
time: 0.94 seconds

## Relational analysis of NS_A2_B2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8224.9387057, upper bound: 8224.9391290
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3377.5024414, 7466.7236328, -3226.2919922, 7140.0424805, -10513.2246094, 10688.4892578
1: -3226.3676758, 6619.1162109, -3086.6608887, 6325.5874023, -9547.4072266, 9700.9609375
2: -2645.4055176, 7052.1118164, -2526.6550293, 6744.1152344, -9386.5097656, 9575.4072266
3: -4769.1718750, 6702.6010742, -4573.7587891, 6408.3671875, -11171.6259766, 11270.3349609
4: -3161.0864258, 7541.7163086, -3019.3017578, 7218.6523438, -10375.9013672, 10557.1113281

Time for backsubstitution: 1.68 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.81 + 416.38 = 420.19 seconds
